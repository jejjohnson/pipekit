"""Which steps of a pipeline are jittable, and why the others are not.

`jit_report` runs a pipeline once, eagerly, and tries each step under a JAX
trace on the way. Nothing is compiled. An app calls ``report.require()`` at
start-up or in a test; everyone else never pays for the check.

JAX is imported inside `jit_report` only.
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Iterator
from types import ModuleType
from typing import Any

from pipekit_ops._base.graph import Graph, Node
from pipekit_ops._base.operator import Operator
from pipekit_ops._base.sequential import Sequential
from pipekit_ops.adapters import BackendError
from pipekit_ops.guards import guard


__all__ = ["JitReport", "JitStep", "jit_report"]


@dataclasses.dataclass(frozen=True)
class JitStep:
    """One step of a pipeline, and whether it runs under ``jax.jit``.

    Args:
        name: Where the step is and its class, e.g. ``"[1] Smooth"``; a step
            of a nested pipeline is ``"[1][0] Smooth"``.
        jittable: Whether it runs under a trace and keeps nothing of it.
        reason: Why not; ``None`` if it is jittable.
    """

    name: str
    jittable: bool
    reason: str | None = None

    def __str__(self) -> str:
        verdict = "jittable" if self.jittable else f"no ({self.reason})"
        return f"{self.name}: {verdict}"


@dataclasses.dataclass(frozen=True)
class JitReport:
    """The steps of a pipeline, in the order they run; printed one per line."""

    steps: tuple[JitStep, ...]

    @property
    def jittable(self) -> bool:
        """Whether every step is."""
        return all(step.jittable for step in self.steps)

    def require(self) -> None:
        """Do nothing if every step is jittable.

        Raises:
            BackendError: Naming every step that is not, and why.
        """
        failing = [str(step) for step in self.steps if not step.jittable]
        if failing:
            raise BackendError("not jittable: " + "; ".join(failing))

    def __str__(self) -> str:
        return "\n".join(map(str, self.steps))


def jit_report(op: Operator, *inputs: Any) -> JitReport:
    """Try each step of ``op`` under a JAX trace, and say which ones work.

    Algorithm:
        1. Walk ``op``: the operators of a `Sequential` and the nodes of a
           `Graph`, nested ones included. Any other operator is one step.
        2. A step that holds an operator with ``traceable = False`` is not
           jittable, and is not traced.
        3. Trace the step on its input with ``jax.eval_shape`` inside
           ``jax.checking_leaks``: what raises is not jittable, and neither is
           a step that stores a traced value.
        4. Run the step eagerly, for the next step's input.

    The steps are deep copies where Python can copy them, so ``op`` keeps no
    result and no traced value. A closure is not copied: a `Tap` calls its
    function twice, once with a traced value. A step that writes a file writes
    it. No `pipekit_ops.guard` reports the run.

    Args:
        op: A pipeline or a single operator.
        *inputs: What ``op`` is called with, by position (and the state, for a
            stateful pipeline).

    Raises:
        ImportError: JAX is not installed.
        Exception: Whatever the eager run raises.

    Example:
        >>> import pytest
        >>> jnp = pytest.importorskip("jax.numpy")
        >>> from pipekit_ops import Lambda
        >>> pipe = Lambda(lambda x: 2 * x) | Lambda(lambda x: x if x[0] > 0 else -x)
        >>> first, second = jit_report(pipe, jnp.ones(3)).steps
        >>> first.jittable, second.jittable
        (True, False)
    """
    try:
        import jax
    except ImportError as exc:
        raise ImportError("jit_report needs JAX: pip install jax") from exc
    steps: list[JitStep] = []
    with guard("off"):
        _walk(jax, op, inputs, "", steps)
    return JitReport(tuple(steps))


def _walk(
    jax: ModuleType,
    op: Operator,
    inputs: tuple[Any, ...],
    at: str,
    steps: list[JitStep],
) -> Any:
    """``op(*inputs)``, with a `JitStep` added for each step inside it."""
    if isinstance(op, Sequential) and op.operators:
        return _walk_sequential(jax, op, inputs, at, steps)
    if isinstance(op, Graph):
        return _walk_graph(jax, op, inputs, at, steps)
    name = f"{at} {type(op).__name__}".strip()
    reason = _why_not(jax, op, inputs)
    steps.append(JitStep(name, reason is None, reason))
    return _copied(op)(*inputs)


def _walk_sequential(
    jax: ModuleType,
    op: Sequential,
    inputs: tuple[Any, ...],
    at: str,
    steps: list[JitStep],
) -> Any:
    """The steps left to right, threading the state as `Sequential` does."""
    stateful = len(inputs) == 2
    out, state = inputs if stateful else (inputs, None)
    for i, step in enumerate(op.operators):
        where = f"{at}[{i}]"
        if not stateful:
            # the first step takes every input, which may be none
            args = out if i == 0 else (out,)
            out = _walk(jax, step, args, where, steps)
        elif getattr(step, "_is_stateful", False):
            out, state = _walk(jax, step, (out, state), where, steps)
        else:
            out = _walk(jax, step, (out,), where, steps)
    return (out, state) if stateful else out


def _walk_graph(
    jax: ModuleType, op: Graph, inputs: tuple[Any, ...], at: str, steps: list[JitStep]
) -> Any:
    """The nodes in the order `Graph` runs them."""
    stateful = len(inputs) == len(op.inputs) + 1
    values = inputs[:-1] if stateful else inputs
    state = inputs[-1] if stateful else None
    cache = dict(zip(map(id, op.inputs.values()), values, strict=True))
    for i, node in enumerate(op._order):
        assert node.operator is not None  # an Input is not in the order
        args = tuple(cache[id(parent)] for parent in node.parents)
        where = f"{at}[{i}]"
        if stateful and getattr(node.operator, "_is_stateful", False):
            out, state = _walk(jax, node.operator, (*args, state), where, steps)
        else:
            out = _walk(jax, node.operator, args, where, steps)
        cache[id(node)] = out
    outputs = {name: cache[id(node)] for name, node in op.outputs.items()}
    return (outputs, state) if stateful else outputs


def _why_not(jax: ModuleType, op: Operator, inputs: tuple[Any, ...]) -> str | None:
    """Why ``op`` does not run under a trace, or ``None``."""
    declared = next((o for o in _operators(op) if not o.traceable), None)
    if declared is not None:
        return f"{type(declared).__name__} declares traceable = False"
    traced = _copied(op)
    try:
        with jax.checking_leaks():
            jax.eval_shape(lambda *xs: traced(*xs), *inputs)
    except Exception as exc:
        first = str(exc).strip().partition("\n")[0]
        if first.startswith("Leaked"):
            return "it stores a traced value"
        return f"{type(exc).__name__}: {first}"
    return None


def _copied(op: Operator) -> Operator:
    """A deep copy of ``op``, or ``op`` itself if it holds what cannot be copied."""
    try:
        return copy.deepcopy(op)
    except Exception:
        return op


def _operators(op: Operator) -> Iterator[Operator]:
    """``op`` and every operator it holds, in its attributes and their containers."""
    seen: set[int] = set()
    stack: list[Any] = [op]
    while stack:
        value = stack.pop()
        if isinstance(value, (Operator, Node)):
            if id(value) in seen:
                continue
            seen.add(id(value))
            if isinstance(value, Operator):
                yield value
            stack.extend(vars(value).values())
        elif isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, (list, tuple, set, frozenset)):
            stack.extend(value)
