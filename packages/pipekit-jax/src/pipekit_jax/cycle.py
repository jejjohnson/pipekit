"""`ScanCycle` and `Rollout` — ``jax.lax.scan``-based time stepping.

`pipekit_cycle.Cycle` is carrier-agnostic and steps in a Python ``for``
loop. Under ``jax.jit`` that loop unrolls: the traced program, compile
time and memory all grow linearly with ``n_steps``, and
``save_history`` appends to ``self.history`` from inside traced code.
That is the right tool for host-side stepping (chunked runs with
monitors between chunks) and the wrong one for anything differentiated
end to end — 4DVar windows, rollout training.

- `ScanCycle` — the JAX specialisation of `Cycle`: same
  ``(carrier, state) -> (carrier, state)`` contract and the same
  history convention, one ``lax.scan`` regardless of ``n_steps``, and
  the history is *returned* by `ScanCycle.run`, never stored.
- `Rollout` — the carrier-only form most PDE use needs:
  ``carrier -> trajectory`` sampled every ``save_every`` steps.

Both accept a pipekit operator or a `pipekit_cycle.ForwardModel` as the
step, are `JaxOperator` s (so learnable leaves of the step are reachable
by ``eqx.filter_grad``), and handle arbitrary pytree carriers — dicts,
``eqx.Module`` states, custom carriers — by stacking with
``jax.tree.map``.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from typing import Any, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from pipekit import Operator, StatefulOperator
from pipekit._base.operator import nested_config
from pipekit_cycle.protocols import ForwardModel

from pipekit_jax.operator import JaxOperator


class _ForwardModelStep(StatefulOperator, JaxOperator):
    """A `ForwardModel` as a `StatefulOperator`: ``(c, s) -> (step(c, dt), s)``.

    The JAX counterpart of ``pipekit_cycle.cycle._ForwardModelAdapter``:
    the model is a pytree field, so its leaves stay visible to
    ``eqx.filter_grad`` through the cycle.
    """

    # The model is arbitrary user code (callables, diffrax vector fields):
    # flag it so `pipekit.check_pickleable` reports it, as
    # `pipekit_cycle`'s adapter does.
    forbid_in_yaml: ClassVar[bool] = True

    model: Any

    def _apply(self, carrier: Any, state: Any) -> tuple[Any, Any]:
        return self.model.step(carrier, self.model.dt), state

    def get_config(self) -> dict[str, Any]:
        return {"model_class": type(self.model).__name__}


class _ForwardModelCarrierStep(JaxOperator):
    """A `ForwardModel` as a carrier-only operator: ``c -> step(c, dt)``."""

    forbid_in_yaml: ClassVar[bool] = True

    model: Any

    def _apply(self, carrier: Any) -> Any:
        return self.model.step(carrier, self.model.dt)

    def get_config(self) -> dict[str, Any]:
        return {"model_class": type(self.model).__name__}


def _coerce_step(step_op: Any) -> StatefulOperator:
    """Return ``step_op`` as a `StatefulOperator` (wrapping a `ForwardModel`)."""
    if isinstance(step_op, StatefulOperator):
        return step_op
    if isinstance(step_op, ForwardModel):
        return _ForwardModelStep(step_op)
    raise TypeError(
        "step_op must be a StatefulOperator or a ForwardModel, "
        f"got {type(step_op).__name__}."
    )


def _coerce_carrier_step(step: Any) -> Operator:
    """Return ``step`` as a carrier-only operator (wrapping a `ForwardModel`)."""
    if isinstance(step, StatefulOperator):
        # A stateful step needs (carrier, state); Rollout calls step(carrier).
        raise TypeError(
            f"Rollout steps carrier -> carrier, but {type(step).__name__} is a "
            "StatefulOperator (carrier, state) -> (carrier, state). Use "
            "ScanCycle for stateful steps."
        )
    if isinstance(step, ForwardModel):
        return _ForwardModelCarrierStep(step)
    if isinstance(step, Operator):
        return step
    raise TypeError(
        f"step must be a pipekit Operator or a ForwardModel, got {type(step).__name__}."
    )


def _empty_history(x: Any) -> Any:
    """A length-0 stack of ``x``, keeping its dtype."""
    x = jnp.asarray(x)
    return jnp.zeros((0, *x.shape), dtype=x.dtype)


def _scan_steps(
    step: Callable[[Any, Any], Any],
    carry: Any,
    start: int,
    length: int,
    chunk: int | None = None,
) -> Any:
    """Apply ``step(carry, i)`` for global indices ``start … start+length-1``.

    With ``chunk``, steps run in checkpointed chunks of ``chunk`` steps:
    reverse mode stores the carry only at chunk boundaries and recomputes
    the steps inside a chunk, so a length-``n`` scan keeps
    ``n / chunk + chunk`` carries instead of ``n``.
    """
    if length == 0:
        return carry

    def body(c: Any, i: Any) -> tuple[Any, None]:
        return step(c, i), None

    if chunk is None or chunk >= length:
        carry, _ = jax.lax.scan(body, carry, start + jnp.arange(length))
        return carry

    n_chunks, rem = divmod(length, chunk)

    @jax.checkpoint
    def run_chunk(c: Any, j: Any) -> tuple[Any, None]:
        c, _ = jax.lax.scan(body, c, start + j * chunk + jnp.arange(chunk))
        return c, None

    carry, _ = jax.lax.scan(run_chunk, carry, jnp.arange(n_chunks))
    return _scan_steps(step, carry, start + n_chunks * chunk, rem)


def _stack(trees: list[Any]) -> Any:
    return jax.tree.map(lambda *xs: jnp.stack(xs), *trees)


class ScanCycle(StatefulOperator, JaxOperator):
    """Apply ``step_op`` for ``n_steps`` inside one ``jax.lax.scan``.

    The jit-native counterpart of `pipekit_cycle.Cycle`.

    .. code-block:: text

        (cᵢ₊₁, sᵢ₊₁) = step_op(cᵢ, sᵢ),   i = 0 … n_steps - 1

    Calling the operator returns ``(c_n, s_n)`` exactly like ``Cycle``.
    `run` additionally returns the history, recorded with ``Cycle``'s
    convention — after step ``i + 1`` for every ``i`` with
    ``i % history_stride == 0`` — so there are ``⌈n_steps / stride⌉``
    entries.

    Attributes:
        step_op: A `StatefulOperator` ``(carrier, state) -> (carrier,
            state)``, or a `pipekit_cycle.ForwardModel` (wrapped so the
            state passes through unchanged).
        n_steps: Number of steps (``>= 0``; ``0`` is a no-op).
        history_stride: Record every ``stride``-th step in `run`.
        checkpoint: Rematerialise on the backward pass. The forward pass
            keeps the carry only at chunk boundaries — every ``⌈√n⌉``
            steps when called, every ``history_stride`` steps in `run` —
            and reverse mode recomputes the steps inside a chunk. Memory
            for ``grad`` drops from O(n) carries to O(√n) (or
            O(n / stride + stride)) at the cost of one extra forward pass.
        truncate: If set to ``k``, gradients flow through the carry of
            the trailing ``k`` steps only (the carry leaving step ``i``
            is ``stop_gradient``-ed when ``i < n_steps - k``), matching
            `pipekit_jax.truncated_scan`: history entries recorded by
            `run` keep the gradient of their own step, as scan outputs
            do. Forward values are unchanged.

    Raises:
        TypeError: If ``step_op`` is neither a `StatefulOperator` nor a
            `ForwardModel`.
        ValueError: If ``n_steps < 0``, ``history_stride < 1`` or
            ``truncate < 1``.

    Example:
        >>> cycle = ScanCycle(step_op=model_op, n_steps=1000, history_stride=100)
        >>> carrier, state = eqx.filter_jit(cycle)(c0, s0)
        >>> carrier, state, (carriers, states) = eqx.filter_jit(cycle.run)(c0, s0)
    """

    step_op: Any = eqx.field(converter=_coerce_step)
    n_steps: int = eqx.field(static=True, default=1)
    history_stride: int = eqx.field(static=True, default=1)
    checkpoint: bool = eqx.field(static=True, default=False)
    truncate: int | None = eqx.field(static=True, default=None)

    def __check_init__(self) -> None:
        if self.n_steps < 0:
            raise ValueError(f"ScanCycle.n_steps must be >= 0, got {self.n_steps}.")
        if self.history_stride < 1:
            raise ValueError(
                f"ScanCycle.history_stride must be >= 1, got {self.history_stride}."
            )
        if self.truncate is not None and self.truncate < 1:
            raise ValueError(f"ScanCycle.truncate must be >= 1, got {self.truncate}.")

    def _cut(self, carry: tuple[Any, Any], i: Any) -> tuple[Any, Any]:
        """Stop gradients through the carry leaving step ``i`` (``truncate``)."""
        if self.truncate is None:
            return carry
        cut = i < self.n_steps - self.truncate
        return jax.tree.map(
            lambda x: jnp.where(cut, jax.lax.stop_gradient(x), x), carry
        )

    def _step(self, carry: tuple[Any, Any], i: Any) -> tuple[Any, Any]:
        return self._cut(self.step_op(*carry), i)

    def _apply(self, carrier: Any, state: Any) -> tuple[Any, Any]:
        chunk = max(1, math.isqrt(self.n_steps)) if self.checkpoint else None
        return _scan_steps(self._step, (carrier, state), 0, self.n_steps, chunk)

    def run(self, carrier: Any, state: Any) -> tuple[Any, Any, Any]:
        """Run the cycle and return ``(carrier, state, (carriers, states))``.

        ``carriers`` / ``states`` stack the recorded entries along a new
        leading axis of length ``⌈n_steps / history_stride⌉`` (zero when
        ``n_steps == 0``).
        """
        step, k, n = self._step, self.history_stride, self.n_steps
        carry = (carrier, state)
        if n == 0:
            return carrier, state, jax.tree.map(_empty_history, carry)

        def body(c: Any, i: Any) -> tuple[Any, None]:
            return step(c, i), None

        def group(c: Any, j: Any) -> tuple[Any, Any]:
            # one recorded step (global index j·k), then k - 1 unrecorded ones.
            # Record *before* the truncation cut, like truncated_scan's
            # outputs: an entry keeps the gradient of its own step even
            # when the carry it passes on is cut.
            recorded = self.step_op(*c)
            c = self._cut(recorded, j * k)
            if k > 1:
                c, _ = jax.lax.scan(body, c, j * k + jnp.arange(1, k))
            return c, recorded

        if self.checkpoint:  # keep carries only at group boundaries
            group = jax.checkpoint(group)

        full, rem = divmod(n, k)
        history = []
        if full:
            carry, hist = jax.lax.scan(group, carry, jnp.arange(full))
            history.append(hist)
        if rem:
            recorded = self.step_op(*carry)
            history.append(_stack([recorded]))
            carry = self._cut(recorded, full * k)
            carry = _scan_steps(step, carry, full * k + 1, rem - 1)
        hist = (
            history[0]
            if len(history) == 1
            else jax.tree.map(lambda *xs: jnp.concatenate(xs), *history)
        )
        return carry[0], carry[1], hist

    def get_config(self) -> dict[str, Any]:
        return {
            "step_op": nested_config(self.step_op),
            "n_steps": self.n_steps,
            "history_stride": self.history_stride,
            "checkpoint": self.checkpoint,
            "truncate": self.truncate,
        }


class _ReadOnlyState(StatefulOperator, JaxOperator):
    """Run ``inner`` but hand the *incoming* state to the next step.

    ``(c, s) -> (inner(c, s)[0], s)``: whatever state the step returns is
    dropped, so every step of the cycle reads the same shared state.
    """

    inner: Any

    def _apply(self, carrier: Any, state: Any) -> tuple[Any, Any]:
        return self.inner(carrier, state)[0], state

    def get_config(self) -> dict[str, Any]:
        return {"inner": nested_config(self.inner)}


class EnsembleScanCycle(StatefulOperator, JaxOperator):
    """Run a `ScanCycle` on every ensemble member at once, with ``jax.vmap``.

    The carrier is a pytree whose leaves share a leading member axis of
    size ``M``; one compiled program advances all members.

    .. code-block:: text

        cⱼ ← cycle(cⱼ, s),    j = 1 … M   (in parallel)

    State handling (``state_axis``):

    - ``None`` (default) — one state shared by all members and treated
      as **read-only**: every step of every member reads the incoming
      state; the state a step returns is discarded (also between the
      steps of the cycle), and the incoming state is returned unchanged.
    - ``0`` — one state per member, stacked on a leading axis like the
      carrier; each member threads its own state.

    Not a drop-in for `pipekit_cycle.EnsembleCycle` when the step
    *writes* shared state: ``EnsembleCycle`` threads one state through
    the members sequentially (member ``j``'s output state feeds member
    ``j + 1``), which a ``vmap`` cannot reproduce. Use ``state_axis=0``
    or ``EnsembleCycle`` for that.

    Attributes:
        cycle: The per-member `ScanCycle`.
        state_axis: ``None`` (shared, read-only) or ``0`` (per member).

    Raises:
        ValueError: If ``state_axis`` is not ``None`` or ``0``.
    """

    cycle: ScanCycle
    state_axis: int | None = eqx.field(static=True, default=None)

    def __check_init__(self) -> None:
        if self.state_axis not in (None, 0):
            raise ValueError(
                "EnsembleScanCycle.state_axis must be None or 0, "
                f"got {self.state_axis}."
            )

    def _read_only_cycle(self) -> ScanCycle:
        """The cycle with its step wrapped so state never changes between steps."""
        return dataclasses.replace(
            self.cycle, step_op=_ReadOnlyState(self.cycle.step_op)
        )

    def _apply(self, carrier: Any, state: Any) -> tuple[Any, Any]:
        # carrier: every leaf has a leading member axis [M, ...]
        if self.state_axis is None:
            cycle = self._read_only_cycle()
            out = jax.vmap(lambda c: cycle(c, state)[0])(carrier)
            return out, state
        return jax.vmap(self.cycle)(carrier, state)

    def run(self, carrier: Any, state: Any) -> tuple[Any, Any, Any]:
        """Like `ScanCycle.run`, per member: histories gain a leading ``M`` axis.

        Returns ``(members, state, (carriers, states))`` with
        ``carriers`` shaped ``[M, n_records, …]``.
        """
        if self.state_axis is None:
            cycle = self._read_only_cycle()
            c, _, hist = jax.vmap(lambda m: cycle.run(m, state))(carrier)
            return c, state, hist
        return jax.vmap(self.cycle.run)(carrier, state)

    def get_config(self) -> dict[str, Any]:
        return {"cycle": nested_config(self.cycle), "state_axis": self.state_axis}


class Rollout(JaxOperator):
    """Carrier-only rollout: ``c₀ -> [c₀, c_k, c_2k, …, c_n]``.

    .. code-block:: text

        cᵢ₊₁ = step(cᵢ),   i = 0 … n_steps - 1
        trajectory = (c₀?, c_k, c_2k, …, c_n),   k = save_every

    The trajectory is stacked along a new leading axis with
    ``jax.tree.map``, so any pytree carrier works.

    Attributes:
        step: A pipekit operator ``carrier -> carrier``, or a
            `pipekit_cycle.ForwardModel` (wrapped to call ``step(c, dt)``;
            the wrapper is flagged ``forbid_in_yaml`` so
            `pipekit.check_pickleable` reports it).
        n_steps: Number of steps; must be a multiple of ``save_every``.
        save_every: Record every ``save_every``-th state.
        include_initial: Prepend ``c₀`` to the trajectory.
        checkpoint: Rematerialise each ``save_every`` block on the
            backward pass, so ``grad`` stores carries only at save points
            (which the trajectory keeps anyway) — O(n / save_every +
            save_every) instead of O(n).

    Raises:
        TypeError: If ``step`` is a `StatefulOperator` (use `ScanCycle`),
            or neither an operator nor a `ForwardModel`.
        ValueError: If ``n_steps < 0``, ``save_every < 1``, or
            ``n_steps`` is not a multiple of ``save_every``.
    """

    step: Any = eqx.field(converter=_coerce_carrier_step)
    n_steps: int = eqx.field(static=True, default=1)
    save_every: int = eqx.field(static=True, default=1)
    include_initial: bool = eqx.field(static=True, default=True)
    checkpoint: bool = eqx.field(static=True, default=False)

    def __check_init__(self) -> None:
        if self.n_steps < 0:
            raise ValueError(f"Rollout.n_steps must be >= 0, got {self.n_steps}.")
        if self.save_every < 1:
            raise ValueError(f"Rollout.save_every must be >= 1, got {self.save_every}.")
        if self.n_steps % self.save_every:
            raise ValueError(
                f"Rollout.n_steps ({self.n_steps}) must be a multiple of "
                f"save_every ({self.save_every})."
            )

    def _apply(self, carrier: Any) -> Any:
        def inner(c: Any, _: Any) -> tuple[Any, None]:
            return self.step(c), None

        def outer(c: Any, _: Any) -> tuple[Any, Any]:
            c, _ = jax.lax.scan(inner, c, None, length=self.save_every)
            return c, c

        if self.checkpoint:  # keep carries only at save points
            outer = jax.checkpoint(outer)

        n_saves = self.n_steps // self.save_every
        if n_saves:
            _, traj = jax.lax.scan(outer, carrier, None, length=n_saves)
        else:
            traj = jax.tree.map(_empty_history, carrier)
        if not self.include_initial:
            return traj
        return jax.tree.map(
            lambda c0, t: jnp.concatenate([jnp.asarray(c0)[None], t]), carrier, traj
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "step": nested_config(self.step),
            "n_steps": self.n_steps,
            "save_every": self.save_every,
            "include_initial": self.include_initial,
            "checkpoint": self.checkpoint,
        }
