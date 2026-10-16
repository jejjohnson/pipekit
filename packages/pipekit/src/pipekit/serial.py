"""Group L — serialisation glue.

A small surface for JSON serialisation, plus a sandboxed loader for
user-uploaded pipelines that only constructs operators registered in
an allowlist.

- `dumps(op) -> str` — JSON-encode ``op.state``.
- `loads(s) -> Operator` — round-trip via `Operator.from_state`. Any
  imported `Operator` subclass resolves, at any depth: a `Sequential`,
  `Graph`, `Cache`, `Try`, `Fanout` or parallel map comes back with its
  children. Operators that hold closures (``forbid_in_yaml``) refuse.
- `register(cls)` — add an `Operator` subclass to the sandboxed
  loader's allowlist.
- `loads_sandboxed(s) -> Operator` — like ``loads`` but rejects any
  class, at any depth, not in the allowlist. Nothing is allowed by
  default, pipekit's own containers included: register `Sequential`,
  `Graph`, … as well as your steps. Use this for untrusted input
  (use case 6: user-uploaded pipelines).

YAML / Hydra loaders are intentionally not shipped here — they live
in optional extras (`pipekit[yaml]`, `pipekit[hydra]`).

See master plan Report 2, Group L.
"""

from __future__ import annotations

import json
from typing import Any

from pipekit._base.operator import Operator, build_operator


def dumps(op: Operator) -> str:
    """JSON-encode an operator's state record.

    Round-trip: ``loads(dumps(op))`` reconstructs an operator with the
    same config, nested operators included. Operators with
    `forbid_in_yaml = True` emit debug-only payloads (their constructors
    need closures that aren't in the config), so ``loads`` refuses them
    with `RuntimeError`, wherever they sit in the tree.

    Args:
        op: The operator to encode.

    Returns:
        A JSON string suitable for round-trip via `loads`.

    Raises:
        TypeError: if ``op`` isn't an `Operator`.
    """
    if not isinstance(op, Operator):
        raise TypeError(f"dumps expects an Operator, got {type(op).__name__}.")
    return json.dumps(op.state, sort_keys=True)


def loads(s: str) -> Operator:
    """Decode an operator from its JSON state record.

    Resolves the target class via `Operator.from_state`, walking all
    imported subclasses. The caller must have imported the module
    declaring the target class — pipekit doesn't auto-import.

    Args:
        s: A JSON string produced by `dumps` (or compatible).

    Raises:
        ValueError: if the JSON is malformed.
        TypeError: if a named class isn't a loaded `Operator` subclass.
        RuntimeError: if a class in the tree is ``forbid_in_yaml``, or a
            config holds a value that is not JSON.
    """
    state = json.loads(s)
    if not isinstance(state, dict):
        raise ValueError(
            f"loads expects a JSON object encoding an operator state, "
            f"got a {type(state).__name__}."
        )
    return Operator.from_state(state)


_ALLOWED: dict[str, type[Operator]] = {}


def register(cls: type[Operator]) -> type[Operator]:
    """Register ``cls`` as loadable via `loads_sandboxed`.

    Idempotent — re-registering the same class is a no-op. The class
    must subclass `Operator`.

    Returns ``cls`` so the function doubles as a class decorator::

        @register
        class MyOp(Operator):
            ...
    """
    if not (isinstance(cls, type) and issubclass(cls, Operator)):
        raise TypeError(f"register requires an Operator subclass, got {cls!r}.")
    key = f"{cls.__module__}.{cls.__name__}"
    _ALLOWED[key] = cls
    return cls


def unregister(cls: type[Operator]) -> None:
    """Remove ``cls`` from the sandboxed loader's allowlist."""
    key = f"{cls.__module__}.{cls.__name__}"
    _ALLOWED.pop(key, None)


def allowed() -> tuple[str, ...]:
    """Return the registered class names (``"module.Class"`` keys)."""
    return tuple(sorted(_ALLOWED))


def loads_sandboxed(s: str) -> Operator:
    """Like `loads`, but reject any class, at any depth, not in the allowlist.

    Use for untrusted input (user-uploaded pipelines, use case 6).
    Domain libraries register their operators via `register`; the
    sandboxed loader refuses to construct anything else. Every class in
    the tree is checked before any operator is built, so a refused
    pipeline constructs nothing. Exception classes (`Try.on`,
    `Retry.on`) resolve only among already-imported modules.

    Raises:
        PermissionError: if any class in the tree isn't registered.
        ValueError / TypeError / RuntimeError: as `loads`.
    """
    state = json.loads(s)
    if not isinstance(state, dict):
        raise ValueError(
            f"loads_sandboxed expects a JSON object, got a {type(state).__name__}."
        )
    module = state.get("module")
    cls = state.get("class")
    if not isinstance(module, str):
        raise ValueError(
            "loads_sandboxed: state must include a string `module` field, "
            f"got {type(module).__name__}."
        )
    if not isinstance(cls, str):
        raise ValueError(
            "loads_sandboxed: state must include a string `class` field, "
            f"got {type(cls).__name__}."
        )
    # Check every class in the tree first, so a refused pipeline
    # constructs nothing.
    _check_allowed(state)
    return build_operator(state, _resolve_allowed)


def _check_allowed(value: Any) -> None:
    """Raise `PermissionError` for the first unregistered class in ``value``."""
    if isinstance(value, list):
        for item in value:
            _check_allowed(item)
        return
    if not isinstance(value, dict):
        return
    if set(value) == {"module", "class", "config"}:
        _resolve_allowed(value["module"], value["class"])
    for item in value.values():
        _check_allowed(item)


def _resolve_allowed(module: Any, cls: Any) -> type[Operator]:
    key = f"{module}.{cls}"
    if key not in _ALLOWED:
        raise PermissionError(
            f"loads_sandboxed: {key!r} is not in the allowlist. "
            f"Registered classes: {allowed()}."
        )
    return _ALLOWED[key]
