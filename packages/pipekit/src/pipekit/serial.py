"""Group L — serialisation glue.

A small surface for JSON serialisation, plus a sandboxed loader for
user-uploaded pipelines that only constructs operators registered in
an allowlist.

- `dumps(op) -> str` — JSON-encode ``op.state``.
- `loads(s) -> Operator` — round-trip via `Operator.from_state`. Any
  imported `Operator` subclass resolves; non-primitive config values
  raise (use a YAML / Hydra loader for nested-operator configs).
- `register(cls)` — add an `Operator` subclass to the sandboxed
  loader's allowlist.
- `loads_sandboxed(s) -> Operator` — like ``loads`` but rejects any
  class not in the allowlist. Use this for untrusted input
  (use case 6: user-uploaded pipelines).

YAML / Hydra loaders are intentionally not shipped here — they live
in optional extras (`pipekit[yaml]`, `pipekit[hydra]`).

See master plan Report 2, Group L.
"""

from __future__ import annotations

import json

from pipekit._base.operator import Operator


def dumps(op: Operator) -> str:
    """JSON-encode an operator's state record.

    Round-trip: ``loads(dumps(op))`` reconstructs an operator with the
    same config. Operators with `forbid_in_yaml = True` emit
    debug-only payloads — their constructors typically need closure
    arguments that aren't in the config, so the round-trip raises
    `TypeError` from the constructor at ``loads`` time.

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
        TypeError: if the named class isn't a loaded `Operator` subclass.
        RuntimeError: if the config contains non-primitive values.
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
    """Like `loads`, but reject any class not in the allowlist.

    Use for untrusted input (user-uploaded pipelines, use case 6).
    Domain libraries register their operators via `register`; the
    sandboxed loader refuses to construct anything else.

    Raises:
        PermissionError: if the named class isn't registered.
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
    key = f"{module}.{cls}"
    if key not in _ALLOWED:
        raise PermissionError(
            f"loads_sandboxed: {key!r} is not in the allowlist. "
            f"Registered classes: {allowed()}."
        )
    return Operator.from_state(state)
