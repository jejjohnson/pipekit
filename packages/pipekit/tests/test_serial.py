"""Tests for `pipekit.serial` — Group L."""

from __future__ import annotations

import json

import pytest
from pipekit import (
    Const,
    Identity,
    Lambda,
    Operator,
)
from pipekit.serial import (
    allowed,
    dumps,
    loads,
    loads_sandboxed,
    register,
    unregister,
)


def test_dumps_loads_round_trip_identity():
    s = dumps(Identity())
    op = loads(s)
    assert isinstance(op, Identity)
    assert op(42) == 42


def test_dumps_loads_round_trip_const():
    s = dumps(Const(7))
    op = loads(s)
    assert isinstance(op, Const)
    assert op() == 7


def test_dumps_emits_json():
    s = dumps(Const(1))
    payload = json.loads(s)
    assert payload["class"] == "Const"
    assert payload["config"] == {"value": 1}


def test_dumps_rejects_non_operator():
    with pytest.raises(TypeError):
        dumps(42)  # type: ignore[arg-type]


def test_loads_rejects_non_object_json():
    with pytest.raises(ValueError):
        loads("[1, 2, 3]")


def test_loads_with_closure_config_raises():
    """Lambda emits a debug-only config — loads can't reconstruct it."""
    s = dumps(Lambda(lambda x: x, name="noop"))
    # Class loads, but constructor needs `fn` — absent from debug config.
    with pytest.raises(TypeError):
        loads(s)


def test_register_adds_to_allowlist():
    @register
    class MyOp(Operator):
        def _apply(self, x):
            return x

    try:
        assert any("MyOp" in n for n in allowed())
        # Sandboxed loader accepts it now.
        s = dumps(MyOp())
        op = loads_sandboxed(s)
        assert isinstance(op, MyOp)
    finally:
        unregister(MyOp)


def test_loads_sandboxed_rejects_unregistered():
    s = dumps(Identity())  # Identity is not registered by default
    with pytest.raises(PermissionError):
        loads_sandboxed(s)


def test_loads_sandboxed_validates_json_shape():
    with pytest.raises(ValueError):
        loads_sandboxed("null")


def test_register_rejects_non_operator_class():
    with pytest.raises(TypeError):
        register(int)  # type: ignore[arg-type]


def test_unregister_is_idempotent():
    class T(Operator):
        def _apply(self, x):
            return x

    unregister(T)  # not registered — should not raise


def test_loads_sandboxed_rejects_missing_module_field():
    """Malformed state without a `module` field raises ValueError, not Permission."""
    import json as _json

    bogus = _json.dumps({"class": "Identity", "config": {}})
    with pytest.raises(ValueError, match="module"):
        loads_sandboxed(bogus)


def test_loads_sandboxed_rejects_missing_class_field():
    """Malformed state without a `class` field raises ValueError, not Permission."""
    import json as _json

    bogus = _json.dumps({"module": "pipekit.blocks", "config": {}})
    with pytest.raises(ValueError, match="class"):
        loads_sandboxed(bogus)


def test_loads_sandboxed_rejects_non_string_module():
    """A non-string `module` field is a shape error, not a permission error."""
    import json as _json

    bogus = _json.dumps({"module": 42, "class": "Identity", "config": {}})
    with pytest.raises(ValueError, match="module"):
        loads_sandboxed(bogus)
