"""Tests for `pipekit.signature` — Group I."""

from __future__ import annotations

import pytest
from pipekit import Operator, Sequential, Signature
from pipekit.signature import compute_output_signature


def test_signature_format():
    sig = Signature(dims=(("time", 12), ("y", 256), ("x", 256)), dtype="float32")
    assert sig.format() == "(time=12, y=256, x=256) float32"


def test_signature_format_unknown_sizes():
    sig = Signature(dims=(("batch", None), ("c", 3)), dtype=None)
    assert sig.format() == "(batch=?, c=3) ?"


def test_signature_drop_dims():
    sig = Signature(dims=(("t", 5), ("y", 4), ("x", 3)), dtype="i4")
    dropped = sig.drop_dims(("t",))
    assert dropped == Signature(dims=(("y", 4), ("x", 3)), dtype="i4")


def test_signature_drop_dims_unknown_raises():
    sig = Signature(dims=(("y", 4),))
    with pytest.raises(KeyError):
        sig.drop_dims(("nope",))


def test_signature_replace_dims():
    sig = Signature(dims=(("y", 4), ("x", 3)), dtype="f4")
    new = sig.replace_dims({"x": 128})
    assert new == Signature(dims=(("y", 4), ("x", 128)), dtype="f4")


def test_signature_replace_dims_unknown_raises():
    sig = Signature(dims=(("y", 4),))
    with pytest.raises(KeyError):
        sig.replace_dims({"nope": 1})


def test_signature_rejects_duplicate_dim_names():
    with pytest.raises(ValueError):
        Signature(dims=(("y", 4), ("y", 5)))


def test_signature_is_immutable():
    sig = Signature(dims=(("y", 4),))
    with pytest.raises((AttributeError, Exception)):  # FrozenInstanceError
        sig.dtype = "f4"  # type: ignore[misc]


def test_operator_default_compute_output_signature_passes_through(Scale_cls):
    sig = Signature(dims=(("x", 3),), dtype="f4")
    op = Scale_cls(2.0)
    assert op.compute_output_signature(sig) is sig


def test_sequential_threads_signature_through_passthrough_ops(Scale_cls):
    pipe = Sequential([Scale_cls(2.0), Scale_cls(3.0)])
    sig = Signature(dims=(("x", 3),), dtype="f4")
    assert pipe.compute_output_signature(sig) == sig


def test_sequential_summary_renders_table(Scale_cls):
    pipe = Sequential([Scale_cls(2.0), Scale_cls(3.0)])
    table = pipe.summary(Signature(dims=(("x", 3),), dtype="f4"))
    assert "Scale" in table
    assert "x=3" in table


def test_operator_overrides_compute_output_signature():
    class DropTime(Operator):
        def _apply(self, x):
            return x

        def compute_output_signature(self, sig):
            if sig is None:
                return None
            return sig.drop_dims(("time",))

    sig = Signature(dims=(("time", 5), ("y", 4)), dtype="f4")
    out = DropTime().compute_output_signature(sig)
    assert out == Signature(dims=(("y", 4),), dtype="f4")


def test_free_function_dispatch(Scale_cls):
    sig = Signature(dims=(("x", 3),))
    assert compute_output_signature(Scale_cls(2.0), sig) is sig
