"""Tests for `pipekit_array.reduce` — `MeanScalar`."""

from __future__ import annotations

import pytest
from pipekit import Operator
from pipekit.signature import Signature
from pipekit_array import MeanScalar


def test_mean_scalar_full_reduction(backend, make_array):
    x = make_array(backend, shape=(4, 8), seed=0)
    out = MeanScalar()(x)
    assert out.ndim == 0


def test_mean_scalar_axis_int(backend, make_array):
    x = make_array(backend, shape=(4, 8), seed=0)
    out = MeanScalar(axis=0)(x)
    assert out.shape == (8,)


def test_mean_scalar_axis_tuple(backend, make_array):
    x = make_array(backend, shape=(4, 8, 16), seed=0)
    out = MeanScalar(axis=(-2, -1))(x)
    assert out.shape == (4,)


def test_mean_scalar_keepdims(backend, make_array):
    x = make_array(backend, shape=(4, 8), seed=0)
    out = MeanScalar(axis=0, keepdims=True)(x)
    assert out.shape == (1, 8)


def test_mean_scalar_matches_native(backend, make_array):
    """The dispatch should produce the same value as the native backend."""
    x = make_array(backend, shape=(4, 8), seed=42)
    out = MeanScalar()(x)
    native_xp = x.__array_namespace__() if hasattr(x, "__array_namespace__") else None
    if native_xp is None:
        # torch path — verify via array_api_compat
        from pipekit_array._namespace import array_namespace

        native_xp = array_namespace(x)
    expected = native_xp.mean(x)
    assert float(out) == pytest.approx(float(expected))


def test_mean_scalar_get_config_round_trip():
    op = MeanScalar(axis=(0, 1), keepdims=True)
    config = op.get_config()
    assert config == {"axis": (0, 1), "keepdims": True}
    revived = Operator.from_state(op.state)
    assert isinstance(revived, MeanScalar)
    assert revived.axis == (0, 1)
    assert revived.keepdims is True


def test_mean_scalar_compute_output_signature_full_reduction():
    sig = Signature(dims=(("y", 256), ("x", 256)), dtype="float32")
    out_sig = MeanScalar().compute_output_signature(sig)
    assert out_sig is not None
    assert out_sig.dims == ()
    assert out_sig.dtype == "float32"


def test_mean_scalar_compute_output_signature_axis_int():
    sig = Signature(dims=(("band", 3), ("y", 256), ("x", 256)), dtype="float32")
    out_sig = MeanScalar(axis=0).compute_output_signature(sig)
    assert out_sig is not None
    assert out_sig.dims == (("y", 256), ("x", 256))


def test_mean_scalar_compute_output_signature_keepdims():
    sig = Signature(dims=(("band", 3), ("y", 256), ("x", 256)), dtype="float32")
    out_sig = MeanScalar(axis=0, keepdims=True).compute_output_signature(sig)
    assert out_sig is not None
    assert out_sig.dims == (("band", 1), ("y", 256), ("x", 256))


def test_mean_scalar_compute_output_signature_none_input():
    assert MeanScalar().compute_output_signature(None) is None


def test_mean_scalar_compute_output_signature_out_of_range_int_axis():
    sig = Signature(dims=(("y", 256), ("x", 256)), dtype="float32")
    with pytest.raises(ValueError, match="out of range"):
        MeanScalar(axis=3).compute_output_signature(sig)


def test_mean_scalar_compute_output_signature_out_of_range_negative_axis():
    sig = Signature(dims=(("band", 3), ("y", 256), ("x", 256)), dtype="float32")
    with pytest.raises(ValueError, match="out of range"):
        MeanScalar(axis=-4).compute_output_signature(sig)


def test_mean_scalar_compute_output_signature_out_of_range_tuple_axis():
    sig = Signature(dims=(("y", 256), ("x", 256)), dtype="float32")
    with pytest.raises(ValueError, match="out of range"):
        MeanScalar(axis=(0, 5)).compute_output_signature(sig)
