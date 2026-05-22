"""Tests for `pipekit.qc` — Group H."""

from __future__ import annotations

import pytest
from pipekit import (
    AssertCallable,
    AssertDType,
    AssertHasAttribute,
    AssertShape,
    QCError,
    Quarantine,
)


class FakeArray:
    """Tiny duck-typed stand-in for an array with shape/dtype/attrs."""

    def __init__(self, shape, dtype="float32", **attrs):
        self.shape = shape
        self.dtype = dtype
        for k, v in attrs.items():
            setattr(self, k, v)


def test_quarantine_passes_through_when_check_passes():
    op = Quarantine(check=lambda x: True, sentinel=None)
    assert op(42) == 42


def test_quarantine_returns_sentinel_when_check_fails():
    op = Quarantine(check=lambda x: False, sentinel="bad")
    assert op(42) == "bad"


def test_quarantine_fires_on_quarantine_callback():
    seen = []
    op = Quarantine(
        check=lambda x: x > 0,
        sentinel=None,
        on_quarantine=lambda x: seen.append(x),
    )
    op(-1)
    op(5)
    assert seen == [-1]


def test_assert_shape_pass():
    arr = FakeArray(shape=(3, 256, 256))
    op = AssertShape((3, 256, 256))
    assert op(arr) is arr


def test_assert_shape_wildcard_pass():
    arr = FakeArray(shape=(8, 3, 256, 256))
    op = AssertShape((None, 3, 256, 256))
    assert op(arr) is arr


def test_assert_shape_mismatch():
    arr = FakeArray(shape=(3, 128, 128))
    op = AssertShape((3, 256, 256))
    with pytest.raises(QCError, match="axis 1"):
        op(arr)


def test_assert_shape_wrong_rank():
    arr = FakeArray(shape=(3, 256, 256))
    op = AssertShape((3, 256))
    with pytest.raises(QCError, match="rank"):
        op(arr)


def test_assert_shape_no_shape_attribute():
    op = AssertShape((3,))
    with pytest.raises(QCError, match="no `shape`"):
        op(42)


def test_assert_dtype_pass():
    arr = FakeArray(shape=(3,), dtype="float32")
    assert AssertDType("float32")(arr) is arr


def test_assert_dtype_mismatch():
    arr = FakeArray(shape=(3,), dtype="float64")
    with pytest.raises(QCError):
        AssertDType("float32")(arr)


def test_assert_dtype_compares_via_str():
    """Numpy dtypes stringify to e.g. 'float32' — we compare via str()."""

    class Dt:
        def __str__(self):
            return "float32"

    arr = FakeArray(shape=(3,), dtype=Dt())
    assert AssertDType("float32")(arr) is arr


def test_assert_has_attribute_presence():
    arr = FakeArray(shape=(3,), name="x")
    assert AssertHasAttribute("name")(arr) is arr


def test_assert_has_attribute_missing():
    arr = FakeArray(shape=(3,))
    with pytest.raises(QCError, match="name"):
        AssertHasAttribute("name")(arr)


def test_assert_has_attribute_value_match():
    arr = FakeArray(shape=(3,), name="x")
    assert AssertHasAttribute("name", value="x")(arr) is arr


def test_assert_has_attribute_value_mismatch():
    arr = FakeArray(shape=(3,), name="y")
    with pytest.raises(QCError):
        AssertHasAttribute("name", value="x")(arr)


def test_assert_callable_pass():
    op = AssertCallable(lambda x: x > 0, message="must be positive")
    assert op(5) == 5


def test_assert_callable_fail():
    op = AssertCallable(lambda x: x > 0, message="must be positive")
    with pytest.raises(QCError, match="must be positive"):
        op(-1)


def test_qc_error_is_assertion_error():
    assert issubclass(QCError, AssertionError)
