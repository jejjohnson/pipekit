"""Tests for `pipekit.observe` — Group E."""

from __future__ import annotations

from pipekit import (
    Histogram,
    Profile,
    Sequential,
    ShapeTrace,
    Snapshot,
    Tap,
)


def test_tap_passes_through():
    seen = []
    op = Tap(seen.append, name="capture")
    assert op(42) == 42
    assert seen == [42]


def test_tap_uses_callable_name_default():
    def my_logger(x):
        return None

    assert Tap(my_logger).name == "my_logger"


def test_tap_is_forbid_in_yaml():
    assert Tap.forbid_in_yaml is True


def test_tap_repr():
    assert repr(Tap(lambda x: None, name="log")) == "Tap(name='log')"


def test_snapshot_captures_intermediate_values(Scale_cls, AddConst_cls):
    snap = Snapshot()
    pipe = Sequential(
        [snap.at("pre"), Scale_cls(2.0), snap.at("mid"), AddConst_cls(1.0)]
    )
    out = pipe(3.0)
    assert out == 7.0
    assert snap.captures == {"pre": 3.0, "mid": 6.0}


def test_snapshot_clear_drops_captures():
    snap = Snapshot()
    snap.at("k")(1)
    snap.clear()
    assert snap.captures == {}


def test_shape_trace_every_mode():
    lines: list[str] = []

    class Arr:
        def __init__(self, shape, dtype):
            self.shape = shape
            self.dtype = dtype

    op = ShapeTrace(printer=lines.append, mode="every")
    op(Arr((3, 256), "float32"))
    op(Arr((3, 256), "float32"))
    assert len(lines) == 2


def test_shape_trace_diff_only_mode():
    lines: list[str] = []

    class Arr:
        def __init__(self, shape, dtype):
            self.shape = shape
            self.dtype = dtype

    op = ShapeTrace(printer=lines.append, mode="diff_only")
    op(Arr((3, 256), "float32"))
    op(Arr((3, 256), "float32"))  # no change → no print
    op(Arr((3, 128), "float32"))  # different → print
    assert len(lines) == 2


def test_shape_trace_handles_missing_shape():
    lines: list[str] = []
    op = ShapeTrace(printer=lines.append, mode="every")
    op(42)  # no shape attr — None reported
    assert "shape=None" in lines[0]


def test_shape_trace_label_in_output():
    lines: list[str] = []

    class Arr:
        shape = (1,)
        dtype = "i4"

    op = ShapeTrace(printer=lines.append, mode="every", label="post")
    op(Arr())
    assert lines[0].startswith("[post]")


def test_profile_records_per_call_timings(Scale_cls):
    prof = Profile()
    pipe = Sequential([prof.wrap(Scale_cls(2.0))])
    pipe(1.0)
    pipe(2.0)
    rep = prof.report()
    assert "Scale" in rep
    assert rep["Scale"]["calls"] == 2
    assert rep["Scale"]["total"] >= 0
    assert rep["Scale"]["mean"] >= 0


def test_profile_clear_drops_timings(Scale_cls):
    prof = Profile()
    prof.wrap(Scale_cls(2.0))(1.0)
    prof.clear()
    assert prof.timings == {}


def test_histogram_records_distribution():
    h = Histogram(bins=4)
    op = h.at("k")
    op([0.0, 1.0, 2.0, 3.0, 4.0])
    res = h.results["k"]
    assert sum(res["counts"]) == 5
    assert len(res["counts"]) == 4
    assert len(res["edges"]) == 5


def test_histogram_passes_through_carrier():
    h = Histogram(bins=4)
    op = h.at("k")
    arr = [0.0, 1.0, 2.0]
    assert op(arr) is arr


def test_histogram_degenerate_range_uses_single_bin_count():
    h = Histogram(bins=5)
    h.at("k")([3.0, 3.0, 3.0])
    res = h.results["k"]
    assert sum(res["counts"]) == 3


def test_histogram_custom_to_array():
    """Pass a tuple-to-list adapter; values come from a non-list carrier."""

    class Arr:
        def __init__(self, vals):
            self.vals = vals

    h = Histogram(bins=3, to_array=lambda a: list(a.vals))
    h.at("k")(Arr((1.0, 2.0, 3.0)))
    assert sum(h.results["k"]["counts"]) == 3


def test_histogram_rejects_zero_bins():
    import pytest

    with pytest.raises(ValueError):
        Histogram(bins=0)


def test_observer_reprs_are_readable():
    """Every observer renders a compact repr, not a raw config dict."""
    assert repr(Snapshot().at("k")).startswith("Snapshot.at(")
    assert repr(ShapeTrace()).startswith("ShapeTrace(")
    assert repr(Histogram(bins=4)).startswith("Histogram(")
