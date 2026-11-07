"""Phase P5 of the pipekit-ops design: `Log`, `Map(progress=…)`, `AssertThat`
and `AssertSignature`, one block per roadmap gate (G5-1 to G5-5).
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Iterator
from typing import Any

import pytest
from pipekit_ops import (
    AssertSignature,
    AssertThat,
    Const,
    Identity,
    Lambda,
    Log,
    Map,
    Operator,
    QCError,
    Sequential,
    Signature,
    dumps,
    loads,
    pipeline_hash,
    register_adapter,
    unregister_adapter,
)


@dataclasses.dataclass(frozen=True)
class Cube:
    """A toy carrier: only its dims and dtype matter here."""

    dims: tuple[str, ...]
    dtype: str = "uint16"


class CubeAdapter:
    """The minimal `NamedArrayAdapter` for `Cube`: every dim has size 2."""

    def dims(self, x: Cube) -> tuple[str, ...]:
        return x.dims

    def data(self, x: Cube) -> Any:
        return None

    def labels(self, x: Cube, dim: str) -> tuple[Any, ...] | None:
        return None

    def isel(self, x: Cube, indexers: dict[str, Any]) -> Cube:
        raise NotImplementedError

    def rewrap(self, x: Cube, data: Any, dims: tuple[str, ...]) -> Cube:
        raise NotImplementedError

    def check_compatible(self, *xs: Cube) -> tuple[Cube, ...]:
        return xs

    def signature(self, x: Cube) -> Signature:
        return Signature(dims=tuple((d, 2) for d in x.dims), dtype=x.dtype)


@pytest.fixture
def cube_adapter() -> Iterator[None]:
    register_adapter(Cube, CubeAdapter())
    yield
    unregister_adapter(Cube)


class Counting(Operator):
    """Identity that counts its calls, to prove no data runs."""

    def __init__(self) -> None:
        self.calls = 0

    def _apply(self, x: Any) -> Any:
        self.calls += 1
        return x


# --- G5-1: Log pass-through --------------------------------------------------


def test_g5_1_log_returns_its_input_and_writes_one_record_per_call(caplog):
    note = Log(logger="p5.log")
    x = [1.0, 2.0]
    with caplog.at_level(logging.INFO, logger="p5.log"):
        for _ in range(3):
            assert note(x) is x
    records = [r for r in caplog.records if r.name == "p5.log"]
    assert len(records) == 3
    assert [r.getMessage() for r in records][-1] == "call 3: list ?"


def test_g5_1_log_formats_the_signature_through_the_adapter(caplog, cube_adapter):
    note = Log("{n} {type} {signature}", level="DEBUG", logger="p5.log")
    with caplog.at_level(logging.DEBUG, logger="p5.log"):
        note(Cube(("band", "y", "x")))
    (record,) = [r for r in caplog.records if r.name == "p5.log"]
    assert record.levelno == logging.DEBUG
    assert record.getMessage() == "1 Cube (band=2, y=2, x=2) uint16"


def test_g5_1_log_below_the_level_writes_nothing_but_still_counts(caplog):
    note = Log("{n}", level="DEBUG", logger="p5.log")
    with caplog.at_level(logging.INFO, logger="p5.log"):
        note(1)
    with caplog.at_level(logging.DEBUG, logger="p5.log"):
        note(1)
    assert [r.getMessage() for r in caplog.records if r.name == "p5.log"] == ["2"]


@pytest.mark.parametrize(
    "kwargs", [{"message": "{shape}"}, {"message": "{0}"}, {"level": "LOUD"}]
)
def test_g5_1_log_rejects_bad_config(kwargs):
    with pytest.raises(ValueError):
        Log(**kwargs)


def test_g5_1_log_round_trips_and_is_not_flagged():
    note = Log("{type}", level="WARNING", logger="p5.log")
    assert not Log.forbid_in_yaml
    assert pipeline_hash(loads(dumps(note))) == pipeline_hash(note)


# --- G5-2: Map progress --------------------------------------------------------


def _progress_records(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if hasattr(r, "done")]


@pytest.mark.parametrize("executor", ["serial", "thread"])
def test_g5_2_progress_every_10_over_100_items(caplog, executor):
    mapped = Map(Identity(), executor=executor, progress=10)
    with caplog.at_level(logging.INFO, logger="pipekit_ops.parallel"):
        assert mapped(range(100)) == list(range(100))
    records = _progress_records(caplog)
    assert len(records) == 10
    assert all(r.name == "pipekit_ops.parallel" for r in records)
    assert all(r.levelno == logging.INFO for r in records)
    assert [r.done for r in records] == list(range(10, 101, 10))
    last = records[-1]
    assert (last.done, last.failed, last.in_flight, last.n) == (100, 0, 0, 100)
    assert last.rate > 0


def test_g5_2_progress_counts_failures_and_unknown_length(caplog):
    inverse = Lambda(lambda x: 1 / x)
    mapped = Map(inverse, on_error="log_and_continue", progress=2)
    with caplog.at_level(logging.INFO, logger="pipekit_ops.parallel"):
        mapped(iter([1, 0, 2, 0]))
    records = _progress_records(caplog)
    assert [(r.done, r.failed, r.n) for r in records] == [(2, 1, None), (4, 2, None)]
    assert "4/? done" in records[-1].getMessage()


def test_g5_2_no_progress_no_records(caplog):
    with caplog.at_level(logging.INFO, logger="pipekit_ops.parallel"):
        Map(Identity())(range(20))
    assert _progress_records(caplog) == []


def test_g5_2_progress_rejects_zero():
    with pytest.raises(ValueError):
        Map(Identity(), progress=0)


def test_g5_2_unset_progress_keeps_the_map_hash():
    # values from the parent commit ac27ab1, before `progress` existed
    assert pipeline_hash(Map(Identity())) == (
        "b5098ac523432d9a9fae2d2a49280cb8d1ebf09e37ed6fdcfc068cd4a78731c8"
    )
    threaded = Map(Identity(), executor="thread", max_in_flight=4)
    assert pipeline_hash(threaded) == (
        "cb159a8d5dad05498c9651ee80f6698be23f2e5e6b5d14c3b50e0b7258ed0498"
    )
    assert "progress" not in Map(Identity()).get_config()


def test_g5_2_progress_round_trips_and_changes_the_hash():
    mapped = Map(Identity(), progress=5)
    loaded = loads(dumps(mapped))
    assert loaded.progress == 5
    assert pipeline_hash(loaded) == pipeline_hash(mapped)
    assert pipeline_hash(mapped) != pipeline_hash(Map(Identity()))


# --- G5-3: AssertThat ------------------------------------------------------------


def test_g5_3_assert_that():
    x = object()
    with pytest.raises(QCError, match="too cloudy"):
        AssertThat(Const(False), "too cloudy")(x)
    assert AssertThat(Const(True))(x) is x


def test_g5_3_assert_that_needs_an_operator():
    with pytest.raises(TypeError):
        AssertThat(bool)  # type: ignore[arg-type]


def test_g5_3_assert_that_round_trips_with_its_predicate():
    check = AssertThat(Const(True) | Const(False), "never true")
    loaded = loads(dumps(check))
    assert pipeline_hash(loaded) == pipeline_hash(check)
    with pytest.raises(QCError, match="never true"):
        loaded(1)


# --- G5-4: AssertSignature -------------------------------------------------------


@pytest.mark.parametrize("pattern", ["band y x", "... y x", "... band y x"])
def test_g5_4_matching_patterns_pass(cube_adapter, pattern):
    cube = Cube(("band", "y", "x"))
    assert AssertSignature(pattern)(cube) is cube


@pytest.mark.parametrize("pattern", ["time band y x", "y x", "... x y"])
def test_g5_4_mismatched_patterns_raise(cube_adapter, pattern):
    with pytest.raises(QCError, match="do not fit"):
        AssertSignature(pattern)(Cube(("band", "y", "x")))


def test_g5_4_dtype(cube_adapter):
    cube = Cube(("band", "y", "x"), dtype="uint16")
    assert AssertSignature("band y x", dtype="uint16")(cube) is cube
    with pytest.raises(QCError, match="dtype"):
        AssertSignature("band y x", dtype="float32")(cube)


def test_g5_4_unregistered_carrier_raises_or_skips():
    cube = Cube(("band", "y", "x"))
    with pytest.raises(QCError, match="no named-array adapter"):
        AssertSignature("band y x")(cube)
    assert AssertSignature("band y x", missing="skip")(cube) is cube


@pytest.mark.parametrize(
    "kwargs", [{"pattern": "... y ..."}, {"pattern": "y x", "missing": "warn"}]
)
def test_g5_4_rejects_bad_config(kwargs):
    with pytest.raises(ValueError):
        AssertSignature(**kwargs)


# --- G5-5: build-time check ------------------------------------------------------


def test_g5_5_summary_reports_a_mismatch_without_running_data():
    upstream = Counting()
    pipe = Sequential([upstream, AssertSignature("band y x")])
    time_y_x = Signature(dims=(("time", 3), ("y", 4), ("x", 4)), dtype="f4")
    with pytest.raises(QCError, match=r"\(time, y, x\) do not fit 'band y x'"):
        pipe.summary(time_y_x)
    assert upstream.calls == 0


def test_g5_5_summary_passes_a_match_and_an_unknown_signature():
    band_y_x = Signature(dims=(("band", 4), ("y", 4), ("x", 4)), dtype="uint16")
    pipe = Sequential([Identity(), AssertSignature("band y x", dtype="uint16")])
    assert "AssertSignature" in pipe.summary(band_y_x)
    assert pipe.compute_output_signature(band_y_x) == band_y_x
    # no input signature: nothing to check
    assert "AssertSignature" in pipe.summary()


def test_g5_5_unknown_dtype_is_not_a_build_time_mismatch():
    no_dtype = Signature(dims=(("band", 4), ("y", 4), ("x", 4)))
    check = AssertSignature("band y x", dtype="uint16")
    assert check.compute_output_signature(no_dtype) == no_dtype
