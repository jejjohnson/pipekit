"""Tests for `pipekit_ops.parallel` — Group J."""

from __future__ import annotations

import asyncio
import multiprocessing
import threading
import time
import warnings
from pathlib import Path

import pytest
from pipekit_ops import (
    AsyncMap,
    BatchedMap,
    Branch,
    Cache,
    Identity,
    Lambda,
    Operator,
    ProcessMap,
    Sequential,
    ThreadMap,
    check_pickleable,
    pipeline_hash,
)

from tests._execution_ops import SlowFailFirst


class Double(Operator):
    """Module-level operator so it pickles (needed by ProcessMap)."""

    def _apply(self, x):
        return x * 2


class SometimesFail(Operator):
    def _apply(self, x):
        if x < 0:
            raise ValueError("negative")
        return x


def test_threadmap_returns_results_in_order():
    op = ThreadMap(Double(), n_workers=4)
    assert op([1, 2, 3, 4]) == [2, 4, 6, 8]


def test_threadmap_releases_overlap():
    """Workers run concurrently — proven via a Barrier, not wall-clock.

    Each worker waits on a 4-party `Barrier`. If `ThreadMap` were
    sequential, the first worker would block forever (no other party
    reaches the barrier) and the test would time out. If it's
    concurrent, all four parties trip the barrier and the call returns.
    """

    barrier = threading.Barrier(4, timeout=5.0)

    class BarrierOp(Operator):
        def _apply(self, x):
            barrier.wait()
            return x

    op = ThreadMap(BarrierOp(), n_workers=4)
    assert op([1, 2, 3, 4]) == [1, 2, 3, 4]


def test_threadmap_rejects_zero_workers():
    with pytest.raises(ValueError):
        ThreadMap(Double(), n_workers=0)


def test_threadmap_rejects_non_operator():
    with pytest.raises(TypeError):
        ThreadMap(lambda x: x)  # type: ignore[arg-type]


def test_processmap_returns_results_in_order():
    op = ProcessMap(Double(), n_workers=2)
    assert op([1, 2, 3, 4]) == [2, 4, 6, 8]


def test_processmap_log_and_continue(caplog):
    op = ProcessMap(SometimesFail(), n_workers=2, on_error="log_and_continue")
    with caplog.at_level("WARNING", logger="pipekit_ops.parallel"):
        out = op([1, -1, 2])
    assert out[0] == 1
    assert out[1] is None
    assert out[2] == 2
    # reported through logging, with the traceback, not printed to stderr
    (record,) = caplog.records
    assert record.name == "pipekit_ops.parallel"
    assert "item 1 failed with ValueError" in record.getMessage()
    assert record.exc_info is not None


def test_asyncmap_inside_a_running_loop_points_to_run():
    import asyncio

    op = AsyncMap(Double())

    async def handler():
        with pytest.raises(RuntimeError, match=r"await async_map\.run"):
            op([1, 2])
        return await op.run([1, 2])

    assert asyncio.run(handler()) == [2, 4]


def test_processmap_raises_by_default():
    op = ProcessMap(SometimesFail(), n_workers=2)
    with pytest.raises(ValueError):
        op([1, -1, 2])


def test_processmap_validates_on_error():
    with pytest.raises(ValueError):
        ProcessMap(Double(), on_error="bogus")


def test_asyncmap_returns_results_in_order():
    op = AsyncMap(Double(), semaphore=4)
    assert op([1, 2, 3]) == [2, 4, 6]


def test_asyncmap_with_async_apply():
    class AsyncDouble(Operator):
        async def _apply(self, x):  # type: ignore[override]
            await asyncio.sleep(0)
            return x * 2

    op = AsyncMap(AsyncDouble(), semaphore=2)
    assert op([1, 2, 3]) == [2, 4, 6]


def test_asyncmap_validates_semaphore():
    with pytest.raises(ValueError):
        AsyncMap(Double(), semaphore=0)


def test_batchedmap_chunks_input():
    class FirstOnly(Operator):
        """Returns the first element of the batch as the batch result."""

        def _apply(self, batch):
            return batch[0]

    op = BatchedMap(FirstOnly(), batch_size=2, flatten=False)
    assert op([10, 20, 30, 40, 50]) == [10, 30, 50]


def test_batchedmap_flattens_list_outputs():
    class Mul(Operator):
        def _apply(self, batch):
            return [b * 2 for b in batch]

    op = BatchedMap(Mul(), batch_size=2, flatten=True)
    assert op([1, 2, 3, 4, 5]) == [2, 4, 6, 8, 10]


def test_batchedmap_validates_batch_size():
    with pytest.raises(ValueError):
        BatchedMap(Double(), batch_size=0)


def test_check_pickleable_finds_lambda_in_sequential():
    pipe = Sequential([Identity(), Lambda(lambda x: x), Double()])
    flagged = check_pickleable(pipe)
    assert len(flagged) == 1
    assert type(flagged[0]).__name__ == "Lambda"


def test_check_pickleable_walks_cache_inner():
    op = Cache(Lambda(lambda x: x))
    flagged = check_pickleable(op)
    assert any(type(o).__name__ == "Lambda" for o in flagged)


def test_check_pickleable_walks_branch():
    op = Branch(
        predicate=lambda x: x > 0,
        if_true=Lambda(lambda x: x),
        if_false=Identity(),
    )
    flagged = check_pickleable(op)
    classes = {type(o).__name__ for o in flagged}
    assert "Branch" in classes  # Branch itself is flagged
    assert "Lambda" in classes


def test_check_pickleable_clean_pipeline_returns_empty():
    pipe = Sequential([Identity(), Double()])
    assert check_pickleable(pipe) == []


def test_check_pickleable_walks_graph():
    """A flagged operator buried in a Graph node must be surfaced."""
    from pipekit_ops import Graph, Input

    x = Input("x")
    y = Lambda(lambda v: v + 1, name="bump")(x)
    z = Double()(y)
    g = Graph(inputs={"x": x}, outputs={"out": z})
    flagged = check_pickleable(g)
    assert any(type(o).__name__ == "Lambda" for o in flagged)


def test_check_pickleable_clean_graph_returns_empty():
    from pipekit_ops import Graph, Input

    x = Input("x")
    y = Double()(x)
    g = Graph(inputs={"x": x}, outputs={"out": y})
    assert check_pickleable(g) == []


def test_maps_are_not_forbid_in_yaml():
    """The maps hold only a nested operator — the walker finds flagged
    children structurally, so the wrappers themselves stay unflagged."""
    for cls in (ThreadMap, ProcessMap, AsyncMap, BatchedMap):
        assert cls.forbid_in_yaml is False, cls.__name__


# --- presets on Map (roadmap P2) -------------------------------------------


def test_g2_6_preset_hashes_are_kept():
    """Full digests recorded at ``903b0c3``, before the presets moved onto Map."""
    presets = [
        ThreadMap(Identity(), n_workers=4),
        ProcessMap(Identity()),
        AsyncMap(Identity()),
        BatchedMap(Identity()),
    ]
    assert [pipeline_hash(preset) for preset in presets] == [
        "be9cbf3c8a67e2dca1aa8571b8945a5627f3e49bce6a1a3ba89582a7cf2420bd",
        "df4f9dd1232c0a142f62627958e0d0c5e3459a9ea1ef2180c144ce72ad95765b",
        "2132e0bfb39a71e00728523de07ec9d0dd72cebbe812abf1e56b6265a701b081",
        "70dec7fb25292cc5a75b625f92ebe1c976d76dd4d8ea91f5596659d7d988bf90",
    ]


class TouchThenFailFirst(Operator):
    """Leave a file per item started; item 0 raises at once, the rest sleep."""

    def __init__(self, directory: str, seconds: float) -> None:
        self.directory = directory
        self.seconds = seconds

    def _apply(self, x):
        Path(self.directory, str(x)).touch()
        if x == 0:
            raise ValueError("item 0 failed")
        time.sleep(self.seconds)
        return x


def test_processmap_raise_cancels_what_has_not_started(tmp_path):
    op = ProcessMap(TouchThenFailFirst(str(tmp_path), 0.05), n_workers=2)
    with pytest.raises(ValueError, match="item 0 failed"):
        op(range(100))
    # the window holds 2 n_w = 4 items; before Map every item ran
    assert len(list(tmp_path.iterdir())) <= 10


@pytest.mark.slow
def test_g2_3_processmap_fails_fast_wall_time():
    op = ProcessMap(SlowFailFirst(0.2), n_workers=2)
    start = time.perf_counter()
    with pytest.raises(ValueError, match="item 0 failed"):
        op(range(100))
    elapsed = time.perf_counter() - start
    print(f"G2-3 ProcessMap: raised after {elapsed:.3f} s")
    assert elapsed < 1.5


def test_processmap_log_keeps_its_prefix(caplog):
    op = ProcessMap(SometimesFail(), n_workers=2, on_error="log_and_continue")
    with caplog.at_level("WARNING", logger="pipekit_ops.parallel"):
        op([-1])
    (record,) = caplog.records
    assert record.getMessage().startswith("ProcessMap: item 0 failed")


@pytest.mark.parametrize(
    ("platform_default", "warns"), [("fork", True), ("spawn", False)]
)
def test_processmap_keeps_the_platform_start_method(
    monkeypatch, platform_default, warns
):
    monkeypatch.setattr(
        multiprocessing, "get_start_method", lambda allow_none=False: platform_default
    )
    with warnings.catch_warnings(record=True, action="always") as caught:
        op = ProcessMap(Double())
    future = [w for w in caught if issubclass(w.category, FutureWarning)]
    assert bool(future) is warns
    assert op.start_method is None
    assert op._default_start_method is None
