"""Tests for `pipekit.parallel` — Group J."""

from __future__ import annotations

import asyncio
import threading

import pytest
from pipekit import (
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
)


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


def test_processmap_log_and_continue():
    op = ProcessMap(SometimesFail(), n_workers=2, on_error="log_and_continue")
    out = op([1, -1, 2])
    assert out[0] == 1
    assert out[1] is None
    assert out[2] == 2


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
    from pipekit import Graph, Input

    x = Input("x")
    y = Lambda(lambda v: v + 1, name="bump")(x)
    z = Double()(y)
    g = Graph(inputs={"x": x}, outputs={"out": z})
    flagged = check_pickleable(g)
    assert any(type(o).__name__ == "Lambda" for o in flagged)


def test_check_pickleable_clean_graph_returns_empty():
    from pipekit import Graph, Input

    x = Input("x")
    y = Double()(x)
    g = Graph(inputs={"x": x}, outputs={"out": y})
    assert check_pickleable(g) == []
