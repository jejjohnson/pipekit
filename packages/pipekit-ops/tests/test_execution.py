"""Tests for `pipekit_ops.execution`: `Map`, `executor_scope`, `Prefetch`, `Fold`.

Gate IDs refer to ``packages/pipekit-ops/docs/design/roadmap.md`` (P2 and
G0). Window gates count items from inside the input generator: at each
pull it records pulled - consumed, which includes results buffered
behind a slower item. Wall-clock gates (G2-3) are marked ``slow``.
"""

from __future__ import annotations

import asyncio
import logging
import random
import subprocess
import sys
import threading
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pipekit_ops
import pytest
from pipekit_ops import (
    Cache,
    Failed,
    Fold,
    Identity,
    Lambda,
    Map,
    MapError,
    Operator,
    Prefetch,
    execution as execution_module,
    executor_scope,
    pipeline_hash,
)

from tests._execution_ops import (
    AddPair,
    Double,
    FailOn,
    HoldsLock,
    JitterDouble,
    SlowFailFirst,
)


# --- instruments (threads only) -------------------------------------------


class Window:
    """A counting input: records the most items pulled and not yet consumed."""

    def __init__(self) -> None:
        self.pulled = 0
        self.consumed = 0
        self.max_held = 0

    def source(self, n: int):
        for i in range(n):
            self.pulled += 1
            self.max_held = max(self.max_held, self.pulled - self.consumed)
            yield i


class Counted(Operator):
    """Identity that counts finished calls, thread-safely."""

    def __init__(self) -> None:
        self.finished = 0
        self._lock = threading.Lock()

    def _apply(self, x: int) -> int:
        with self._lock:
            self.finished += 1
        return x


class BlockHead(Operator):
    """x ↦ 2x; item 0 waits for ``gate``; counts the later items that finish."""

    def __init__(self) -> None:
        self.gate = threading.Event()
        self.later_finished = 0
        self.changed = threading.Condition()

    def _apply(self, x: int) -> int:
        if x == 0:
            assert self.gate.wait(timeout=10.0), "the head item was never released"
            return 0
        with self.changed:
            self.later_finished += 1
            self.changed.notify_all()
        return 2 * x

    def wait_for_later(self, n: int) -> None:
        with self.changed:
            reached = self.changed.wait_for(
                lambda: self.later_finished >= n, timeout=10.0
            )
        assert reached, f"only {self.later_finished} later items finished"


class AsyncDouble(Operator):
    """x ↦ 2x with a coroutine ``_apply``."""

    async def _apply(self, x: int) -> int:
        await asyncio.sleep(0)
        return 2 * x


# --- G2-1: same outputs ------------------------------------------------------


def test_g2_1_every_executor_returns_the_same_list():
    items = list(range(1000))
    expected = [2 * x for x in items]
    for executor in ("serial", "thread", "process", "async"):
        out = Map(JitterDouble(), executor=executor, n_workers=4)(items)
        assert out == expected, executor
    pool = ThreadPoolExecutor(max_workers=4)
    with executor_scope(pool):
        assert Map(JitterDouble(), executor="scope")(items) == expected
    pool.shutdown()


def test_scope_without_an_executor_runs_serially():
    assert Map(Double(), executor="scope")([1, 2]) == [2, 4]


def test_borrowed_process_pool_is_pickle_checked():
    from concurrent.futures import ProcessPoolExecutor

    pool = ProcessPoolExecutor(max_workers=1)
    with executor_scope(pool), pytest.raises(TypeError, match="Lambda"):
        Map(Lambda(abs), executor="scope")([1])
    pool.shutdown()


# --- G2-2: lazy input --------------------------------------------------------


def test_g2_2_thread_map_pulls_lazily():
    op = Counted()
    pulled = 0
    max_held = 0

    def source(n: int):
        nonlocal pulled, max_held
        for i in range(n):
            pulled += 1
            # "collected" happens inside Map; finished ≥ collected, so
            # pulled - finished ≤ pulled - collected
            max_held = max(max_held, pulled - op.finished)
            yield i

    mapped = Map(op, executor="thread", n_workers=4, max_in_flight=8)
    assert mapped(source(10_000)) == list(range(10_000))
    print(f"G2-2: max pulled and not finished {max_held}")
    assert max_held <= 8


def test_g2_2_iter_unordered_never_holds_more_than_w():
    window = Window()
    mapped = Map(Counted(), executor="thread", n_workers=4, max_in_flight=8)
    seen = []
    for index, value in mapped.iter(window.source(1000), ordered=False):
        window.consumed += 1
        seen.append((index, value))
    assert sorted(seen) == [(i, i) for i in range(1000)]
    assert window.max_held <= 8


# --- G2-3: fail fast -------------------------------------------------------


class ReleaseOnFailure(Operator):
    """Item 0 raises at once; the rest wait until it has raised."""

    def __init__(self) -> None:
        self.failed = threading.Event()
        self.started = 0
        self._lock = threading.Lock()

    def _apply(self, x: int) -> int:
        with self._lock:
            self.started += 1
        if x == 0:
            self.failed.set()
            raise ValueError("item 0 failed")
        assert self.failed.wait(timeout=10.0)
        return x


def test_g2_3_raise_stops_pulling_and_cancels_pending():
    window = Window()
    op = ReleaseOnFailure()
    mapped = Map(op, executor="thread", n_workers=2)
    with pytest.raises(ValueError, match="item 0 failed") as caught:
        mapped(window.source(100))
    assert "Map: item 0 failed" in caught.value.__notes__
    # W = 2 n_w = 4: nothing beyond the window was pulled or started
    assert window.pulled == 4
    assert op.started <= 4


@pytest.mark.slow
@pytest.mark.parametrize(("executor", "bound"), [("thread", 0.5), ("process", 1.5)])
def test_g2_3_fail_fast_wall_time(executor, bound):
    mapped = Map(SlowFailFirst(0.2), executor=executor, n_workers=2)
    start = time.perf_counter()
    with pytest.raises(ValueError, match="item 0 failed"):
        mapped(range(100))
    elapsed = time.perf_counter() - start
    print(f"G2-3 {executor}: raised after {elapsed:.3f} s")
    assert elapsed < bound


# --- G2-4: collect -------------------------------------------------------


@pytest.mark.parametrize("executor", ["serial", "thread"])
def test_g2_4_collect_raises_map_error_with_partial_results(executor):
    mapped = Map(FailOn((3, 7)), executor=executor, on_error="collect")
    with pytest.raises(MapError) as caught:
        mapped(range(10))
    error = caught.value
    assert len(error.exceptions) == 2
    notes = [exc.__notes__ for exc in error.exceptions]
    assert notes == [["Map: item 3 failed"], ["Map: item 7 failed"]]
    assert error.results is not None
    assert [r for i, r in enumerate(error.results) if i not in (3, 7)] == [
        0,
        1,
        2,
        4,
        5,
        6,
        8,
        9,
    ]
    assert isinstance(error.results[3], Failed)
    assert error.results[3].index == 3
    assert error.results[7].exception is error.exceptions[1]


def test_collect_survives_except_star():
    mapped = Map(FailOn((1,)), on_error="collect")
    try:
        mapped(range(3))
    except* ValueError as group:
        assert isinstance(group, MapError)
        assert group.results is not None
        assert group.results[2] == 2


def test_iter_collect_yields_failed_records_then_raises():
    stream = Map(FailOn((1,)), executor="thread", on_error="collect").iter(range(3))
    out = []
    with pytest.raises(MapError) as caught:
        for value in stream:
            out.append(value)
    assert out[0] == 0 and out[2] == 2
    assert isinstance(out[1], Failed)
    assert caught.value.results is None


# --- G2-5: log and continue -------------------------------------------------


def test_g2_5_log_and_continue(caplog):
    mapped = Map(FailOn((3,)), executor="thread", on_error="log_and_continue")
    with caplog.at_level(logging.WARNING, logger="pipekit_ops.parallel"):
        out = mapped(range(10))
    assert out == [0, 1, 2, None, 4, 5, 6, 7, 8, 9]
    records = [r for r in caplog.records if r.name == "pipekit_ops.parallel"]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    assert "item 3" in records[0].getMessage()


def test_keyboard_interrupt_is_never_collected():
    class Interrupt(Operator):
        def _apply(self, x: int) -> int:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        Map(Interrupt(), on_error="collect")([1, 2])
    with pytest.raises(KeyboardInterrupt):
        Map(Interrupt(), executor="thread", on_error="log_and_continue")([1, 2])


# --- G2-7: pickling caught early -------------------------------------------


class _NoPool:
    """Stands in for `ProcessPoolExecutor`: fails if a pool is ever built."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("a process pool was started")


def test_g2_7_flagged_operator_refused_before_any_worker(monkeypatch):
    monkeypatch.setattr(execution_module, "ProcessPoolExecutor", _NoPool)
    with pytest.raises(TypeError, match="Lambda"):
        Map(Lambda(abs), executor="process")([1])


def test_g2_7_unflagged_unpicklable_operator_refused(monkeypatch):
    monkeypatch.setattr(execution_module, "ProcessPoolExecutor", _NoPool)
    with pytest.raises(TypeError, match="HoldsLock does not pickle"):
        Map(HoldsLock(), executor="process")([1])


def test_g2_7_cache_goes_to_a_process_pool():
    assert Map(Cache(Identity()), executor="process", n_workers=2)([1, 2]) == [1, 2]


def test_process_errors_keep_their_index():
    mapped = Map(FailOn((2,)), executor="process", n_workers=2, on_error="collect")
    with pytest.raises(MapError) as caught:
        mapped(range(4))
    assert caught.value.exceptions[0].__notes__ == ["Map: item 2 failed"]
    assert caught.value.results is not None
    assert caught.value.results[3] == 3


# --- G2-8: tracers ---------------------------------------------------------


def test_g2_8_tracers():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    x = jnp.array([1.0, 2.0, 3.0])

    def serial(v):
        return jnp.stack(Map(Double(), executor="serial")(list(v)))

    assert bool(jnp.array_equal(jax.jit(serial)(x), serial(x)))

    for executor in ("thread", "async"):

        def pooled(v, executor=executor):
            return jnp.stack(Map(Double(), executor=executor)(list(v)))

        with pytest.raises(TypeError, match=r"jax\.vmap"):
            jax.jit(pooled)(x)

    with pytest.raises(TypeError, match="tracer"):
        jax.jit(lambda v: jnp.stack(list(Prefetch(Double(), executor="serial")(v))))(x)


# --- G2-9: running event loop ------------------------------------------------


def test_g2_9_running_loop():
    mapped = Map(Double(), executor="async")

    async def handler():
        with pytest.raises(RuntimeError, match="run"):
            mapped([1, 2])
        return await mapped.run([1, 2, 3])

    assert asyncio.run(handler()) == mapped([1, 2, 3]) == [2, 4, 6]


def test_async_awaits_a_coroutine_apply():
    mapped = Map(AsyncDouble(), executor="async", max_in_flight=3)
    assert mapped(range(10)) == [2 * x for x in range(10)]
    assert asyncio.run(mapped.run(range(3))) == [0, 2, 4]


def test_run_on_a_thread_executor():
    assert asyncio.run(Map(Double(), executor="thread").run([1, 2])) == [2, 4]


# --- G2-10: borrowed executor stays open -----------------------------------


def test_g2_10_borrowed_executor_stays_open():
    pool = ThreadPoolExecutor(max_workers=2)
    with executor_scope(pool):
        assert Map(Double(), executor="scope")([1, 2]) == [2, 4]
        with pytest.raises(ValueError):
            Map(FailOn((0,)), executor="scope")([0, 1])
    assert pool.submit(abs, -1).result() == 1
    pool.shutdown()


def test_executor_scope_is_restored():
    pool = ThreadPoolExecutor(max_workers=1)
    with executor_scope(pool):
        with executor_scope(None):
            assert execution_module._SCOPED_EXECUTOR.get() is None
        assert execution_module._SCOPED_EXECUTOR.get() is pool
    assert execution_module._SCOPED_EXECUTOR.get() is None
    pool.shutdown()


# --- G2-11: Prefetch depth ---------------------------------------------------


def test_g2_11_prefetch_depth_with_a_blocked_head():
    window = Window()
    op = BlockHead()
    stream = Prefetch(op, depth=4)(window.source(100))
    out: list[int] = []

    def consume() -> None:
        for value in stream:
            window.consumed += 1
            out.append(value)

    consumer = threading.Thread(target=consume)
    consumer.start()
    # the 3 later items in the window finish; nothing more may be pulled
    op.wait_for_later(3)
    pulled_before_release = window.pulled
    finished_before_release = op.later_finished
    op.gate.set()
    consumer.join(timeout=10.0)
    assert not consumer.is_alive()
    assert pulled_before_release <= 4
    assert finished_before_release <= 3
    assert window.max_held <= 4
    assert out == [2 * x for x in range(100)]


# --- G2-12: Fold equals reduce -----------------------------------------------


def test_g2_12_fold_equals_sum():
    rng = random.Random(0)
    items = [rng.randint(-1000, 1000) for _ in range(10_000)]
    fold = Fold(AddPair(), 0, chunksize=128, executor="thread")
    assert fold(items) == sum(items)


def test_fold_empty_and_combine():
    assert Fold(AddPair(), 0)([]) == 0
    assert Fold(AddPair(), 0, chunksize=2, combine=AddPair())(range(5)) == 10


def test_fold_on_processes():
    fold = Fold(AddPair(), 0, chunksize=50, executor="process")
    assert fold(range(200)) == sum(range(200))


# --- G2-13: nested maps ------------------------------------------------------


def test_g2_13_nested_maps_do_not_deadlock():
    inner = Map(Double(), executor="scope")
    outer = Map(inner, executor="scope")
    rows = [list(range(10)) for _ in range(10)]
    out: list[Any] = []
    pool = ThreadPoolExecutor(max_workers=2)

    def run() -> None:
        with executor_scope(pool):
            out.extend(outer(rows))

    start = time.perf_counter()
    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout=5.0)
    elapsed = time.perf_counter() - start
    print(f"G2-13: 10 x 10 nested items in {elapsed:.3f} s")
    assert not worker.is_alive(), "nested Map deadlocked"
    assert out == [[2 * x for x in range(10)]] * 10
    pool.shutdown()


# --- G2-14: ordered iterator window ----------------------------------------


def test_g2_14_ordered_iterator_window():
    window = Window()
    op = BlockHead()
    mapped = Map(op, executor="thread", n_workers=4, max_in_flight=4)
    snapshot: dict[str, int] = {}

    def release() -> None:
        snapshot["pulled"] = window.pulled
        snapshot["later_finished"] = op.later_finished
        op.gate.set()

    # the gate's 1 s block on item 0
    timer = threading.Timer(1.0, release)
    timer.start()
    start = time.perf_counter()
    out = []
    for value in mapped.iter(window.source(100), ordered=True):
        window.consumed += 1
        out.append(value)
    elapsed = time.perf_counter() - start
    timer.join()
    print(f"G2-14: max held {window.max_held}, {elapsed:.3f} s, {snapshot}")
    assert out == [2 * x for x in range(100)]
    assert window.max_held <= 4
    assert snapshot == {"pulled": 4, "later_finished": 3}


# --- chunking, ordering, validation --------------------------------------


def test_chunks_keep_errors_per_item():
    mapped = Map(FailOn((5,)), executor="thread", chunksize=4, on_error="collect")
    with pytest.raises(MapError) as caught:
        mapped(range(10))
    assert len(caught.value.exceptions) == 1
    assert caught.value.results is not None
    assert caught.value.results[4] == 4 and caught.value.results[6] == 6


def test_chunks_on_processes():
    mapped = Map(Double(), executor="process", n_workers=2, chunksize=8)
    assert mapped(range(50)) == [2 * x for x in range(50)]


def test_iter_unordered_yields_index_pairs():
    pairs = Map(Double(), executor="thread").iter([5, 6], ordered=False)
    assert sorted(pairs) == [(0, 10), (1, 12)]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"executor": "dask"},
        {"on_error": "ignore"},
        {"n_workers": 0},
        {"max_in_flight": 0},
        {"chunksize": 0},
        {"start_method": "vfork"},
    ],
)
def test_map_validates_arguments(kwargs):
    with pytest.raises(ValueError):
        Map(Double(), **kwargs)


def test_map_rejects_a_callable():
    with pytest.raises(TypeError):
        Map(abs)  # type: ignore[arg-type]


def test_the_executor_is_not_in_the_hash():
    pool = ThreadPoolExecutor(max_workers=1)
    mapped = Map(Double(), executor="scope")
    before = pipeline_hash(mapped)
    with executor_scope(pool):
        assert pipeline_hash(mapped) == before
    pool.shutdown()


# --- G0-3 / G0-4 -----------------------------------------------------------


def test_g0_3_no_dependencies():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    project = tomllib.loads(pyproject.read_text())["project"]
    assert project["dependencies"] == []
    script = (
        "import sys\n"
        "before = set(sys.modules)\n"
        "import pipekit_ops\n"
        "new = {m.split('.')[0] for m in set(sys.modules) - before}\n"
        # multiprocessing aliases __main__ as __mp_main__
        "new.discard('__mp_main__')\n"
        "print(sorted(new - set(sys.stdlib_module_names) - {'pipekit_ops'}))\n"
    )
    found = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert found.stdout.strip() == "[]"


def test_g0_4_no_new_class_is_flagged():
    for cls in (Map, Prefetch, Fold, execution_module._ReduceChunk):
        assert cls.forbid_in_yaml is False
    assert pipekit_ops.check_pickleable(Map(Prefetch(Fold(AddPair(), 0)))) == []
