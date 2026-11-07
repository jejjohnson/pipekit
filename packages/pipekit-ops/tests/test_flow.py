"""Flow limits: `Timeout`, `RateLimit` and `Until` (phase P3, gates G3-1 to G3-7).

Every wall-clock gate (G3-1, G3-3, G3-4) is marked ``slow`` and has a
deterministic twin: `Timeout` tests block on events instead of sleeping,
and the `RateLimit` twin runs on a fake clock.
"""

from __future__ import annotations

import contextvars
import logging
import pickle
import threading
import time
from collections import deque
from typing import Any

import pytest
from pipekit_ops import (
    Const,
    Identity,
    Lambda,
    Operator,
    RateLimit,
    Retry,
    Signature,
    Timeout,
    Until,
    dumps,
    flow,
    loads,
)


# --- test operators ---------------------------------------------------------


class Sleep(Operator):
    """Sleep ``seconds``, then return the input."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def _apply(self, x: Any) -> Any:
        time.sleep(self.seconds)
        return x


class Blocked(Operator):
    """Block the first ``n_hangs`` calls on ``release``, return the input after.

    ``finished`` is set whenever a blocked call returns, so a test can wait
    for its abandoned threads before it ends.
    """

    def __init__(self, n_hangs: int) -> None:
        self.n_hangs = n_hangs
        self.n_calls = 0
        self.release = threading.Event()
        self.finished = threading.Semaphore(0)

    def _apply(self, x: Any) -> Any:
        self.n_calls += 1
        if self.n_calls <= self.n_hangs:
            self.release.wait()
            self.finished.release()
        return x

    def unblock(self, n_blocked: int) -> None:
        """Release every blocked call and wait for ``n_blocked`` to return."""
        self.release.set()
        for _ in range(n_blocked):
            assert self.finished.acquire(timeout=5.0)


class HangThenReturn(Operator):
    """Sleep ``seconds`` on the first ``n_hangs`` calls, return at once after."""

    def __init__(self, n_hangs: int, seconds: float) -> None:
        self.n_hangs = n_hangs
        self.seconds = seconds
        self.n_calls = 0

    def _apply(self, x: Any) -> Any:
        self.n_calls += 1
        if self.n_calls <= self.n_hangs:
            time.sleep(self.seconds)
        return x


class RecordStart(Operator):
    """Append the clock reading at each start, then return the input."""

    def __init__(self, clock: Any = time.monotonic) -> None:
        self.clock = clock
        self.starts: list[float] = []

    def _apply(self, x: Any) -> Any:
        self.starts.append(self.clock())
        return x


class Increment(Operator):
    """x ↦ x + 1, counting its calls."""

    def __init__(self) -> None:
        self.n_calls = 0

    def _apply(self, x: int) -> int:
        self.n_calls += 1
        return x + 1


class AtLeast(Operator):
    """x ↦ x ≥ ``bound``."""

    def __init__(self, bound: int) -> None:
        self.bound = bound

    def _apply(self, x: int) -> bool:
        return x >= self.bound


class AsyncRead(Operator):
    """An operator with an ``async def _apply``."""

    async def _apply(self, x: Any) -> Any:
        return x


class RecordingDeque(deque):
    """A deque that also logs every value appended to it.

    Swapped in for `RateLimit._starts`, it records the admission times the
    limiter itself decides, under its lock, free of thread-scheduling
    jitter between admission and the operator's first line.
    """

    def __init__(self, maxlen: int) -> None:
        super().__init__(maxlen=maxlen)
        self.log: list[float] = []

    def append(self, value: float) -> None:
        self.log.append(value)
        super().append(value)


class FakeTime:
    """A shared fake clock that only `sleep` advances.

    Each thread's sleep wakes at its own last clock reading plus the delay,
    as a real sleep would, so concurrent sleepers do not add up.
    """

    def __init__(self) -> None:
        self.now = 0.0
        self.lock = threading.Lock()
        self.last_read = threading.local()

    def clock(self) -> float:
        with self.lock:
            self.last_read.value = self.now
            return self.now

    def sleep(self, seconds: float) -> None:
        with self.lock:
            self.now = max(self.now, self.last_read.value + seconds)


def _fake_limiter(calls: int, period: float) -> tuple[RateLimit, RecordStart]:
    fake = FakeTime()
    record = RecordStart(clock=fake.clock)
    limiter = RateLimit(record, calls=calls, period=period)
    limiter._clock = fake.clock  # type: ignore[method-assign]
    limiter._sleep = fake.sleep  # type: ignore[method-assign]
    return limiter, record


def _max_starts_per_window(starts: list[float], period: float) -> int:
    """The most start times in any half-open window [t, t + period)."""
    ordered = sorted(starts)
    most = 0
    for i, first in enumerate(ordered):
        inside = [t for t in ordered[i:] if t < first + period]
        most = max(most, len(inside))
    return most


def _call_from_threads(op: Operator, n_calls: int, n_threads: int) -> None:
    """Call ``op`` ``n_calls`` times in total, spread over ``n_threads``."""
    shares = [
        n_calls // n_threads + (i < n_calls % n_threads) for i in range(n_threads)
    ]

    def worker(share: int) -> None:
        for _ in range(share):
            op(0)

    threads = [threading.Thread(target=worker, args=(share,)) for share in shares]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


@pytest.fixture(autouse=True)
def _no_abandoned_threads():
    """Start each test with no abandoned `Timeout` thread from an earlier one."""
    deadline = time.monotonic() + 5.0
    while flow._ABANDONED and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not flow._ABANDONED, "an earlier test left a Timeout thread running"


# --- Timeout ----------------------------------------------------------------


@pytest.mark.slow
def test_g3_1_timeout_fires_within_0_15_s():
    timed = Timeout(Sleep(1.0), seconds=0.1)
    start = time.perf_counter()
    with pytest.raises(TimeoutError, match=r"Sleep did not return within 0\.1 s"):
        timed(1)
    assert time.perf_counter() - start < 0.15


def test_g3_1_timeout_fires_on_a_blocked_call():
    blocked = Blocked(n_hangs=1)
    with pytest.raises(TimeoutError):
        Timeout(blocked, seconds=0.05)(1)
    assert len(flow._ABANDONED) == 1
    blocked.unblock(n_blocked=1)
    # the thread left the abandoned set on its way out
    assert not flow._ABANDONED


def test_g3_2_timeout_passes_the_output_through():
    assert Timeout(Sleep(0.01), seconds=0.1)("tile") == "tile"


@pytest.mark.slow
def test_g3_3_retry_on_timeout_returns_within_0_35_s():
    read = HangThenReturn(n_hangs=2, seconds=1.0)
    sturdy = Retry(Timeout(read, 0.1), attempts=3, base_delay=0, on=(TimeoutError,))
    start = time.perf_counter()
    assert sturdy("tile") == "tile"
    assert time.perf_counter() - start < 0.35
    assert read.n_calls == 3


def test_g3_3_retry_on_timeout_with_blocked_calls():
    blocked = Blocked(n_hangs=2)
    sturdy = Retry(Timeout(blocked, 0.05), attempts=3, base_delay=0, on=(TimeoutError,))
    assert sturdy("tile") == "tile"
    assert blocked.n_calls == 3
    blocked.unblock(n_blocked=2)


def test_timeout_reraises_the_operators_own_error():
    with pytest.raises(ZeroDivisionError):
        Timeout(Lambda(lambda x: 1 / x), seconds=1.0)(0)
    assert not flow._ABANDONED


def test_timeout_warns_when_abandoned_threads_pile_up(caplog):
    blocked = Blocked(n_hangs=2)
    timed = Timeout(blocked, seconds=0.01, max_abandoned=1)
    with caplog.at_level(logging.WARNING, logger="pipekit_ops.flow"):
        for _ in range(2):
            with pytest.raises(TimeoutError):
                timed(1)
    blocked.unblock(n_blocked=2)
    warnings = [r for r in caplog.records if r.name == "pipekit_ops.flow"]
    # the first abandoned thread is within the limit, the second is not
    assert len(warnings) == 1
    assert "2 abandoned threads" in warnings[0].getMessage()


def test_timeout_runs_in_the_callers_context():
    tile = contextvars.ContextVar("tile", default="none")
    read_tile = Lambda(lambda _: tile.get())
    token = tile.set("T31TBF")
    try:
        assert Timeout(read_tile, seconds=1.0)(0) == "T31TBF"
    finally:
        tile.reset(token)


def test_timeout_reports_the_wrapped_signature():
    signature = Signature(dims=(("y", 4), ("x", 4)), dtype="float32")
    assert Timeout(Identity(), 1.0).compute_output_signature(signature) == signature


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"seconds": 0.0}, "seconds"), ({"seconds": 1.0, "max_abandoned": -1}, "max_")],
)
def test_timeout_refuses_bad_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Timeout(Identity(), **kwargs)


# --- RateLimit --------------------------------------------------------------


@pytest.mark.slow
def test_g3_4_rate_limit_from_8_threads():
    record = RecordStart()
    limiter = RateLimit(record, calls=10, period=1.0)
    admitted = RecordingDeque(maxlen=10)
    limiter._starts = admitted
    start = time.perf_counter()
    _call_from_threads(limiter, n_calls=50, n_threads=8)
    elapsed = time.perf_counter() - start
    assert len(record.starts) == 50
    assert 4.0 <= elapsed <= 4.5
    assert _max_starts_per_window(admitted.log, period=1.0) <= 10


def test_g3_4_rate_limit_on_a_fake_clock_one_thread():
    limiter, record = _fake_limiter(calls=10, period=1.0)
    for _ in range(50):
        limiter(0)
    # five full windows: starts at 0, 1, 2, 3 and 4 s, ten in each
    assert record.starts == [float(t // 10) for t in range(50)]


def test_g3_4_rate_limit_on_a_fake_clock_8_threads():
    limiter, record = _fake_limiter(calls=10, period=1.0)
    _call_from_threads(limiter, n_calls=50, n_threads=8)
    assert len(record.starts) == 50
    assert 4.0 <= max(record.starts) <= 4.5
    assert _max_starts_per_window(record.starts, period=1.0) <= 10


def test_rate_limit_copies_start_with_no_history():
    limiter = RateLimit(Identity(), calls=1, period=60.0)
    limiter(0)
    copy = pickle.loads(pickle.dumps(limiter))
    assert len(copy._starts) == 0
    assert copy._starts.maxlen == 1
    assert copy._lock is not limiter._lock
    reloaded = loads(dumps(RateLimit(Identity(), calls=1, period=60.0)))
    assert len(reloaded._starts) == 0
    assert isinstance(reloaded._lock, type(threading.Lock()))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"calls": 0, "period": 1.0}, "calls"), ({"calls": 1, "period": 0.0}, "period")],
)
def test_rate_limit_refuses_bad_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        RateLimit(Identity(), **kwargs)


# --- Until ------------------------------------------------------------------


def test_g3_5_until_raises_after_exactly_max_iter_calls():
    step = Increment()
    with pytest.raises(RuntimeError, match="max_iter=5"):
        Until(step, Const(False), max_iter=5)(0)
    assert step.n_calls == 5


def test_g3_5_until_returns_x_5_on_max():
    step = Increment()
    assert Until(step, Const(False), max_iter=5, on_max="return")(0) == 5
    assert step.n_calls == 5


def test_g3_6_until_checks_the_input_first():
    step = Increment()
    assert Until(step, Const(True), max_iter=5)(0) == 0
    assert step.n_calls == 0


def test_until_returns_the_first_value_that_passes():
    step = Increment()
    assert Until(step, AtLeast(3), max_iter=10)(0) == 3
    assert step.n_calls == 3
    # pred(x_m) holding is a success, not a budget overrun
    assert Until(Increment(), AtLeast(3), max_iter=3)(0) == 3


def test_until_tracks_no_signature():
    assert (
        Until(Increment(), AtLeast(3), max_iter=3).compute_output_signature(1) is None
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"max_iter": -1}, "max_iter"), ({"max_iter": 3, "on_max": "skip"}, "on_max")],
)
def test_until_refuses_bad_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Until(Increment(), AtLeast(3), **kwargs)


# --- arguments that must be operators ---------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda f: Timeout(f, seconds=1.0),
        lambda f: RateLimit(f, calls=1, period=1.0),
        lambda f: Until(f, Const(True), max_iter=1),
        lambda f: Until(Identity(), f, max_iter=1),
    ],
    ids=["Timeout.op", "RateLimit.op", "Until.op", "Until.pred"],
)
def test_bare_callables_are_pointed_to_lambda(build):
    with pytest.raises(TypeError, match="Lambda"):
        build(lambda x: x)


def test_non_callables_are_refused():
    with pytest.raises(TypeError, match="must be an Operator"):
        Timeout(3, seconds=1.0)  # type: ignore[arg-type]


# --- G3-7: async children ---------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda op: Retry(op, attempts=2, base_delay=0.0, on=(OSError,)),
        lambda op: Timeout(op, seconds=1.0),
        lambda op: RateLimit(op, calls=1, period=1.0),
        lambda op: Until(op, Const(True), max_iter=1),
        lambda op: Until(Identity(), op, max_iter=1),
    ],
    ids=["Retry", "Timeout", "RateLimit", "Until.op", "Until.pred"],
)
def test_g3_7_async_children_are_refused(build):
    with pytest.raises(TypeError, match="AsyncRead, whose _apply is async"):
        build(AsyncRead())


# --- JAX tracers (decision O16) ---------------------------------------------


@pytest.mark.parametrize(
    "build",
    [lambda op: Timeout(op, seconds=1.0), lambda op: RateLimit(op, 1, 1.0)],
    ids=["Timeout", "RateLimit"],
)
def test_tracers_are_refused(build):
    jax = pytest.importorskip("jax")
    wrapped = build(Identity())
    with pytest.raises(TypeError, match="JAX tracer"):
        jax.jit(wrapped)(1.0)


# --- G0-4 -------------------------------------------------------------------


def test_g0_4_no_flow_operator_is_flagged():
    assert not any(cls.forbid_in_yaml for cls in (Timeout, RateLimit, Until))
