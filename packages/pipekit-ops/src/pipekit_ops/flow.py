"""Flow limits: `Timeout`, `RateLimit` and `Until`.

`Timeout` bounds how long one call may take, `RateLimit` bounds how often
calls may start, and `Until` repeats an operator up to a fixed number of
times. All three are sync wrappers: a child whose ``_apply`` is an
``async def`` is refused at construction (decision O15), as `Retry` does.

Design: ``packages/pipekit-ops/docs/design/api/operators.md`` §3.
"""

from __future__ import annotations

import contextvars
import inspect
import logging
import threading
import time
from collections import deque
from concurrent.futures import Future, wait
from typing import Any

from pipekit_ops._base.operator import Operator, nested_config, require_operator
from pipekit_ops._base.tracing import is_traced


__all__ = ["RateLimit", "Timeout", "Until"]

log = logging.getLogger("pipekit_ops.flow")

ON_MAX = ("raise", "return")

#: futures of `Timeout` calls that timed out while their thread still runs
_ABANDONED: set[Future] = set()
_ABANDONED_LOCK = threading.Lock()


def require_sync(op: Operator, owner: str, field: str) -> None:
    """Raise `TypeError` if ``op`` has an ``async def _apply`` (decision O15).

    A sync wrapper calling such a child gets an un-awaited coroutine back,
    so nothing runs; refusing at construction makes that failure loud.

    Args:
        op: The child operator.
        owner: The wrapping class's name, for the message.
        field: The constructor argument that holds ``op``.
    """
    if inspect.iscoroutinefunction(type(op)._apply):
        raise TypeError(
            f"{owner}.{field} is {type(op).__name__}, whose _apply is async; "
            f"{owner} is sync and would return an un-awaited coroutine. Run it "
            'with Map(executor="async") instead; async-aware wrappers are '
            "planned (phase P6)."
        )


def _require_child(value: Any, owner: str, field: str) -> None:
    """Raise `TypeError` unless ``value`` is a sync `Operator`.

    A bare callable gets a message pointing to `Lambda` (the conventions in
    ``api/README.md``).
    """
    if callable(value) and not isinstance(value, Operator):
        name = getattr(value, "__name__", type(value).__name__)
        raise TypeError(
            f"{owner}.{field} must be an Operator, got the callable {name!r}. "
            "Wrap it in Lambda(...), or write a small Operator so the "
            "pipeline still round-trips."
        )
    require_operator(value, owner, field)
    require_sync(value, owner, field)


class Timeout(Operator):
    """Raise `TimeoutError` if ``op`` has not returned after ``seconds``.

    y = op(x) if op returns within s seconds, else `TimeoutError`.

    Algorithm:
        1. Start a daemon thread running ``op(x)`` in a copy of the
           caller's context, so `executor_scope` stays visible.
        2. Wait at most ``seconds`` for it to finish.
        3. If it finished, return its result or re-raise its error;
           otherwise abandon the thread and raise `TimeoutError`.

    The thread is abandoned, not killed: Python cannot stop a thread
    safely, so it runs until ``op`` returns and its result is dropped.
    That suits the case this exists for, a network read with no timeout of
    its own. When more than ``max_abandoned`` abandoned threads are still
    running in the process, each further timeout logs a warning on
    ``pipekit_ops.flow``. A process-based timeout is a later phase.

    Args:
        op: The operator to bound.
        seconds: The wait s > 0, in seconds.
        max_abandoned: Abandoned threads tolerated, process-wide, before
            each timeout logs a warning.

    Raises:
        TypeError: ``op`` is not an `Operator`, or its ``_apply`` is async;
            or, at call time, the input is a JAX tracer.
        ValueError: ``seconds`` ≤ 0 or ``max_abandoned`` < 0.

    Examples:
        A call that returns in time passes through:

        >>> from pipekit_ops import Identity
        >>> Timeout(Identity(), seconds=1.0)(3)
        3

        A call that hangs raises the builtin `TimeoutError`:

        >>> import time
        >>> from pipekit_ops import Lambda
        >>> hangs = Lambda(lambda x: time.sleep(0.2))
        >>> Timeout(hangs, seconds=0.01)(1)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        TimeoutError: Timeout: Lambda did not return within 0.01 s

        Retry a read that sometimes hangs:

        >>> from pipekit_ops import Retry
        >>> sturdy = Retry(
        ...     Timeout(Identity(), seconds=1.0),
        ...     attempts=3,
        ...     base_delay=0.0,
        ...     on=(TimeoutError,),
        ... )
        >>> sturdy(5)
        5
    """

    __config_mixin_auto__ = False

    def __init__(self, op: Operator, seconds: float, max_abandoned: int = 8) -> None:
        _require_child(op, "Timeout", "op")
        if seconds <= 0:
            raise ValueError(f"Timeout.seconds must be > 0, got {seconds}.")
        if max_abandoned < 0:
            raise ValueError(
                f"Timeout.max_abandoned must be >= 0, got {max_abandoned}."
            )
        self.op = op
        self.seconds = seconds
        self.max_abandoned = max_abandoned

    def _apply(self, x: Any) -> Any:
        if is_traced(x):
            raise TypeError(
                "Timeout cannot run on a JAX tracer: it calls the operator in "
                "another thread, and JAX's trace state is thread-local."
            )
        future: Future = Future()
        # scopes (executor_scope, store_scope) stay visible in the thread
        context = contextvars.copy_context()
        # a daemon thread: an abandoned call must not keep the process alive
        thread = threading.Thread(
            target=_run_in_thread,
            args=(future, context, self.op, x),
            name="pipekit-timeout",
            daemon=True,
        )
        thread.start()
        wait([future], timeout=self.seconds)
        if not self._abandon(future):
            return future.result()
        raise TimeoutError(
            f"Timeout: {type(self.op).__name__} did not return within "
            f"{self.seconds} s; its thread is abandoned and keeps running."
        )

    def _abandon(self, future: Future) -> bool:
        """Mark ``future`` abandoned unless it has finished; warn on a pile-up."""
        with _ABANDONED_LOCK:
            # checked under the lock: the thread may finish right after `wait`
            if future.done():
                return False
            _ABANDONED.add(future)
            n_abandoned = len(_ABANDONED)
        if n_abandoned > self.max_abandoned:
            log.warning(
                "Timeout: %d abandoned threads are still running (max_abandoned"
                "=%d); %s may be hanging for good.",
                n_abandoned,
                self.max_abandoned,
                type(self.op).__name__,
            )
        return True

    def get_config(self) -> dict[str, Any]:
        return {
            "op": nested_config(self.op),
            "seconds": self.seconds,
            "max_abandoned": self.max_abandoned,
        }

    def compute_output_signature(self, input_signature: Any) -> Any:
        """The wrapped operator's: a time limit does not change the output."""
        return self.op.compute_output_signature(input_signature)


def _run_in_thread(
    future: Future, context: contextvars.Context, op: Operator, x: Any
) -> None:
    """`Timeout`'s thread body: settle ``future``, then leave the abandoned set."""
    try:
        future.set_result(context.run(op, x))
    except BaseException as exc:
        # the caller re-raises it; a thread has nowhere else to send it
        future.set_exception(exc)
    finally:
        with _ABANDONED_LOCK:
            _ABANDONED.discard(future)


class RateLimit(Operator):
    """At most ``calls`` starts of ``op`` in any window of ``period`` seconds.

    #{starts in [t, t + T)} ≤ n for every t, with n = ``calls``, T = ``period``.

    Algorithm:
        1. Under a lock, read the clock t and the oldest of the last n
           start times, t₀.
        2. If fewer than n calls have started, or t - t₀ ≥ T, record t as
           a start, release the lock and call ``op(x)``.
        3. Otherwise release the lock, sleep t₀ + T - t, and go to step 1.

    The limiter is shared by every thread in the process; the sleep
    happens outside the lock. It does not cross processes: on a pool of p
    processes each worker has its own limiter, so the service sees up to
    p n starts per T. A pickled or reloaded copy starts with no history.

    Placement matters: ``Retry(RateLimit(read))`` makes every attempt wait
    for a slot, while ``RateLimit(Retry(read))`` lets one slot cover all of
    them.

    Args:
        op: The operator to limit.
        calls: The number of starts n ≥ 1 allowed per window.
        period: The window T > 0, in seconds.

    Raises:
        TypeError: ``op`` is not an `Operator`, or its ``_apply`` is async;
            or, at call time, the input is a JAX tracer.
        ValueError: ``calls`` < 1 or ``period`` ≤ 0.

    Examples:
        Within the limit, calls pass straight through:

        >>> from pipekit_ops import Identity
        >>> limited = RateLimit(Identity(), calls=2, period=60.0)
        >>> [limited(1), limited(2)]
        [1, 2]

        A pickled copy forgets the history, so it may start at once:

        >>> import pickle
        >>> pickle.loads(pickle.dumps(limited))(3)
        3

        Inside `Retry`, every attempt waits for its own slot:

        >>> from pipekit_ops import Retry
        >>> read = RateLimit(Identity(), calls=5, period=1.0)
        >>> Retry(read, attempts=2, base_delay=0.0, on=(OSError,))(4)
        4
    """

    __config_mixin_auto__ = False

    def __init__(self, op: Operator, calls: int, period: float) -> None:
        _require_child(op, "RateLimit", "op")
        if calls < 1:
            raise ValueError(f"RateLimit.calls must be >= 1, got {calls}.")
        if period <= 0:
            raise ValueError(f"RateLimit.period must be > 0, got {period}.")
        self.op = op
        self.calls = calls
        self.period = period
        self._starts: deque[float] = deque(maxlen=calls)
        self._lock = threading.Lock()

    def _apply(self, x: Any) -> Any:
        if is_traced(x):
            raise TypeError(
                "RateLimit cannot run on a JAX tracer: under jax.jit the "
                "operator runs once, at trace time, so there is nothing to limit."
            )
        self._acquire()
        return self.op(x)

    def _acquire(self) -> None:
        """Block until a start is allowed, then record it."""
        while True:
            with self._lock:
                now = self._clock()
                # the deque holds the last n starts; [0] is the oldest
                if len(self._starts) < self.calls:
                    self._starts.append(now)
                    return
                # t - t₀ ≥ T: the oldest start has left the window
                if now - self._starts[0] >= self.period:
                    self._starts.append(now)
                    return
                # t₀ + T - t
                delay = self._starts[0] + self.period - now
            self._sleep(delay)

    def _clock(self) -> float:
        """Hook for tests to substitute a fake clock."""
        return time.monotonic()

    def _sleep(self, seconds: float) -> None:
        """Hook for tests to monkey-patch out real sleeping."""
        time.sleep(seconds)

    def __getstate__(self) -> dict[str, Any]:
        # A lock does not pickle; a copy sent to a process pool starts with
        # no history and its own lock (decision O10).
        state = dict(vars(self))
        del state["_lock"]
        state["_starts"] = deque(maxlen=self.calls)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        vars(self).update(state)
        self._lock = threading.Lock()

    def get_config(self) -> dict[str, Any]:
        return {
            "op": nested_config(self.op),
            "calls": self.calls,
            "period": self.period,
        }

    def compute_output_signature(self, input_signature: Any) -> Any:
        """The wrapped operator's: a rate limit does not change the output."""
        return self.op.compute_output_signature(input_signature)


class Until(Operator):
    """Apply ``op`` until ``pred`` holds, at most ``max_iter`` times.

    x₀ = x,  xᵢ₊₁ = f(xᵢ);  returns x_k for the least k with pred(x_k)
    truthy, k ≤ m, where f is ``op`` and m is ``max_iter``.

    Algorithm:
        1. Set x₀ = x and i = 0; if pred(xᵢ) is truthy, return xᵢ.
        2. If i = m, raise `RuntimeError` (``on_max="raise"``) or return
           x_m (``on_max="return"``).
        3. Set xᵢ₊₁ = f(xᵢ), i ← i + 1, and go to step 1.

    The stateless sibling of ``pipekit_cycle.Recurrence``, which threads a
    carry state. ``max_iter`` is required, so a pipeline cannot loop
    forever.

    Args:
        op: The step f.
        pred: The stopping test, an `Operator` returning a truthy value.
        max_iter: The most calls of f allowed, m ≥ 0.
        on_max: ``"raise"`` or ``"return"``, when pred(x_m) is falsy.

    Raises:
        TypeError: ``op`` or ``pred`` is not an `Operator` (a bare callable
            is pointed to `Lambda`), or its ``_apply`` is async.
        ValueError: ``max_iter`` < 0 or ``on_max`` is not a choice.
        RuntimeError: At call time, under ``on_max="raise"``, if pred(x_m)
            is falsy.

    Examples:
        Double until past 10:

        >>> from pipekit_ops import Lambda
        >>> double = Lambda(lambda x: 2 * x)
        >>> past_ten = Lambda(lambda x: x > 10)
        >>> Until(double, past_ten, max_iter=10)(1)
        16

        The input is checked first, so f may run zero times:

        >>> Until(double, past_ten, max_iter=10)(50)
        50

        Return x_m instead of raising when the budget runs out:

        >>> Until(double, past_ten, max_iter=2, on_max="return")(1)
        4
        >>> Until(double, past_ten, max_iter=2)(1)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        RuntimeError: Until: pred still false after max_iter=2 calls of Lambda
    """

    __config_mixin_auto__ = False

    def __init__(
        self, op: Operator, pred: Operator, max_iter: int, on_max: str = "raise"
    ) -> None:
        _require_child(op, "Until", "op")
        _require_child(pred, "Until", "pred")
        if max_iter < 0:
            raise ValueError(f"Until.max_iter must be >= 0, got {max_iter}.")
        if on_max not in ON_MAX:
            raise ValueError(f"Until.on_max must be one of {ON_MAX}, got {on_max!r}.")
        self.op = op
        self.pred = pred
        self.max_iter = max_iter
        self.on_max = on_max

    def _apply(self, x: Any) -> Any:
        value = x
        n_calls = 0
        while not self.pred(value):
            if n_calls == self.max_iter:
                if self.on_max == "return":
                    return value
                raise RuntimeError(
                    f"Until: pred still false after max_iter={self.max_iter} "
                    f"calls of {type(self.op).__name__}."
                )
            # xᵢ₊₁ = f(xᵢ)
            value = self.op(value)
            n_calls += 1
        return value

    def get_config(self) -> dict[str, Any]:
        return {
            "op": nested_config(self.op),
            "pred": nested_config(self.pred),
            "max_iter": self.max_iter,
            "on_max": self.on_max,
        }

    def compute_output_signature(self, input_signature: Any) -> Any:
        """``None``: the number of steps, and so the output, is data-dependent."""
        return None
