"""Execution: one engine, `Map`, that runs an operator over many items.

Where the work runs is an argument (``"serial"``, ``"thread"``,
``"process"``, ``"async"``) or a deployment choice bound around the call
(``"scope"`` with `executor_scope`). Every executor shares one window
loop (`Map._drive`): input is pulled lazily, at most ``max_in_flight``
items are held at a time, and ``on_error`` acts on items, not on chunks.
`Prefetch` and `Fold` are built on it.

Design: ``packages/pipekit-ops/docs/design/api/execution.md``.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import functools
import inspect
import itertools
import logging
import multiprocessing
import os
import pickle
import threading
import time
from collections.abc import Callable, Generator, Iterable, Iterator, Sized
from concurrent.futures import (
    FIRST_COMPLETED,
    Executor,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass
from typing import Any, ClassVar

from pipekit_ops._base.operator import Operator, nested_config, require_operator
from pipekit_ops._base.tracing import is_traced
from pipekit_ops.store import Store, bind_stores, bound_stores


__all__ = ["Failed", "Fold", "Map", "MapError", "Prefetch", "executor_scope"]

# The design names this logger for every map's failure log (§2.4).
log = logging.getLogger("pipekit_ops.parallel")

EXECUTORS = ("serial", "thread", "process", "async", "scope")
ON_ERROR = ("raise", "collect", "log_and_continue")
START_METHODS = ("spawn", "fork", "forkserver")
#: default window for a borrowed executor, whose worker count is unknown
BORROWED_WINDOW = 32

#: the executor bound by `executor_scope`; ``None`` means serial
_SCOPED_EXECUTOR: contextvars.ContextVar[Executor | None] = contextvars.ContextVar(
    "pipekit_ops_executor", default=None
)

#: one task's outcomes: ``(index, succeeded, value or exception)`` per item
Outcome = tuple[int, bool, Any]
Chunk = list[tuple[int, Any]]


# --- public records --------------------------------------------------------


@dataclass(frozen=True)
class Failed:
    """The slot of an item that failed under ``on_error="collect"``.

    Args:
        index: The item's position in the input.
        exception: What the item raised.

    Examples:
        >>> Failed(3, ValueError("bad"))
        Failed(index=3, exception=ValueError('bad'))
    """

    index: int
    exception: Exception


class MapError(ExceptionGroup):
    """Every failure of a ``Map(on_error="collect")`` run, with the partial results.

    An `ExceptionGroup` with one exception per failed item, in input
    order; each carries the note ``"Map: item i failed"``.

    Attributes:
        results: The output list, input order, with a `Failed` record in
            each failed slot; ``None`` when raised by `Map.iter`, which
            keeps no results.

    Examples:
        >>> error = MapError("1 of 2 items failed", [ValueError("x")], [1, None])
        >>> len(error.exceptions), error.results
        (1, [1, None])
    """

    results: list[Any] | None

    def __new__(
        cls,
        message: str,
        exceptions: list[Exception],
        results: list[Any] | None = None,
    ) -> MapError:
        self = super().__new__(cls, message, exceptions)
        self.results = results
        return self

    def __init__(
        self,
        message: str,
        exceptions: list[Exception],
        results: list[Any] | None = None,
    ) -> None:
        super().__init__(message, exceptions)

    def derive(self, excs: Any) -> MapError:
        # `except*` splits keep the partial results
        return MapError(self.message, list(excs), self.results)


@contextlib.contextmanager
def executor_scope(executor: Executor | None) -> Iterator[Executor | None]:
    """Bind the executor that ``Map(executor="scope")`` borrows inside the block.

    The executor is not part of any config, so a pipeline hashes the same
    on a laptop and on a cluster. `Map` never shuts a borrowed executor
    down. Inside a `Map` task the scope is reset to serial, so a nested
    ``Map(executor="scope")`` cannot deadlock the pool that runs it.

    Args:
        executor: Any `concurrent.futures.Executor` (a thread pool, a
            process pool, ``distributed.Client.get_executor()``), or
            ``None`` for serial.

    Examples:
        >>> from concurrent.futures import ThreadPoolExecutor
        >>> from pipekit_ops import Identity
        >>> pool = ThreadPoolExecutor(max_workers=2)
        >>> with executor_scope(pool):
        ...     Map(Identity(), executor="scope")([1, 2, 3])
        [1, 2, 3]
        >>> pool.submit(abs, -1).result()
        1
        >>> pool.shutdown()
    """
    token = _SCOPED_EXECUTOR.set(executor)
    try:
        yield executor
    finally:
        _SCOPED_EXECUTOR.reset(token)


# --- worker side: what a task runs ----------------------------------------


def _run_chunk(op: Operator, chunk: Chunk, isolate: bool = True) -> list[Outcome]:
    """Apply ``op`` to each item of a chunk, catching each item's error.

    Errors stay per item, so ``on_error`` acts on items, not on chunks.
    `KeyboardInterrupt` and `SystemExit` are not caught. With ``isolate``
    the executor scope is serial for the task (no nested pools, §2.7).
    """
    token = _SCOPED_EXECUTOR.set(None) if isolate else None
    try:
        outcomes: list[Outcome] = []
        for index, item in chunk:
            try:
                outcomes.append((index, True, op(item)))
            except Exception as exc:
                outcomes.append((index, False, exc))
        return outcomes
    finally:
        if token is not None:
            _SCOPED_EXECUTOR.reset(token)


#: the operator a process-pool worker runs, unpickled once per worker
_WORKER_OP: Operator | None = None


def _init_process_worker(op: Operator, stores: dict[str, Store] | None = None) -> None:
    """Process-pool initializer: keep ``op``, pickled once per worker.

    ``stores`` are the caller's `store_scope` bindings, pickled with it,
    so a ``Cache(store=…)`` inside ``op`` finds its store in the worker.
    """
    global _WORKER_OP
    _WORKER_OP = op
    if stores:
        bind_stores(stores)


def _run_chunk_in_process(chunk: Chunk) -> list[Outcome]:
    """Run a chunk with the operator `_init_process_worker` stored."""
    assert _WORKER_OP is not None, "process worker started without an operator"
    return _run_chunk(_WORKER_OP, chunk)


def _run_chunk_with_stores(
    op: Operator, stores: dict[str, Store], chunk: Chunk
) -> list[Outcome]:
    """`_run_chunk` under the caller's store bindings (a remote task)."""
    context = contextvars.copy_context()

    def run() -> list[Outcome]:
        bind_stores(stores)
        return _run_chunk(op, chunk)

    return context.run(run)


async def _run_chunk_async(op: Operator, chunk: Chunk) -> list[Outcome]:
    """Async twin of `_run_chunk`: await a coroutine ``_apply``, else use a thread."""
    _SCOPED_EXECUTOR.set(None)
    is_coroutine = inspect.iscoroutinefunction(op._apply)
    outcomes: list[Outcome] = []
    for index, item in chunk:
        try:
            if is_coroutine:
                op._check_backends((item,), {})
                value = await op._apply(item)
            else:
                value = await asyncio.to_thread(op, item)
            outcomes.append((index, True, value))
        except Exception as exc:
            outcomes.append((index, False, exc))
    return outcomes


class _InlineExecutor(Executor):
    """Runs each task in the calling thread at submit: the serial executor."""

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future:
        future: Future = Future()
        future.set_result(fn(*args, **kwargs))
        return future


# --- the engine -------------------------------------------------------------

#: ``submit(chunk) -> Future`` for one executor
Submit = Callable[[Chunk], Future]


class Map(Operator):
    """Apply ``op`` to every item, on any executor, with a bounded window.

    yᵢ = op(xᵢ) for i = 0 … n - 1, returned in input order.

    Algorithm:
        1. Resolve the executor; for a process pool or a borrowed
           non-thread executor, check that ``op`` pickles.
        2. Pull items lazily, ``chunksize`` at a time, while fewer than
           W = ``max_in_flight`` items are held, and submit each chunk.
        3. Wait for the first completion, buffer its results, release
           them (as completed, or in input order for ``iter``), and
           apply ``on_error`` to each failed item.
        4. On a failure under ``"raise"``, or any exit, cancel what has
           not started and shut down an owned pool.

    Args:
        op: The operator run on each item.
        executor: ``"serial"`` (the calling thread), ``"thread"``,
            ``"process"``, ``"async"``, or ``"scope"`` (the executor bound
            by `executor_scope`, serial when none is).
        n_workers: Workers of an owned pool. Default 1 serial, 8 thread
            and async, the CPU count for process; ignored for ``"scope"``.
        max_in_flight: The window W: items pulled and not yet released.
            Default 2 n_w for an owned pool and 32 for a borrowed one.
        on_error: ``"raise"`` fails fast and cancels pending work;
            ``"collect"`` runs every item, then raises `MapError`;
            ``"log_and_continue"`` logs a warning on
            ``pipekit_ops.parallel`` and puts ``None`` in the slot.
        chunksize: Items per task c. A chunk is pulled whole, so up to
            W + c - 1 items can be held.
        start_method: Process start method; default ``"spawn"``.
        progress: If set, every ``progress`` released items log one INFO
            record on ``pipekit_ops.parallel`` whose ``done``, ``failed``,
            ``in_flight``, ``n`` (``None`` without ``len``) and ``rate``
            (items per second) attributes hold the counts so far.

    Raises:
        TypeError: ``op`` is flagged by `check_pickleable` or does not
            pickle (process and borrowed non-thread executors), or an item
            is a JAX tracer on any executor but serial.
        RuntimeError: ``executor="async"`` called inside a running event
            loop; ``await map_op.run(items)`` there instead.
        MapError: Under ``on_error="collect"``, if any item failed.

    Examples:
        Threads, results in input order:

        >>> from pipekit_ops import Identity
        >>> Map(Identity(), executor="thread", n_workers=2)([3, 1, 2])
        [3, 1, 2]

        Collect failures, keep the rest:

        >>> from pipekit_ops import Lambda
        >>> inverse = Lambda(lambda x: 1 / x)
        >>> try:
        ...     Map(inverse, on_error="collect")([1, 0, 2])
        ... except MapError as error:
        ...     error.results
        [1.0, Failed(index=1, exception=ZeroDivisionError('division by zero')), 0.5]

        Stream lazily, in completion order:

        >>> sorted(Map(Identity(), executor="thread").iter("ab", ordered=False))
        [(0, 'a'), (1, 'b')]

        Log progress every 2 items:

        >>> records = []
        >>> handler = logging.Handler()
        >>> handler.emit = records.append
        >>> log.addHandler(handler)
        >>> log.setLevel(logging.INFO)
        >>> Map(Identity(), progress=2)(range(5))
        [0, 1, 2, 3, 4]
        >>> log.removeHandler(handler)
        >>> log.setLevel(logging.NOTSET)
        >>> [record.done for record in records]
        [2, 4]
    """

    __config_mixin_auto__ = False
    #: start method when ``start_method`` is ``None``; ``None`` is the platform's
    _default_start_method: ClassVar[str | None] = "spawn"

    def __init__(
        self,
        op: Operator,
        executor: str = "serial",
        n_workers: int | None = None,
        max_in_flight: int | None = None,
        on_error: str = "raise",
        chunksize: int = 1,
        start_method: str | None = None,
        progress: int | None = None,
    ) -> None:
        # a preset's errors name the preset
        owner = type(self).__name__
        require_operator(op, owner, "op")
        _require_choice(f"{owner}.executor", executor, EXECUTORS)
        _require_choice(f"{owner}.on_error", on_error, ON_ERROR)
        if start_method is not None:
            _require_choice(f"{owner}.start_method", start_method, START_METHODS)
        for name, value in (
            ("n_workers", n_workers),
            ("max_in_flight", max_in_flight),
            ("progress", progress),
        ):
            if value is not None and value < 1:
                raise ValueError(f"{owner}.{name} must be >= 1, got {value}.")
        if chunksize < 1:
            raise ValueError(f"{owner}.chunksize must be >= 1, got {chunksize}.")
        self.op = op
        self.executor = executor
        self.n_workers = n_workers
        self.max_in_flight = max_in_flight
        self.on_error = on_error
        self.chunksize = chunksize
        self.start_method = start_method
        self.progress = progress

    # --- entry points ---

    def _apply(self, items: Iterable[Any]) -> list[Any]:
        if self.executor == "async":
            _refuse_running_loop()
        return self._collect(self._drive(items, ordered=False))

    async def run(self, items: Iterable[Any]) -> list[Any]:
        """Return the same list as a call, from inside a running event loop.

        With ``executor="async"``, coroutines run on the caller's loop and
        a sync ``op`` on that loop's default thread pool; the window loop
        itself waits in a thread, so the caller's loop is never blocked.

        Examples:
            >>> import asyncio
            >>> from pipekit_ops import Identity
            >>> asyncio.run(Map(Identity(), executor="async").run([1, 2]))
            [1, 2]
        """
        if self.executor != "async":
            return await asyncio.to_thread(self._apply, items)
        loop = asyncio.get_running_loop()
        outcomes = self._drive(items, ordered=False, loop=loop)
        return await asyncio.to_thread(self._collect, outcomes)

    def iter(self, items: Iterable[Any], ordered: bool = True) -> Iterator[Any]:
        """Yield results lazily, holding at most W items at any point.

        With ``ordered=True``, results come in input order, and a finished
        result waiting behind a slower item still counts against W. With
        ``ordered=False``, ``(i, result)`` pairs come as items complete.
        Under ``"collect"``, a failed slot yields a `Failed` record and
        `MapError` (with ``results=None``) is raised once the input is
        exhausted. Close the iterator, or consume it, to free the pool.

        Examples:
            >>> from pipekit_ops import Identity
            >>> stream = Map(Identity(), executor="thread", max_in_flight=2)
            >>> list(stream.iter(range(5)))
            [0, 1, 2, 3, 4]
        """
        if self.executor == "async":
            _refuse_running_loop()
        failures: list[tuple[int, Exception]] = []
        with contextlib.closing(self._drive(items, ordered)) as outcomes:
            for index, succeeded, value in outcomes:
                slot = self._settle(index, succeeded, value, failures)
                yield slot if ordered else (index, slot)
        if failures:
            raise _map_error(failures, n_items=None, results=None)

    def compute_output_signature(self, input_signature: Any) -> Any:
        """``None``: the output is a list of results, not one carrier."""
        return None

    def get_config(self) -> dict[str, Any]:
        config = {
            "op": nested_config(self.op),
            "executor": self.executor,
            "n_workers": self.n_workers,
            "max_in_flight": self.max_in_flight,
            "on_error": self.on_error,
            "chunksize": self.chunksize,
            "start_method": self.start_method,
        }
        # left out when unset, so a Map without progress keeps its hash
        if self.progress is not None:
            config["progress"] = self.progress
        return config

    # --- the error policy ---

    def _collect(self, outcomes: Generator[Outcome]) -> list[Any]:
        """Write each released outcome into its slot; raise per ``on_error``."""
        slots: dict[int, Any] = {}
        failures: list[tuple[int, Exception]] = []
        with contextlib.closing(outcomes):
            for index, succeeded, value in outcomes:
                slots[index] = self._settle(index, succeeded, value, failures)
        results = [slots[i] for i in range(len(slots))]
        if failures:
            raise _map_error(failures, n_items=len(results), results=results)
        return results

    def _settle(
        self,
        index: int,
        succeeded: bool,
        value: Any,
        failures: list[tuple[int, Exception]],
    ) -> Any:
        """The slot value of one outcome under ``on_error``."""
        if succeeded:
            return value
        value.add_note(f"Map: item {index} failed")
        if self.on_error == "raise":
            raise value
        if self.on_error == "log_and_continue":
            log.warning(
                "%s: item %d failed with %s: %s",
                type(self).__name__,
                index,
                type(value).__name__,
                value,
                exc_info=value,
            )
            return None
        failures.append((index, value))
        return Failed(index, value)

    # --- the window loop (§2.3) ---

    def _drive(
        self,
        items: Iterable[Any],
        ordered: bool,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> Generator[Outcome]:
        """Yield ``(index, succeeded, value)`` as items are released.

        Held items are those pulled and not yet released: running or
        queued tasks plus results buffered behind a slower item. At most
        W are held; an item is released when it is yielded.
        """
        n_items = len(items) if isinstance(items, Sized) else None
        with self._open_executor(loop) as (submit, window, refuse_tracers):
            numbered = enumerate(items)
            # each pending task -> the indices of its chunk
            pending: dict[Future, list[int]] = {}
            # finished results waiting to be released, by index
            buffered: dict[int, tuple[bool, Any]] = {}
            held = 0
            next_to_yield = 0
            exhausted = False
            released = 0
            failed = 0
            started = time.perf_counter()
            try:
                while True:
                    # top up the window: pull only while fewer than W items are held
                    while not exhausted and held < window:
                        chunk = list(itertools.islice(numbered, self.chunksize))
                        exhausted = not chunk
                        if chunk:
                            if refuse_tracers:
                                _refuse_tracers(chunk, self.executor)
                            pending[submit(chunk)] = [i for i, _ in chunk]
                            held += len(chunk)
                    if not pending and not buffered:
                        return
                    # release: the next indices in input order, or all finished ones
                    if ordered:
                        upcoming = itertools.count(next_to_yield)
                        run = itertools.takewhile(buffered.__contains__, upcoming)
                        releasable = list(run)
                    else:
                        releasable = list(buffered)
                    for index in releasable:
                        succeeded, value = buffered.pop(index)
                        held -= 1
                        next_to_yield = index + 1
                        released += 1
                        failed += not succeeded
                        if self.progress and released % self.progress == 0:
                            elapsed = time.perf_counter() - started
                            _log_progress(released, failed, held, n_items, elapsed)
                        # each released item frees one slot of the window
                        yield index, succeeded, value
                    if pending:
                        # completion moves results from pending to buffered:
                        # held stays the same
                        done, _ = wait(pending, return_when=FIRST_COMPLETED)
                        for future in done:
                            indices = pending.pop(future)
                            for outcome in _task_outcomes(future, indices):
                                buffered[outcome[0]] = outcome[1:]
            finally:
                # fail fast or early exit: drop what has not started
                for future in pending:
                    future.cancel()

    @contextlib.contextmanager
    def _open_executor(
        self, loop: asyncio.AbstractEventLoop | None
    ) -> Iterator[tuple[Submit, int, bool]]:
        """Yield ``(submit, window W, refuse tracers)``; shut down what is owned."""
        op = self.op
        executor = self.executor
        borrowed = _SCOPED_EXECUTOR.get() if executor == "scope" else None
        submit: Submit

        # --- serial: the calling thread; tracers allowed (§2.9) ---
        if executor == "serial" or (executor == "scope" and borrowed is None):
            window = self.max_in_flight or 2 * (self.n_workers or 1)
            inline = _InlineExecutor()
            submit = functools.partial(inline.submit, _run_chunk, op, isolate=False)
            yield submit, window, False
            return

        # --- borrowed: never shut down (§2.5) ---
        if borrowed is not None:
            window = self.max_in_flight or BORROWED_WINDOW
            if isinstance(borrowed, ThreadPoolExecutor):
                submit = _thread_submit(borrowed, op)
            else:
                _require_pickleable(op, "a borrowed executor")
                # the store_scope bindings travel with each task, pickled
                submit = functools.partial(
                    borrowed.submit, _run_chunk_with_stores, op, bound_stores()
                )
            yield submit, window, True
            return

        # --- owned: created per call, shut down on exit ---
        n_workers = self.n_workers or _default_workers(executor)
        window = self.max_in_flight or 2 * n_workers
        if executor == "thread":
            with _owned(ThreadPoolExecutor(max_workers=n_workers)) as pool:
                yield _thread_submit(pool, op), window, True
        elif executor == "process":
            _require_pickleable(op, 'executor="process"')
            method = self.start_method or self._default_start_method
            context = multiprocessing.get_context(method)
            # op is pickled once per worker, through the initializer
            pool = ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=context,
                initializer=_init_process_worker,
                initargs=(op, bound_stores()),
            )
            with _owned(pool):
                submit = functools.partial(pool.submit, _run_chunk_in_process)
                yield submit, window, True
        else:
            with _event_loop(loop, window) as running_loop:
                yield _loop_submit(running_loop, op), window, True


class Prefetch(Operator):
    """Run ``op`` up to ``depth`` items ahead of the consumer, in input order.

    ``Map(op, executor, max_in_flight=depth).iter(items, ordered=True)``:
    ``depth`` bounds items running, queued, or finished and not yet
    consumed, so a slow item at the head cannot make it read further ahead.
    The result is a lazy iterator; consume or close it to free the pool.

    Args:
        op: The operator run ahead, typically a read.
        depth: Items held ahead of the consumer, d ≥ 1.
        executor: Where ``op`` runs, as for `Map`.

    Raises:
        TypeError: If the input is, or holds, a JAX tracer.

    Examples:
        Read ahead, then work on each result as it arrives:

        >>> from pipekit_ops import Identity
        >>> ahead = Prefetch(Identity(), depth=2)
        >>> list(ahead(range(4)))
        [0, 1, 2, 3]

        Compose with a `Map` that consumes the stream:

        >>> (ahead | Map(Identity()))(range(3))
        [0, 1, 2]
    """

    __config_mixin_auto__ = False

    def __init__(self, op: Operator, depth: int = 2, executor: str = "thread") -> None:
        require_operator(op, "Prefetch", "op")
        if depth < 1:
            raise ValueError(f"Prefetch.depth must be >= 1, got {depth}.")
        self.op = op
        self.depth = depth
        self.executor = executor
        self._map = Map(op, executor=executor, max_in_flight=depth)

    def _apply(self, items: Iterable[Any]) -> Iterator[Any]:
        # a generator is never traced, and JAX warns when handed one
        if not isinstance(items, Iterator) and is_traced(items):
            raise TypeError(_tracer_message("Prefetch"))
        return self._map.iter(items, ordered=True)

    def compute_output_signature(self, input_signature: Any) -> Any:
        """``None``: the output is an iterator of results, not one carrier."""
        return None

    def get_config(self) -> dict[str, Any]:
        return {
            "op": nested_config(self.op),
            "depth": self.depth,
            "executor": self.executor,
        }


class _ReduceChunk(Operator):
    """Left fold of one chunk from ``initial``: the task `Fold` maps."""

    __config_mixin_auto__ = False

    def __init__(self, op: Operator, initial: Any) -> None:
        self.op = op
        self.initial = initial

    def _apply(self, chunk: Iterable[Any]) -> Any:
        return _left_fold(self.op, self.initial, chunk)

    def get_config(self) -> dict[str, Any]:
        return {"op": nested_config(self.op), "initial": self.initial}


class Fold(Operator):
    """Associative fold over chunks, reduced in parallel through `Map`.

    a = c(… c(c(e, f(chunk₀)), f(chunk₁)) …), with f(chunk) the left fold
    of ``op`` from e over the chunk and c the ``combine`` operator.
    Equals the left fold of ``op`` over all items when ``op`` is
    associative and e = ``initial`` is its identity (toolz
    ``sandbox.fold``).

    Algorithm:
        1. Split the input into chunks of ``chunksize`` items.
        2. Left-fold each chunk from ``initial`` with ``op``, through
           ``Map(executor=executor)``.
        3. Left-fold the partial results from ``initial`` with
           ``combine``.

    Args:
        op: Binary step called on one tuple ``(acc, item)``.
        initial: The identity e of ``op``; also the result for no items.
        chunksize: Items per chunk.
        combine: Binary step for the partial results, called on
            ``(acc, partial)``; default ``op``.
        executor: Where the chunks are folded, as for `Map`.

    Examples:
        >>> from pipekit_ops import Operator
        >>> class AddPair(Operator):
        ...     def _apply(self, pair):
        ...         acc, item = pair
        ...         return acc + item
        >>> Fold(AddPair(), 0, chunksize=3, executor="thread")(range(10))
        45
        >>> Fold(AddPair(), 0)([])
        0
    """

    __config_mixin_auto__ = False

    def __init__(
        self,
        op: Operator,
        initial: Any,
        chunksize: int = 128,
        combine: Operator | None = None,
        executor: str = "scope",
    ) -> None:
        require_operator(op, "Fold", "op")
        if combine is not None:
            require_operator(combine, "Fold", "combine")
        if chunksize < 1:
            raise ValueError(f"Fold.chunksize must be >= 1, got {chunksize}.")
        self.op = op
        self.initial = initial
        self.chunksize = chunksize
        self.combine = combine
        self.executor = executor
        self._map = Map(_ReduceChunk(op, initial), executor=executor)

    def _apply(self, items: Iterable[Any]) -> Any:
        chunks = itertools.batched(items, self.chunksize)
        partials = self._map(chunks)
        return _left_fold(self.combine or self.op, self.initial, partials)

    def compute_output_signature(self, input_signature: Any) -> Any:
        """``None``: the fold's value has no tracked signature."""
        return None

    def get_config(self) -> dict[str, Any]:
        combine = None if self.combine is None else nested_config(self.combine)
        return {
            "op": nested_config(self.op),
            "initial": self.initial,
            "chunksize": self.chunksize,
            "combine": combine,
            "executor": self.executor,
        }


# --- helpers ----------------------------------------------------------------


def _left_fold(op: Operator, initial: Any, items: Iterable[Any]) -> Any:
    """aₙ with a₀ = ``initial`` and aᵢ₊₁ = op((aᵢ, xᵢ))."""
    accumulated = initial
    for item in items:
        accumulated = op((accumulated, item))
    return accumulated


def _log_progress(
    done: int, failed: int, in_flight: int, n_items: int | None, elapsed: float
) -> None:
    """One progress record; the counts are also record attributes."""
    rate = done / elapsed if elapsed > 0 else float("inf")
    total = "?" if n_items is None else n_items
    counts = {
        "done": done,
        "failed": failed,
        "in_flight": in_flight,
        "n": n_items,
        "rate": rate,
    }
    log.info(
        "Map: %d/%s done, %d failed, %d in flight, %.1f items/s",
        done,
        total,
        failed,
        in_flight,
        rate,
        extra=counts,
    )


def _task_outcomes(future: Future, indices: list[int]) -> list[Outcome]:
    """The outcomes of a finished task; a whole-task error fails each item."""
    try:
        return future.result()
    except Exception as exc:
        # the task failed as a whole: its result did not pickle, or the
        # pool broke; `KeyboardInterrupt` and `SystemExit` propagate
        return [(index, False, exc) for index in indices]


def _thread_submit(pool: Executor, op: Operator) -> Submit:
    """Submit to a thread pool inside a copy of the caller's context."""

    def submit(chunk: Chunk) -> Future:
        # scopes (executor_scope, store_scope) stay visible in the worker
        context = contextvars.copy_context()
        return pool.submit(context.run, _run_chunk, op, chunk)

    return submit


def _loop_submit(loop: asyncio.AbstractEventLoop, op: Operator) -> Submit:
    """Submit a chunk as a task on ``loop``; cancelling the future cancels it."""

    def submit(chunk: Chunk) -> Future:
        # the task starts in a copy of the submitting thread's context
        coroutine = _run_chunk_async(op, chunk)
        return asyncio.run_coroutine_threadsafe(coroutine, loop)

    return submit


@contextlib.contextmanager
def _owned(pool: Executor) -> Iterator[Executor]:
    """Shut an owned pool down on exit, cancelling queued work: no work outlives it."""
    try:
        yield pool
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


@contextlib.contextmanager
def _event_loop(
    loop: asyncio.AbstractEventLoop | None, window: int
) -> Iterator[asyncio.AbstractEventLoop]:
    """The caller's running loop, or an owned loop on a helper thread.

    The owned loop's default executor has W threads, so a sync ``op`` is
    not capped at ``asyncio.to_thread``'s default pool size.
    """
    if loop is not None:
        yield loop
        return
    owned = asyncio.new_event_loop()
    owned.set_default_executor(ThreadPoolExecutor(max_workers=window))
    thread = threading.Thread(
        target=owned.run_forever, name="pipekit-map-loop", daemon=True
    )
    thread.start()
    try:
        yield owned
    finally:
        # cancelled tasks need the loop to unwind before it stops
        asyncio.run_coroutine_threadsafe(_drain_tasks(), owned).result()
        owned.call_soon_threadsafe(owned.stop)
        thread.join()
        owned.run_until_complete(owned.shutdown_default_executor())
        owned.close()


async def _drain_tasks() -> None:
    """Wait for every other task on the running loop to finish or cancel."""
    current = asyncio.current_task()
    others = [task for task in asyncio.all_tasks() if task is not current]
    await asyncio.gather(*others, return_exceptions=True)


def _default_workers(executor: str) -> int:
    """n_w for an owned pool: 8, or the CPU count for processes."""
    if executor != "process":
        return 8
    # os.process_cpu_count is new in 3.13
    count = getattr(os, "process_cpu_count", os.cpu_count)()
    return count or 1


def _require_pickleable(op: Operator, where: str) -> None:
    """Raise `TypeError` unless ``op`` can travel to another process."""
    # imported here: the presets in `parallel` subclass `Map`
    from pipekit_ops.parallel import check_pickleable

    flagged = check_pickleable(op)
    if flagged:
        names = ", ".join(type(found).__name__ for found in flagged)
        raise TypeError(
            f"Map on {where} needs an operator that pickles; {names} hold "
            "closures (forbid_in_yaml). Replace them with module-level operators."
        )
    try:
        pickle.dumps(op)
    except Exception as exc:
        raise TypeError(
            f"Map on {where}: {type(op).__name__} does not pickle "
            f"({type(exc).__name__}: {exc})."
        ) from exc
    # the store_scope bindings go to the workers too
    for name, store in bound_stores().items():
        try:
            pickle.dumps(store)
        except Exception as exc:
            raise TypeError(
                f"Map on {where}: the store bound to {name!r} "
                f"({type(store).__name__}) does not pickle "
                f"({type(exc).__name__}: {exc})."
            ) from exc


def _refuse_tracers(chunk: Chunk, executor: str) -> None:
    """Raise `TypeError` if an item is a JAX tracer (§2.9)."""
    if any(is_traced(item) for _, item in chunk):
        raise TypeError(_tracer_message(f'Map(executor="{executor}")'))


def _tracer_message(name: str) -> str:
    return (
        f"{name} cannot run on a JAX tracer: JAX's trace state is thread-local "
        "and a tracer has no value to send to a process. Inside jax.jit, use "
        'jax.vmap or jax.lax.map, or Map(executor="serial").'
    )


def _refuse_running_loop() -> None:
    """Raise `RuntimeError` inside a running event loop (§2.8)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise RuntimeError(
        'Map(executor="async") was called inside a running event loop (a '
        "server handler, a notebook cell running async code). Use "
        "`await map_op.run(items)` there instead."
    )


def _map_error(
    failures: list[tuple[int, Exception]],
    n_items: int | None,
    results: list[Any] | None,
) -> MapError:
    """A `MapError` with the failures in input order."""
    failures.sort(key=lambda failure: failure[0])
    of = "" if n_items is None else f" of {n_items}"
    message = f"Map: {len(failures)}{of} items failed"
    return MapError(message, [exc for _, exc in failures], results)


def _require_choice(name: str, value: str, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}.")
