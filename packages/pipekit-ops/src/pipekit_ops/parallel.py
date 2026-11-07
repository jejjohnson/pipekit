"""Group J — parallelism.

Four presets of `Map` for the four parallelism shapes pipekit deployments
actually hit. Each takes an inner `Operator` and applies it across a
collection; the constructors, configs and hashes are those of the
classes they replace (decision O6).

- Thread-pool: ``ThreadMap(op, n_workers)`` — I/O-bound (S3, network).
- Process-pool: ``ProcessMap(op, n_workers, on_error)`` — CPU-bound.
- Async: ``AsyncMap(op, semaphore)`` — async I/O stacks.
- Batched: ``BatchedMap(op, batch_size)`` — vectorised / GPU inference.

All four take an iterable as input and return a list of results
(`AsyncMap` returns the list from inside an async coroutine).

The `check_pickleable` helper walks an operator tree and surfaces
flagged operators (``forbid_in_yaml = True``) as warnings — pre-deployment
lint for `ProcessMap` shapes.

See master plan Report 2, Group J, and
``packages/pipekit-ops/docs/design/api/execution.md`` §2.11.
"""

from __future__ import annotations

import asyncio
import itertools
import multiprocessing
import warnings
from collections.abc import Iterable
from typing import Any

from pipekit_ops._base.operator import Operator, nested_config
from pipekit_ops.execution import Map


class ThreadMap(Map):
    """Apply ``op`` to each item of an iterable using a thread pool.

    Preset: ``Map(op, executor="thread", n_workers=n_workers)``.
    Use for I/O-bound workloads where the GIL releases inside the
    underlying call (network, ``rasterio`` reads, etc.).

    Args:
        op: The inner `Operator` invoked per item.
        n_workers: Maximum concurrent threads.

    Returns from ``__call__(iterable)``: a list of results in input order.
    """

    def __init__(self, op: Operator, n_workers: int = 8) -> None:
        super().__init__(op, executor="thread", n_workers=n_workers)

    def get_config(self) -> dict[str, Any]:
        return {
            "op": nested_config(self.op),
            "n_workers": self.n_workers,
        }


class ProcessMap(Map):
    """Apply ``op`` to each item of an iterable using a process pool.

    Preset: ``Map(op, executor="process", n_workers=n_workers,
    on_error=on_error)``, started with the platform's default start
    method; a `FutureWarning` says when that is not spawn, which the
    preset adopts in the next minor release. Use for CPU-bound,
    pickleable workloads. Every operator in the chain — and every
    closure inside it — must be pickleable.

    Args:
        op: The inner `Operator` invoked per item.
        n_workers: Maximum concurrent processes.
        on_error: ``"raise"`` (default) re-raises the first worker
            error and cancels the items not yet started;
            ``"log_and_continue"`` logs it as a warning on the
            ``pipekit_ops.parallel`` logger (with the traceback) and
            substitutes ``None`` for the failing item; ``"collect"`` as
            for `Map`.

    Returns from ``__call__(iterable)``: a list of results in input order.
    """

    # roadmap P2 fallback: spawn becomes the preset's default in one minor
    _default_start_method = None

    def __init__(
        self,
        op: Operator,
        n_workers: int = 8,
        on_error: str = "raise",
    ) -> None:
        super().__init__(op, executor="process", n_workers=n_workers, on_error=on_error)
        # get_start_method() without allow_none would fix the default
        method = multiprocessing.get_start_method(allow_none=True)
        default = method or multiprocessing.get_all_start_methods()[0]
        if default != "spawn":
            warnings.warn(
                f"ProcessMap starts workers with {default!r}, the platform "
                "default; it will use 'spawn' in the next minor release. Use "
                'Map(op, executor="process", start_method=...) to choose now.',
                FutureWarning,
                stacklevel=2,
            )

    def get_config(self) -> dict[str, Any]:
        return {
            "op": nested_config(self.op),
            "n_workers": self.n_workers,
            "on_error": self.on_error,
        }


class AsyncMap(Map):
    """Apply ``op`` to each item of an iterable inside an async event loop.

    Preset: ``Map(op, executor="async", max_in_flight=semaphore)``.
    If ``op._apply`` is a coroutine function, it's awaited directly.
    Otherwise it runs in a thread. At most ``semaphore`` items are in
    flight.

    `AsyncMap.__call__` is itself synchronous — it drives an event loop
    so callers don't need to be async. For callers already inside an
    event loop (a FastAPI or LitServe handler), use
    ``await op.run(items)``: calling the operator there raises a
    `RuntimeError` that says so.

    Args:
        op: The inner `Operator` invoked per item.
        semaphore: Maximum number of concurrent in-flight tasks.

    Returns from ``__call__(iterable)``: a list of results in input order.
    """

    def __init__(self, op: Operator, semaphore: int = 8) -> None:
        if semaphore < 1:
            raise ValueError(f"AsyncMap.semaphore must be >= 1, got {semaphore}.")
        super().__init__(op, executor="async", max_in_flight=semaphore)
        self.semaphore = semaphore

    def _apply(self, items: Iterable[Any]) -> list[Any]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # no loop running: drive one ourselves
            return super()._apply(items)
        raise RuntimeError(
            "AsyncMap was called inside a running event loop (a server "
            "handler, a notebook cell running async code). Use "
            "`await async_map.run(items)` there instead."
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "op": nested_config(self.op),
            "semaphore": self.semaphore,
        }


class BatchedMap(Map):
    """Split an iterable into batches and call ``op`` on each batch as a list.

    Preset: ``Map(op)`` (serial) over lists of ``batch_size`` items, then
    an optional flatten. The carrier-agnostic batching primitive. The
    inner operator receives a *list* of items at a time; what it does
    with that list is its problem (vectorise, send to GPU, concatenate
    via ``np.stack``, …). Array-aware variants that split along axis 0
    of a single array live in `pipekit-array`.

    Args:
        op: The inner `Operator` invoked per batch. It receives a
            ``list[item]`` and may return any value; the per-batch
            outputs are collected into a flat list if they're lists,
            otherwise appended as-is.
        batch_size: Maximum items per batch.
        flatten: If ``True`` (default) and every batch result is a
            list, the per-batch lists are concatenated. If ``False``,
            the per-batch results are returned as-is.

    Returns from ``__call__(iterable)``: a list — either of items
    (when ``flatten=True`` and the inner returns lists) or of batch
    outputs (otherwise).
    """

    def __init__(
        self,
        op: Operator,
        batch_size: int = 8,
        flatten: bool = True,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"BatchedMap.batch_size must be >= 1, got {batch_size}.")
        super().__init__(op)
        self.batch_size = batch_size
        self.flatten = flatten

    def _apply(self, items: Iterable[Any]) -> list[Any]:
        batches = map(list, itertools.batched(items, self.batch_size))
        out: list[Any] = []
        for result in super()._apply(batches):
            if self.flatten and isinstance(result, list):
                out.extend(result)
            else:
                out.append(result)
        return out

    def get_config(self) -> dict[str, Any]:
        return {
            "op": nested_config(self.op),
            "batch_size": self.batch_size,
            "flatten": self.flatten,
        }


def check_pickleable(op: Operator) -> list[Operator]:
    """Return the list of `forbid_in_yaml`-flagged operators inside ``op``.

    Pre-deployment lint for `ProcessMap` shapes: any operator carrying
    ``forbid_in_yaml = True`` (closures, callables, user RNGs) is a
    pickleability hazard.

    Walks the operator tree generically: every instance attribute of
    every operator is scanned for nested operators — held directly,
    inside list / tuple / set / dict containers, or inside `Graph`
    nodes — so new operator shapes are covered without registering
    their attribute names here. Returns the flagged operators in
    encounter order (deduplicated by ``id``).
    """
    from pipekit_ops._base.graph import Node

    found: list[Operator] = []
    seen: set[int] = set()

    def scan(value: Any) -> None:
        if isinstance(value, Operator):
            visit(value)
        elif isinstance(value, Node):
            if id(value) in seen:
                return
            seen.add(id(value))
            scan(value.operator)
            for parent in value.parents:
                scan(parent)
        elif isinstance(value, dict):
            for child in value.values():
                scan(child)
        elif isinstance(value, (list, tuple, set, frozenset)):
            for child in value:
                scan(child)

    def visit(node: Operator) -> None:
        if id(node) in seen:
            return
        seen.add(id(node))
        if getattr(type(node), "forbid_in_yaml", False):
            found.append(node)
        for value in vars(node).values():
            scan(value)

    visit(op)
    return found
