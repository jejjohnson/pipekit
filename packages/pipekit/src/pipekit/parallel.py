"""Group J — parallelism.

Four primitives for the four parallelism shapes pipekit deployments
actually hit. Each takes an inner `Operator` and applies it across a
collection.

- Thread-pool: ``ThreadMap(op, n_workers)`` — I/O-bound (S3, network).
- Process-pool: ``ProcessMap(op, n_workers, on_error)`` — CPU-bound.
- Async: ``AsyncMap(op, semaphore)`` — async I/O stacks.
- Batched: ``BatchedMap(op, batch_size)`` — vectorised / GPU inference.

All four take an iterable as input and return a list of results
(`AsyncMap` returns the list from inside an async coroutine).

The `check_pickleable` helper walks an operator tree and surfaces
flagged operators (``forbid_in_yaml = True``) as warnings — pre-deployment
lint for `ProcessMap` shapes.

See master plan Report 2, Group J.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any

from pipekit._base.operator import Operator, nested_config, require_operator


class ThreadMap(Operator):
    """Apply ``op`` to each item of an iterable using a thread pool.

    Use for I/O-bound workloads where the GIL releases inside the
    underlying call (network, ``rasterio`` reads, etc.).

    Args:
        op: The inner `Operator` invoked per item.
        n_workers: Maximum concurrent threads.

    Returns from ``__call__(iterable)``: a list of results in input order.
    """

    __config_mixin_auto__ = False

    def __init__(self, op: Operator, n_workers: int = 8) -> None:
        require_operator(op, "ThreadMap", "op")
        if n_workers < 1:
            raise ValueError(f"ThreadMap.n_workers must be >= 1, got {n_workers}.")
        self.op = op
        self.n_workers = n_workers

    def _apply(self, iterable: Iterable[Any]) -> list[Any]:
        items = list(iterable)
        with ThreadPoolExecutor(max_workers=self.n_workers) as ex:
            return list(ex.map(self.op, items))

    def get_config(self) -> dict[str, Any]:
        return {
            "op": nested_config(self.op),
            "n_workers": self.n_workers,
        }


class ProcessMap(Operator):
    """Apply ``op`` to each item of an iterable using a process pool.

    Use for CPU-bound, pickleable workloads. Every operator in the
    chain — and every closure inside it — must be pickleable.

    Args:
        op: The inner `Operator` invoked per item.
        n_workers: Maximum concurrent processes.
        on_error: ``"raise"`` (default) re-raises the first worker
            error; ``"log_and_continue"`` logs to ``stderr`` and
            substitutes ``None`` for the failing item.

    Returns from ``__call__(iterable)``: a list of results in input order.
    """

    __config_mixin_auto__ = False

    def __init__(
        self,
        op: Operator,
        n_workers: int = 8,
        on_error: str = "raise",
    ) -> None:
        require_operator(op, "ProcessMap", "op")
        if n_workers < 1:
            raise ValueError(f"ProcessMap.n_workers must be >= 1, got {n_workers}.")
        if on_error not in ("raise", "log_and_continue"):
            raise ValueError(
                "ProcessMap.on_error must be 'raise' or 'log_and_continue', "
                f"got {on_error!r}."
            )
        self.op = op
        self.n_workers = n_workers
        self.on_error = on_error

    def _apply(self, iterable: Iterable[Any]) -> list[Any]:
        items = list(iterable)
        with ProcessPoolExecutor(max_workers=self.n_workers) as ex:
            futures = [ex.submit(self.op, item) for item in items]
            results: list[Any] = []
            for i, fut in enumerate(futures):
                try:
                    results.append(fut.result())
                except Exception as exc:
                    if self.on_error == "raise":
                        raise
                    print(
                        f"ProcessMap: item {i} failed with {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                    results.append(None)
            return results

    def get_config(self) -> dict[str, Any]:
        return {
            "op": nested_config(self.op),
            "n_workers": self.n_workers,
            "on_error": self.on_error,
        }


class AsyncMap(Operator):
    """Apply ``op`` to each item of an iterable inside an async event loop.

    If ``op._apply`` is a coroutine function, it's awaited directly.
    Otherwise it runs via ``asyncio.to_thread``. Concurrency is bounded
    by an internal semaphore.

    `AsyncMap.__call__` is itself synchronous — it drives an event loop
    via ``asyncio.run`` so callers don't need to be async. For callers
    already inside an event loop, use `AsyncMap.run` directly (an
    ``async def`` method).

    Args:
        op: The inner `Operator` invoked per item.
        semaphore: Maximum number of concurrent in-flight tasks.

    Returns from ``__call__(iterable)``: a list of results in input order.
    """

    __config_mixin_auto__ = False

    def __init__(self, op: Operator, semaphore: int = 8) -> None:
        require_operator(op, "AsyncMap", "op")
        if semaphore < 1:
            raise ValueError(f"AsyncMap.semaphore must be >= 1, got {semaphore}.")
        self.op = op
        self.semaphore = semaphore

    def _apply(self, iterable: Iterable[Any]) -> list[Any]:
        return asyncio.run(self.run(iterable))

    async def run(self, iterable: Iterable[Any]) -> list[Any]:
        """Async entry-point. Use this when already inside an event loop."""
        items = list(iterable)
        sem = asyncio.Semaphore(self.semaphore)
        is_coro = asyncio.iscoroutinefunction(self.op._apply)

        async def _one(item: Any) -> Any:
            async with sem:
                if is_coro:
                    return await self.op._apply(item)
                return await asyncio.to_thread(self.op, item)

        return await asyncio.gather(*(_one(it) for it in items))

    def get_config(self) -> dict[str, Any]:
        return {
            "op": nested_config(self.op),
            "semaphore": self.semaphore,
        }


class BatchedMap(Operator):
    """Split an iterable into batches and call ``op`` on each batch as a list.

    The carrier-agnostic batching primitive. The inner operator
    receives a *list* of items at a time; what it does with that list
    is its problem (vectorise, send to GPU, concatenate via
    ``np.stack``, …). Array-aware variants that split along axis 0 of
    a single array live in `pipekit-array`.

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

    __config_mixin_auto__ = False

    def __init__(
        self,
        op: Operator,
        batch_size: int = 8,
        flatten: bool = True,
    ) -> None:
        require_operator(op, "BatchedMap", "op")
        if batch_size < 1:
            raise ValueError(f"BatchedMap.batch_size must be >= 1, got {batch_size}.")
        self.op = op
        self.batch_size = batch_size
        self.flatten = flatten

    def _apply(self, iterable: Iterable[Any]) -> list[Any]:
        items = list(iterable)
        out: list[Any] = []
        for start in range(0, len(items), self.batch_size):
            batch = items[start : start + self.batch_size]
            result = self.op(batch)
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
    from pipekit._base.graph import Node

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
