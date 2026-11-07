"""Module-level operators and tasks for the store tests (gates G4-2, G4-3).

The tests run these in spawned processes, and a spawned process imports
each function and operator by module and name, so they cannot live in a
test function.
"""

from __future__ import annotations

import random
import time
from typing import Any

from pipekit_ops import Cache, DirectoryStore, Operator, store_scope


class Payload(Operator):
    """i ↦ ``(i, size bytes)``: large enough to tear if written in place."""

    def __init__(self, size: int = 20_000) -> None:
        self.size = size

    def _apply(self, i: int) -> tuple[int, bytes]:
        return i, bytes([i % 256]) * self.size


def run_cached(root: str, items: list[int]) -> tuple[list[Any], int, int]:
    """Run a fresh ``Cache(Payload(), store="s")`` over ``items`` in this process.

    Returns:
        The outputs, then the cache's hits and misses.
    """
    cached = Cache(Payload(), store="s")
    with store_scope(s=DirectoryStore(root)):
        outputs = [cached(i) for i in items]
    return outputs, cached.hits, cached.misses


def write_all(root: str, items: list[int], seed: int, start_at: float) -> int:
    """Save every item's value under its cache key, in a shuffled order.

    Every writer overwrites every key (no hit short-cuts the write), and all
    writers start at ``start_at`` so that their writes overlap.

    Returns:
        The number of saves made.
    """
    store = DirectoryStore(root)
    cached = Cache(Payload(), store="s")
    order = list(items)
    random.Random(seed).shuffle(order)
    time.sleep(max(0.0, start_at - time.time()))
    for i in order:
        store.save(cached._make_key(i), cached.inner(i))
    return len(order)
