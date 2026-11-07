"""Module-level operators for the execution tests.

``Map(executor="process")`` starts workers with spawn, and a spawned
worker imports each operator's module by name, so these cannot live in a
test function. They hold no events or counters for the same reason; the
thread-only operators that do live in ``test_execution.py``.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from pipekit_ops import Operator


class Double(Operator):
    """x ↦ 2x; works on numbers and on JAX tracers."""

    def _apply(self, x: Any) -> Any:
        return x * 2


class JitterDouble(Operator):
    """x ↦ 2x after a pseudo-random sleep of 0 to 2 ms, fixed per item."""

    def _apply(self, x: int) -> int:
        # a fixed permutation of 0 … 2000 µs, so every run sleeps the same
        time.sleep((x * 7919 % 2001) / 1e6)
        return x * 2


class FailOn(Operator):
    """Raise `ValueError` on the listed items, return the rest unchanged."""

    def __init__(self, bad: tuple[int, ...]) -> None:
        self.bad = bad

    def _apply(self, x: int) -> int:
        if x in self.bad:
            raise ValueError(f"item {x} is bad")
        return x


class SlowFailFirst(Operator):
    """Sleep ``seconds``, then raise on item 0 and return the rest."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def _apply(self, x: int) -> int:
        time.sleep(self.seconds)
        if x == 0:
            raise ValueError("item 0 failed")
        return x


class AddPair(Operator):
    """(acc, item) ↦ acc + item: the binary step of a sum."""

    def _apply(self, pair: tuple[int, int]) -> int:
        acc, item = pair
        return acc + item


class HoldsLock(Operator):
    """An unflagged operator that does not pickle: it holds a lock."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def _apply(self, x: Any) -> Any:
        return x
