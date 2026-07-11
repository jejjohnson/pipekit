"""Group G — caching.

Hash-keyed memoisation as a composable operator. In-memory backend
in v0.1; a disk backend lands in v0.2.

- `Cache(inner)` — wraps any operator; caches outputs keyed by a hash
  of ``(input, inner.get_config())``. Exposes ``hits`` / ``misses``
  counters for cache-effectiveness reporting.
- `Memoize` — alias of `Cache`, named after ``toolz.memoize``.

Hash key shape: ``(input_key_hash, config_canonical_json_hash)``.
Array-likes (anything exposing ``tobytes`` / ``shape`` / ``dtype``)
are keyed by their full byte content — plain ``repr`` would be
truncated by NumPy's print options, letting two different large
arrays silently collide onto one key. Other inputs fall back to
``repr(x)``; this is deliberately conservative — `Cache` is opt-in
and the user should know whether their carrier is hash-friendly.

`Cache` is safe to share across `ThreadMap` workers: bookkeeping is
lock-protected. The inner operator runs *outside* the lock, so
concurrent misses on the same key may compute it more than once (the
usual memoisation trade-off; correctness is unaffected).

See master plan Report 2, Group G.
"""

from __future__ import annotations

import threading
from typing import Any

from pipekit._base.operator import Operator
from pipekit.hashing import sha256_hex, stable_json, stable_repr


class Cache(Operator):
    """Memoise an inner operator by hashing its input and config.

    Args:
        inner: The `Operator` whose outputs to cache.

    Attributes:
        hits: Number of cache hits since construction.
        misses: Number of cache misses since construction.

    Example::

        slow = Cache(ExpensiveDownload(uri))
        slow(key)   # miss — runs ExpensiveDownload
        slow(key)   # hit — returns cached value
        slow.hits, slow.misses
        # (1, 1)
    """

    __config_mixin_auto__ = False
    __config_exclude__ = ("hits", "misses")

    def __init__(self, inner: Operator) -> None:
        if not isinstance(inner, Operator):
            raise TypeError(
                f"Cache.inner must be an Operator, got {type(inner).__name__}."
            )
        self.inner = inner
        self._store: dict[str, Any] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _apply(self, x: Any) -> Any:
        key = _make_key(x, self.inner.get_config())
        with self._lock:
            if key in self._store:
                self.hits += 1
                return self._store[key]
            self.misses += 1
        # Run the inner operator unlocked: memoisation must not serialise
        # expensive work across threads. Two threads racing on the same
        # fresh key may both compute; last write wins.
        out = self.inner(x)
        with self._lock:
            self._store[key] = out
        return out

    def clear(self) -> None:
        """Drop all cached entries (counters preserved)."""
        with self._lock:
            self._store.clear()

    def get_config(self) -> dict[str, Any]:
        return {
            "inner": {
                "class": type(self.inner).__name__,
                "config": self.inner.get_config(),
            }
        }

    def __repr__(self) -> str:
        return f"Cache({self.inner!r})"


# `toolz.memoize`-flavoured alias. Same semantics, different name.
Memoize = Cache


def _input_key(value: Any) -> str:
    """Stable key material for a cache input.

    Array-likes are keyed by ``(dtype, shape, raw bytes)`` because their
    ``repr`` is truncated by print options and would collide for large
    arrays that differ only in elided elements. Everything else uses
    `stable_repr`.
    """
    tobytes = getattr(value, "tobytes", None)
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if callable(tobytes) and shape is not None and dtype is not None:
        try:
            return sha256_hex(
                str(dtype), b"\x00", repr(tuple(shape)), b"\x00", tobytes()
            )
        except Exception:
            pass  # exotic array-likes fall through to repr
    return stable_repr(value)


def _make_key(value: Any, config: dict[str, Any]) -> str:
    """Build a stable hash key from ``(value, config)``."""
    return sha256_hex(_input_key(value), b"\x00", stable_json(config))
