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

`Cache` raises on a JAX tracer: a tracer has no value to key on, and inside
``jax.jit`` JAX's compilation cache already does this job.

See master plan Report 2, Group G.
"""

from __future__ import annotations

import threading
from typing import Any

from pipekit_ops._base.operator import Operator, nested_config, require_operator
from pipekit_ops._base.tracing import is_traced
from pipekit_ops.hashing import sha256_hex, stable_json, stable_repr


class Cache(Operator):
    """Memoise an inner operator by hashing its input and config.

    Args:
        inner: The `Operator` whose outputs to cache.

    Attributes:
        hits: Number of cache hits since construction.
        misses: Number of cache misses since construction.

    Raises:
        TypeError: If called on a JAX tracer, for example inside ``jax.jit``.

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
        require_operator(inner, "Cache", "inner")
        self.inner = inner
        self._store: dict[str, Any] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _apply(self, x: Any) -> Any:
        if is_traced(x):
            raise TypeError(
                "Cache cannot key a traced value: it has no bytes to hash, so "
                "different inputs share one entry. Inside jax.jit, JAX's "
                "compilation cache already does this job."
            )
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

    def __getstate__(self) -> dict[str, Any]:
        # A lock does not pickle; a copy sent to a process pool starts empty,
        # with fresh counters and its own lock (decision O10).
        state = dict(vars(self))
        del state["_lock"]
        state["_store"] = {}
        state["hits"] = 0
        state["misses"] = 0
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        vars(self).update(state)
        self._lock = threading.Lock()

    def get_config(self) -> dict[str, Any]:
        return {"inner": nested_config(self.inner)}

    def compute_output_signature(self, input_signature: Any) -> Any:
        """The inner operator's: caching does not change the output."""
        return self.inner.compute_output_signature(input_signature)

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
