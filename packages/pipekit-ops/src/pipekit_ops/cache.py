"""Group G — caching.

Hash-keyed memoisation as a composable operator, in memory by default
or in a named `Store` (`pipekit_ops.store`), such as a `DirectoryStore`
that survives a restart.

- `Cache(inner, store=None, key=None)` — wraps any operator; caches
  outputs keyed by a hash of the input (or of ``key(input)``) and of
  ``pipeline_hash(inner)``. Exposes ``hits`` / ``misses`` counters for
  cache-effectiveness reporting.
- `Memoize` — alias of `Cache`, named after ``toolz.memoize``.

Hash key shape: ``sha256(input key ‖ 0x00 ‖ pipeline_hash(inner))``, with
``‖ 0x00 ‖ pipeline_hash(key)`` appended when a ``key`` operator is set.
`pipeline_hash` records the inner operator's module and class, so two
classes with equal configs never share an entry.
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
from pipekit_ops.dicts import require_child
from pipekit_ops.hashing import pipeline_hash, sha256_hex, stable_repr
from pipekit_ops.store import MemoryStore, Store, resolve_store


class Cache(Operator):
    """Memoise an inner operator by hashing its input and the inner pipeline.

    With no ``store`` the entries live in a private `MemoryStore` owned by
    this instance, as before P4. A ``store`` name is looked up in the
    enclosing `store_scope` at call time, so the same pipeline can cache
    to a scratch directory on one machine and in memory on another; the
    name is in the config, the store is not.

    Args:
        inner: The `Operator` whose outputs to cache.
        store: The name of a store bound by `store_scope`, or ``None`` for
            a private in-memory store.
        key: An `Operator` that computes the cache key's input from the
            input, such as ``Get("id")`` to key a STAC item by its ID
            (``toolz.memoize(key=…)``); ``None`` keys on the whole input.

    Attributes:
        hits: Number of cache hits since construction, in this process.
        misses: Number of cache misses since construction, in this process.

    Raises:
        TypeError: If ``inner`` or ``key`` is not an `Operator` (a plain
            function is told to use `Lambda`), ``store`` is not a string, or
            the call's input is a JAX tracer, for example inside ``jax.jit``.
        LookupError: At call time, if no `store_scope` binds ``store``.

    Examples:
        >>> from pipekit_ops import Get, Identity
        >>> cached = Cache(Identity())
        >>> cached(1), cached(1), cached(2)
        (1, 1, 2)
        >>> cached.hits, cached.misses
        (1, 2)
        >>> by_id = Cache(Identity(), key=Get("id"))
        >>> by_id({"id": "a", "cloud": 3})
        {'id': 'a', 'cloud': 3}
        >>> by_id({"id": "a", "cloud": 9})
        {'id': 'a', 'cloud': 3}
    """

    __config_mixin_auto__ = False
    __config_exclude__ = ("hits", "misses")

    def __init__(
        self,
        inner: Operator,
        store: str | None = None,
        key: Operator | None = None,
    ) -> None:
        require_operator(inner, "Cache", "inner")
        if store is not None and not isinstance(store, str):
            raise TypeError(
                f"Cache.store must be a store name (a str bound by store_scope), "
                f"got {type(store).__name__}"
            )
        if key is not None:
            require_child(key, "Cache", "key")
        self.inner = inner
        self.store = store
        self.key = key
        self._store: MemoryStore = MemoryStore()
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
        store = self._resolve()
        key = self._make_key(x)
        # a store may drop an entry between the check and the load
        found = False
        if key in store:
            try:
                out = store.load(key)
                found = True
            except KeyError:
                pass
        with self._lock:
            if found:
                self.hits += 1
            else:
                self.misses += 1
        if found:
            return out
        # Run the inner operator unlocked: memoisation must not serialise
        # expensive work across threads. Two threads racing on the same
        # fresh key may both compute; last write wins.
        out = self.inner(x)
        store.save(key, out)
        return out

    def _resolve(self) -> Store:
        """The private store, or the store bound to ``self.store``."""
        if self.store is None:
            return self._store
        return resolve_store(self.store, owner=f"Cache(store={self.store!r})")

    def _make_key(self, x: Any) -> str:
        """``sha256(input key ‖ 0x00 ‖ pipeline_hash(inner))``, plus the key's hash.

        With a ``key`` operator, ``‖ 0x00 ‖ pipeline_hash(key)`` is appended,
        so two key operators that return equal values never share an entry.
        """
        if self.key is None:
            return sha256_hex(_input_key(x), b"\x00", pipeline_hash(self.inner))
        return sha256_hex(
            _input_key(self.key(x)),
            b"\x00",
            pipeline_hash(self.inner),
            b"\x00",
            pipeline_hash(self.key),
        )

    def clear(self) -> None:
        """Drop the private in-memory entries (counters preserved).

        A named store may be shared with other caches and processes, so it
        is left alone; clear it through the store itself.
        """
        self._store.clear()

    def __getstate__(self) -> dict[str, Any]:
        # A lock does not pickle; a copy sent to a process pool starts empty,
        # with fresh counters and its own lock (decision O10).
        state = dict(vars(self))
        del state["_lock"]
        state["_store"] = MemoryStore()
        state["hits"] = 0
        state["misses"] = 0
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        vars(self).update(state)
        self._lock = threading.Lock()

    def get_config(self) -> dict[str, Any]:
        # `None` arguments are left out, so the config and `pipeline_hash`
        # of a plain `Cache(inner)` are the same as before P4 (gate G4-1)
        config: dict[str, Any] = {"inner": nested_config(self.inner)}
        if self.store is not None:
            config["store"] = self.store
        if self.key is not None:
            config["key"] = nested_config(self.key)
        return config

    def compute_output_signature(self, input_signature: Any) -> Any:
        """The inner operator's: caching does not change the output."""
        return self.inner.compute_output_signature(input_signature)

    def __repr__(self) -> str:
        extra = ""
        if self.store is not None:
            extra += f", store={self.store!r}"
        if self.key is not None:
            extra += f", key={self.key!r}"
        return f"Cache({self.inner!r}{extra})"


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
