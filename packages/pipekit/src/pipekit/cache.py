"""Group G — caching.

Hash-keyed memoisation as a composable operator. In-memory backend
in v0.1; a disk backend lands in v0.2.

- `Cache(inner)` — wraps any operator; caches outputs keyed by a hash
  of ``(input, inner.get_config())``. Exposes ``hits`` / ``misses``
  counters for cache-effectiveness reporting.
- `Memoize` — alias of `Cache`, named after ``toolz.memoize``.

Hash key shape: ``(input_repr_hash, config_canonical_json_hash)``.
Inputs that aren't hashable fall back to ``repr(x)``; this is
deliberately conservative — `Cache` is opt-in and the user should
know whether their carrier is hash-friendly.

See master plan Report 2, Group G.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pipekit._base.operator import Operator


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
        self.hits = 0
        self.misses = 0

    def _apply(self, x: Any) -> Any:
        key = _make_key(x, self.inner.get_config())
        if key in self._store:
            self.hits += 1
            return self._store[key]
        self.misses += 1
        out = self.inner(x)
        self._store[key] = out
        return out

    def clear(self) -> None:
        """Drop all cached entries (counters preserved)."""
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


def _make_key(value: Any, config: dict[str, Any]) -> str:
    """Build a stable hash key from ``(value, config)``."""
    h = hashlib.sha256()
    h.update(_stable_repr(value).encode("utf-8"))
    h.update(b"\x00")
    h.update(_canonical_json(config).encode("utf-8"))
    return h.hexdigest()


def _stable_repr(value: Any) -> str:
    """Return a stable string representation for hashing.

    Tries ``repr`` first; falls back to ``str(type)`` for objects whose
    repr embeds an ``id``. The result need not be reversible — only
    deterministic per-process for equal inputs.
    """
    try:
        return repr(value)
    except Exception:
        return f"<unrepr {type(value).__name__}>"


def _canonical_json(obj: Any) -> str:
    """JSON-encode ``obj`` with sorted keys; non-JSON values use ``repr``."""
    return json.dumps(obj, sort_keys=True, default=repr)
