"""Stores: where a `Cache` keeps its entries.

A `Store` is three methods over string keys: ``key in store``,
``store.load(key)`` and ``store.save(key, value)``. `Cache` keys are
64-character hex digests, so a store never has to sanitise them.

- `MemoryStore` — a dict and a lock; threads in one process.
- `DirectoryStore` — one pickle file per key under a root directory,
  written atomically, so several processes on one file system can share it.
- `store_scope` — binds store names to stores for the duration of a
  block. ``Cache(op, store="scenes")`` names a store; which directory it
  is, is a deployment choice made around the call, not part of the
  pipeline's config or `pipeline_hash`.

Carrier-specific stores (Zarr, COG, an object store) live in the sister
packages and only implement the three methods.

Design: ``packages/pipekit-ops/docs/design/api/operators.md``, section 4.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import os
import pickle
import re
import tempfile
import threading
from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable


__all__ = ["DirectoryStore", "MemoryStore", "Store", "store_scope"]

#: the version a `DirectoryStore` writes to, and accepts in, its marker file
FORMAT_VERSION = 1
MARKER = "pipekit-store.json"
CODECS = ("pickle",)
_KEY = re.compile(r"[0-9a-f]{64}")

#: the stores bound by `store_scope`, by name; read-only so a scope cannot leak
_STORES: contextvars.ContextVar[Mapping[str, Store]] = contextvars.ContextVar(
    "pipekit_ops_stores", default=MappingProxyType({})
)


@runtime_checkable
class Store(Protocol):
    """What `Cache` needs from a store: membership, load and save by key.

    Keys are 64-character lowercase hex digests. ``load`` of a missing
    key raises `KeyError`. A store shared between threads or processes
    must make ``save`` atomic: a reader sees the old value or the new
    one, never part of one.

    Examples:
        >>> isinstance(MemoryStore(), Store)
        True
        >>> isinstance({}, Store)
        False
    """

    def __contains__(self, key: str) -> bool: ...

    def load(self, key: str) -> Any: ...

    def save(self, key: str, value: Any) -> None: ...


class MemoryStore:
    """A dict guarded by a lock: shared by the threads of one process.

    A pickled copy keeps the entries and gets its own lock, so a copy in
    another process is a snapshot, not a shared store.

    Examples:
        >>> store = MemoryStore()
        >>> key = "0" * 64
        >>> key in store
        False
        >>> store.save(key, [1, 2])
        >>> store.load(key), len(store)
        ([1, 2], 1)
    """

    def __init__(self) -> None:
        self._entries: dict[str, Any] = {}
        self._lock = threading.Lock()

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._entries

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def load(self, key: str) -> Any:
        """Return the value saved under ``key``; `KeyError` if there is none."""
        with self._lock:
            return self._entries[key]

    def save(self, key: str, value: Any) -> None:
        """Save ``value`` under ``key``, replacing any earlier value."""
        with self._lock:
            self._entries[key] = value

    def clear(self) -> None:
        """Drop every entry."""
        with self._lock:
            self._entries.clear()

    def __getstate__(self) -> dict[str, Any]:
        # a lock does not pickle; the copy gets a fresh one
        with self._lock:
            return {"_entries": dict(self._entries)}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._entries = state["_entries"]
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"MemoryStore(<{len(self)} entries>)"


class DirectoryStore:
    """One file per key under ``root``, written atomically.

    The entry for key ``abcd…`` is ``root/ab/abcd….pkl``. A save writes a
    temporary file in the same directory and renames it into place with
    `os.replace`, so a reader never sees a partial value, and processes on
    one file system can share the directory; when two write one key, the
    last rename wins. ``root/pipekit-store.json`` records the format
    version, and a directory with another version is refused.

    Pickle loads run code: point a `DirectoryStore` only at a directory
    you control. A `DirectoryStore` pickles as its root and codec, so it
    can travel to a process pool.

    Args:
        root: The directory; created, with its parents, if missing.
        codec: How values are written; ``"pickle"`` is the only codec.

    Raises:
        ValueError: If ``codec`` is unknown, or ``root`` holds a store of
            another format version.

    Examples:
        >>> import tempfile
        >>> root = tempfile.mkdtemp()
        >>> store = DirectoryStore(root)
        >>> key = "ab" + "0" * 62
        >>> store.save(key, {"id": "S2A"})
        >>> key in store, store.load(key)
        (True, {'id': 'S2A'})
        >>> key in DirectoryStore(root)
        True
    """

    def __init__(self, root: str | os.PathLike[str], codec: str = "pickle") -> None:
        if codec not in CODECS:
            raise ValueError(
                f"DirectoryStore codec must be one of {CODECS}, got {codec!r}"
            )
        self.root = os.fspath(root)
        self.codec = codec
        os.makedirs(self.root, exist_ok=True)
        self._check_marker()

    def __contains__(self, key: str) -> bool:
        return os.path.isfile(self._path(key))

    def load(self, key: str) -> Any:
        """Return the value saved under ``key``; `KeyError` if there is none."""
        try:
            with open(self._path(key), "rb") as file:
                return pickle.load(file)
        except FileNotFoundError:
            raise KeyError(key) from None

    def save(self, key: str, value: Any) -> None:
        """Save ``value`` under ``key`` atomically, replacing any earlier value."""
        path = self._path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _write_atomic(path, pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))

    def _path(self, key: str) -> str:
        """``root/ab/abcd….pkl``; a key that is not a hex digest is refused."""
        if not isinstance(key, str) or not _KEY.fullmatch(key):
            raise ValueError(
                f"DirectoryStore keys are 64-character hex digests, got {key!r}"
            )
        return os.path.join(self.root, key[:2], f"{key}.pkl")

    def _check_marker(self) -> None:
        """Write the format marker, or refuse a directory of another version."""
        marker = os.path.join(self.root, MARKER)
        if not os.path.exists(marker):
            record = {"format": "pipekit-store", "version": FORMAT_VERSION}
            payload = json.dumps(record, sort_keys=True).encode("utf-8")
            # concurrent constructors write the same bytes; any rename wins
            _write_atomic(marker, payload)
        with open(marker, encoding="utf-8") as file:
            version = json.load(file).get("version")
        if version != FORMAT_VERSION:
            raise ValueError(
                f"{self.root} holds a pipekit store of format version {version!r}; "
                f"this DirectoryStore reads version {FORMAT_VERSION} only"
            )

    def __reduce__(self) -> tuple[type[DirectoryStore], tuple[str, str]]:
        return (DirectoryStore, (self.root, self.codec))

    def __repr__(self) -> str:
        return f"DirectoryStore({self.root!r}, codec={self.codec!r})"


@contextlib.contextmanager
def store_scope(**stores: Store) -> Iterator[Mapping[str, Store]]:
    """Bind store names to stores inside the block.

    Bindings nest: an inner scope adds names to the outer ones and may
    rebind a name for its block. The stores are not part of any config,
    so a pipeline hashes the same on a laptop and on a cluster. `Map`
    worker threads see the scope; process workers get the stores pickled.

    Args:
        **stores: ``name=store`` pairs; each value must be a `Store`.

    Raises:
        TypeError: If a value is not a `Store`.

    Examples:
        >>> from pipekit_ops import Cache, Identity
        >>> cached = Cache(Identity(), store="scenes")
        >>> scenes = MemoryStore()
        >>> with store_scope(scenes=scenes):
        ...     cached(1), cached(1)
        (1, 1)
        >>> cached.hits, cached.misses, len(scenes)
        (1, 1, 1)
    """
    for name, store in stores.items():
        if not isinstance(store, Store):
            raise TypeError(
                f"store_scope({name}=...) must be a Store (with __contains__, "
                f"load and save), got {type(store).__name__}"
            )
    token = _STORES.set(MappingProxyType({**_STORES.get(), **stores}))
    try:
        yield _STORES.get()
    finally:
        _STORES.reset(token)


def resolve_store(name: str, owner: str = "Cache") -> Store:
    """The store bound to ``name`` by the innermost `store_scope`.

    Raises:
        LookupError: If no scope binds ``name``.
    """
    try:
        return _STORES.get()[name]
    except KeyError:
        raise LookupError(
            f"{owner}: no store named {name!r} is bound; run the call inside "
            f"store_scope({name}=...)"
        ) from None


def bound_stores() -> dict[str, Store]:
    """The current bindings, as a plain dict that pickles (for process workers)."""
    return dict(_STORES.get())


def bind_stores(stores: Mapping[str, Store]) -> None:
    """Bind ``stores`` for the rest of the current context (a process worker)."""
    _STORES.set(MappingProxyType(dict(stores)))


def _write_atomic(path: str, payload: bytes) -> None:
    """Write ``payload`` to a temporary file beside ``path``, then rename it over."""
    directory, name = os.path.split(path)
    # the dot prefix and .part suffix keep a crashed write from looking like an entry
    descriptor, temporary = tempfile.mkstemp(
        dir=directory, prefix=f".{name}.", suffix=".part"
    )
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(payload)
            file.flush()
            # the bytes reach the disk before the rename makes them visible
            os.fsync(file.fileno())
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise
