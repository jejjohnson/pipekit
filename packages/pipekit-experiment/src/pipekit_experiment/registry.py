"""Content-addressed model registries.

A trained model is identified by ``hash(operator_config, weights)``;
loading by hash is the canonical path. Names (``"methane_emulator_v3"``)
are *tags* that resolve to hashes.

- `LocalModelRegistry` — local-filesystem backend. Useful for dev,
  CI, and as the reference implementation.
- `S3ModelRegistry` — fsspec-backed (works for s3, gcs, az, file …).
  Behind the ``[s3]`` extra; raises a clean ``ImportError`` if
  ``fsspec`` isn't installed.

Layout under ``root/`` (or ``s3://bucket/prefix/``):

    <hash>/operator.json   # pipekit operator state (round-trip via Operator.from_state)
    <hash>/weights.bin     # optional raw weight bytes
    <hash>/metadata.json   # tags, training run id, …
    _tags/<name>           # one-line file holding the target hash

The ``weights.bin`` blob is opaque to the registry — domain
operators round-trip their own weights via a separate channel
(`pipekit_jax.JaxModelOp.with_weights` etc.). The registry just
persists / retrieves the bytes alongside the operator config.

Both backends share their control flow via `_ModelRegistryBase`; the
subclasses only supply the storage primitives (local ``Path`` I/O vs
fsspec). References (hashes) and tag names are restricted to single
path segments so a caller-supplied string can never escape the
registry root.

See master plan Report 12, section 2.4 and section 5.
"""

from __future__ import annotations

import builtins
import json
import os
from pathlib import Path
from typing import Any

from pipekit import Operator
from pipekit.hashing import sha256_hex, stable_json


def _hash_payload(config: dict[str, Any], weights: bytes | None) -> str:
    """Content hash of ``(operator state, optional weight bytes)``."""
    return sha256_hex(stable_json(config), b"\x00", weights or b"")


def _check_component(value: str, kind: str) -> str:
    """Validate that ``value`` is a safe single path segment.

    Hashes, tag names, and refs are all used to build storage paths, so
    they must not contain separators or traversal sequences and must not
    collide with the reserved ``_tags`` directory.

    Args:
        value: The caller-supplied hash / tag name / ref.
        kind: Human-readable noun for the error message.

    Returns:
        ``value`` unchanged, if valid.

    Raises:
        ValueError: If ``value`` is empty, contains a path separator,
            is a traversal component (``.``/``..``), or is ``_tags``.
    """
    if (
        not value
        or "/" in value
        or "\\" in value
        or value in (".", "..", "_tags")
    ):
        raise ValueError(
            f"Invalid {kind} {value!r}: must be a non-empty name without "
            "path separators (and not '_tags')."
        )
    return value


class _ModelRegistryBase:
    """Shared store/load/tag control flow for model registries.

    Subclasses provide the storage primitives (``_exists``,
    ``_read_text`` / ``_write_text``, ``_read_bytes`` / ``_write_bytes``,
    ``_children``, ``_where``); everything user-facing lives here so the
    local and fsspec backends cannot drift apart.
    """

    # -- storage primitives (implemented by subclasses) -------------------

    def _exists(self, *parts: str) -> bool:
        raise NotImplementedError

    def _read_text(self, *parts: str) -> str:
        raise NotImplementedError

    def _write_text(self, text: str, *parts: str) -> None:
        raise NotImplementedError

    def _read_bytes(self, *parts: str) -> bytes:
        raise NotImplementedError

    def _write_bytes(self, data: bytes, *parts: str) -> None:
        raise NotImplementedError

    def _children(self) -> list[str]:
        raise NotImplementedError

    def _where(self) -> str:
        raise NotImplementedError

    # -- public API --------------------------------------------------------

    def store(
        self,
        model_op: Operator,
        *,
        name: str | None = None,
        tags: dict[str, Any] | None = None,
        weights: bytes | None = None,
    ) -> str:
        """Store ``model_op`` (and optional weights) under its content hash.

        Re-storing identical content is idempotent and *merges* ``tags``
        into any previously stored ones (existing tags are never wiped by
        a later ``store`` call that omits them).

        Args:
            model_op: The operator whose state is persisted.
            name: Optional tag name bound to the new hash (``force=True``
                semantics — an existing binding is moved).
            tags: Key/value metadata used by :meth:`list` filtering.
            weights: Optional opaque weight bytes. Empty bytes are
                normalized to "absent" so ``b""`` and ``None`` cannot
                produce divergent layouts for the same content hash.

        Returns:
            The content hash the model was stored under.

        Raises:
            TypeError: If ``model_op`` is not an `Operator`.
        """
        if not isinstance(model_op, Operator):
            raise TypeError(
                f"store: model_op must be an Operator, got {type(model_op).__name__}."
            )
        if not weights:
            # b"" and None hash identically; storing them differently would
            # make load_weights() depend on write order.
            weights = None
        state = model_op.state
        h = _hash_payload(state, weights)
        self._write_text(stable_json(state), h, "operator.json")
        if weights is not None:
            self._write_bytes(weights, h, "weights.bin")
        merged: dict[str, Any] = {}
        if self._exists(h, "metadata.json"):
            merged = json.loads(self._read_text(h, "metadata.json")).get("tags", {})
        if tags:
            merged.update(tags)
        self._write_text(stable_json({"tags": merged}), h, "metadata.json")
        if name is not None:
            self.tag(h, name, force=True)
        return h

    def load(self, ref: str) -> Operator:
        """Reconstruct the operator stored under ``ref`` (hash or tag)."""
        h = self._resolve(ref)
        if not self._exists(h, "operator.json"):
            raise KeyError(f"No model with hash {h!r} in {self._where()}.")
        state = json.loads(self._read_text(h, "operator.json"))
        return Operator.from_state(state)

    def load_weights(self, ref: str) -> bytes | None:
        """Return the raw weight bytes for ``ref``, or ``None`` if absent."""
        h = self._resolve(ref)
        if not self._exists(h, "weights.bin"):
            return None
        return self._read_bytes(h, "weights.bin")

    def list(
        self,
        *,
        tags: dict[str, Any] | None = None,
    ) -> builtins.list[str]:
        """Return stored model hashes, optionally filtered by tag values."""
        hashes: list[str] = []
        for name in self._children():
            if name == "_tags":
                continue
            if tags:
                if not self._exists(name, "metadata.json"):
                    continue
                meta = json.loads(self._read_text(name, "metadata.json"))
                stored = meta.get("tags", {})
                if not all(stored.get(k) == v for k, v in tags.items()):
                    continue
            hashes.append(name)
        return sorted(hashes)

    def tag(self, hash: str, name: str, *, force: bool = False) -> None:
        """Bind tag ``name`` to ``hash``.

        Args:
            hash: An existing content hash.
            name: The tag name (a single path segment).
            force: Overwrite an existing binding instead of raising.

        Raises:
            ValueError: If ``hash`` or ``name`` is not a safe path segment.
            KeyError: If ``hash`` is not stored in this registry.
            FileExistsError: If ``name`` exists and ``force`` is false.
        """
        _check_component(hash, "hash")
        _check_component(name, "tag name")
        if not self._exists(hash, "operator.json"):
            raise KeyError(f"Cannot tag unknown hash {hash!r}.")
        if self._exists("_tags", name) and not force:
            raise FileExistsError(
                f"Tag {name!r} already exists. Pass force=True to overwrite."
            )
        self._write_text(hash, "_tags", name)

    def resolve_tag(self, name: str) -> str:
        """Return the hash bound to ``name``.

        Raises:
            ValueError: If ``name`` is not a safe path segment.
            KeyError: If the tag does not exist.
        """
        _check_component(name, "tag name")
        if not self._exists("_tags", name):
            raise KeyError(f"Unknown tag {name!r}.")
        return self._read_text("_tags", name).strip()

    def _resolve(self, ref: str) -> str:
        """Treat ``ref`` as a hash if its ``operator.json`` exists,
        otherwise as a tag.
        """
        _check_component(ref, "ref")
        if self._exists(ref, "operator.json"):
            return ref
        return self.resolve_tag(ref)


class LocalModelRegistry(_ModelRegistryBase):
    """Local-filesystem model registry.

    Writes are atomic (temp file + ``os.replace``), so a crash mid-write
    can never leave a truncated ``operator.json`` or tag file behind.

    Args:
        root: Directory under which models are stored. Created on
            construction if it doesn't exist.

    Storage layout (under ``root``):

        ``<hash>/operator.json`` — operator state
        ``<hash>/weights.bin``   — optional weight bytes
        ``<hash>/metadata.json`` — tags + arbitrary metadata
        ``_tags/<name>``         — file containing the target hash
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "_tags").mkdir(exist_ok=True)

    def _exists(self, *parts: str) -> bool:
        return self.root.joinpath(*parts).exists()

    def _read_text(self, *parts: str) -> str:
        return self.root.joinpath(*parts).read_text()

    def _write_text(self, text: str, *parts: str) -> None:
        self._atomic_write(text.encode("utf-8"), *parts)

    def _read_bytes(self, *parts: str) -> bytes:
        return self.root.joinpath(*parts).read_bytes()

    def _write_bytes(self, data: bytes, *parts: str) -> None:
        self._atomic_write(data, *parts)

    def _atomic_write(self, data: bytes, *parts: str) -> None:
        path = self.root.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    def _children(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def _where(self) -> str:
        return str(self.root)


class S3ModelRegistry(_ModelRegistryBase):
    """fsspec-backed model registry.

    Works with any fsspec-supported scheme (``s3://``, ``gs://``,
    ``az://``, ``memory://``, plain ``file://`` paths). Behind the
    ``[s3]`` extra.

    Unlike `LocalModelRegistry`, writes are only as atomic as the
    backing store makes them (object stores typically are; plain
    ``file://`` through fsspec is not).

    Args:
        uri: Root URI (e.g. ``"s3://bucket/models/"``).
        storage_options: Passed straight to ``fsspec.open``. Use this
            for credentials / endpoints.

    Raises:
        ImportError: If ``fsspec`` is not installed.
    """

    def __init__(
        self,
        uri: str,
        storage_options: dict[str, Any] | None = None,
    ) -> None:
        try:
            import fsspec  # noqa: F401  # ty: ignore[unresolved-import]
        except ImportError as e:
            raise ImportError(
                "S3ModelRegistry requires fsspec. Install with "
                "`pip install pipekit-experiment[s3]`."
            ) from e
        self.uri = uri.rstrip("/") + "/"
        self.storage_options = dict(storage_options) if storage_options else {}

    def _fs(self):
        import fsspec  # ty: ignore[unresolved-import]

        fs, _ = fsspec.core.url_to_fs(self.uri, **self.storage_options)
        return fs

    def _path(self, *parts: str) -> str:
        return self.uri + "/".join(parts)

    def _exists(self, *parts: str) -> bool:
        return self._fs().exists(self._path(*parts))

    def _read_text(self, *parts: str) -> str:
        with self._fs().open(self._path(*parts), "r") as f:
            return f.read()

    def _write_text(self, text: str, *parts: str) -> None:
        with self._fs().open(self._path(*parts), "w") as f:
            f.write(text)

    def _read_bytes(self, *parts: str) -> bytes:
        with self._fs().open(self._path(*parts), "rb") as f:
            return f.read()

    def _write_bytes(self, data: bytes, *parts: str) -> None:
        with self._fs().open(self._path(*parts), "wb") as f:
            f.write(data)

    def _children(self) -> list[str]:
        # detail=False explicitly: several backends (memory, local) default
        # to detail=True and return dicts instead of path strings.
        entries = self._fs().ls(self.uri.rstrip("/"), detail=False)
        return sorted(entry.rstrip("/").split("/")[-1] for entry in entries)

    def _where(self) -> str:
        return self.uri
