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
(`pipekit-array.ModelOp.with_weights` etc.). The registry just
persists / retrieves the bytes alongside the operator config.

See master plan Report 12, section 2.4 and section 5.
"""

from __future__ import annotations

import builtins
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from pipekit import Operator


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=repr)


def _hash_payload(config: dict[str, Any], weights: bytes | None) -> str:
    h = hashlib.sha256()
    h.update(_stable_json(config).encode("utf-8"))
    h.update(b"\x00")
    if weights is not None:
        h.update(weights)
    return h.hexdigest()


class LocalModelRegistry:
    """Local-filesystem model registry.

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

    def store(
        self,
        model_op: Operator,
        *,
        name: str | None = None,
        tags: dict[str, Any] | None = None,
        weights: bytes | None = None,
    ) -> str:
        if not isinstance(model_op, Operator):
            raise TypeError(
                f"store: model_op must be an Operator, got {type(model_op).__name__}."
            )
        state = model_op.state
        h = _hash_payload(state, weights)
        dest = self.root / h
        dest.mkdir(exist_ok=True)
        (dest / "operator.json").write_text(_stable_json(state))
        if weights is not None:
            (dest / "weights.bin").write_bytes(weights)
        metadata = {"tags": dict(tags) if tags else {}}
        (dest / "metadata.json").write_text(_stable_json(metadata))
        if name is not None:
            self.tag(h, name, force=True)
        return h

    def load(self, ref: str) -> Operator:
        h = self._resolve(ref)
        path = self.root / h / "operator.json"
        if not path.exists():
            raise KeyError(f"No model with hash {h!r} in {self.root}.")
        state = json.loads(path.read_text())
        return Operator.from_state(state)

    def load_weights(self, ref: str) -> bytes | None:
        """Return the raw weight bytes for ``ref``, or ``None`` if absent."""
        h = self._resolve(ref)
        path = self.root / h / "weights.bin"
        return path.read_bytes() if path.exists() else None

    def list(
        self,
        *,
        tags: dict[str, Any] | None = None,
    ) -> builtins.list[str]:
        """Return stored model hashes, optionally filtered by tag values."""
        hashes: list[str] = []
        for child in sorted(self.root.iterdir()):
            if not child.is_dir() or child.name == "_tags":
                continue
            if tags:
                meta_path = child / "metadata.json"
                if not meta_path.exists():
                    continue
                meta = json.loads(meta_path.read_text())
                stored = meta.get("tags", {})
                if not all(stored.get(k) == v for k, v in tags.items()):
                    continue
            hashes.append(child.name)
        return hashes

    def tag(self, hash: str, name: str, *, force: bool = False) -> None:
        if not (self.root / hash).exists():
            raise KeyError(f"Cannot tag unknown hash {hash!r}.")
        path = self.root / "_tags" / name
        if path.exists() and not force:
            raise FileExistsError(
                f"Tag {name!r} already exists. Pass force=True to overwrite."
            )
        path.write_text(hash)

    def resolve_tag(self, name: str) -> str:
        """Return the hash bound to ``name``."""
        path = self.root / "_tags" / name
        if not path.exists():
            raise KeyError(f"Unknown tag {name!r}.")
        return path.read_text().strip()

    def _resolve(self, ref: str) -> str:
        """Treat ``ref`` as a hash if a matching directory exists,
        otherwise as a tag.
        """
        if (self.root / ref).is_dir() and ref != "_tags":
            return ref
        return self.resolve_tag(ref)


class S3ModelRegistry:
    """fsspec-backed model registry.

    Works with any fsspec-supported scheme (``s3://``, ``gs://``,
    ``az://``, plain ``file://`` paths). Behind the ``[s3]`` extra.

    Args:
        uri: Root URI (e.g. ``"s3://bucket/models/"``).
        storage_options: Passed straight to ``fsspec.open``. Use this
            for credentials / endpoints.
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

    def store(
        self,
        model_op: Operator,
        *,
        name: str | None = None,
        tags: dict[str, Any] | None = None,
        weights: bytes | None = None,
    ) -> str:
        if not isinstance(model_op, Operator):
            raise TypeError(
                f"store: model_op must be an Operator, got {type(model_op).__name__}."
            )
        state = model_op.state
        h = _hash_payload(state, weights)
        fs = self._fs()
        with fs.open(self._path(h, "operator.json"), "w") as f:
            f.write(_stable_json(state))
        if weights is not None:
            with fs.open(self._path(h, "weights.bin"), "wb") as f:
                f.write(weights)
        metadata = {"tags": dict(tags) if tags else {}}
        with fs.open(self._path(h, "metadata.json"), "w") as f:
            f.write(_stable_json(metadata))
        if name is not None:
            self.tag(h, name, force=True)
        return h

    def load(self, ref: str) -> Operator:
        h = self._resolve(ref)
        fs = self._fs()
        path = self._path(h, "operator.json")
        if not fs.exists(path):
            raise KeyError(f"No model with hash {h!r} at {self.uri}.")
        with fs.open(path, "r") as f:
            state = json.loads(f.read())
        return Operator.from_state(state)

    def load_weights(self, ref: str) -> bytes | None:
        h = self._resolve(ref)
        fs = self._fs()
        path = self._path(h, "weights.bin")
        if not fs.exists(path):
            return None
        with fs.open(path, "rb") as f:
            return f.read()

    def list(
        self,
        *,
        tags: dict[str, Any] | None = None,
    ) -> builtins.list[str]:
        fs = self._fs()
        entries = fs.ls(self.uri.rstrip("/"))
        hashes: list[str] = []
        for entry in entries:
            name = entry.rstrip("/").split("/")[-1]
            if name in ("_tags",):
                continue
            if tags:
                meta_path = self._path(name, "metadata.json")
                if not fs.exists(meta_path):
                    continue
                with fs.open(meta_path, "r") as f:
                    meta = json.loads(f.read())
                stored = meta.get("tags", {})
                if not all(stored.get(k) == v for k, v in tags.items()):
                    continue
            hashes.append(name)
        return sorted(hashes)

    def tag(self, hash: str, name: str, *, force: bool = False) -> None:
        fs = self._fs()
        target = self._path(hash, "operator.json")
        if not fs.exists(target):
            raise KeyError(f"Cannot tag unknown hash {hash!r}.")
        tag_path = self._path("_tags", name)
        if fs.exists(tag_path) and not force:
            raise FileExistsError(
                f"Tag {name!r} already exists. Pass force=True to overwrite."
            )
        with fs.open(tag_path, "w") as f:
            f.write(hash)

    def resolve_tag(self, name: str) -> str:
        fs = self._fs()
        path = self._path("_tags", name)
        if not fs.exists(path):
            raise KeyError(f"Unknown tag {name!r}.")
        with fs.open(path, "r") as f:
            return f.read().strip()

    def _resolve(self, ref: str) -> str:
        fs = self._fs()
        # Try as hash first
        if fs.exists(self._path(ref, "operator.json")):
            return ref
        return self.resolve_tag(ref)
