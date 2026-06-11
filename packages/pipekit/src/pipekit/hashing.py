"""Stable hashing primitives shared across the workspace.

One canonical implementation of the content-hash building blocks that
`pipekit.Cache`, `pipekit_train.TrainingDataset.content_hash`, and the
content-addressed registries in `pipekit_experiment` all rely on.
Before this module each of those sites carried its own copy of the
same ``json.dumps(sort_keys=True, default=repr)`` + sha256 dance;
keeping the primitives here means a framing change can never silently
diverge between the cache key, the dataset hash, and the registry hash.

The helpers are deliberately tiny and dependency-free:

- `stable_json` — deterministic JSON encoding (sorted keys, ``repr``
  fallback for non-JSON values).
- `stable_repr` — ``repr`` with a safe fallback for objects whose
  ``repr`` raises.
- `sha256_hex` — hex digest over a sequence of chunks; callers control
  the framing (separators, optional parts) so existing digests stay
  byte-for-byte stable.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


__all__ = ["sha256_hex", "stable_json", "stable_repr"]


def stable_json(obj: Any) -> str:
    """JSON-encode ``obj`` deterministically for hashing.

    Keys are sorted so dict ordering never changes the digest; values
    that are not JSON-serialisable fall back to ``repr``. The result is
    stable across processes for JSON-native payloads (the common case:
    ``Operator.get_config()`` dicts).

    Args:
        obj: Any value. Non-JSON values are encoded via ``repr``.

    Returns:
        The canonical JSON string.
    """
    return json.dumps(obj, sort_keys=True, default=repr)


def stable_repr(value: Any) -> str:
    """Return a stable string representation of ``value`` for hashing.

    Tries ``repr`` first; falls back to a type-name placeholder for
    objects whose ``repr`` raises. The result need not be reversible —
    only deterministic per-process for equal inputs.

    Args:
        value: Any value.

    Returns:
        ``repr(value)``, or ``"<unrepr TypeName>"`` if ``repr`` raises.
    """
    try:
        return repr(value)
    except Exception:
        return f"<unrepr {type(value).__name__}>"


def sha256_hex(*chunks: bytes | str) -> str:
    """Hex sha256 digest of the concatenated ``chunks``.

    String chunks are UTF-8 encoded; bytes chunks are hashed as-is.
    Callers own the framing — pass explicit separator chunks (e.g.
    ``b"\\x00"``) between variable-length parts to avoid ambiguity.

    Args:
        *chunks: The parts to fold into the digest, in order.

    Returns:
        The 64-character lowercase hex digest.
    """
    h = hashlib.sha256()
    for chunk in chunks:
        h.update(chunk.encode("utf-8") if isinstance(chunk, str) else chunk)
    return h.hexdigest()
