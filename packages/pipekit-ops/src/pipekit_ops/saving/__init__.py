"""Saving a pipeline, naming it, and keeping what it computes.

`dumps` and `loads` write and read a pipeline's state, `loads_sandboxed` reads one that
may only use registered classes, and `upgrade_state` renames the modules of an old
state. `pipeline_hash` is the content hash of a pipeline, built on `stable_json`,
`stable_repr` and `sha256_hex`. `Store`, `MemoryStore` and `DirectoryStore` hold what
`Cache` keeps, bound by `store_scope`.
"""

from __future__ import annotations

from pipekit_ops.saving._hashing import (
    pipeline_hash,
    sha256_hex,
    stable_json,
    stable_repr,
)
from pipekit_ops.saving._serial import (
    allowed,
    dumps,
    from_state_sandboxed,
    loads,
    loads_sandboxed,
    register,
    unregister,
    upgrade_state,
)
from pipekit_ops.saving._store import (
    DirectoryStore,
    MemoryStore,
    Store,
    store_scope,
)


__all__ = [
    "DirectoryStore",
    "MemoryStore",
    "Store",
    "allowed",
    "dumps",
    "from_state_sandboxed",
    "loads",
    "loads_sandboxed",
    "pipeline_hash",
    "register",
    "sha256_hex",
    "stable_json",
    "stable_repr",
    "store_scope",
    "unregister",
    "upgrade_state",
]
