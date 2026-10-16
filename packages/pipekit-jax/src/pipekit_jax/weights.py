"""Shared weight-blob round trip for pipekit-jax operators.

`WeightsMixin` gives an operator ``serialize_weights`` / ``with_weights``
/ ``from_registry``: the array leaves of a pytree are written with
``equinox.tree_serialise_leaves`` and read back into a **skeleton** — an
instance with the same pytree structure, whose non-array leaves
(activations, dtypes, sizes, static config) cannot be recovered from
bytes alone.

Equinox loads leaves *positionally*, so a skeleton whose fields are in a
different order, or a different class with the same field names — but
the same number of leaves and compatible shapes — would silently receive
weights it was never trained with. The blob therefore starts with a
signature header (`tree_signature`): the node types of the tree
(``module.QualName`` of every pytree node, from the root down, with the
structural aux data of non-Equinox nodes such as a `pipekit.Graph`'s
wiring) and the key path, shape and dtype of every array leaf. Only
array leaves are stored; every non-array value comes from the skeleton.
`with_weights` compares the signature with the skeleton before loading.
Node types are read from the treedef, not its repr, and aux data is
encoded by `_stable`, so nothing process-specific (memory addresses)
enters the signature.

Blob layout::

    b"PKJXW1\\n" | header length (8 bytes, little endian) | JSON signature
               | equinox.tree_serialise_leaves payload

Blobs without the magic prefix (written before the header existed) are
still accepted and loaded positionally.

Two hooks pick *which* pytree is persisted:

- ``_weights_tree()`` — the tree to serialise (default: ``self``).
- ``_from_weights_tree(tree)`` — wrap a deserialised tree back into an
  operator (default: return ``tree``).

`JaxModelOp` persists its wrapped ``module``; `JaxOperator` persists
itself.
"""

from __future__ import annotations

import io
import json
from typing import Any

import equinox as eqx
import jax


_MAGIC = b"PKJXW1\n"


def _stable(value: Any) -> Any:
    """A process-independent JSON form of pytree aux data.

    Primitives as-is, containers recursively, functions / classes by
    ``module.qualname`` and any other object by its type — never a repr
    that could embed a memory address.
    """
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list | tuple):
        return [_stable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _stable(v) for k, v in value.items()}
    if hasattr(value, "__qualname__"):
        return f"{getattr(value, '__module__', '?')}.{value.__qualname__}"
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _node_types(treedef: Any) -> Any:
    """Nested ``[type name, aux, [children...]]`` of a treedef; ``None`` for a leaf.

    Aux data that encodes *structure* is included — `pipekit.Graph` keeps
    its whole wiring there, dicts their keys. An ``eqx.Module``'s aux is
    its static field values, i.e. configuration; like every non-array
    value it comes from the skeleton, so it is left out.
    """
    data = treedef.node_data()
    if data is None:
        return None
    node_type, aux = data
    structural = None if issubclass(node_type, eqx.Module) else _stable(aux)
    return [
        f"{node_type.__module__}.{node_type.__qualname__}",
        structural,
        [_node_types(child) for child in treedef.children()],
    ]


def tree_signature(tree: Any) -> dict[str, Any]:
    """Node types plus `leaf_signature` of ``tree``: what a weight blob records."""
    arrays_only = eqx.filter(tree, eqx.is_array)
    return {
        "nodes": _node_types(jax.tree.structure(arrays_only)),
        "leaves": leaf_signature(tree),
    }


def leaf_signature(tree: Any) -> list[list[Any]]:
    """``[key path, shape, dtype]`` of every array leaf of ``tree``, in order."""
    flat, _ = jax.tree_util.tree_flatten_with_path(tree)
    return [
        [jax.tree_util.keystr(path), list(leaf.shape), str(leaf.dtype)]
        for path, leaf in flat
        if eqx.is_array(leaf)
    ]


class WeightsMixin:
    """``serialize_weights`` / ``with_weights`` / ``from_registry``."""

    def _weights_tree(self) -> Any:
        return self

    def _from_weights_tree(self, tree: Any) -> Any:
        return tree

    def serialize_weights(self) -> bytes:
        """Serialise the persisted tree's array leaves to bytes.

        A signature header (key path, shape and dtype of every array
        leaf) precedes the ``equinox.tree_serialise_leaves`` payload, so
        `with_weights` can refuse a skeleton with a different layout.
        """
        header = json.dumps(tree_signature(self._weights_tree())).encode()
        buf = io.BytesIO()
        buf.write(_MAGIC)
        buf.write(len(header).to_bytes(8, "little"))
        buf.write(header)
        # Array leaves only: non-array values (a Python-float temperature,
        # a flag) are configuration and come from the skeleton.
        eqx.tree_serialise_leaves(buf, eqx.filter(self._weights_tree(), eqx.is_array))
        return buf.getvalue()

    def with_weights(self, blob: bytes) -> Any:
        """Return a new operator with weights loaded from ``blob``.

        ``self`` is the **skeleton** / template — its non-array leaves
        (activations, dtypes, layer sizes) determine the structure into
        which the persisted bytes are loaded. The bytes must come from a
        `serialize_weights` call on an operator with the same pytree
        structure.

        Args:
            blob: Bytes produced by `serialize_weights`.

        Returns:
            A fresh operator carrying the persisted weight values.

        Raises:
            TypeError: if ``blob`` is not bytes.
            ValueError: if the blob's signature (node types, key paths,
                shapes, dtypes) differs from the skeleton's, or — for a legacy
                blob without a signature — if it holds more array leaves
                than the skeleton requests.
            RuntimeError: from newer equinox versions when a
                deserialised leaf's shape/dtype doesn't match the
                skeleton (``eqx.tree_deserialise_leaves``).
        """
        name = type(self).__name__
        if not isinstance(blob, (bytes, bytearray)):
            raise TypeError(
                f"{name}.with_weights(blob): blob must be bytes "
                f"(produced by serialize_weights); got {type(blob).__name__}."
            )
        skeleton = self._weights_tree()
        buf = io.BytesIO(blob)
        if bytes(blob[: len(_MAGIC)]) == _MAGIC:
            buf.seek(len(_MAGIC))
            size = int.from_bytes(buf.read(8), "little")
            saved = json.loads(buf.read(size))
            expected = tree_signature(skeleton)
            if saved["nodes"] != expected["nodes"]:
                raise ValueError(
                    f"{name}.with_weights: the blob's tree structure does not "
                    "match the skeleton: the pytree node types differ (saved "
                    f"root {saved['nodes'][0]!r}, skeleton root "
                    f"{expected['nodes'][0]!r}). Load weights only into the "
                    "class and structure they were saved from."
                )
            saved_leaves, expected_leaves = saved["leaves"], expected["leaves"]
            if saved_leaves != expected_leaves:
                diff = next(
                    (
                        (a, b)
                        for a, b in zip(saved_leaves, expected_leaves, strict=False)
                        if a != b
                    ),
                    (
                        f"{len(saved_leaves)} array leaves",
                        f"{len(expected_leaves)} array leaves",
                    ),
                )
                raise ValueError(
                    f"{name}.with_weights: the blob's tree structure does not "
                    f"match the skeleton (saved {diff[0]!r}, skeleton "
                    f"{diff[1]!r}). Build the skeleton with the same structure, "
                    "field order and shapes as the operator that was serialised."
                )
            arrays = eqx.tree_deserialise_leaves(
                buf, eqx.filter(skeleton, eqx.is_array)
            )
            tree = eqx.combine(arrays, skeleton)
        else:
            # Legacy blob: written with Equinox's default filter.
            tree = eqx.tree_deserialise_leaves(buf, skeleton)
        unread = len(blob) - buf.tell()
        if unread:
            # The skeleton asked for fewer leaves than were saved (e.g. an
            # optional bias that is None in the skeleton): a partial restore
            # would silently drop weights.
            raise ValueError(
                f"{name}.with_weights: {unread} bytes of the blob "
                "were not consumed — the saved tree has more array leaves than "
                "the skeleton. Build the skeleton with the same structure as the "
                "operator that was serialised."
            )
        return self._from_weights_tree(tree)

    def from_registry(self, registry: Any, ref: str) -> Any:
        """Reload weights stored under ``ref`` into a copy of ``self``.

        The inverse of::

            registry.store(trained_op, weights=trained_op.serialize_weights())

        Args:
            registry: Anything satisfying
                ``pipekit_experiment.ModelRegistry`` (which includes
                ``load_weights(ref) -> bytes | None``).
            ref: Hash or tag identifying the stored model.

        Returns:
            A new operator with the persisted weights loaded into a
            copy of ``self``.

        Raises:
            TypeError: if ``registry`` doesn't expose a way to read
                the raw weight bytes.
            ValueError: if no weights are stored, or they don't match
                ``self``'s structure.
        """
        name = type(self).__name__
        loader = getattr(registry, "load_weights", None)
        if loader is None:
            raise TypeError(
                f"{name}.from_registry: registry must satisfy the "
                "ModelRegistry protocol (missing `load_weights`). Got "
                f"{type(registry).__name__}."
            )
        blob = loader(ref)
        if blob is None:
            raise ValueError(
                f"{name}.from_registry: no weights blob stored "
                f"for ref={ref!r}. The model was registered without "
                "weights bytes — pass weights=op.serialize_weights() "
                "to registry.store() when persisting it."
            )
        return self.with_weights(blob)
