"""Shared weight-blob round trip for pipekit-jax operators.

`WeightsMixin` gives an operator ``serialize_weights`` / ``with_weights``
/ ``from_registry``: the array leaves of a pytree are written with
``equinox.tree_serialise_leaves`` and read back into a **skeleton** — an
instance with the same pytree structure, whose non-array leaves
(activations, dtypes, sizes, static config) cannot be recovered from
bytes alone.

Two hooks pick *which* pytree is persisted:

- ``_weights_tree()`` — the tree to serialise (default: ``self``).
- ``_from_weights_tree(tree)`` — wrap a deserialised tree back into an
  operator (default: return ``tree``).

`JaxModelOp` persists its wrapped ``module``; `JaxOperator` persists
itself.
"""

from __future__ import annotations

import io
from typing import Any

import equinox as eqx


class WeightsMixin:
    """``serialize_weights`` / ``with_weights`` / ``from_registry``."""

    def _weights_tree(self) -> Any:
        return self

    def _from_weights_tree(self, tree: Any) -> Any:
        return tree

    def serialize_weights(self) -> bytes:
        """Serialise the persisted tree's array leaves to bytes.

        Uses ``equinox.tree_serialise_leaves``, which writes each
        array leaf's raw bytes plus a tree-shape header. The
        complementary `with_weights` reads the same format.
        """
        buf = io.BytesIO()
        eqx.tree_serialise_leaves(buf, self._weights_tree())
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
            ValueError: if ``blob`` doesn't deserialise into a tree
                with the same structure as the skeleton, including a blob
                with more array leaves than the skeleton requests.
            RuntimeError: from newer equinox versions when a
                deserialised leaf's shape/dtype doesn't match the
                skeleton (``eqx.tree_deserialise_leaves``).
        """
        if not isinstance(blob, (bytes, bytearray)):
            raise TypeError(
                f"{type(self).__name__}.with_weights(blob): blob must be bytes "
                f"(produced by serialize_weights); got {type(blob).__name__}."
            )
        buf = io.BytesIO(blob)
        tree = eqx.tree_deserialise_leaves(buf, self._weights_tree())
        unread = len(blob) - buf.tell()
        if unread:
            # The skeleton asked for fewer leaves than were saved (e.g. an
            # optional bias that is None in the skeleton): a partial restore
            # would silently drop weights.
            raise ValueError(
                f"{type(self).__name__}.with_weights: {unread} bytes of the blob "
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
