"""`JaxModelOp` — pipekit.Operator wrapper around an `eqx.Module`.

Closes the v0.1 reproducibility caveat documented in
``pipekit-train/docs/design/boundaries.md §13.1``: the in-package
``EquinoxModelOp`` stand-in shipped with v0.1 couldn't round-trip
trained weights through a ``ModelRegistry`` from JSON config alone,
because an ``eqx.Module`` is a JAX PyTree of arrays whose values
aren't recoverable from a class name.

`JaxModelOp` adds two methods on top of the same wrapper surface:

- `serialize_weights() -> bytes` — `eqx.tree_serialise_leaves` of
  the wrapped module's array leaves. Suitable for the registry's
  ``weights`` blob.
- `with_weights(blob: bytes) -> JaxModelOp` — load the bytes back
  into the module's array leaves, returning a new `JaxModelOp` whose
  ``module`` is structurally identical to ``self.module`` but with
  the persisted weight values. ``self`` is the **skeleton** /
  template.

The skeleton-from-template requirement is honest: an ``eqx.Module``'s
non-array leaves (activation functions, dtypes, hidden-size ints) are
*Python objects* not reproducible from JSON alone. The user always
needs to build a fresh module of the same shape before calling
`with_weights`. `from_registry` packages the
"reload weights from a stored hash into self" pattern into one call.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any, ClassVar, cast

import equinox as eqx
from pipekit import Operator


if TYPE_CHECKING:
    pass


class JaxModelOp(Operator):
    """Wrap an `eqx.Module` as a `pipekit.Operator` with weight round-trip.

    Drop-in replacement for the in-package
    ``pipekit_train.adapters.equinox.EquinoxModelOp`` stand-in.
    Same constructor signature; same ``_apply`` shape (calls the
    module on the input); additionally supports
    `serialize_weights` / `with_weights` for byte-identical
    persistence through any `pipekit_experiment.ModelRegistry`.

    The wrapper is marked ``forbid_in_yaml = True`` because an
    ``eqx.Module`` is a PyTree of arrays — the *structural* config
    round-trips (we surface ``module_class`` in `get_config`) but
    the array values do not. The round-trip path is:
    ``registry.store(op, weights=op.serialize_weights())`` to
    persist, then ``template.with_weights(blob)`` to reload into
    a fresh skeleton.

    Args:
        module: An ``eqx.Module``.
    """

    forbid_in_yaml: ClassVar[bool] = True
    __config_mixin_auto__ = False

    def __init__(self, module: eqx.Module) -> None:
        if not isinstance(module, eqx.Module):
            raise TypeError(
                f"JaxModelOp.module must be an eqx.Module; got {type(module).__name__}."
            )
        self.module = module

    def _apply(self, x: Any) -> Any:
        # eqx.Module is callable but ty can't see that statically.
        return cast(Any, self.module)(x)

    def get_config(self) -> dict[str, Any]:
        return {"module_class": type(self.module).__qualname__}

    # --- Weight round-trip --------------------------------------------

    def serialize_weights(self) -> bytes:
        """Serialise the wrapped module's array leaves to bytes.

        Uses ``equinox.tree_serialise_leaves``, which writes each
        array leaf's raw bytes plus a tree-shape header. The
        complementary `with_weights` reads the same format.
        """
        buf = io.BytesIO()
        eqx.tree_serialise_leaves(buf, self.module)
        return buf.getvalue()

    def with_weights(self, blob: bytes) -> JaxModelOp:
        """Return a new `JaxModelOp` with weights loaded from ``blob``.

        ``self.module`` is used as the **skeleton** / template — its
        non-array leaves (activations, dtypes, layer sizes) determine
        the structure into which the persisted bytes are loaded. The
        bytes must come from a `serialize_weights` call on a module
        with the same PyTree structure (typically the same module
        class with the same constructor arguments).

        Args:
            blob: Bytes produced by `serialize_weights`.

        Returns:
            A fresh `JaxModelOp` whose ``module`` carries the
            persisted weight values.

        Raises:
            ValueError: if ``blob`` doesn't deserialise into a tree
                with the same structure as ``self.module``.
            RuntimeError: from newer equinox versions when a
                deserialised leaf's shape/dtype doesn't match the
                skeleton (``eqx.tree_deserialise_leaves``).
        """
        if not isinstance(blob, (bytes, bytearray)):
            raise TypeError(
                "JaxModelOp.with_weights(blob): blob must be bytes "
                f"(produced by serialize_weights); got {type(blob).__name__}."
            )
        buf = io.BytesIO(blob)
        new_module = eqx.tree_deserialise_leaves(buf, self.module)
        return JaxModelOp(new_module)

    # --- Registry round-trip ------------------------------------------

    def from_registry(self, registry: Any, ref: str) -> JaxModelOp:
        """Reload weights stored under ``ref`` from ``registry`` into
        ``self`` as the skeleton, returning a fresh `JaxModelOp`.

        This is the inverse of:

            registry.store(trained_op, weights=trained_op.serialize_weights())

        Args:
            registry: Anything satisfying
                ``pipekit_experiment.ModelRegistry`` (which includes
                ``load_weights(ref) -> bytes | None``).
            ref: Hash or tag identifying the stored model.

        Returns:
            A new `JaxModelOp` with the persisted weights loaded
            into a copy of ``self.module``.

        Raises:
            TypeError: if ``registry`` doesn't expose a way to read
                the raw weight bytes.
            ValueError: if the stored weights don't match
                ``self.module``'s structure.
        """
        loader = getattr(registry, "load_weights", None)
        if loader is None:
            raise TypeError(
                "JaxModelOp.from_registry: registry must satisfy the "
                "ModelRegistry protocol (missing `load_weights`). Got "
                f"{type(registry).__name__}."
            )
        blob = loader(ref)
        if blob is None:
            raise ValueError(
                f"JaxModelOp.from_registry: no weights blob stored "
                f"for ref={ref!r}. The model was registered without "
                "weights bytes — pass weights=op.serialize_weights() "
                "to registry.store() when persisting a JaxModelOp."
            )
        return self.with_weights(blob)
