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

from typing import TYPE_CHECKING, Any, ClassVar, cast

import equinox as eqx
from pipekit import Operator

from pipekit_jax.weights import WeightsMixin


if TYPE_CHECKING:
    pass


class JaxModelOp(WeightsMixin, Operator):
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

    # --- Weight round-trip (shared via WeightsMixin) -------------------
    # The persisted tree is the wrapped module, not the wrapper: an
    # eqx.Module's array leaves round-trip, the Operator shell doesn't.

    def _weights_tree(self) -> Any:
        return self.module

    def _from_weights_tree(self, tree: Any) -> JaxModelOp:
        return JaxModelOp(tree)
