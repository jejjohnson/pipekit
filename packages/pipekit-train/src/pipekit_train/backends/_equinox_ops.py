"""Equinox modules inside pipekit operators, shared by the Equinox and Keras backends.

`EquinoxModelOp` wraps a trained ``eqx.Module`` as an operator. `unwrap_module`
finds the module an operator trains, and `rewrap_module` puts the trained module
back in the form the operator had. Both backends train Equinox models (ADR
D21), and one backend area does not import another (D17), so they live here.
The Equinox area re-exports `EquinoxModelOp` and records it under its own name.
"""

from __future__ import annotations

import copy
import io
from typing import Any, ClassVar, cast

import equinox as eqx
from pipekit_ops import Operator


class EquinoxModelOp(Operator):
    """Wrap an `eqx.Module` as a `pipekit_ops.Operator`.

    Public surface used in notebooks and downstream packages. Same
    constructor signature and ``_apply`` shape as
    ``pipekit_jax.JaxModelOp``. It writes its array leaves with
    ``serialize_weights``, so a registry can store them and two trained
    models get different hashes. Swap the import when you need to reload
    them (``with_weights`` / ``from_registry``), which this in-package twin
    deliberately omits so pipekit-train depends only on pipekit core.

    The wrapper is marked ``forbid_in_yaml = True`` because an
    ``eqx.Module`` is a JAX PyTree of arrays — the round-trip path is
    the model registry's ``weights`` blob plus the module's
    structural config, not JSON of the operator itself.

    Args:
        module: An ``eqx.Module``.
    """

    forbid_in_yaml: ClassVar[bool] = True
    __config_mixin_auto__ = False

    def __init__(self, module: eqx.Module) -> None:
        if not isinstance(module, eqx.Module):
            raise TypeError(
                "EquinoxModelOp.module must be an eqx.Module; got "
                f"{type(module).__name__}."
            )
        self.module = module

    def _apply(self, x: Any) -> Any:
        # eqx.Module is callable (instance method __call__), but ty
        # doesn't know that statically.
        return cast(Any, self.module)(x)

    def serialize_weights(self) -> bytes:
        """Return the module's leaves as ``tree_serialise_leaves`` bytes.

        Unfiltered, so that Python scalar fields are written too.
        ``pipekit_jax.JaxModelOp.with_weights`` reads these bytes against an
        unfiltered skeleton of the same structure.
        """
        buffer = io.BytesIO()
        eqx.tree_serialise_leaves(buffer, self.module)
        return buffer.getvalue()

    def get_config(self) -> dict[str, Any]:
        # __qualname__ matches pipekit_jax.JaxModelOp so the two twins emit
        # identical configs for the same module.
        return {"module_class": type(self.module).__qualname__}


def unwrap_module(model_op: Any) -> eqx.Module | None:
    """Return the ``eqx.Module`` that ``model_op`` trains, or ``None``.

    The operator itself when it is a module (``pipekit_jax.JaxOperator``),
    else its ``.module`` when that is one (`EquinoxModelOp`,
    ``pipekit_jax.JaxModelOp``). `rewrap_module` follows the same order.
    """
    if isinstance(model_op, eqx.Module):
        return model_op
    module = getattr(model_op, "module", None)
    return module if isinstance(module, eqx.Module) else None


def rewrap_module(model_op: Any, trained: eqx.Module) -> Operator:
    """Return ``trained`` in the form the user passed the model in.

    Args:
        model_op: The operator given to `TrainingLoop`.
        trained: The module after training.

    Returns:
        ``trained`` itself when ``model_op`` was already a module, such as a
        `JaxOperator`; a copy of ``model_op`` holding ``trained`` when it wraps
        a ``.module``, so a `JaxModelOp` keeps its weight round trip; otherwise
        a new `EquinoxModelOp`. The order matches how ``run`` unwraps its input:
        a module may also have a child named ``module``, and it is frozen.
    """
    if isinstance(model_op, eqx.Module):
        return trained if isinstance(trained, Operator) else EquinoxModelOp(trained)
    if isinstance(getattr(model_op, "module", None), eqx.Module):
        wrapped = copy.copy(model_op)
        wrapped.module = trained
        return wrapped
    return EquinoxModelOp(trained)
