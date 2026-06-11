"""Group N — carrier-agnostic structural protocols.

The duck-typed seams that model-inference wrappers across the sister
libraries (geotoolz ``ModelOp``, xrtoolz ``ModelOp`` / ``SklearnOp``,
pipekit-jax ``JaxModelOp``) all rely on, promoted to runtime-checkable
Protocols so the contract is written down once instead of re-implied in
each wrapper's docstring. Domain-specific protocols stay in their
domain packages (`pipekit_cycle.protocols`,
`pipekit_experiment.protocols`); this module holds only the
carrier-agnostic ones.

``runtime_checkable`` checks method *presence*, not signatures — an
``isinstance(model, Predictor)`` check is a structural sniff, not a
guarantee that the call will succeed with your carrier.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


__all__ = ["FittableTransformer", "Predictor"]


@runtime_checkable
class Predictor(Protocol):
    """Anything with a sklearn-style ``predict`` method.

    The structural contract behind model-inference wrappers: scikit-learn
    estimators, Keras models, and the marshalling wrappers in the sister
    libraries all satisfy it. Models invoked via bare ``__call__``
    (PyTorch modules, JAX functions) are covered by ``callable(model)``
    instead — a Protocol member named ``__call__`` would match nearly
    every object, so it is deliberately left out.

    Members:
        predict(x): Map a batch of inputs to a batch of outputs.
    """

    def predict(self, x: Any) -> Any: ...


@runtime_checkable
class FittableTransformer(Protocol):
    """Anything with sklearn-style ``fit`` + ``transform`` methods.

    The split-object pattern's structural contract: an estimator that
    learns state from data (``fit``) and applies it (``transform``).
    scikit-learn transformers satisfy it as-is; the sister libraries'
    carrier-aware wrappers (GeoTensor / xarray marshalling) satisfy it
    by construction.

    Members:
        fit(x): Learn internal state from ``x``; return the fitted
            estimator (conventionally ``self``).
        transform(x): Apply the learned state to ``x``.
    """

    def fit(self, x: Any) -> Any: ...

    def transform(self, x: Any) -> Any: ...
