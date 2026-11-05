"""Loss protocol + common loss operators.

The carrier-agnostic loss surface from the v0.1 design.

- `Loss` (Protocol) — runtime-checkable; signature
  ``(predicted, target) -> float | tuple[float, dict]``.
- `MSE` — mean squared error. Carrier-agnostic via duck typing on
  ``-``, ``**``, ``.mean``; works for both numpy and JAX arrays.
- `NLL` — assumes ``predicted`` has ``.log_prob(target)``. Distribution-
  shaped outputs (Equinox distributions, numpyro distributions, custom
  density heads). For Gaussian-NLL on raw (mean, log_var) tuples,
  wrap them in a small density head that exposes ``.log_prob``.
- `KL` — Kullback-Leibler divergence. Prefers ``predicted.kl_divergence
  (target)`` if available; otherwise estimates it with samples drawn from
  ``predicted``, which needs a PRNG ``key``.
- `Composite` — weighted sum of `Loss` components, emitting the
  per-component values as auxiliary metrics.

See ``docs/design/api/primitives.md``.
"""

from __future__ import annotations

import inspect
from typing import Any, Protocol, runtime_checkable

from pipekit import Operator

from pipekit_train._signatures import accepts_keyword


@runtime_checkable
class Loss(Protocol):
    """Loss function — per-batch scalar called by the adapter.

    Implementations may return either a scalar loss directly, or a
    ``(scalar, aux_metrics_dict)`` tuple. The adapter distinguishes
    by introspection. A loss that needs randomness may also take a keyword
    ``key``, which the adapter supplies per batch; see `call_loss`.

    See ``docs/design/api/primitives.md`` and ADR D4 in
    ``docs/design/decisions.md``.
    """

    def __call__(
        self,
        predicted: Any,
        target: Any,
    ) -> float | tuple[float, dict[str, float]]: ...


def call_loss(loss: Any, predicted: Any, target: Any, key: Any = None) -> Any:
    """Call ``loss``, passing ``key`` only if it takes one.

    Args:
        loss: Any `Loss`-shaped callable.
        predicted: The model's prediction.
        target: The target.
        key: Optional PRNG key.

    Returns:
        Whatever ``loss`` returns.
    """
    if key is not None and accepts_keyword(loss, "key"):
        return loss(predicted, target, key=key)
    return loss(predicted, target)


class MSE(Operator):
    """Mean squared error loss.

    Args:
        reduction: ``"mean"`` (default), ``"sum"``, or ``"none"``.
            ``"none"`` returns the elementwise squared error without
            reduction — the adapter is then responsible for reducing.

    Returns from ``__call__``:
        ``(loss, {"mse": loss})`` so adapters see a consistent
        ``(scalar, aux)`` shape.
    """

    def __init__(self, reduction: str = "mean") -> None:
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(
                f"MSE.reduction must be one of 'mean', 'sum', 'none'; "
                f"got {reduction!r}."
            )
        self.reduction = reduction

    def _apply(
        self,
        predicted: Any,
        target: Any,
    ) -> tuple[Any, dict[str, Any]]:
        diff = predicted - target
        sq = diff * diff
        if self.reduction == "mean":
            loss = sq.mean()
        elif self.reduction == "sum":
            loss = sq.sum()
        else:
            loss = sq
        return loss, {"mse": loss}


class NLL(Operator):
    """Negative log-likelihood for distribution-shaped predictions.

    Contract: ``predicted`` must expose ``.log_prob(target)`` returning
    a per-sample log-probability array. The loss reduces over the
    leading batch axis with ``reduction``.

    Typical wrappers — your model emits a distribution-shaped output:

    - For amortized inference: a ``pyrox`` / ``distrax`` / ``numpyro``
      distribution whose ``.log_prob`` is differentiable through the
      model parameters.
    - For Gaussian regression: a tiny head operator that wraps
      ``(mean, log_var)`` model outputs into a ``Normal``-like object.

    Args:
        reduction: ``"mean"`` (default), ``"sum"``, or ``"none"``.

    Returns from ``__call__``:
        ``(loss, {"nll": loss})``.
    """

    def __init__(self, reduction: str = "mean") -> None:
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(
                f"NLL.reduction must be one of 'mean', 'sum', 'none'; "
                f"got {reduction!r}."
            )
        self.reduction = reduction

    def _apply(
        self,
        predicted: Any,
        target: Any,
    ) -> tuple[Any, dict[str, Any]]:
        if not hasattr(predicted, "log_prob"):
            raise TypeError(
                "NLL expects `predicted` to expose a `.log_prob(target)` "
                "method (a distribution-shaped output). Got "
                f"{type(predicted).__name__}. Wrap your model output in a "
                "density head, or use MSE for plain-array regression."
            )
        log_probs = predicted.log_prob(target)
        if self.reduction == "mean":
            loss = -log_probs.mean()
        elif self.reduction == "sum":
            loss = -log_probs.sum()
        else:
            loss = -log_probs
        return loss, {"nll": loss}


class KL(Operator):
    """Kullback-Leibler divergence between two distributions.

    KL(q ‖ p) = E_q[log q(z) - log p(z)], where q is ``predicted`` and p is
    ``target``. The analytic path is used when ``predicted`` has
    ``.kl_divergence(target)``. Otherwise it is estimated by Monte Carlo:
    (1 / S) Σ_s [log q(z_s) - log p(z_s)], with S draws z_s from q.

    Algorithm:
        1. Use ``predicted.kl_divergence(target)`` if it exists.
        2. Otherwise draw ``n_samples`` from ``predicted`` with ``key``.
        3. Average ``log_prob`` differences over draws and any batch dims.

    Gradients flow through the draws only if ``predicted`` samples by
    reparameterisation.

    Args:
        n_samples: Monte Carlo draws S for the estimate. Its error falls as
            1 / √S.

    Returns from ``__call__``:
        ``(loss, {"kl": loss})``.

    Raises:
        ValueError: If ``n_samples`` is below 1.
        TypeError: If the estimate is needed and no ``key`` was given.

    Examples:
        >>> class Point:
        ...     def kl_divergence(self, other):
        ...         import numpy as np
        ...
        ...         return np.array([0.5, 1.5])
        >>> KL()(Point(), None)[0].item()
        1.0
    """

    def __init__(self, n_samples: int = 32) -> None:
        if n_samples < 1:
            raise ValueError(f"KL.n_samples must be >= 1; got {n_samples}.")
        self.n_samples = n_samples

    def _apply(
        self,
        predicted: Any,
        target: Any,
        key: Any = None,
    ) -> tuple[Any, dict[str, Any]]:
        if hasattr(predicted, "kl_divergence"):
            loss = predicted.kl_divergence(target).mean()
            return loss, {"kl": loss}
        if key is None:
            raise TypeError(
                "Sample-based KL requires a PRNG key. The Equinox adapter "
                "supplies one per batch; for ad-hoc use, pass "
                "key=jax.random.key(seed)."
            )
        # z_s ~ q
        samples = _draw_samples(predicted, key, self.n_samples)
        # log q(z_s)
        log_q = predicted.log_prob(samples)
        # log p(z_s)
        log_p = target.log_prob(samples)
        # KL(q ‖ p) ≈ (1 / S) Σ_s [log q(z_s) - log p(z_s)]
        loss = (log_q - log_p).mean()
        return loss, {"kl": loss}


def _draw_samples(distribution: Any, key: Any, n_samples: int) -> Any:
    """Draw from a distribution in the library's own calling convention.

    NumPyro takes ``sample(key, sample_shape)``, whereas distrax and TFP take
    ``sample(seed=..., sample_shape=...)``.
    """
    parameters = inspect.signature(distribution.sample).parameters
    if "seed" in parameters:
        return distribution.sample(seed=key, sample_shape=(n_samples,))
    return distribution.sample(key, sample_shape=(n_samples,))


class Composite(Operator):
    """Weighted sum of `Loss` components.

    The aggregate loss is ``sum(w_i * L_i(predicted, target))``. Each
    component's scalar value is surfaced as an auxiliary metric so
    the adapter logs them individually. A component that takes a ``key`` gets
    its own, split from the one passed in.

    Args:
        components: Tuple of ``(weight, loss)`` pairs. ``weight`` is
            a Python float; ``loss`` is any `Loss`-shaped callable.

    Raises:
        ValueError: if ``components`` is empty.

    Returns from ``__call__``:
        ``(total, {"<name>_<i>": value, ..., "total": total})``.
    """

    __config_mixin_auto__ = False

    def __init__(
        self,
        components: tuple[tuple[float, Loss], ...],
    ) -> None:
        if not components:
            raise ValueError("Composite requires at least one component.")
        self.components = tuple(components)

    def _apply(
        self,
        predicted: Any,
        target: Any,
        key: Any = None,
    ) -> tuple[Any, dict[str, Any]]:
        aux: dict[str, Any] = {}
        total = None
        keys: Any = [None] * len(self.components)
        if key is not None:
            import jax

            keys = jax.random.split(key, len(self.components))
        for i, (weight, loss) in enumerate(self.components):
            result = call_loss(loss, predicted, target, keys[i])
            value = result[0] if isinstance(result, tuple) else result
            name = f"{type(loss).__name__}_{i}"
            aux[name] = value
            weighted = weight * value
            total = weighted if total is None else total + weighted
        aux["total"] = total
        return total, aux

    def get_config(self) -> dict[str, Any]:
        # The components tuple holds Loss callables that may be
        # arbitrary operators — surface their class names and weights.
        return {
            "components": [
                {"weight": float(w), "loss_class": type(loss).__name__}
                for w, loss in self.components
            ]
        }
