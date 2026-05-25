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
  (target)`` if available; otherwise falls back to
  ``-(target.log_prob(samples) - predicted.log_prob(samples)).mean()``
  with samples drawn from ``predicted``.
- `Composite` — weighted sum of `Loss` components, emitting the
  per-component values as auxiliary metrics.

See ``docs/design/api/primitives.md``.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pipekit import Operator


@runtime_checkable
class Loss(Protocol):
    """Loss function — per-batch scalar called by the adapter.

    Implementations may return either a scalar loss directly, or a
    ``(scalar, aux_metrics_dict)`` tuple. The adapter distinguishes
    by introspection.

    See ``docs/design/api/primitives.md`` and ADR D4 in
    ``docs/design/decisions.md``.
    """

    def __call__(
        self,
        predicted: Any,
        target: Any,
    ) -> float | tuple[float, dict[str, float]]: ...


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

    Contract:
        ``predicted`` and ``target`` must both expose either
        ``.kl_divergence(other)`` (fast analytic path) or
        ``.log_prob(samples)`` + ``.sample(key)`` (sample-based
        fallback). The fast path is tried first.

    Returns from ``__call__``:
        ``(loss, {"kl": loss})``.

    Note:
        The sample-based fallback requires a PRNG key. v0.1 ships the
        analytic path only; sample-based KL lands in v0.1.1 when the
        Equinox adapter exposes the per-batch RNG.
    """

    def _apply(
        self,
        predicted: Any,
        target: Any,
    ) -> tuple[Any, dict[str, Any]]:
        if hasattr(predicted, "kl_divergence"):
            loss = predicted.kl_divergence(target).mean()
            return loss, {"kl": loss}
        raise TypeError(
            "KL requires `predicted` to expose `.kl_divergence(target)`. "
            "Sample-based KL fallback is scheduled for v0.1.1; see "
            "docs/design/api/primitives.md."
        )


class Composite(Operator):
    """Weighted sum of `Loss` components.

    The aggregate loss is ``sum(w_i * L_i(predicted, target))``. Each
    component's scalar value is surfaced as an auxiliary metric so
    the adapter logs them individually.

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
    ) -> tuple[Any, dict[str, Any]]:
        aux: dict[str, Any] = {}
        total = None
        for i, (weight, loss) in enumerate(self.components):
            result = loss(predicted, target)
            if isinstance(result, tuple):
                value = result[0]
            else:
                value = result
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
