"""Shared Bayesian-backend seam for the NumPyro (and BlackJAX) adapters.

Per ADR D13/D14, the NumPyro `svi` / `mcmc` adapters (and the planned
BlackJAX adapter) are *task-first*: their objective is a probabilistic
``model(x, y=None)`` rather than a carrier-agnostic ``Loss(pred, target)``.
This module holds what they share:

- `NumpyroTask` — the per-backend task (the model + predictive config plus
  per-paradigm fields). No ``method`` field: the **backend string** selects
  SVI vs MCMC, so the same task works under either.
- `NumpyroPredictiveOp` — the trained artifact: a `pipekit.Operator` wrapping
  a ``numpyro.infer.Predictive`` that returns the posterior-predictive mean
  of a chosen site.
- small helpers: dataset materialisation, the optax optimiser builder, and
  task validation.

Importing this module needs ``numpyro`` (the ``[numpyro]`` extra); it is
imported lazily inside the adapters' ``run`` so the package loads without
the extra.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from pipekit import Operator


if TYPE_CHECKING:
    from pipekit_train.dataset import TrainingDataset

# optax optimiser names supported by ``optimizer_config`` (mirrors the
# Equinox adapter's surface, kept here so the [numpyro] extra needn't pull in
# the equinox adapter module).
_OPTIMIZER_NAMES: tuple[str, ...] = ("adam", "adamw", "sgd", "rmsprop")


@dataclass
class NumpyroTask:
    """A NumPyro inference target (``loop.task`` for the numpyro backends).

    The same task is consumed by ``backend="numpyro-svi"`` and
    ``backend="numpyro-mcmc"``; each reads only the fields it needs (there is
    no ``method`` field — the backend selects the paradigm).

    Attributes:
        model: A NumPyro model ``model(x, y=None)`` placing priors on the
            latents and a likelihood ``numpyro.sample("<site>", ..., obs=y)``.
        guide: SVI guide; defaults to ``AutoNormal(model)`` when ``None``.
        loss: SVI objective; defaults to ``Trace_ELBO()`` when ``None``.
        predictive_site: The model site the trained operator returns.
        predictive_samples: Number of posterior-predictive draws averaged.
        num_warmup, num_samples, num_chains: MCMC (NUTS) controls.
    """

    model: Callable[..., Any]
    guide: Callable[..., Any] | None = None
    loss: Any = None
    predictive_site: str = "obs"
    predictive_samples: int = 200
    num_warmup: int = 1000
    num_samples: int = 1000
    num_chains: int = 1


class NumpyroPredictiveOp(Operator):
    """Wrap a NumPyro posterior as a posterior-predictive `pipekit.Operator`.

    ``_apply(x)`` draws from the wrapped ``Predictive`` and returns the mean
    over draws of ``predictive_site``. Marked ``forbid_in_yaml`` because it
    carries the model closure + a sample/param PyTree; the round-trip path is
    the model registry weight-blob (the posterior samples / variational
    params), analogous to ``pipekit-jax.JaxModelOp``.

    Args:
        predictive: A configured ``numpyro.infer.Predictive``.
        predictive_site: Site to read from the predictive dict.
        seed: PRNG seed for the (stochastic) predictive draw.
    """

    forbid_in_yaml: ClassVar[bool] = True
    __config_mixin_auto__ = False

    def __init__(
        self,
        predictive: Any,
        predictive_site: str = "obs",
        seed: int = 0,
        posterior: dict[str, Any] | None = None,
    ) -> None:
        self.predictive = predictive
        self.predictive_site = predictive_site
        self.seed = int(seed)
        # The variational params (SVI) or posterior samples (MCMC) — the
        # meaningful "weights" of this operator, persisted via
        # ``serialize_weights`` for the model-registry round-trip.
        self.posterior = posterior

    def _apply(self, x: Any) -> Any:
        import jax  # ty: ignore[unresolved-import]
        import jax.numpy as jnp  # ty: ignore[unresolved-import]

        draws = self.predictive(jax.random.key(self.seed), jnp.asarray(x))
        if self.predictive_site not in draws:
            raise KeyError(
                f"predictive_site {self.predictive_site!r} not in the model's "
                f"predictive sites {sorted(draws)}."
            )
        return jnp.mean(draws[self.predictive_site], axis=0)

    def serialize_weights(self) -> bytes:
        """Return the posterior PyTree as a portable weight blob.

        This is the model-registry round-trip path (see the Equinox
        adapter's ``JaxModelOp`` analogue): ``Checkpoint(registry=…)`` stores
        these bytes alongside the operator state. The model/guide closures
        are *not* serialised — like ``EquinoxModelOp``, a full reload
        re-attaches the persisted posterior to a re-supplied model.
        """
        import pickle

        payload = {k: np.asarray(v) for k, v in (self.posterior or {}).items()}
        return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)

    def get_config(self) -> dict[str, Any]:
        return {"predictive_site": self.predictive_site, "seed": self.seed}


def validate_task(loop: Any) -> NumpyroTask:
    """Return ``loop.task`` if it is a `NumpyroTask`; raise otherwise.

    The numpyro backends are task-first: a carrier-agnostic ``Loss`` does not
    define a probabilistic model, so ``loop.task`` is required.
    """
    task = getattr(loop, "task", None)
    if task is None:
        raise ValueError(
            "The numpyro backends are task-first: pass a "
            "pipekit_train.adapters._bayes.NumpyroTask as loop.task "
            "(a Loss does not define a probabilistic model)."
        )
    if not isinstance(task, NumpyroTask):
        raise TypeError(
            "loop.task must be a NumpyroTask for the numpyro backends; got "
            f"{type(task).__name__}."
        )
    return task


def materialize(dataset: TrainingDataset) -> tuple[Any, Any]:
    """Stack a dataset's ``(x, y)`` pairs into full ``(X, y)`` arrays.

    The numpyro backends are full-batch in v1 (SVI minibatching via
    ``numpyro.plate(subsample_size=…)`` is a follow-on), so a single pass
    over ``dataset`` materialises the whole likelihood.
    """
    import jax.numpy as jnp  # ty: ignore[unresolved-import]

    # Single pass: a streaming/stateful dataset would be exhausted by a
    # second iteration, so collect both halves of each pair together.
    xs: list[Any] = []
    ys: list[Any] = []
    for x, y in dataset:
        xs.append(np.asarray(x))
        ys.append(np.asarray(y))
    if not xs:
        raise ValueError("dataset yielded no (x, y) pairs.")
    return jnp.asarray(np.stack(xs)), jnp.asarray(np.stack(ys))


def numpyro_logdensity(model: Callable[..., Any], x: Any, y: Any, key: Any) -> Any:
    """Bridge a NumPyro model to a BlackJAX-style target.

    Returns ``(logdensity_fn, init_position, constrain_fn)`` via
    ``numpyro.infer.util.initialize_model(dynamic_args=True)``:
    ``logdensity_fn(position)`` is the (unconstrained) log-posterior,
    ``init_position`` is the unconstrained starting point, and
    ``constrain_fn(position)`` maps an unconstrained sample back to the
    model's constrained sites (needed before ``Predictive``).
    """
    from numpyro.infer.util import initialize_model  # ty: ignore[unresolved-import]

    info = initialize_model(key, model, model_args=(x, y), dynamic_args=True)
    param_info, potential_fn_gen, postprocess_fn_gen, _ = info

    def logdensity_fn(position: Any) -> Any:
        return -potential_fn_gen(x, y)(position)

    return logdensity_fn, param_info.z, postprocess_fn_gen(x, y)


def build_optimizer(config: dict[str, Any]) -> Any:
    """Translate ``optimizer_config`` into an optax ``GradientTransformation``.

    Mirrors the Equinox adapter: supports ``adam``/``adamw``/``sgd``/
    ``rmsprop``; ``lr`` is normalised to ``learning_rate``.
    """
    import optax  # ty: ignore[unresolved-import]

    config = dict(config) if config else {"name": "adam", "lr": 1e-3}
    name = config.pop("name", "adam")
    if "lr" in config and "learning_rate" not in config:
        config["learning_rate"] = config.pop("lr")
    if name not in _OPTIMIZER_NAMES:
        raise ValueError(
            f"Unknown optimizer {name!r}. Supported: {sorted(_OPTIMIZER_NAMES)}."
        )
    return getattr(optax, name)(**config)


def dispatch(callbacks: tuple[Any, ...], hook: str, *args: Any) -> None:
    """Call ``hook`` on each callback that implements it (mirrors equinox)."""
    for cb in callbacks:
        fn = getattr(cb, hook, None)
        if fn is not None:
            fn(*args)


def carry_state(loop: Any, model_op: Operator, step: int, metrics: dict[str, float]):
    """Build a `TrainerCarryState` snapshot for callback dispatch."""
    from pipekit_train.loop import TrainerCarryState

    epoch = step // loop.steps_per_epoch if loop.steps_per_epoch else 0
    return TrainerCarryState(
        model=model_op,
        opt_state=None,
        step=step,
        epoch=epoch,
        metrics=dict(metrics),
        rng=None,
    )
