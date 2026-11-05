"""Shared Bayesian-backend seam for the NumPyro and BlackJAX adapters.

Per ADR D13/D14, the NumPyro `svi` / `mcmc` adapters and the BlackJAX
adapter are *task-first*: their objective is a probabilistic
``model(x, y=None)`` rather than a carrier-agnostic ``Loss(pred, target)``.
This module holds what they share:

- `NumpyroTask` — the per-backend task (the model + predictive config plus
  per-paradigm fields). No ``method`` field: the **backend string** selects
  SVI vs MCMC, so the same task works under either.
- `NumpyroPredictiveOp` — the trained artifact: a `pipekit.Operator` wrapping
  a ``numpyro.infer.Predictive`` that returns the posterior-predictive mean
  of a chosen site, and gives access to the full draws.
- small helpers: dataset materialisation, the optax optimiser builder, and
  task validation.

Importing this module needs ``numpyro`` (the ``[numpyro]`` extra); it is
imported lazily inside the adapters' ``run`` so the package loads without
the extra.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from pipekit import Operator

from pipekit_train.adapters import _common
from pipekit_train.adapters._common import any_should_stop, dispatch  # noqa: F401


if TYPE_CHECKING:
    from pipekit_train.dataset import TrainingDataset


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
        num_warmup, num_samples, num_chains: MCMC controls.
        kernel_kwargs: Keyword arguments for the MCMC kernel, such as
            ``{"target_accept_prob": 0.995, "max_tree_depth": 12}``.
        kernel_factory: Replaces ``NUTS``: called as
            ``kernel_factory(model, **kernel_kwargs)``. ``None`` uses ``NUTS``.
        supervised: ``True`` for a dataset of ``(x, y)`` pairs and a call of
            ``model(x, y, **model_kwargs)``. ``False`` for a dataset of single
            items, stacked, and a call of ``model(data, **model_kwargs)``.
        model_kwargs: Keyword arguments for every call of the model, such as
            prior settings. They are constants under ``jit``.
    """

    model: Callable[..., Any]
    guide: Callable[..., Any] | None = None
    loss: Any = None
    predictive_site: str = "obs"
    predictive_samples: int = 200
    num_warmup: int = 1000
    num_samples: int = 1000
    num_chains: int = 1
    kernel_kwargs: dict[str, Any] = field(default_factory=dict)
    kernel_factory: Callable[..., Any] | None = None
    supervised: bool = True
    model_kwargs: dict[str, Any] = field(default_factory=dict)


class NumpyroPredictiveOp(Operator):
    """Wrap a NumPyro posterior as a posterior-predictive `pipekit.Operator`.

    ``_apply(x)`` draws from the wrapped ``Predictive`` and returns the mean
    over draws of ``predictive_site``. `predictive_draws` and
    `posterior_samples` return the full draws, for credible intervals on derived
    quantities. Marked ``forbid_in_yaml`` because it
    carries the model closure + a sample/param PyTree; the round-trip path is
    the model registry weight-blob (the posterior samples / variational
    params), analogous to `pipekit_jax.JaxModelOp`.

    Args:
        predictive: A configured ``numpyro.infer.Predictive``.
        predictive_site: Site to read from the predictive dict.
        seed: PRNG seed for the (stochastic) predictive draw.
        posterior: The posterior samples (MCMC) or variational params (SVI).
        guide: The fitted SVI guide; ``None`` means ``posterior`` holds samples.
        model_kwargs: Keyword arguments for every call of the model.

    Examples:
        The posterior of a fit, without running a model:

        >>> import numpy as np
        >>> samples = {"mu": np.array([0.5, 1.5])}
        >>> op = NumpyroPredictiveOp(None, posterior=samples)
        >>> op.posterior_samples()["mu"].shape
        (2,)
    """

    forbid_in_yaml: ClassVar[bool] = True
    __config_mixin_auto__ = False

    def __init__(
        self,
        predictive: Any,
        predictive_site: str = "obs",
        seed: int = 0,
        posterior: dict[str, Any] | None = None,
        *,
        guide: Any = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.predictive = predictive
        self.predictive_site = predictive_site
        self.seed = int(seed)
        # The variational params (SVI) or posterior samples (MCMC) — the
        # meaningful "weights" of this operator, persisted via
        # ``serialize_weights`` for the model-registry round-trip.
        self.posterior = posterior
        self.guide = guide
        self.model_kwargs = dict(model_kwargs or {})

    def _apply(self, x: Any) -> Any:
        import jax.numpy as jnp

        draws = self.predictive_draws(x)
        if self.predictive_site not in draws:
            raise KeyError(
                f"predictive_site {self.predictive_site!r} not in the model's "
                f"predictive sites {sorted(draws)}."
            )
        return jnp.mean(draws[self.predictive_site], axis=0)

    def predictive_draws(self, *args: Any, key: Any = None) -> dict[str, Any]:
        """Return the full posterior-predictive draws of every site.

        Args:
            *args: The model's positional arguments. Pass ``None`` for an
                unsupervised model's data to draw new data rather than condition
                on it.
            key: PRNG key; defaults to one built from ``seed``.

        Returns:
            A dict from site name to draws, with a leading sample dim.

        Raises:
            TypeError: If this operator was reloaded without a model.
        """
        import jax
        import jax.numpy as jnp

        if self.predictive is None:
            raise TypeError(
                "This NumpyroPredictiveOp has no model; pass model= to from_registry."
            )
        if key is None:
            key = jax.random.key(self.seed)
        arrays = [None if arg is None else jnp.asarray(arg) for arg in args]
        return self.predictive(key, *arrays, **self.model_kwargs)

    def posterior_samples(
        self, *args: Any, num_samples: int | None = None, key: Any = None
    ) -> dict[str, Any]:
        """Return full posterior draws of the latent sites.

        MCMC and BlackJAX fits return their stored samples. An SVI fit draws from
        its guide, which must have ``sample_posterior``, as the autoguides do.

        Args:
            *args: SVI only; the model's positional arguments, needed by a guide
                that replays the model, such as ``AutoDelta`` with deterministic
                sites. ``model_kwargs`` are passed too.
            num_samples: SVI only; defaults to the fit's ``predictive_samples``.
            key: SVI only; PRNG key, defaulting to one built from ``seed``.

        Returns:
            A dict from site name to draws, with a leading sample dim.

        Raises:
            TypeError: If there is no posterior, or a custom SVI guide can't
                sample it.
            ValueError: If an SVI fit has no sample count.
        """
        if self.posterior is None:
            raise TypeError("This NumpyroPredictiveOp holds no posterior.")
        if self.guide is None:
            return dict(self.posterior)
        import jax

        if not hasattr(self.guide, "sample_posterior"):
            raise TypeError(
                f"{type(self.guide).__name__} has no sample_posterior; use "
                "predictive_draws, or an autoguide."
            )
        if num_samples is None:
            num_samples = getattr(self.predictive, "num_samples", None)
        if num_samples is None:
            raise ValueError("Pass num_samples to draw from this guide.")
        if key is None:
            key = jax.random.key(self.seed)
        return self.guide.sample_posterior(
            key,
            self.posterior,
            *args,
            sample_shape=(num_samples,),
            **self.model_kwargs,
        )

    @classmethod
    def from_registry(
        cls,
        registry: Any,
        ref: str,
        *,
        model: Callable[..., Any] | None = None,
        guide: Any = None,
        predictive_site: str = "obs",
        seed: int = 0,
        num_samples: int = 200,
        model_kwargs: dict[str, Any] | None = None,
    ) -> NumpyroPredictiveOp:
        """Reload an operator from the weights stored under ``ref``.

        The model and guide are closures, so they aren't stored. Unpickles the
        stored bytes, so use only a registry you trust.

        Args:
            registry: Anything with ``load_weights(ref) -> bytes | None``.
            ref: Hash or tag of the stored model.
            model: The model. Without it, `posterior_samples` works and
                `predictive_draws` doesn't.
            guide: For an SVI fit, the guide after fitting or ``svi.init``, so
                that it can sample.
            predictive_site: The site the operator returns.
            seed: PRNG seed for draws.
            num_samples: SVI only; the number of predictive draws.
            model_kwargs: Keyword arguments for every call of the model.

        Returns:
            The reloaded operator.

        Raises:
            ValueError: If ``ref`` has no stored weights.
        """
        import pickle

        blob = registry.load_weights(ref)
        if blob is None:
            raise ValueError(f"The registry holds no weights for {ref!r}.")
        posterior = pickle.loads(blob)
        predictive = None
        if model is not None:
            from numpyro.infer import Predictive  # ty: ignore[unresolved-import]

            if guide is None:
                predictive = Predictive(model, posterior_samples=posterior)
            else:
                predictive = Predictive(
                    model, guide=guide, params=posterior, num_samples=num_samples
                )
        return cls(
            predictive,
            predictive_site=predictive_site,
            seed=seed,
            posterior=posterior,
            guide=guide,
            model_kwargs=model_kwargs,
        )

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
            "pipekit_train.adapters.bayes.NumpyroTask as loop.task "
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
    import jax.numpy as jnp

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


def materialize_unpaired(dataset: TrainingDataset) -> Any:
    """Stack a dataset of single items into one array, for unsupervised fits."""
    import jax.numpy as jnp

    items = [np.asarray(item) for item in dataset]
    if not items:
        raise ValueError("dataset yielded no items.")
    return jnp.asarray(np.stack(items))


def model_args(task: NumpyroTask, dataset: TrainingDataset) -> tuple[Any, ...]:
    """Return the model's positional arguments for a whole dataset."""
    if task.supervised:
        return materialize(dataset)
    return (materialize_unpaired(dataset),)


def numpyro_logdensity(
    model: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    key: Any,
) -> Any:
    """Bridge a NumPyro model to a BlackJAX-style target.

    Returns ``(logdensity_fn, init_position, constrain_fn)`` via
    ``numpyro.infer.util.initialize_model(dynamic_args=True)``:
    ``logdensity_fn(position)`` is the (unconstrained) log-posterior,
    ``init_position`` is the unconstrained starting point, and
    ``constrain_fn(position)`` maps an unconstrained sample back to the
    model's constrained sites (needed before ``Predictive``).
    """
    from numpyro.infer.util import initialize_model  # ty: ignore[unresolved-import]

    info = initialize_model(
        key, model, model_args=args, model_kwargs=kwargs, dynamic_args=True
    )
    param_info, potential_fn_gen, postprocess_fn_gen, _ = info

    def logdensity_fn(position: Any) -> Any:
        return -potential_fn_gen(*args, **kwargs)(position)

    return logdensity_fn, param_info.z, postprocess_fn_gen(*args, **kwargs)


def build_optimizer(config: dict[str, Any]) -> Any:
    """Translate ``optimizer_config`` into an optax ``GradientTransformation``.

    Thin wrapper over `adapters._common.build_optimizer` with the
    Bayesian backends' default optimiser (``adam``).
    """
    return _common.build_optimizer(config, default_name="adam")


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
