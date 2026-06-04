"""BlackJAX adapter — ``backend="blackjax"`` (sampler-library backend).

BlackJAX is a sampler library on a ``logdensity_fn`` (PPL-agnostic), not a
PPL — so it is its own backend, but it interoperates with NumPyro: a
`BlackjaxTask` accepts a raw ``logdensity_fn`` *or* a `NumpyroTask` (bridged
via ``numpyro.infer.util.initialize_model``), reusing the shared
``adapters._bayes`` seam (`NumpyroPredictiveOp`, the constrain map). Its
per-step ``kernel.step`` maps onto the per-step `TrainingLoop` (per-sample
diagnostics). See ADR D14 and ``docs/design/blackjax_adapter.md``.

v1 implements the full-batch **NUTS** path (the "same model, different
engine" headline). MCLMC / HMC and the stochastic-gradient samplers
(``sgld`` / ``sghmc``) are staged follow-ons and currently raise
``NotImplementedError`` with guidance.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np
from pipekit import Operator

from pipekit_train.adapters import _bayes


if TYPE_CHECKING:
    from pipekit_train.loop import TrainingLoop

_SUPPORTED_SAMPLERS: tuple[str, ...] = ("nuts",)
_PLANNED_SAMPLERS: tuple[str, ...] = ("mclmc", "hmc", "sgld", "sghmc")


@dataclass
class BlackjaxTask:
    """A BlackJAX inference target (``loop.task`` for ``backend="blackjax"``).

    Supply either a raw target (``logdensity_fn`` + ``init_position``) or a
    `NumpyroTask` (``numpyro_task``) — the latter is bridged to a logdensity
    + constrain map and reuses the model's `Predictive`.

    Attributes:
        logdensity_fn: ``position -> log p(position)`` (raw, full-batch).
        init_position: Starting position for the raw path.
        predict_fn: ``(params, x) -> prediction`` for the raw predictive op.
        numpyro_task: A `NumpyroTask` to bridge into a logdensity.
        sampler: One of ``"nuts"`` (v1) | ``"mclmc"``/``"hmc"``/``"sgld"``/
            ``"sghmc"`` (planned).
        num_warmup: ``window_adaptation`` steps.
        num_chains: Reserved for multi-chain (vmap); v1 uses 1.
        num_integration_steps: Required for ``hmc``/``sghmc`` (planned).
        grad_estimator, step_size: Reserved for SG-MCMC (planned).
        predictive_site: Model site the trained operator returns.
    """

    logdensity_fn: Callable[..., Any] | None = None
    init_position: Any = None
    predict_fn: Callable[..., Any] | None = None
    numpyro_task: _bayes.NumpyroTask | None = None
    sampler: Literal["nuts", "mclmc", "hmc", "sgld", "sghmc"] = "nuts"
    num_warmup: int = 1000
    num_chains: int = 1
    num_integration_steps: int | None = None
    grad_estimator: Callable[..., Any] | None = None
    step_size: float = 1e-3
    predictive_site: str = "obs"


class BlackjaxPredictiveOp(Operator):
    """Predictive operator for a raw-logdensity BlackJAX run.

    ``_apply(x)`` averages ``predict_fn(theta_s, x)`` over the posterior
    samples. When no ``predict_fn`` is given the op cannot predict — call
    ``.posterior`` for the raw samples or supply ``predict_fn``.
    """

    forbid_in_yaml: ClassVar[bool] = True
    __config_mixin_auto__ = False

    def __init__(
        self,
        predict_fn: Callable[..., Any] | None,
        posterior: dict[str, Any],
        predictive_site: str = "obs",
    ) -> None:
        self.predict_fn = predict_fn
        self.posterior = posterior
        self.predictive_site = predictive_site

    def _apply(self, x: Any) -> Any:
        import jax  # ty: ignore[unresolved-import]
        import jax.numpy as jnp  # ty: ignore[unresolved-import]

        if self.predict_fn is None:
            raise RuntimeError(
                "BlackjaxPredictiveOp has no predict_fn; pass predict_fn to "
                "BlackjaxTask to get a predictive operator, or read .posterior."
            )
        preds = jax.vmap(lambda theta: self.predict_fn(theta, jnp.asarray(x)))(
            self.posterior
        )
        return jnp.mean(preds, axis=0)

    def serialize_weights(self) -> bytes:
        import pickle

        payload = {k: np.asarray(v) for k, v in self.posterior.items()}
        return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)

    def get_config(self) -> dict[str, Any]:
        return {"predictive_site": self.predictive_site}


def _validate_task(loop: Any) -> BlackjaxTask:
    task = getattr(loop, "task", None)
    if task is None:
        raise ValueError(
            "The blackjax backend is task-first: pass a "
            "pipekit_train.adapters.blackjax.BlackjaxTask as loop.task."
        )
    if not isinstance(task, BlackjaxTask):
        raise TypeError(
            f"loop.task must be a BlackjaxTask for backend='blackjax'; got "
            f"{type(task).__name__}."
        )
    return task


def run(loop: TrainingLoop) -> tuple[Operator, dict[str, Any]]:
    """Sample ``loop.task`` with BlackJAX (NUTS); return ``(op, info)``."""
    import blackjax  # ty: ignore[unresolved-import]
    import jax  # ty: ignore[unresolved-import]
    import jax.numpy as jnp  # ty: ignore[unresolved-import]

    t0 = time.time()
    task = _validate_task(loop)
    if task.sampler in _PLANNED_SAMPLERS:
        raise NotImplementedError(
            f"sampler={task.sampler!r} is a planned BlackJAX follow-on; v1 "
            f"implements {sorted(_SUPPORTED_SAMPLERS)}. See "
            "docs/design/blackjax_adapter.md (build order)."
        )
    if task.sampler not in _SUPPORTED_SAMPLERS:
        raise ValueError(f"Unknown sampler {task.sampler!r}.")

    x_train, y_train = _bayes.materialize(loop.dataset)
    rng = jax.random.key(loop.seed)
    rng, init_key, warm_key = jax.random.split(rng, 3)

    # --- resolve the target (numpyro model or raw logdensity) -------------
    model = None
    constrain_fn = None
    if task.numpyro_task is not None:
        model = task.numpyro_task.model
        logdensity_fn, init_position, constrain_fn = _bayes.numpyro_logdensity(
            model, x_train, y_train, init_key
        )
    elif task.logdensity_fn is not None and task.init_position is not None:
        logdensity_fn, init_position = task.logdensity_fn, task.init_position
    else:
        raise ValueError(
            "BlackjaxTask needs either numpyro_task, or logdensity_fn + "
            "init_position (raw target)."
        )

    # --- warmup (window adaptation) + tuned NUTS kernel -------------------
    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity_fn)
    (state, tuned), _ = warmup.run(warm_key, init_position, num_steps=task.num_warmup)
    kernel = blackjax.nuts(logdensity_fn, **tuned)
    # JIT the transition once — calling the kernel step un-jitted in a Python
    # loop would re-trace/compile every iteration (pathologically slow).
    step_fn = jax.jit(kernel.step)

    initial = _bayes.carry_state(loop, loop.model_op, 0, {})
    _bayes.dispatch(loop.callbacks, "on_train_begin", loop, initial)

    # --- per-step sampling loop ------------------------------------------
    positions: list[Any] = []
    num_divergent = 0
    step = 0
    for step in range(1, loop.max_steps + 1):
        rng, step_key = jax.random.split(rng)
        state, info = step_fn(step_key, state)
        positions.append(state.position)
        num_divergent += int(bool(info.is_divergent))
        metrics = {
            "acceptance_rate": float(info.acceptance_rate),
            "is_divergent": float(info.is_divergent),
        }
        if loop.metric_writer is not None and step % loop.log_every_n_steps == 0:
            loop.metric_writer.write(step, metrics)
        carry = _bayes.carry_state(loop, loop.model_op, step, metrics)
        _bayes.dispatch(loop.callbacks, "on_step_end", loop, carry, metrics)
        if loop.steps_per_epoch and step % loop.steps_per_epoch == 0:
            _bayes.dispatch(loop.callbacks, "on_epoch_end", loop, carry, metrics)
        if _any_should_stop(loop.callbacks):
            break

    # stack the per-step positions into a pytree with a leading sample axis
    samples = jax.tree.map(lambda *leaves: jnp.stack(leaves), *positions)

    # --- build the predictive artifact -----------------------------------
    if model is not None:
        from numpyro.infer import Predictive  # ty: ignore[unresolved-import]

        assert constrain_fn is not None  # set whenever model is (numpyro path)
        constrained = jax.vmap(constrain_fn)(samples)
        predictive = Predictive(model, posterior_samples=constrained)
        model_op: Operator = _bayes.NumpyroPredictiveOp(
            predictive,
            predictive_site=task.predictive_site,
            seed=loop.seed,
            posterior=dict(constrained),
        )
    else:
        model_op = BlackjaxPredictiveOp(
            task.predict_fn, dict(samples), predictive_site=task.predictive_site
        )

    final = _bayes.carry_state(loop, model_op, int(step), {})
    _bayes.dispatch(loop.callbacks, "on_train_end", loop, final)
    if loop.metric_writer is not None:
        loop.metric_writer.close()

    backend_info = {
        "backend": "blackjax",
        "sampler": task.sampler,
        "jax_version": jax.__version__,
        "blackjax_version": blackjax.__version__,
        "devices": [str(d) for d in jax.devices()],
        "total_seconds": time.time() - t0,
        "final_step": int(step),
        "num_warmup": task.num_warmup,
        "num_samples": int(step),
        "num_divergences": num_divergent,
        "final_metrics": {},
    }
    return model_op, backend_info


def _any_should_stop(callbacks: tuple[Any, ...]) -> bool:
    return any(getattr(cb, "should_stop", False) for cb in callbacks)
