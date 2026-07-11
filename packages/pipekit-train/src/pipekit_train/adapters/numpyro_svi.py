"""NumPyro SVI adapter — ``backend="numpyro-svi"`` (variational inference).

A per-step backend: ``svi.update`` returns ``(state, loss)`` each step, so it
maps onto the existing per-step `TrainingLoop` (callbacks, metric writer,
periodic eval, ``max_steps``, ``optimizer_config``). v1 is full-batch — the
whole dataset is materialised once and each step evaluates the ELBO on all of
it; minibatch SVI (via ``numpyro.plate(subsample_size=…)``) is a follow-on.

The artifact is a `NumpyroPredictiveOp` around
``Predictive(model, guide, params, num_samples)``. See ADR D13 and
``docs/design/numpyro_adapter.md``.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from pipekit_train.adapters import bayes


if TYPE_CHECKING:
    from pipekit import Operator

    from pipekit_train.loop import TrainingLoop


def run(loop: TrainingLoop) -> tuple[Operator, dict[str, Any]]:
    """Fit ``loop.task`` with NumPyro SVI.

    Returns:
        A pair ``(predictive_op, backend_info)``. ``predictive_op`` is a
        `bayes.NumpyroPredictiveOp` over the fitted guide;
        ``backend_info`` carries the keys ``backend``, ``jax_version``,
        ``numpyro_version``, ``devices``, ``total_seconds``,
        ``final_step``, ``final_elbo``, ``final_metrics``, ``guide``,
        and ``param_shapes``.
    """
    import jax
    import numpy as np

    try:
        import numpyro  # ty: ignore[unresolved-import]
    except ImportError as exc:
        raise ImportError(
            "backend='numpyro-svi' needs the [numpyro] extra. Install with "
            "`pip install pipekit-train[numpyro]`."
        ) from exc
    from numpyro.infer import (  # ty: ignore[unresolved-import]
        SVI,
        Predictive,
        Trace_ELBO,
    )
    from numpyro.infer.autoguide import AutoNormal  # ty: ignore[unresolved-import]

    t0 = time.time()
    task = bayes.validate_task(loop)
    guide = task.guide if task.guide is not None else AutoNormal(task.model)
    elbo = task.loss if task.loss is not None else Trace_ELBO()
    optimizer = bayes.build_optimizer(loop.optimizer_config)

    x_train, y_train = bayes.materialize(loop.dataset)
    rng = jax.random.key(loop.seed)
    rng, init_key = jax.random.split(rng)

    svi = SVI(task.model, guide, optimizer, loss=elbo)
    state = svi.init(init_key, x_train, y_train)

    initial = bayes.carry_state(loop, loop.model_op, 0, {})
    bayes.dispatch(loop.callbacks, "on_train_begin", loop, initial)

    last_loss = float("nan")
    step = 0
    # try/finally: a raising callback must not leak the writer's file
    # handle (dropping its buffered lines).
    try:
        for step in range(1, loop.max_steps + 1):
            state, loss = svi.update(state, x_train, y_train)
            last_loss = float(loss)
            metrics = {"elbo": -last_loss, "loss": last_loss}

            if loop.metric_writer is not None and step % loop.log_every_n_steps == 0:
                loop.metric_writer.write(step, metrics)

            carry = bayes.carry_state(loop, loop.model_op, step, metrics)
            bayes.dispatch(loop.callbacks, "on_step_end", loop, carry, metrics)

            if loop.steps_per_epoch and step % loop.steps_per_epoch == 0:
                bayes.dispatch(loop.callbacks, "on_epoch_end", loop, carry, metrics)

            if loop.val_dataset is not None and step % loop.eval_every_n_steps == 0:
                x_val, y_val = bayes.materialize(loop.val_dataset)
                val_elbo = -float(svi.evaluate(state, x_val, y_val))
                bayes.dispatch(
                    loop.callbacks, "on_eval_end", loop, carry, {"val_elbo": val_elbo}
                )

            if bayes.any_should_stop(loop.callbacks):
                break
    finally:
        if loop.metric_writer is not None:
            loop.metric_writer.close()

    params = svi.get_params(state)
    predictive = Predictive(
        task.model, guide=guide, params=params, num_samples=task.predictive_samples
    )
    model_op = bayes.NumpyroPredictiveOp(
        predictive,
        predictive_site=task.predictive_site,
        seed=loop.seed,
        posterior=dict(params),
    )

    final = bayes.carry_state(
        loop, model_op, int(min(loop.max_steps, step)), {"elbo": -last_loss}
    )
    bayes.dispatch(loop.callbacks, "on_train_end", loop, final)

    backend_info = {
        "backend": "numpyro-svi",
        "jax_version": jax.__version__,
        "numpyro_version": numpyro.__version__,
        "devices": [str(d) for d in jax.devices()],
        "total_seconds": time.time() - t0,
        "final_step": int(step),
        "final_elbo": -last_loss,
        "final_metrics": {"elbo": -last_loss},
        "guide": type(guide).__name__,
        "param_shapes": {k: list(np.shape(v)) for k, v in params.items()},
    }
    return model_op, backend_info
