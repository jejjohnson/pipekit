"""NumPyro MCMC adapter — ``backend="numpyro-mcmc"`` (NUTS sampling).

A single-call backend: ``mcmc.run`` is one blocking warmup+sampling pass over
the **full** dataset, so the per-step loop fields (``max_steps``,
``optimizer_config``, per-step eval) do not apply — only ``on_train_begin`` /
``on_train_end`` callbacks fire. The artifact is a `NumpyroPredictiveOp`
around ``Predictive(model, posterior_samples)``. See ADR D13 and
``docs/design/numpyro_adapter.md``.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from pipekit_train.adapters import _bayes


if TYPE_CHECKING:
    from pipekit import Operator

    from pipekit_train.loop import TrainingLoop


def run(loop: TrainingLoop) -> tuple[Operator, dict[str, Any]]:
    """Fit ``loop.task`` with NumPyro NUTS; return ``(predictive_op, info)``."""
    import jax
    import numpy as np
    import numpyro  # ty: ignore[unresolved-import]
    from numpyro.infer import MCMC, NUTS, Predictive  # ty: ignore[unresolved-import]

    t0 = time.time()
    task = _bayes.validate_task(loop)
    x_train, y_train = _bayes.materialize(loop.dataset)
    rng = jax.random.key(loop.seed)

    initial = _bayes.carry_state(loop, loop.model_op, 0, {})
    _bayes.dispatch(loop.callbacks, "on_train_begin", loop, initial)

    mcmc = MCMC(
        NUTS(task.model),
        num_warmup=task.num_warmup,
        num_samples=task.num_samples,
        num_chains=task.num_chains,
        progress_bar=False,
    )
    # try/finally: close the writer even if sampling raises, for symmetry
    # with the per-step backends.
    try:
        mcmc.run(rng, x_train, y_train, extra_fields=("diverging",))
    finally:
        if loop.metric_writer is not None:
            loop.metric_writer.close()
    samples = mcmc.get_samples()
    extra = mcmc.get_extra_fields()
    num_divergences = int(np.sum(np.asarray(extra["diverging"]))) if extra else 0

    predictive = Predictive(task.model, posterior_samples=samples)
    model_op = _bayes.NumpyroPredictiveOp(
        predictive,
        predictive_site=task.predictive_site,
        seed=loop.seed,
        posterior=dict(samples),
    )

    final_step = task.num_samples * task.num_chains
    final = _bayes.carry_state(loop, model_op, final_step, {})
    _bayes.dispatch(loop.callbacks, "on_train_end", loop, final)

    backend_info = {
        "backend": "numpyro-mcmc",
        "jax_version": jax.__version__,
        "numpyro_version": numpyro.__version__,
        "devices": [str(d) for d in jax.devices()],
        "total_seconds": time.time() - t0,
        "final_step": final_step,
        "num_warmup": task.num_warmup,
        "num_samples": task.num_samples,
        "num_chains": task.num_chains,
        "num_divergences": num_divergences,
        "sites": sorted(samples),
        "final_metrics": {},
    }
    return model_op, backend_info
