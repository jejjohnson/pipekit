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

from pipekit_train.adapters import bayes


if TYPE_CHECKING:
    from pipekit import Operator

    from pipekit_train.loop import TrainingLoop


def run(loop: TrainingLoop) -> tuple[Operator, dict[str, Any]]:
    """Fit ``loop.task`` with NumPyro NUTS.

    Returns:
        A pair ``(predictive_op, backend_info)``. ``predictive_op`` is a
        `bayes.NumpyroPredictiveOp` over the posterior samples;
        ``backend_info`` carries the keys ``backend``, ``jax_version``,
        ``numpyro_version``, ``devices``, ``total_seconds``,
        ``final_step``, ``num_warmup``, ``num_samples``, ``num_chains``,
        ``num_divergences``, ``final_metrics``, and ``sites``.
        ``num_divergences`` is ``None`` for a kernel that doesn't report them.
    """
    import jax
    import numpy as np

    try:
        import numpyro  # ty: ignore[unresolved-import]
    except ImportError as exc:
        raise ImportError(
            "backend='numpyro-mcmc' needs the [numpyro] extra. Install with "
            "`pip install pipekit-train[numpyro]`."
        ) from exc
    from numpyro.infer import (  # ty: ignore[unresolved-import]
        HMC,
        MCMC,
        NUTS,
        SA,
        Predictive,
    )

    t0 = time.time()
    task = bayes.validate_task(loop)
    args = bayes.model_args(task, loop.dataset)
    rng = jax.random.key(loop.seed)

    initial = bayes.carry_state(loop, loop.model_op, 0, {})
    bayes.dispatch(loop.callbacks, "on_train_begin", loop, initial)

    kernel_factory = task.kernel_factory or NUTS
    kernel = kernel_factory(task.model, **task.kernel_kwargs)
    # Only these kernels' states have the field; asking any other kernel for it
    # raises, for example BarkerMH.
    extra_fields = ("diverging",) if isinstance(kernel, HMC | SA) else ()
    mcmc = MCMC(
        kernel,
        num_warmup=task.num_warmup,
        num_samples=task.num_samples,
        num_chains=task.num_chains,
        progress_bar=False,
    )
    # try/finally: close the writer even if sampling raises, for symmetry
    # with the per-step backends.
    try:
        mcmc.run(rng, *args, extra_fields=extra_fields, **task.model_kwargs)
    finally:
        if loop.metric_writer is not None:
            loop.metric_writer.close()
    samples = mcmc.get_samples()
    extra = mcmc.get_extra_fields()
    num_divergences = int(np.sum(np.asarray(extra["diverging"]))) if extra else None

    predictive = Predictive(task.model, posterior_samples=samples)
    model_op = bayes.NumpyroPredictiveOp(
        predictive,
        predictive_site=task.predictive_site,
        seed=loop.seed,
        posterior=dict(samples),
        model_kwargs=task.model_kwargs,
    )

    final_step = task.num_samples * task.num_chains
    final = bayes.carry_state(loop, model_op, final_step, {})
    bayes.dispatch(loop.callbacks, "on_train_end", loop, final)

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
