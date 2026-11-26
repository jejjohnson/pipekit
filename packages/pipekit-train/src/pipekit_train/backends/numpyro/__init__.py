"""The NumPyro backends: SVI and NUTS (``[numpyro]``).

`run_svi(loop)` fits a guide by stochastic variational inference and `run_mcmc(loop)`
samples with NUTS. Both take a `NumpyroTask` as ``loop.task`` and return the posterior
predictive as a `NumpyroPredictiveOp`.
"""

from __future__ import annotations

from pipekit_ops.core import record_moves, record_under

from pipekit_train.backends._bayes import NumpyroPredictiveOp, NumpyroTask
from pipekit_train.backends.numpyro._mcmc import run as run_mcmc
from pipekit_train.backends.numpyro._svi import run as run_svi


__all__ = [
    "NumpyroPredictiveOp",
    "NumpyroTask",
    "run_mcmc",
    "run_svi",
]


record_under(__name__, NumpyroPredictiveOp)
record_moves(__name__, "pipekit_train.adapters.bayes")
