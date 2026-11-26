"""The BlackJAX backend: NUTS from BlackJAX on NumPyro models (``[blackjax]``).

`run(loop)` takes a `BlackjaxTask` as ``loop.task`` and returns the posterior predictive
as a `BlackjaxPredictiveOp`; the NumPyro model, its constrain map and the predictive
come through the seam it shares with `pipekit_train.backends.numpyro`.
"""

from __future__ import annotations

from pipekit_ops.core import record_moves, record_under

from pipekit_train.backends.blackjax._run import BlackjaxPredictiveOp, BlackjaxTask, run


__all__ = [
    "BlackjaxPredictiveOp",
    "BlackjaxTask",
    "run",
]


record_under(__name__, BlackjaxPredictiveOp)
record_moves(__name__, "pipekit_train.adapters.blackjax")
