"""The losses a training run minimises.

`Loss` is the protocol; `MSE`, `NLL`, `KL` and `Composite` are losses as operators;
`call_loss` calls any loss, passing a key to the ones that take it.
"""

from __future__ import annotations

from pipekit_train.losses._losses import KL, MSE, NLL, Composite, Loss, call_loss


__all__ = [
    "KL",
    "MSE",
    "NLL",
    "Composite",
    "Loss",
    "call_loss",
]
