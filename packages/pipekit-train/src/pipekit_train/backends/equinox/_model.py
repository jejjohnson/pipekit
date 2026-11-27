"""The wrapper that makes a trained ``eqx.Module`` a pipekit operator.

`EquinoxModelOp` and the unwrap and rewrap of an Equinox-backed operator live in
``pipekit_train.backends._equinox_ops``, which the Keras backend shares.
"""

from __future__ import annotations

from pipekit_train.backends._equinox_ops import EquinoxModelOp
from pipekit_train.backends._equinox_ops import rewrap_module as _wrap_trained_model


__all__ = ["EquinoxModelOp", "_wrap_trained_model"]
