"""pipekit-train — training pipelines on top of pipekit.

v0.0 ships the carrier-agnostic Protocols (`Loss`, `Callback`,
`MetricWriter`), the default `JSONLWriter`, and stub adapter modules
(Lightning, Keras) that raise ``NotImplementedError`` per the v0.1
design. The full surface (datasets, training loop, Equinox adapter)
is documented under ``docs/design/`` and lands in v0.1.

See ``docs/design/`` for the design.
"""

from pipekit_train.callbacks import Callback
from pipekit_train.loss import Loss
from pipekit_train.writer import JSONLWriter, MetricWriter


__all__ = [
    "Callback",
    "JSONLWriter",
    "Loss",
    "MetricWriter",
]

__version__ = "0.0.1"  # x-release-please-version
