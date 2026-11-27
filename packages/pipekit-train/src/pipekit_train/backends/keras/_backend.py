"""Import Keras on the JAX backend, the only one this backend runs on (ADR D19)."""

from __future__ import annotations

import os
import sys
from typing import Any


def import_keras() -> Any:
    """Import Keras, choosing the JAX backend when Keras is not imported yet.

    Keras reads ``KERAS_BACKEND`` once, at its first import, and defaults to
    TensorFlow, so without this a machine without TensorFlow cannot import it.
    An explicit ``KERAS_BACKEND`` is left alone.

    Returns:
        The ``keras`` module.

    Raises:
        ImportError: If Keras is not installed.
        RuntimeError: If Keras runs on a backend other than JAX.
    """
    if "keras" not in sys.modules:
        os.environ.setdefault("KERAS_BACKEND", "jax")
    try:
        import keras  # ty: ignore[unresolved-import]
    except ImportError as exc:
        raise ImportError(
            "backend='keras' needs the [keras] extra. Install with "
            "`pip install pipekit-train[keras]`."
        ) from exc
    backend = keras.config.backend()
    if backend != "jax":
        raise RuntimeError(
            f"Keras runs on the {backend!r} backend, and pipekit-train's Keras "
            "backend runs on JAX only. Set KERAS_BACKEND=jax before Keras is "
            "first imported."
        )
    return keras
