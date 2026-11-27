"""The Keras backend's area without Keras installed.

Importing ``pipekit_train.backends.keras`` must not import Keras, so that a CLI
can branch on what is installed; using it without Keras must name the extra.
The backend itself is tested in ``tests/adapters/test_keras_run.py``.
"""

from __future__ import annotations

import subprocess
import sys


def test_using_the_keras_backend_without_keras_names_the_extra() -> None:
    code = (
        "import sys; sys.modules['keras'] = None\n"
        "from pipekit_train.backends import keras as area\n"
        "try:\n    area.import_keras()\nexcept ImportError as e:\n    print(e)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert "pipekit-train[keras]" in out.stdout
