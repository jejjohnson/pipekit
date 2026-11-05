"""Tests for the Keras adapter stub.

The adapter module ships as a NotImplementedError stub per the design (see
``docs/design/api/adapters.md``). It must be importable without Keras
installed, and ``run`` must raise a clear message. (The Equinox adapter is the
reference implementation and is tested in ``tests/adapters/test_equinox.py``.)
"""

from __future__ import annotations

import pytest
from pipekit_train.adapters import keras as keras_adapter


def test_keras_adapter_run_raises_not_implemented():
    with pytest.raises(NotImplementedError, match=r"v0\.3"):
        keras_adapter.run(loop=None)


def test_keras_message_points_at_design_doc():
    with pytest.raises(NotImplementedError, match=r"docs/design"):
        keras_adapter.run(loop=None)
