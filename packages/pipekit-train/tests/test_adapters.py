"""Tests for the Lightning and Keras adapter stubs.

The v0.1 release ships these two adapter modules as
NotImplementedError stubs per the design (see
``docs/design/api/adapters.md``). Both must be importable without
their underlying tool installed, and ``run`` must raise a clear
message. (The Equinox adapter is the v0.1 reference implementation
and is tested in ``tests/adapters/test_equinox.py``.)
"""

from __future__ import annotations

import pytest
from pipekit_train.adapters import (
    keras as keras_adapter,
    lightning as lightning_adapter,
)


def test_lightning_adapter_run_raises_not_implemented():
    with pytest.raises(NotImplementedError, match=r"v0\.2"):
        lightning_adapter.run(loop=None)


def test_keras_adapter_run_raises_not_implemented():
    with pytest.raises(NotImplementedError, match=r"v0\.3"):
        keras_adapter.run(loop=None)


def test_lightning_message_points_at_design_doc():
    """Sanity check — the error message should mention the design path."""
    with pytest.raises(NotImplementedError, match=r"docs/design"):
        lightning_adapter.run(loop=None)


def test_keras_message_points_at_design_doc():
    with pytest.raises(NotImplementedError, match=r"docs/design"):
        keras_adapter.run(loop=None)
