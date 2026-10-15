"""Names that moved out of pipekit-train point to their new home."""

from __future__ import annotations

import pipekit_train
import pytest


def test_xarray_window_dataset_points_to_geopatcher() -> None:
    with pytest.raises(ImportError, match=r"geopatcher\.integrations\.train"):
        _ = pipekit_train.XarrayWindowDataset


def test_it_is_not_exported() -> None:
    assert "XarrayWindowDataset" not in pipekit_train.__all__
