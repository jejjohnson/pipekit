"""Tests for `pipekit_experiment.adapters.hydra`.

When hydra-core / hydra-zen are absent the adapter raises a clean
ImportError at method invocation. Functional tests are gated with
``pytest.importorskip``.
"""

from __future__ import annotations

import pytest
from pipekit import Const, Identity, Operator
from pipekit_experiment.adapters.hydra import HydraConfigLoader


def test_import_error_when_hydra_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "hydra.utils":
            raise ImportError("simulated absence")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="hydra-core"):
        HydraConfigLoader.from_hydra_cfg({"_target_": "anything"})


def test_to_hydra_cfg_import_error_when_hydra_zen_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "hydra_zen":
            raise ImportError("simulated absence")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="hydra-zen"):
        HydraConfigLoader.to_hydra_cfg(Identity())


def test_to_hydra_cfg_rejects_non_operator():
    with pytest.raises(TypeError):
        HydraConfigLoader.to_hydra_cfg(42)  # type: ignore[arg-type]


def test_from_hydra_cfg_round_trip_with_hydra():
    """End-to-end test when hydra is actually installed."""
    pytest.importorskip("hydra")
    cfg = {
        "_target_": f"{Const.__module__}.{Const.__name__}",
        "value": 7,
    }
    op = HydraConfigLoader.from_hydra_cfg(cfg)
    assert isinstance(op, Const)
    assert op() == 7


def test_from_hydra_cfg_rejects_non_operator_target():
    pytest.importorskip("hydra")
    cfg = {"_target_": "builtins.dict"}
    with pytest.raises(TypeError, match="Operator"):
        HydraConfigLoader.from_hydra_cfg(cfg)


def test_to_yaml_round_trip():
    """to_yaml → from_yaml reconstructs an Operator with the same config."""
    pytest.importorskip("hydra_zen")
    pytest.importorskip("omegaconf")
    op: Operator = Const(99)
    text = HydraConfigLoader.to_yaml(op)
    reconstructed = HydraConfigLoader.from_yaml(text)
    assert isinstance(reconstructed, Const)
    assert reconstructed() == 99
