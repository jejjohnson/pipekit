"""Tests for `pipekit_train.loss` concrete operators."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from pipekit_train import KL, MSE, NLL, Composite, Loss


# --- MSE ----------------------------------------------------------------


def test_mse_mean_default():
    loss_op = MSE()
    pred = np.array([1.0, 2.0, 3.0])
    target = np.array([1.0, 2.0, 2.0])
    loss, aux = loss_op(pred, target)
    assert loss == pytest.approx(1.0 / 3.0)
    assert aux == {"mse": loss}


def test_mse_sum_reduction():
    loss_op = MSE(reduction="sum")
    pred = np.array([1.0, 2.0, 3.0])
    target = np.array([0.0, 0.0, 0.0])
    loss, _ = loss_op(pred, target)
    assert loss == pytest.approx(14.0)


def test_mse_none_reduction_preserves_shape():
    loss_op = MSE(reduction="none")
    pred = np.array([[1.0, 2.0], [3.0, 4.0]])
    target = np.zeros_like(pred)
    loss, _ = loss_op(pred, target)
    np.testing.assert_array_almost_equal(loss, pred * pred)


def test_mse_rejects_unknown_reduction():
    with pytest.raises(ValueError, match="reduction"):
        MSE(reduction="invalid")


def test_mse_satisfies_loss_protocol():
    assert isinstance(MSE(), Loss)


def test_mse_round_trips_config():
    loss_op = MSE(reduction="sum")
    assert loss_op.get_config() == {"reduction": "sum"}


# --- NLL ----------------------------------------------------------------


class _MockDistribution:
    """A minimal distribution-like object exposing .log_prob."""

    def __init__(self, log_probs: np.ndarray) -> None:
        self._log_probs = log_probs

    def log_prob(self, target: Any) -> np.ndarray:
        del target
        return self._log_probs


def test_nll_calls_log_prob_and_negates():
    pred = _MockDistribution(np.array([-1.0, -2.0, -3.0]))
    loss, aux = NLL()(pred, target=np.array([0, 1, 2]))
    assert loss == pytest.approx(2.0)
    assert aux == {"nll": loss}


def test_nll_sum_reduction():
    pred = _MockDistribution(np.array([-1.0, -2.0, -3.0]))
    loss, _ = NLL(reduction="sum")(pred, target=None)
    assert loss == pytest.approx(6.0)


def test_nll_rejects_non_distribution_predicted():
    with pytest.raises(TypeError, match="log_prob"):
        NLL()(predicted=np.array([1.0, 2.0]), target=np.array([1.0, 2.0]))


def test_nll_rejects_unknown_reduction():
    with pytest.raises(ValueError, match="reduction"):
        NLL(reduction="invalid")


def test_nll_satisfies_loss_protocol():
    assert isinstance(NLL(), Loss)


# --- KL -----------------------------------------------------------------


class _MockDistributionWithKL:
    def __init__(self, kl_to_other: float) -> None:
        self._kl = kl_to_other

    def kl_divergence(self, other: Any) -> np.ndarray:
        del other
        return np.array([self._kl])


def test_kl_analytic_path():
    pred = _MockDistributionWithKL(kl_to_other=0.5)
    loss, aux = KL()(pred, target=None)
    assert loss == pytest.approx(0.5)
    assert aux == {"kl": loss}


def test_kl_sample_path_not_implemented_yet():
    pred = _MockDistribution(np.zeros(3))  # has log_prob but no kl_divergence
    with pytest.raises(TypeError, match="kl_divergence"):
        KL()(pred, target=None)


def test_kl_satisfies_loss_protocol():
    assert isinstance(KL(), Loss)


# --- Composite ----------------------------------------------------------


def test_composite_weighted_sum():
    pred = np.array([1.0])
    target = np.array([0.0])
    composite = Composite(
        components=(
            (1.0, MSE()),
            (0.5, MSE()),
        )
    )
    total, aux = composite(pred, target)
    # MSE on these inputs = 1.0; weighted: 1*1 + 0.5*1 = 1.5
    assert total == pytest.approx(1.5)
    assert aux["total"] == pytest.approx(1.5)
    assert "MSE_0" in aux and "MSE_1" in aux


def test_composite_emits_per_component_aux_metrics():
    pred = np.array([2.0])
    target = np.array([0.0])
    composite = Composite(components=((0.25, MSE()),))
    total, aux = composite(pred, target)
    # MSE = 4.0; weighted = 1.0
    assert total == pytest.approx(1.0)
    assert aux["MSE_0"] == pytest.approx(4.0)


def test_composite_rejects_empty_components():
    with pytest.raises(ValueError, match="at least one"):
        Composite(components=())


def test_composite_satisfies_loss_protocol():
    assert isinstance(Composite(components=((1.0, MSE()),)), Loss)


def test_composite_get_config_surfaces_components():
    composite = Composite(components=((1.0, MSE()), (0.5, MSE())))
    config = composite.get_config()
    assert config == {
        "components": [
            {"weight": 1.0, "loss_class": "MSE"},
            {"weight": 0.5, "loss_class": "MSE"},
        ]
    }


# --- Sanity: losses are pipekit Operators (compose with the rest) -------


def test_losses_are_operators():
    from pipekit import Operator

    assert isinstance(MSE(), Operator)
    assert isinstance(NLL(), Operator)
    assert isinstance(KL(), Operator)
    assert isinstance(Composite(components=((1.0, MSE()),)), Operator)


# --- Pi constant used nowhere; keep math import live for future use -----


_ = math.pi
