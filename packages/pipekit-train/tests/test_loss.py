"""Tests for `pipekit_train.loss` concrete operators."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from pipekit import Lambda
from pipekit_train import KL, MSE, NLL, Composite, Loss
from pipekit_train.loss import call_loss


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


def test_kl_without_analytic_path_or_key_says_what_to_pass():
    pred = object()
    with pytest.raises(TypeError, match="PRNG key"):
        KL()(pred, target=None)


def test_kl_rejects_too_few_samples():
    with pytest.raises(ValueError, match="n_samples"):
        KL(n_samples=0)


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


# --- Sample-based KL ------------------------------------------------------


jax = pytest.importorskip("jax")


class _Normal:
    """A Gaussian with distrax's calling convention, ``sample(seed=...)``."""

    def __init__(self, loc: float, scale: float) -> None:
        self.loc = loc
        self.scale = scale

    def sample(self, seed, sample_shape=()):
        draws = jax.random.normal(seed, sample_shape)
        return self.loc + self.scale * draws

    def log_prob(self, x):
        z = (x - self.loc) / self.scale
        return -0.5 * z**2 - jax.numpy.log(self.scale) - 0.9189385332046727


def _closed_form_kl(q: _Normal, p: _Normal) -> float:
    # KL(q ‖ p) = log(s_p / s_q) + (s_q² + (m_q - m_p)²) / (2 s_p²) - 1/2
    ratio = float(jax.numpy.log(p.scale / q.scale))
    numerator = q.scale**2 + (q.loc - p.loc) ** 2
    return ratio + numerator / (2 * p.scale**2) - 0.5


def test_sample_based_kl_matches_the_closed_form():
    q, p = _Normal(0.5, 1.0), _Normal(0.0, 2.0)
    loss, aux = KL(n_samples=200_000)(q, p, key=jax.random.key(0))
    assert float(loss) == pytest.approx(_closed_form_kl(q, p), abs=0.01)
    assert aux == {"kl": loss}


def test_sample_based_kl_of_a_distribution_with_itself_is_zero():
    q = _Normal(0.5, 1.0)
    loss, _ = KL()(q, q, key=jax.random.key(0))
    assert float(loss) == 0.0


def test_sample_based_kl_is_reproducible_and_depends_on_the_key():
    q, p = _Normal(0.5, 1.0), _Normal(0.0, 2.0)
    first, _ = KL()(q, p, key=jax.random.key(0))
    again, _ = KL()(q, p, key=jax.random.key(0))
    other, _ = KL()(q, p, key=jax.random.key(1))
    assert float(first) == float(again)
    assert float(first) != float(other)


def test_analytic_path_ignores_the_key():
    pred = _MockDistributionWithKL(kl_to_other=0.5)
    loss, _ = KL()(pred, target=None, key=jax.random.key(0))
    assert loss == pytest.approx(0.5)


def test_sample_based_kl_works_with_numpyro_distributions():
    dist = pytest.importorskip("numpyro.distributions")
    q, p = dist.Normal(0.5, 1.0), dist.Normal(0.0, 2.0)
    loss, _ = KL(n_samples=200_000)(q, p, key=jax.random.key(0))
    assert float(loss) == pytest.approx(
        _closed_form_kl(_Normal(0.5, 1.0), _Normal(0.0, 2.0)), abs=0.01
    )


def test_composite_gives_each_component_its_own_key():
    seen = []

    def takes_key(predicted, target, key=None):
        seen.append(key)
        return 1.0

    def takes_no_key(predicted, target):
        return 2.0

    total, _ = Composite(
        components=((1.0, takes_key), (1.0, takes_no_key), (1.0, takes_key))
    )(None, None, key=jax.random.key(0))

    assert total == 4.0
    assert len(seen) == 2
    assert not bool(
        jax.numpy.array_equal(
            jax.random.key_data(seen[0]), jax.random.key_data(seen[1])
        )
    )


def test_composite_without_a_key_calls_components_plainly():
    total, _ = Composite(components=((2.0, lambda predicted, target: 3.0),))(None, None)
    assert total == 6.0


def test_call_loss_passes_a_key_only_to_a_loss_that_takes_one():
    key = jax.random.key(0)
    assert call_loss(lambda p, t: "plain", None, None, key) == "plain"
    assert call_loss(lambda p, t, key=None: key is not None, None, None, key)
    assert call_loss(Lambda(lambda p, t: "lambda"), None, None, key) == "lambda"
    assert call_loss(MSE(), np.ones(2), np.ones(2), key)[0] == 0.0
