"""Tests for the ensemble-method and iterative-inversion protocols.

`LatentMap`, `ObservationNoise`, `CovarianceLocalizer`,
`EnsembleInflator`, `StepScheduler`, `IterativeProcess` — the seams
ensemble filtering and ensemble Kalman inversion libraries (filterax, …)
satisfy structurally. Here we only check the structural contract.
"""

from __future__ import annotations

from typing import Any

from pipekit_cycle import (
    CovarianceLocalizer,
    EnsembleInflator,
    IterativeProcess,
    LatentMap,
    ObservationNoise,
    ReducedOrderModel,
    StepScheduler,
)


class _DummyLatentMap:
    """Minimal object satisfying `LatentMap` structurally."""

    def encode(self, state: Any) -> Any:
        return state

    def decode(self, coords: Any) -> Any:
        return coords


class _EncodeOnly:
    def encode(self, state: Any) -> Any:
        return state


def test_dummy_latent_map_satisfies_latent_map() -> None:
    assert isinstance(_DummyLatentMap(), LatentMap)


def test_encode_only_is_not_latent_map() -> None:
    assert not isinstance(_EncodeOnly(), LatentMap)


def test_latent_map_is_subset_of_reduced_order_model() -> None:
    """Every `ReducedOrderModel` is structurally also a `LatentMap`."""

    class _ROM:
        def encode(self, state: Any) -> Any:
            return state

        def decode(self, coords: Any) -> Any:
            return coords

        def step(self, coords: Any, dt: float) -> Any:
            return coords

        @property
        def latent_dim(self) -> int:
            return 4

    rom = _ROM()
    assert isinstance(rom, ReducedOrderModel)
    assert isinstance(rom, LatentMap)


class _DummyNoise:
    """Minimal object satisfying `ObservationNoise` structurally."""

    def covariance(self) -> Any:
        return ((1.0, 0.0), (0.0, 1.0))

    def sample(self, key: Any, shape: tuple[int, ...]) -> Any:
        return [0.0] * shape[0]


class _CovarianceOnly:
    def covariance(self) -> Any:
        return ()


def test_dummy_noise_satisfies_observation_noise() -> None:
    assert isinstance(_DummyNoise(), ObservationNoise)


def test_covariance_only_is_not_observation_noise() -> None:
    assert not isinstance(_CovarianceOnly(), ObservationNoise)


class _DummyLocalizer:
    """Minimal object satisfying `CovarianceLocalizer` structurally."""

    def __call__(self, covariance: Any, coords: Any) -> Any:
        return covariance


def test_dummy_localizer_satisfies_covariance_localizer() -> None:
    assert isinstance(_DummyLocalizer(), CovarianceLocalizer)


def test_non_callable_is_not_covariance_localizer() -> None:
    assert not isinstance(object(), CovarianceLocalizer)


class _DummyInflator:
    """Minimal object satisfying `EnsembleInflator` structurally."""

    def __call__(
        self,
        particles: Any,
        forecast_particles: Any = None,
        **kwargs: Any,
    ) -> Any:
        return particles


def test_dummy_inflator_satisfies_ensemble_inflator() -> None:
    assert isinstance(_DummyInflator(), EnsembleInflator)


class _DummyScheduler:
    """Minimal object satisfying `StepScheduler` structurally."""

    def get_dt(self, state: Any) -> float:
        return 0.5


class _NoGetDt:
    def dt(self, state: Any) -> float:
        return 0.5


def test_dummy_scheduler_satisfies_step_scheduler() -> None:
    assert isinstance(_DummyScheduler(), StepScheduler)


def test_wrong_method_name_is_not_step_scheduler() -> None:
    assert not isinstance(_NoGetDt(), StepScheduler)


class _DummyProcess:
    """Minimal object satisfying `IterativeProcess` structurally."""

    def init(self, particles: Any, obs: Any, noise_cov: Any) -> Any:
        return particles

    def update(self, state: Any, forward_evals: Any, **kwargs: Any) -> Any:
        return state


class _MissingUpdate:
    def init(self, particles: Any, obs: Any, noise_cov: Any) -> Any:
        return particles


def test_dummy_process_satisfies_iterative_process() -> None:
    assert isinstance(_DummyProcess(), IterativeProcess)


def test_incomplete_process_is_not_iterative_process() -> None:
    assert not isinstance(_MissingUpdate(), IterativeProcess)
