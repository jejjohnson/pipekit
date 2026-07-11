"""Tests for the runtime-checkable reduced-order `pipekit_cycle.protocols`.

`ReducedBasis`, `TangentLinearModel`, `ErrorSubspace` — the variational /
reduced-rank seams. The concrete implementations live in algorithm
libraries (vardax, filterx); here we only check the structural contract.
"""

from __future__ import annotations

from typing import Any

from pipekit_cycle import (
    ErrorSubspace,
    ReducedBasis,
    ReducedOrderModel,
    TangentLinearModel,
)


class _DummyBasis:
    """Minimal object satisfying `ReducedBasis` structurally."""

    def operg(self, t: float, X: Any, state: Any = None) -> Any:
        return X

    def prior_inv(self, X: Any) -> Any:
        return X

    @property
    def nbasis(self) -> int:
        return 3


class _MissingPriorInv:
    def operg(self, t: float, X: Any, state: Any = None) -> Any:
        return X

    @property
    def nbasis(self) -> int:
        return 3


def test_dummy_basis_satisfies_reduced_basis() -> None:
    assert isinstance(_DummyBasis(), ReducedBasis)


def test_incomplete_basis_is_not_reduced_basis() -> None:
    assert not isinstance(_MissingPriorInv(), ReducedBasis)


class _DummyTLM:
    """Minimal object satisfying `TangentLinearModel` structurally."""

    def tangent(self, state: Any, dx: Any, dt: float) -> Any:
        return dx

    def adjoint(self, state: Any, dz: Any, dt: float) -> Any:
        return dz


class _MissingAdjoint:
    def tangent(self, state: Any, dx: Any, dt: float) -> Any:
        return dx


def test_dummy_tlm_satisfies_tangent_linear_model() -> None:
    assert isinstance(_DummyTLM(), TangentLinearModel)


def test_incomplete_tlm_is_not_tangent_linear_model() -> None:
    assert not isinstance(_MissingAdjoint(), TangentLinearModel)


class _DummySubspace:
    """Minimal object satisfying `ErrorSubspace` structurally."""

    def propagate(self, model: Any, state: Any, dt: float) -> _DummySubspace:
        return self

    @property
    def modes(self) -> Any:
        return ()

    @property
    def rank(self) -> int:
        return 2


class _MissingRank:
    def propagate(self, model: Any, state: Any, dt: float) -> Any:
        return self

    @property
    def modes(self) -> Any:
        return ()


def test_dummy_subspace_satisfies_error_subspace() -> None:
    assert isinstance(_DummySubspace(), ErrorSubspace)


def test_incomplete_subspace_is_not_error_subspace() -> None:
    assert not isinstance(_MissingRank(), ErrorSubspace)


class _DummyROM:
    """Minimal object satisfying `ReducedOrderModel` structurally."""

    def encode(self, state: Any) -> Any:
        return state

    def decode(self, coords: Any) -> Any:
        return coords

    def step(self, coords: Any, dt: float) -> Any:
        return coords

    @property
    def latent_dim(self) -> int:
        return 4


class _MissingDecode:
    def encode(self, state: Any) -> Any:
        return state

    def step(self, coords: Any, dt: float) -> Any:
        return coords

    @property
    def latent_dim(self) -> int:
        return 4


def test_dummy_rom_satisfies_reduced_order_model() -> None:
    assert isinstance(_DummyROM(), ReducedOrderModel)


def test_incomplete_rom_is_not_reduced_order_model() -> None:
    assert not isinstance(_MissingDecode(), ReducedOrderModel)


class _FullAnalysis:
    def __call__(self, forecast, obs, *, obs_op, obs_err_cov):
        return forecast


class _NotCallableAnalysis:
    """No __call__ — must fail the AnalysisStep check."""


def test_full_analysis_satisfies_analysis_step() -> None:
    from pipekit_cycle import AnalysisStep

    assert isinstance(_FullAnalysis(), AnalysisStep)


def test_non_callable_is_not_analysis_step() -> None:
    from pipekit_cycle import AnalysisStep

    assert not isinstance(_NotCallableAnalysis(), AnalysisStep)
