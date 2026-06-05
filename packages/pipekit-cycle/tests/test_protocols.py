"""Tests for the runtime-checkable `pipekit_cycle.protocols`.

Focused on `ReducedBasis` (the protocol added for variational control
vectors). The concrete bases that satisfy it live in algorithm libraries
(vardax) — here we only check the structural contract.
"""

from __future__ import annotations

from typing import Any

from pipekit_cycle import ReducedBasis


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
