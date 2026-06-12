"""Adjoint spec vocabulary + SupportsAdjoint protocol conformance."""

from __future__ import annotations

import dataclasses

import pytest
from pipekit_cycle import (
    BacksolveAdjoint,
    DirectAdjoint,
    ImplicitAdjoint,
    RecursiveCheckpointAdjoint,
    SupportsAdjoint,
    TruncatedAdjoint,
)


ALL_SPECS = [
    DirectAdjoint(),
    RecursiveCheckpointAdjoint(),
    RecursiveCheckpointAdjoint(checkpoints=16),
    TruncatedAdjoint(),
    TruncatedAdjoint(k=5),
    ImplicitAdjoint(),
    BacksolveAdjoint(),
]


@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: repr(s))
def test_specs_are_frozen_hashable_config(spec):
    """Specs behave like config values: frozen, hashable, equal by value."""
    assert dataclasses.is_dataclass(spec)
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.k = 99  # type: ignore[misc]
    assert hash(spec) == hash(dataclasses.replace(spec))
    assert spec == dataclasses.replace(spec)


def test_truncated_defaults_to_one_step():
    assert TruncatedAdjoint().k == 1


def test_recursive_checkpoint_default_schedule_is_none():
    assert RecursiveCheckpointAdjoint().checkpoints is None


class _ToyDifferentiable:
    """Plain object satisfying `SupportsAdjoint` structurally."""

    def __init__(self, spec=None):
        self._spec = spec if spec is not None else RecursiveCheckpointAdjoint()

    @property
    def adjoint(self):
        return self._spec

    def with_adjoint(self, spec):
        return _ToyDifferentiable(spec)


def test_supports_adjoint_conformance():
    toy = _ToyDifferentiable()
    assert isinstance(toy, SupportsAdjoint)
    swapped = toy.with_adjoint(TruncatedAdjoint(k=3))
    assert swapped.adjoint == TruncatedAdjoint(k=3)
    assert toy.adjoint == RecursiveCheckpointAdjoint()  # original unchanged


def test_non_conforming_object_rejected():
    assert not isinstance(object(), SupportsAdjoint)
