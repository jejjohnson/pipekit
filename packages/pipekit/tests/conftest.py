"""Shared fixtures for pipekit Group A tests."""

from __future__ import annotations

import pytest
from pipekit import Operator


class Scale(Operator):
    """Multiply input by a scalar factor — canonical tiny operator."""

    def __init__(self, factor: float) -> None:
        self.factor = factor

    def _apply(self, x: float) -> float:
        return x * self.factor


class AddConst(Operator):
    """Add a constant to the input."""

    def __init__(self, value: float) -> None:
        self.value = value

    def _apply(self, x: float) -> float:
        return x + self.value


class Sink(Operator):
    """A terminal operator — returns ``None`` and is only valid as the last step."""

    _terminal = True

    def __init__(self, label: str = "sink") -> None:
        self.label = label

    def _apply(self, x: float) -> None:
        return None


class TwoInput(Operator):
    """Take two carriers and return their sum — for Graph testing."""

    def _apply(self, a: float, b: float) -> float:
        return a + b


@pytest.fixture
def Scale_cls() -> type[Operator]:
    return Scale


@pytest.fixture
def AddConst_cls() -> type[Operator]:
    return AddConst


@pytest.fixture
def Sink_cls() -> type[Operator]:
    return Sink


@pytest.fixture
def TwoInput_cls() -> type[Operator]:
    return TwoInput
