"""Tests for the Group N structural protocols (`pipekit.protocols`)."""

from __future__ import annotations

from typing import Any

from pipekit import FittableTransformer, Predictor


class _SklearnishModel:
    """Minimal object satisfying `Predictor` structurally."""

    def predict(self, x: Any) -> Any:
        return x


class _CallableOnly:
    def __call__(self, x: Any) -> Any:
        return x


def test_predict_method_satisfies_predictor() -> None:
    assert isinstance(_SklearnishModel(), Predictor)


def test_bare_callable_is_not_predictor() -> None:
    assert not isinstance(_CallableOnly(), Predictor)


class _SklearnishTransformer:
    """Minimal object satisfying `FittableTransformer` structurally."""

    def fit(self, x: Any) -> _SklearnishTransformer:
        return self

    def transform(self, x: Any) -> Any:
        return x


class _FitOnly:
    def fit(self, x: Any) -> Any:
        return self


def test_fit_transform_satisfies_fittable_transformer() -> None:
    assert isinstance(_SklearnishTransformer(), FittableTransformer)


def test_fit_only_is_not_fittable_transformer() -> None:
    assert not isinstance(_FitOnly(), FittableTransformer)
