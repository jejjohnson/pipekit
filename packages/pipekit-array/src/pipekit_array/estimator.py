"""`Estimator`: one scikit-learn wrapper for every named carrier (geostack OP-O3).

A scikit-learn-style estimator learns from a ``(n_samples, n_features)``
table. `Estimator` chooses the samples by dim name, flattens the carrier
with `pipekit_array.to_rows`, fits or applies the estimator, and rebuilds a
carrier like the input with `pipekit_array.from_rows`:

    cube (time, band, y, x) ── Estimator(PCA(3), sample_dims=("time", "y", "x"),
                                         feature_dim="component") ──>
        (time, component, y, x)

It satisfies `pipekit.FittableTransformer`: ``fit`` returns the operator
with ``estimator_`` and ``layout_`` set, and ``transform`` (and
``predict``, ``predict_proba``, ``inverse_transform``) only read them. Fitting
and applying run on the host: scikit-learn computes on NumPy, so a dask
input is computed and a JAX input converted, eagerly; inside a JAX trace
``fit`` raises.

**NaN policies.** ``nan_policy`` decides what a non-finite entry costs:

- ``"propagate"`` (default): the rows go to the estimator as they are;
- ``"raise"``: any non-finite entry raises, naming the policy;
- ``"mask_samples"``: rows with a non-finite entry are left out of fitting
  and applying, and come back as ``NaN`` in the output;
- ``"mask_features"``: columns non-finite in every fitted row are dropped
  at fit time and at transform time, and come back as ``NaN`` from
  ``inverse_transform``;
- ``"mask"``: ``"mask_features"`` then ``"mask_samples"`` on the columns
  kept, for a land-masked field that also has gaps.

The scikit-learn import is lazy: install the ``[sklearn]`` extra.
"""

from __future__ import annotations

import dataclasses
from typing import Any, ClassVar, Literal, Self

import numpy as np
from pipekit import Operator, backend_of

from pipekit_array.dims import adapter_of
from pipekit_array.rows import RowLayout, from_rows, to_rows


__all__ = ["Estimator", "NanPolicy"]

NanPolicy = Literal["propagate", "raise", "mask_samples", "mask_features", "mask"]
_POLICIES: tuple[str, ...] = (
    "propagate",
    "raise",
    "mask_samples",
    "mask_features",
    "mask",
)
_METHODS: tuple[str, ...] = ("transform", "predict", "predict_proba")


class Estimator(Operator):
    """A scikit-learn-style estimator fitted and applied on a named carrier.

    Args:
        estimator: Any object with scikit-learn's ``fit`` and the methods
            you call (``transform``, ``predict``, …). It is cloned on
            ``fit``; the original is never fitted.
        sample_dims: The dims whose entries are samples; every other dim
            is a feature.
        feature_dim: The name of the output dim when the estimator's output
            columns are not the input's features (PCA components, class
            probabilities). Left ``None``, outputs must have the input's
            feature count.
        nan_policy: ``"propagate"``, ``"raise"``, ``"mask_samples"``,
            ``"mask_features"`` or ``"mask"`` (module docstring).
        method: What calling the operator runs once fitted:
            ``"transform"``, ``"predict"`` or ``"predict_proba"``.

    Raises:
        ValueError: An unknown ``nan_policy`` or ``method``, or no sample
            dims.

    Example:
        >>> import numpy as np
        >>> from sklearn.decomposition import PCA
        >>> from pipekit_array import DimArray
        >>> rng = np.random.default_rng(0)
        >>> cube = DimArray(rng.normal(size=(5, 4, 3)), ("band", "y", "x"))
        >>> pca = Estimator(PCA(2), sample_dims=("y", "x"), feature_dim="component")
        >>> pca.fit(cube).transform(cube).dims
        ('component', 'y', 'x')
    """

    forbid_in_yaml: ClassVar[bool] = True
    backends: ClassVar[tuple[str, ...]] = ("numpy", "dask")
    __config_exclude__: ClassVar[tuple[str, ...]] = (
        "estimator_",
        "layout_",
        "kept_features_",
    )

    def __init__(
        self,
        estimator: Any,
        *,
        sample_dims: tuple[str, ...] | list[str],
        feature_dim: str | None = None,
        nan_policy: NanPolicy = "propagate",
        method: str = "transform",
    ) -> None:
        if nan_policy not in _POLICIES:
            raise ValueError(
                f"nan_policy must be one of {_POLICIES}; got {nan_policy!r}"
            )
        if method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}; got {method!r}")
        if not tuple(sample_dims):
            raise ValueError("Estimator needs at least one sample dim.")
        self.estimator = estimator
        self.sample_dims = tuple(sample_dims)
        self.feature_dim = feature_dim
        self.nan_policy = nan_policy
        self.method = method
        self.estimator_: Any = None
        self.layout_: RowLayout | None = None
        self.kept_features_: np.ndarray | None = None

    # ------------------------------------------------------------------ fit

    def fit(self, x: Any, y: Any = None) -> Self:
        """Learn the estimator from ``x`` (and ``y``, for a supervised one).

        Args:
            x: A named carrier with the sample dims.
            y: Targets: a carrier whose dims are the sample dims (plus at
                most one target dim), or ``None``.

        Returns:
            This operator, fitted.

        Raises:
            TypeError: ``x`` is a JAX tracer (fit eagerly, outside ``jit``).
            ValueError: ``nan_policy="raise"`` and ``x`` has a non-finite
                entry.
        """
        from sklearn.base import clone

        rows, layout = to_rows(x, self.sample_dims)
        table = _host(rows, "fit")
        self._check(table)
        if self.nan_policy in ("mask_features", "mask"):
            self.kept_features_ = np.asarray(np.isfinite(table).any(axis=0))
            table = table[:, self.kept_features_]
        keep = self._rows(table)
        targets = None if y is None else _targets(y, self.sample_dims)
        estimator = clone(self.estimator)
        if targets is None:
            estimator.fit(table[keep])
        else:
            estimator.fit(table[keep], targets[keep])
        self.estimator_ = estimator
        # the layout without its template: the training data is not kept alive
        self.layout_ = dataclasses.replace(layout, template=None)
        return self

    # ------------------------------------------------------------- apply

    def transform(self, x: Any) -> Any:
        """The estimator's ``transform`` of ``x``, as a carrier like ``x``."""
        return self._apply_method("transform", x)

    def predict(self, x: Any) -> Any:
        """One prediction per sample: a carrier with the sample dims only."""
        return self._apply_method("predict", x)

    def predict_proba(self, x: Any) -> Any:
        """Class probabilities per sample, along ``feature_dim`` (``"class"``)."""
        return self._apply_method("predict_proba", x)

    def inverse_transform(self, z: Any) -> Any:
        """Map ``transform``'s output back to the fitted features.

        Args:
            z: A carrier with the sample dims and the output dim (``feature_dim``).

        Returns:
            A carrier with the fitted feature dims and sizes in place of the
            output dim; dropped features (``"mask_features"``) are ``NaN``.
        """
        layout = self._fitted()
        rows, z_layout = to_rows(z, self.sample_dims)
        table = _host(rows, "inverse_transform")
        keep = _finite_rows(table) if self._masks_samples else _all(table)
        out_cols = self._restore_features(
            self.estimator_.inverse_transform(table[keep])
        )
        full = _fill_rows(out_cols, keep, layout.n_features)
        target = RowLayout(
            dims=layout.dims,
            sizes=tuple(
                _size(z_layout, d) if d in layout.sample_dims else _size(layout, d)
                for d in layout.dims
            ),
            sample_dims=layout.sample_dims,
            feature_dims=layout.feature_dims,
            template=z,
        )
        return from_rows(full, target)

    def _apply(self, x: Any) -> Any:
        return getattr(self, self.method)(x)

    # ----------------------------------------------------------- helpers

    @property
    def _masks_samples(self) -> bool:
        return self.nan_policy in ("mask_samples", "mask")

    def _fitted(self) -> RowLayout:
        if self.estimator_ is None or self.layout_ is None:
            raise ValueError("Estimator is not fitted: call fit(x) first.")
        return self.layout_

    def _apply_method(self, name: str, x: Any) -> Any:
        layout = self._fitted()
        rows, x_layout = to_rows(x, self.sample_dims)
        _require_same_features(x_layout, layout)
        table = _host(rows, name)
        self._check(table)
        if self.kept_features_ is not None:
            table = table[:, self.kept_features_]
        keep = self._rows(table)
        out = np.asarray(getattr(self.estimator_, name)(table[keep]))
        if out.ndim == 1:
            return from_rows(_fill_rows(out, keep, None), x_layout)
        full = _fill_rows(out, keep, out.shape[1])
        if self.feature_dim is None and out.shape[1] == x_layout.n_features:
            return from_rows(full, x_layout)
        dim = self.feature_dim or ("class" if name == "predict_proba" else None)
        if dim is None:
            raise ValueError(
                f"Estimator.{name} returned {out.shape[1]} columns for "
                f"{x_layout.n_features} features; pass feature_dim= to name them."
            )
        return from_rows(full, x_layout, feature_dim=dim)

    def _check(self, table: np.ndarray) -> None:
        """Raise on a non-finite entry under ``nan_policy="raise"``."""
        if self.nan_policy == "raise" and not np.isfinite(table).all():
            raise ValueError(
                "Estimator: the input has non-finite entries (nan_policy='raise'); "
                "use 'mask_samples', 'mask_features' or 'mask' to leave them out."
            )

    def _rows(self, table: np.ndarray) -> np.ndarray:
        """The rows to use: the finite ones when the policy masks samples."""
        return _finite_rows(table) if self._masks_samples else _all(table)

    def _restore_features(self, cols: np.ndarray) -> np.ndarray:
        """Re-insert features dropped at fit time as ``NaN`` columns."""
        if self.kept_features_ is None:
            return cols
        out = np.full((cols.shape[0], self.kept_features_.size), np.nan)
        out[:, self.kept_features_] = cols
        return out


def _host(rows: Any, what: str) -> np.ndarray:
    """``rows`` as a NumPy array; a JAX tracer raises, naming eager use."""
    if backend_of(rows) == "jax":
        from jax.core import Tracer

        if isinstance(rows, Tracer):
            raise TypeError(
                f"Estimator.{what} runs scikit-learn on the host and cannot run "
                "inside a JAX trace; call it eagerly, outside jax.jit."
            )
    return np.asarray(rows)


def _targets(y: Any, sample_dims: tuple[str, ...]) -> np.ndarray:
    """``y`` as one value (or one row) per sample, in the samples' order."""
    adapter_of(y, "Estimator.fit")
    rows, layout = to_rows(y, sample_dims)
    table = _host(rows, "fit")
    return table[:, 0] if layout.n_features == 1 else table


def _require_same_features(got: RowLayout, fitted: RowLayout) -> None:
    sizes = tuple(_size(got, d) for d in got.feature_dims)
    want = tuple(_size(fitted, d) for d in fitted.feature_dims)
    if got.feature_dims != fitted.feature_dims or sizes != want:
        fitted_sizes = dict(zip(fitted.feature_dims, want, strict=True))
        got_sizes = dict(zip(got.feature_dims, sizes, strict=True))
        raise ValueError(
            f"Estimator was fitted on features {fitted_sizes}, got {got_sizes}."
        )


def _fill_rows(values: np.ndarray, keep: np.ndarray, n_cols: int | None) -> np.ndarray:
    """``values`` for the kept rows, ``NaN`` in the others."""
    if keep.all():
        return values
    shape = (keep.size,) if n_cols is None else (keep.size, n_cols)
    dtype = np.result_type(values.dtype, np.float32)
    out = np.full(shape, np.nan, dtype=dtype)
    out[keep] = values
    return out


def _finite_rows(table: np.ndarray) -> np.ndarray:
    return np.asarray(np.isfinite(table).all(axis=1))


def _all(table: np.ndarray) -> np.ndarray:
    return np.ones(table.shape[0], dtype=bool)


def _size(layout: RowLayout, dim: str) -> int:
    return layout.sizes[layout.dims.index(dim)]
