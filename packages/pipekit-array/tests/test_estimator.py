"""`Estimator`: one scikit-learn wrapper on any named carrier (#130, OP-O3)."""

from __future__ import annotations

import numpy as np
import pytest
from pipekit_array import DimArray, Estimator, to_rows


sklearn = pytest.importorskip("sklearn")
import pipekit
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression


DIMS = ("band", "y", "x")


def _scene(seed: int = 0) -> DimArray:
    rng = np.random.default_rng(seed)
    return DimArray(rng.normal(size=(4, 6, 5)), DIMS, {"band": ("a", "b", "c", "d")})


def _table(x: DimArray) -> np.ndarray:
    """The hand-flattened table: pixels as rows, bands as columns."""
    return np.moveaxis(x.data, 0, -1).reshape(-1, x.data.shape[0])


def test_pca_matches_sklearn_on_the_flattened_table() -> None:
    scene = _scene()
    op = Estimator(PCA(2), sample_dims=("y", "x"), feature_dim="component")
    out = op.fit(scene).transform(scene)
    assert out.dims == ("component", "y", "x")
    want = PCA(2).fit(_table(scene)).transform(_table(scene))
    np.testing.assert_allclose(_table(out), want, rtol=1e-10, atol=1e-12)
    back = op.inverse_transform(out)
    assert back.dims == DIMS and back.shape == scene.shape


def test_kmeans_predict_gives_the_sample_dims() -> None:
    scene = _scene()
    op = Estimator(KMeans(3, n_init=1, random_state=0), sample_dims=("y", "x"))
    labels = op.fit(scene).predict(scene)
    assert labels.dims == ("y", "x")
    want = KMeans(3, n_init=1, random_state=0).fit(_table(scene)).predict(_table(scene))
    np.testing.assert_array_equal(labels.data.ravel(), want)


def test_isolation_forest_scores_and_method_dispatch() -> None:
    scene = _scene()
    op = Estimator(
        IsolationForest(random_state=0), sample_dims=("y", "x"), method="predict"
    ).fit(scene)
    np.testing.assert_array_equal(op(scene).data, op.predict(scene).data)


def test_supervised_fit_and_predict_proba() -> None:
    scene = _scene()
    target = DimArray((scene.data[0] > 0).astype(int), ("y", "x"))
    op = Estimator(LogisticRegression(), sample_dims=("y", "x")).fit(scene, target)
    proba = op.predict_proba(scene)
    assert proba.dims == ("class", "y", "x") and proba.shape[0] == 2
    np.testing.assert_allclose(proba.data.sum(axis=0), 1.0)


def test_mask_samples_leaves_nan_pixels_out_and_puts_them_back() -> None:
    scene = _scene()
    data = scene.data.copy()
    data[2, 1, 3] = np.nan  # one band of one pixel
    holed = DimArray(data, DIMS)
    op = Estimator(
        PCA(2),
        sample_dims=("y", "x"),
        feature_dim="component",
        nan_policy="mask_samples",
    )
    out = op.fit(holed).transform(holed)
    assert np.isnan(out.data[:, 1, 3]).all()
    valid = np.isfinite(_table(holed)).all(axis=1)
    want = PCA(2).fit(_table(holed)[valid]).transform(_table(holed)[valid])
    np.testing.assert_allclose(_table(out)[valid], want, rtol=1e-10, atol=1e-12)


def test_mask_features_drops_all_nan_bands_and_restores_them() -> None:
    scene = _scene()
    data = scene.data.copy()
    data[1] = np.nan  # a whole band
    holed = DimArray(data, DIMS)
    op = Estimator(
        PCA(2),
        sample_dims=("y", "x"),
        feature_dim="component",
        nan_policy="mask_features",
    ).fit(holed)
    assert op.kept_features_.tolist() == [True, False, True, True]
    back = op.inverse_transform(op.transform(holed))
    assert np.isnan(back.data[1]).all() and np.isfinite(back.data[[0, 2, 3]]).all()


def test_raise_policy_and_propagate_default() -> None:
    data = _scene().data.copy()
    data[0, 0, 0] = np.inf
    holed = DimArray(data, DIMS)
    with pytest.raises(ValueError, match="nan_policy='raise'"):
        Estimator(PCA(1), sample_dims=("y", "x"), nan_policy="raise").fit(holed)
    with pytest.raises(ValueError):  # scikit-learn refuses the inf itself
        Estimator(PCA(1), sample_dims=("y", "x")).fit(holed)


def test_a_feature_mismatch_at_transform_time_names_the_dims() -> None:
    op = Estimator(PCA(2), sample_dims=("y", "x"), feature_dim="c").fit(_scene())
    other = DimArray(np.zeros((3, 6, 5)), DIMS)
    with pytest.raises(ValueError, match="fitted on features"):
        op.transform(other)


def test_unfitted_and_bad_arguments() -> None:
    with pytest.raises(ValueError, match="not fitted"):
        Estimator(PCA(2), sample_dims=("y", "x")).transform(_scene())
    with pytest.raises(ValueError, match="nan_policy"):
        Estimator(PCA(2), sample_dims=("y", "x"), nan_policy="drop")
    with pytest.raises(ValueError, match="method"):
        Estimator(PCA(2), sample_dims=("y", "x"), method="score")
    with pytest.raises(ValueError, match="feature_dim"):
        Estimator(PCA(2), sample_dims=("y", "x")).fit(_scene()).transform(_scene())


def test_the_original_estimator_is_never_fitted() -> None:
    pca = PCA(2)
    Estimator(pca, sample_dims=("y", "x"), feature_dim="c").fit(_scene())
    assert not hasattr(pca, "components_")


def test_it_is_a_fittable_transformer_that_composes() -> None:
    scene = _scene()
    op = Estimator(PCA(2), sample_dims=("y", "x"), feature_dim="component").fit(scene)
    assert isinstance(op, pipekit.FittableTransformer)
    pipe = pipekit.Sequential([op])
    np.testing.assert_allclose(pipe(scene).data, op.transform(scene).data)
    assert op.forbid_in_yaml


def test_xarray_keeps_sample_coordinates() -> None:
    xr = pytest.importorskip("xarray")
    scene = _scene()
    da = xr.DataArray(
        scene.data, dims=DIMS, coords={"band": list("abcd"), "x": np.arange(5.0)}
    )
    out = (
        Estimator(PCA(2), sample_dims=("y", "x"), feature_dim="component")
        .fit(da)
        .transform(da)
    )
    assert isinstance(out, xr.DataArray) and out.dims == ("component", "y", "x")
    np.testing.assert_array_equal(out["x"].values, np.arange(5.0))
    assert "band" not in out.coords


def test_fit_inside_a_jax_trace_raises() -> None:
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    op = Estimator(PCA(1), sample_dims=("y", "x"))

    def fit_in_trace(data):
        op.fit(DimArray(data, DIMS))
        return data

    with pytest.raises(TypeError, match="JAX trace"):
        jax.jit(fit_in_trace)(jnp.asarray(_scene().data))
    # eagerly, a JAX carrier is converted on the host
    rows, _ = to_rows(DimArray(jnp.asarray(_scene().data), DIMS), ("y", "x"))
    assert rows.shape == (30, 4)


def test_mask_drops_land_then_gappy_samples() -> None:
    scene = _scene()
    data = scene.data.copy()
    data[1] = np.nan  # a whole band: dropped as a feature
    data[0, 2, 3] = np.nan  # one gap: that pixel is dropped as a sample
    holed = DimArray(data, DIMS)
    op = Estimator(
        PCA(2), sample_dims=("y", "x"), feature_dim="component", nan_policy="mask"
    ).fit(holed)
    assert op.kept_features_.tolist() == [True, False, True, True]
    out = op.transform(holed)
    assert np.isnan(out.data[:, 2, 3]).all()
    assert int(np.isfinite(out.data[0]).sum()) == 29
    table = _table(holed)[:, [0, 2, 3]]
    valid = np.isfinite(table).all(axis=1)
    want = PCA(2).fit(table[valid]).transform(table[valid])
    np.testing.assert_allclose(_table(out)[valid], want, rtol=1e-10, atol=1e-12)
    back = op.inverse_transform(out)
    assert np.isnan(back.data[1]).all() and np.isnan(back.data[:, 2, 3]).all()
