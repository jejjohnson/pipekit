"""Tests for `pipekit_jax.JaxModelOp`."""

from __future__ import annotations

import pytest


pytest.importorskip("equinox")
pytest.importorskip("jax")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from pipekit import Operator
from pipekit_jax import JaxModelOp


def _toy_mlp(key: int = 0) -> eqx.nn.MLP:
    return eqx.nn.MLP(
        in_size=2,
        out_size=2,
        width_size=8,
        depth=2,
        key=jax.random.key(key),
    )


# ---------------------------------------------------------------------
# Construction + the Operator surface
# ---------------------------------------------------------------------


def test_jax_model_op_wraps_module():
    mlp = _toy_mlp()
    op = JaxModelOp(mlp)
    assert op.module is mlp
    assert isinstance(op, Operator)


def test_jax_model_op_apply_forwards_to_module():
    mlp = _toy_mlp()
    op = JaxModelOp(mlp)
    x = jnp.array([0.5, -0.5], dtype=jnp.float32)
    np.testing.assert_array_almost_equal(op(x), mlp(x))


def test_jax_model_op_get_config_surfaces_class_name():
    op = JaxModelOp(_toy_mlp())
    config = op.get_config()
    assert config == {"module_class": "MLP"}


def test_jax_model_op_forbid_in_yaml():
    """Array leaves don't round-trip via JSON; advertise that."""
    assert JaxModelOp.forbid_in_yaml is True


def test_jax_model_op_rejects_non_eqx_module():
    class _NotAModule:
        pass

    with pytest.raises(TypeError, match=r"eqx\.Module"):
        JaxModelOp(_NotAModule())  # type: ignore[arg-type]


# ---------------------------------------------------------------------
# serialize_weights / with_weights round-trip
# ---------------------------------------------------------------------


def test_serialize_weights_returns_bytes():
    op = JaxModelOp(_toy_mlp())
    blob = op.serialize_weights()
    assert isinstance(blob, bytes)
    assert len(blob) > 0


def test_with_weights_round_trip_is_byte_identical():
    """The headline test: serialise → deserialise into a fresh
    skeleton gives back the same weights leaf-for-leaf."""
    # Train-shaped module with deterministic weights.
    trained = JaxModelOp(_toy_mlp(key=0))
    blob = trained.serialize_weights()

    # Fresh skeleton with *different* weights (key=1) but the same
    # PyTree structure.
    skeleton = JaxModelOp(_toy_mlp(key=1))

    # Pre-condition: trained and skeleton differ.
    trained_leaves = jax.tree.leaves(eqx.filter(trained.module, eqx.is_array))
    skeleton_leaves = jax.tree.leaves(eqx.filter(skeleton.module, eqx.is_array))
    differ = any(
        not np.array_equal(np.asarray(t), np.asarray(s))
        for t, s in zip(trained_leaves, skeleton_leaves, strict=True)
    )
    assert differ, "test fixture broken — fresh skeleton should differ"

    # Post-condition: after with_weights, restored == trained.
    restored = skeleton.with_weights(blob)
    restored_leaves = jax.tree.leaves(eqx.filter(restored.module, eqx.is_array))
    assert len(restored_leaves) == len(trained_leaves)
    for r, t in zip(restored_leaves, trained_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(r), np.asarray(t))

    # The skeleton itself is untouched (with_weights returns a NEW op).
    skeleton_leaves_after = jax.tree.leaves(eqx.filter(skeleton.module, eqx.is_array))
    for s_before, s_after in zip(skeleton_leaves, skeleton_leaves_after, strict=True):
        np.testing.assert_array_equal(np.asarray(s_before), np.asarray(s_after))


def test_with_weights_rejects_non_bytes():
    op = JaxModelOp(_toy_mlp())
    with pytest.raises(TypeError, match="bytes"):
        op.with_weights("not bytes")  # type: ignore[arg-type]


def test_with_weights_returns_new_jax_model_op():
    op = JaxModelOp(_toy_mlp())
    restored = op.with_weights(op.serialize_weights())
    assert isinstance(restored, JaxModelOp)
    # Different object identity — fresh wrapper.
    assert restored is not op


# ---------------------------------------------------------------------
# Registry round-trip
# ---------------------------------------------------------------------


def test_from_registry_with_loader_method():
    """If the registry exposes load_weights(ref), use it."""
    trained = JaxModelOp(_toy_mlp(key=0))
    blob = trained.serialize_weights()

    class _LoaderRegistry:
        def load_weights(self, ref: str) -> bytes:
            assert ref == "my-hash"
            return blob

    skeleton = JaxModelOp(_toy_mlp(key=1))
    restored = skeleton.from_registry(_LoaderRegistry(), "my-hash")
    np.testing.assert_array_equal(
        np.asarray(eqx.filter(restored.module, eqx.is_array).layers[0].weight),
        np.asarray(eqx.filter(trained.module, eqx.is_array).layers[0].weight),
    )


def test_from_registry_with_local_model_registry(tmp_path):
    """End-to-end round-trip through a real LocalModelRegistry."""
    from pipekit_experiment import LocalModelRegistry

    trained = JaxModelOp(_toy_mlp(key=0))
    registry = LocalModelRegistry(tmp_path / "models")
    h = registry.store(trained, weights=trained.serialize_weights())

    # Reload via a fresh skeleton.
    skeleton = JaxModelOp(_toy_mlp(key=1))
    restored = skeleton.from_registry(registry, h)

    # Byte-identical weight equality.
    trained_leaves = jax.tree.leaves(eqx.filter(trained.module, eqx.is_array))
    restored_leaves = jax.tree.leaves(eqx.filter(restored.module, eqx.is_array))
    for r, t in zip(restored_leaves, trained_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(r), np.asarray(t))


def test_from_registry_missing_weights_raises_clear_error(tmp_path):
    """Stored without weights → reload via from_registry raises."""
    from pipekit_experiment import LocalModelRegistry

    op = JaxModelOp(_toy_mlp())
    registry = LocalModelRegistry(tmp_path / "models")
    h = registry.store(op)  # no weights= kwarg

    skeleton = JaxModelOp(_toy_mlp(key=1))
    with pytest.raises(ValueError, match="no weights blob"):
        skeleton.from_registry(registry, h)


def test_from_registry_rejects_bad_registry():
    """A registry without load_weights AND without LocalModelRegistry
    layout raises a clear TypeError."""

    class _BadRegistry:
        pass

    skeleton = JaxModelOp(_toy_mlp())
    with pytest.raises(TypeError, match="registry"):
        skeleton.from_registry(_BadRegistry(), "irrelevant")


# ---------------------------------------------------------------------
# The headline integration test: trained → store → reload by hash
# gives byte-identical predictions
# ---------------------------------------------------------------------


def test_trained_model_reloads_byte_identical_predictions(tmp_path):
    """Pin the headline v0.2 claim: stored trained weights round-trip
    well enough that the reloaded model produces identical predictions."""
    from pipekit_experiment import LocalModelRegistry

    registry = LocalModelRegistry(tmp_path / "models")
    trained = JaxModelOp(_toy_mlp(key=0))

    # Get the trained predictions on a fixed input.
    x = jnp.array([0.3, -0.7], dtype=jnp.float32)
    expected = trained(x)

    # Store with weights, reload via a fresh skeleton.
    h = registry.store(trained, weights=trained.serialize_weights())
    skeleton = JaxModelOp(_toy_mlp(key=999))  # arbitrary fresh weights
    reloaded = skeleton.from_registry(registry, h)

    actual = reloaded(x)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
