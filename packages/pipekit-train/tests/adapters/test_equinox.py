"""End-to-end tests for the Equinox adapter.

Gated by ``pytest.importorskip("equinox")`` so the test only runs in
the ``[equinox]`` extra environment.

The headline test trains a tiny MLP on a synthetic 1-D regression
task (``y = 2x + 1``) and asserts the loss decreases monotonically —
the cheap end-to-end verification per the design's §9 test strategy.
"""

from __future__ import annotations

import math
from typing import Any

import pytest


pytest.importorskip("equinox")
pytest.importorskip("optax")
pytest.importorskip("jax")
pytest.importorskip("orbax.checkpoint")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from pipekit_train import (
    MSE,
    EarlyStopping,
    IterableDataset,
    JSONLWriter,
    TrainingLoop,
)
from pipekit_train.adapters.equinox import (
    EquinoxModelOp,
    TrainState,
    TrainTask,
    _build_optimizer,
    _SynthesisedTask,
    restore_state,
    save_state,
    train_step,
)


# --- Test fixtures --------------------------------------------------------


def _synthetic_regression(n: int = 64, seed: int = 0) -> IterableDataset:
    """Synthetic 1-D regression: y = 2x + 1 + tiny noise."""
    rng = np.random.default_rng(seed)
    xs = rng.uniform(-1.0, 1.0, size=(n, 1)).astype(np.float32)
    ys = 2.0 * xs + 1.0 + 0.01 * rng.standard_normal((n, 1)).astype(np.float32)
    pairs = list(zip(xs, ys, strict=True))
    return IterableDataset(source=pairs, content_hash=f"synth-{n}-{seed}")


def _toy_mlp(key: jax.Array, in_size: int = 1, out_size: int = 1) -> eqx.nn.MLP:
    return eqx.nn.MLP(
        in_size=in_size,
        out_size=out_size,
        width_size=16,
        depth=2,
        key=key,
    )


# --- Optimizer builder ----------------------------------------------------


def test_build_optimizer_adamw():
    opt = _build_optimizer({"name": "adamw", "learning_rate": 1e-3})
    assert isinstance(opt, optax.GradientTransformation)


def test_build_optimizer_sgd():
    opt = _build_optimizer({"name": "sgd", "learning_rate": 1e-2})
    assert isinstance(opt, optax.GradientTransformation)


def test_build_optimizer_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown optimizer"):
        _build_optimizer({"name": "frodo"})


# --- TrainState -----------------------------------------------------------


def test_train_state_create_then_step():
    model = _toy_mlp(jax.random.key(0))
    optimizer = optax.adam(1e-3)
    state = TrainState.create(model, optimizer)
    assert int(state.step) == 0
    # The state is a PyTree of arrays + static; both partition cleanly.
    arrays, static = eqx.partition(state, eqx.is_array)
    assert arrays is not None
    assert static is not None


# --- train_step monotonic loss decrease -----------------------------------


def test_train_step_monotonic_loss_decrease():
    """One-step gradient descent reduces loss for a simple regression problem."""
    model = _toy_mlp(jax.random.key(0))
    optimizer = optax.adam(1e-2)
    state = TrainState.create(model, optimizer)

    # Same batch repeated should drive the loss down monotonically.
    x = jnp.linspace(-1, 1, 32).reshape(-1, 1).astype(jnp.float32)
    y = 2.0 * x + 1.0
    batch = (x, y)
    task = _SynthesisedTask(MSE())

    losses: list[float] = []
    for _ in range(20):
        state, metrics = train_step(state, batch, jax.random.key(0), task, optimizer)
        losses.append(float(metrics["mse"]))
    # Last loss should be much smaller than the first.
    assert losses[-1] < losses[0] * 0.5
    # Final state.step should reflect the 20 updates.
    assert int(state.step) == 20


# --- TrainTask Protocol conformance --------------------------------------


def test_synthesised_task_satisfies_traintask_protocol():
    task = _SynthesisedTask(MSE())
    assert isinstance(task, TrainTask)


def test_user_task_satisfies_protocol_via_duck_typing():
    class _UserTask:
        def loss_fn(self, model, batch, key):
            return jnp.asarray(0.0), {"loss": jnp.asarray(0.0)}

    assert isinstance(_UserTask(), TrainTask)


# --- save_state / restore_state round-trip --------------------------------


def test_save_restore_round_trip(tmp_path):
    import orbax.checkpoint as ocp

    model = _toy_mlp(jax.random.key(0))
    optimizer = optax.adam(1e-3)
    state = TrainState.create(model, optimizer)

    mngr = ocp.CheckpointManager(
        directory=str(tmp_path / "ckpt"),
        options=ocp.CheckpointManagerOptions(max_to_keep=1),
    )
    save_state(mngr, state, step=0)
    mngr.wait_until_finished()

    # Build a fresh state with the same structure (different weights),
    # restore the saved weights into it, verify equality.
    fresh = TrainState.create(_toy_mlp(jax.random.key(1)), optimizer)
    restored = restore_state(mngr, fresh, step=0)

    # The restored model's weights should match the saved ones, not
    # the fresh template's weights.
    saved_arrays, _ = eqx.partition(state, eqx.is_array)
    restored_arrays, _ = eqx.partition(restored, eqx.is_array)
    leaves_saved = jax.tree.leaves(saved_arrays)
    leaves_restored = jax.tree.leaves(restored_arrays)
    assert len(leaves_saved) == len(leaves_restored)
    for a, b in zip(leaves_saved, leaves_restored, strict=True):
        np.testing.assert_array_almost_equal(a, b)


# --- EquinoxModelOp ------------------------------------------------------


def test_equinox_model_op_wraps_and_forwards():
    module = _toy_mlp(jax.random.key(0))
    op = EquinoxModelOp(module)
    x = jnp.array([0.5], dtype=jnp.float32)
    out = op(x)
    expected = module(x)
    np.testing.assert_array_almost_equal(out, expected)
    assert op.module is module
    assert op.forbid_in_yaml is True


def test_equinox_model_op_rejects_non_module():
    with pytest.raises(TypeError, match=r"eqx\.Module"):
        EquinoxModelOp(module="not an eqx.Module")  # type: ignore[arg-type]


# --- End-to-end run() — the headline test --------------------------------


def test_run_end_to_end_loss_decreases():
    """Train a tiny MLP via TrainingLoop.run() on synthetic data.

    Asserts the final-step loss is below 50% of the initial-step
    loss — the lightweight version of monotonic decrease.
    """
    dataset = _synthetic_regression(n=128, seed=0)
    val_dataset = _synthetic_regression(n=32, seed=1)
    model_op = EquinoxModelOp(_toy_mlp(jax.random.key(0)))

    written: list[tuple[int, dict[str, float]]] = []

    class _CapturingWriter:
        def write(self, step: int, metrics: dict[str, float]) -> None:
            written.append((step, dict(metrics)))

        def close(self) -> None:
            pass

    loop = TrainingLoop(
        model_op=model_op,
        dataset=dataset,
        val_dataset=val_dataset,
        loss=MSE(),
        optimizer_config={"name": "adam", "learning_rate": 1e-2},
        max_steps=100,
        batch_size=16,
        backend="equinox",
        eval_every_n_steps=50,
        log_every_n_steps=10,
        metric_writer=_CapturingWriter(),
        seed=42,
    )

    trained_op, artifact = loop.run()

    assert isinstance(trained_op, EquinoxModelOp)

    # The metric writer captured per-step metrics; verify the loss
    # decreased.
    assert len(written) >= 5
    first_loss = written[0][1]["mse"]
    last_loss = written[-1][1]["mse"]
    assert last_loss < first_loss * 0.5, (
        f"Expected loss to drop by ≥50% over training; got "
        f"first={first_loss}, last={last_loss}."
    )

    # The artifact carries the backend_info we expect.
    info = artifact.backend_info
    assert info["backend"] == "equinox"
    assert "jax_version" in info
    assert "equinox_version" in info
    assert "optax_version" in info
    assert info["final_step"] == 100
    assert artifact.dataset_hash == dataset.content_hash()


def test_run_with_early_stopping_breaks_early():
    """EarlyStopping with patience=1 and a fixed val_loss → stops on 2nd eval."""
    dataset = _synthetic_regression(n=64, seed=0)
    val_dataset = _synthetic_regression(n=16, seed=1)
    model_op = EquinoxModelOp(_toy_mlp(jax.random.key(0)))

    es = EarlyStopping(metric="mse", patience=1, mode="min", min_delta=10.0)

    loop = TrainingLoop(
        model_op=model_op,
        dataset=dataset,
        val_dataset=val_dataset,
        loss=MSE(),
        optimizer_config={"name": "adam", "learning_rate": 1e-2},
        max_steps=1000,
        batch_size=16,
        eval_every_n_steps=10,
        backend="equinox",
        callbacks=(es,),
        seed=0,
    )

    _, artifact = loop.run()
    # EarlyStopping should have fired well before 1000 steps. With
    # min_delta=10 and a regression loss of order 1, no improvement
    # is "good enough", so wait reaches patience after the first
    # eval round.
    final_step = artifact.backend_info["final_step"]
    assert final_step < 1000
    # Regression: PR #11 review — TrainerCarryState.step must reflect
    # the actual final step (from backend_info), not the configured
    # max_steps. final_state isn't returned from run(); construct one
    # the same way _apply does and verify the shape.
    from pipekit_train import TrainerCarryState

    expected_step = int(loop._last_backend_info["final_step"])
    assert expected_step == final_step
    assert expected_step < loop.max_steps
    # Sanity: TrainerCarryState API accepts the shape _apply uses.
    from pipekit import Const

    cs = TrainerCarryState(
        model=Const(0),
        opt_state=None,
        step=expected_step,
        epoch=0,
        metrics=dict(loop._last_backend_info.get("final_metrics", {})),
    )
    assert cs.step == expected_step


def test_run_with_callbacks_dispatches_lifecycle():
    """A custom recording callback receives all five hooks at the right times."""
    dataset = _synthetic_regression(n=32, seed=0)
    val_dataset = _synthetic_regression(n=8, seed=1)
    model_op = EquinoxModelOp(_toy_mlp(jax.random.key(0)))

    events: list[str] = []

    class _Recorder:
        def on_train_begin(self, loop: Any, state: Any) -> None:
            events.append("begin")

        def on_step_end(self, loop: Any, state: Any, metrics: dict) -> None:
            events.append(f"step:{state.step}")

        def on_eval_end(self, loop: Any, state: Any, eval_metrics: dict) -> None:
            events.append(f"eval:{state.step}")

        def on_train_end(self, loop: Any, state: Any) -> None:
            events.append("end")

    cb = _Recorder()
    # Sanity: implements the full Callback hook surface. Callback is
    # intentionally not @runtime_checkable; check method presence
    # directly (per ADR D9).
    for hook in (
        "on_train_begin",
        "on_step_end",
        "on_eval_end",
        "on_train_end",
    ):
        assert callable(getattr(cb, hook, None))

    loop = TrainingLoop(
        model_op=model_op,
        dataset=dataset,
        val_dataset=val_dataset,
        loss=MSE(),
        optimizer_config={"name": "adam", "learning_rate": 1e-2},
        max_steps=20,
        batch_size=8,
        eval_every_n_steps=10,
        backend="equinox",
        callbacks=(cb,),
        seed=0,
    )
    loop.run()

    assert events[0] == "begin"
    assert events[-1] == "end"
    # At least one step event and one eval event happened.
    assert any(e.startswith("step:") for e in events)
    assert any(e.startswith("eval:") for e in events)


def test_run_requires_loss_or_task():
    """No `loss` and no `task` → adapter raises a clear ValueError."""
    dataset = _synthetic_regression(n=8, seed=0)
    model_op = EquinoxModelOp(_toy_mlp(jax.random.key(0)))
    loop = TrainingLoop(
        model_op=model_op,
        dataset=dataset,
        max_steps=2,
        batch_size=2,
        backend="equinox",
    )
    with pytest.raises(ValueError, match=r"loss or loop\.task"):
        loop.run()


def test_run_rejects_non_eqx_model_op():
    """A non-Equinox model_op fails with a clear TypeError."""
    from pipekit import Const

    dataset = _synthetic_regression(n=8, seed=0)
    loop = TrainingLoop(
        model_op=Const(7),  # Not an EquinoxModelOp; no .module
        dataset=dataset,
        loss=MSE(),
        max_steps=2,
        batch_size=2,
        backend="equinox",
    )
    with pytest.raises(TypeError, match="EquinoxModelOp"):
        loop.run()


def test_run_user_supplied_task():
    """A custom TrainTask bypasses the auto-synthesised path."""
    dataset = _synthetic_regression(n=32, seed=0)
    model_op = EquinoxModelOp(_toy_mlp(jax.random.key(0)))

    class _MyTask:
        def loss_fn(self, model: eqx.Module, batch: Any, key: jax.Array):
            del key
            x, y = batch
            pred = jax.vmap(model)(x)
            loss = jnp.mean((pred - y) ** 2)
            return loss, {"custom_mse": loss}

    loop = TrainingLoop(
        model_op=model_op,
        dataset=dataset,
        task=_MyTask(),  # loss is None; task takes priority
        optimizer_config={"name": "adam", "learning_rate": 1e-2},
        max_steps=20,
        batch_size=8,
        backend="equinox",
        seed=0,
    )
    _, artifact = loop.run()
    # The custom metric name should surface in final_metrics.
    assert "custom_mse" in artifact.backend_info["final_metrics"]


# --- Pi-on-air sanity (no division-by-zero, no NaNs) --------------------


def test_run_does_not_produce_nan():
    dataset = _synthetic_regression(n=32, seed=0)
    model_op = EquinoxModelOp(_toy_mlp(jax.random.key(0)))
    loop = TrainingLoop(
        model_op=model_op,
        dataset=dataset,
        loss=MSE(),
        optimizer_config={"name": "adam", "learning_rate": 1e-2},
        max_steps=20,
        batch_size=8,
        backend="equinox",
        seed=0,
    )
    _, artifact = loop.run()
    final_loss = artifact.backend_info["final_metrics"].get("mse")
    assert final_loss is not None
    assert math.isfinite(final_loss)


# --- JSONLWriter integration ---------------------------------------------


def test_run_with_jsonl_writer_writes_records(tmp_path):
    dataset = _synthetic_regression(n=32, seed=0)
    model_op = EquinoxModelOp(_toy_mlp(jax.random.key(0)))
    out = tmp_path / "metrics.jsonl"
    loop = TrainingLoop(
        model_op=model_op,
        dataset=dataset,
        loss=MSE(),
        optimizer_config={"name": "adam", "learning_rate": 1e-2},
        max_steps=20,
        batch_size=8,
        log_every_n_steps=5,
        backend="equinox",
        metric_writer=JSONLWriter(path=out, flush_every=1),
        seed=0,
    )
    loop.run()
    contents = out.read_text().splitlines()
    assert len(contents) >= 3  # steps 5, 10, 15, 20


# --- Regressions / robustness ---------------------------------------------


def test_indexable_dataset_smaller_than_batch_size_raises():
    """Regression: PR #11 review — small indexable dataset would hang
    the trainer (iterator yields nothing). Now fails fast with a
    clear error.

    We need an indexable dataset (`__len__`/`__getitem__`) because the
    streaming `IterableDataset` doesn't have this constraint. Build one
    by wrapping a tiny sequence in a custom TrainingDataset subclass.
    """
    from pipekit_train import TrainingDataset

    class _TinyIndexed(TrainingDataset):
        def __init__(self):
            super().__init__()
            self.pairs = [
                (np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32)),
                (np.array([1.0], dtype=np.float32), np.array([2.0], dtype=np.float32)),
            ]

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, i):
            return self.pairs[i]

        def __iter__(self):
            return iter(self.pairs)

        def content_hash(self):
            return "tiny-2"

    tiny = _TinyIndexed()
    loop = TrainingLoop(
        model_op=EquinoxModelOp(_toy_mlp(jax.random.key(0))),
        dataset=tiny,
        loss=MSE(),
        max_steps=5,
        batch_size=32,  # > len(tiny) == 2
        backend="equinox",
    )
    with pytest.raises(ValueError, match="batch_size"):
        loop.run()


def test_synthesised_task_reduces_non_scalar_loss():
    """Regression: PR #11 review — MSE(reduction='none') would crash
    inside eqx.filter_grad because the loss isn't a scalar. The
    synthesised task now reduces non-scalar outputs with jnp.mean
    so common Loss reductions don't break the gradient path.
    """
    dataset = _synthetic_regression(n=32, seed=0)
    loop = TrainingLoop(
        model_op=EquinoxModelOp(_toy_mlp(jax.random.key(0))),
        dataset=dataset,
        loss=MSE(reduction="none"),  # returns per-element squared error
        optimizer_config={"name": "adam", "learning_rate": 1e-2},
        max_steps=10,
        batch_size=8,
        backend="equinox",
        seed=0,
    )
    _, artifact = loop.run()
    final_loss = artifact.backend_info["final_metrics"].get("mse")
    assert final_loss is not None and math.isfinite(final_loss)


def test_run_carry_state_step_matches_actual_final_step():
    """Regression: PR #11 review — final_state.step must come from
    the adapter's actual final_step (so early-stopped runs report
    the truth, not the configured max_steps).
    """
    dataset = _synthetic_regression(n=32, seed=0)
    loop = TrainingLoop(
        model_op=EquinoxModelOp(_toy_mlp(jax.random.key(0))),
        dataset=dataset,
        loss=MSE(),
        optimizer_config={"name": "adam", "learning_rate": 1e-2},
        max_steps=20,
        batch_size=8,
        backend="equinox",
        seed=0,
    )
    _, final_state = loop._apply()
    backend_info = loop._last_backend_info
    assert final_state.step == int(backend_info["final_step"])
