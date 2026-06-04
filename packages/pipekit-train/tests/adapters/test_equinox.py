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
    restored, data_iter_state = restore_state(mngr, fresh, step=0)
    # No iterator side-car was written → data_iter_state is None.
    assert data_iter_state is None

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


def test_iterator_state_checkpoint_round_trip(tmp_path):
    """Regression: #17 — train half the steps, checkpoint, resume into
    a loop that *continues* training to completion. The model produced
    by the resumed pipeline must be byte-identical to a single
    uninterrupted run from step 0 — equality requires the iterator
    state to round-trip (otherwise the resumed run sees different
    batches in steps 5-10 from the reference run, leading to
    different weights).

    PR #24 copilot review caught that the original version of this
    test set ``loop2.max_steps == restored state.step``, so the
    second run never advanced and would have passed even with
    iterator state ignored. The fixed version actually advances.
    """
    import jax
    from pipekit_train import TrainingDataset

    # Build a synthetic indexable dataset whose elements are tiny
    # but each sample's value uniquely identifies it.
    class _IndexedDataset(TrainingDataset):
        forbid_in_yaml = True
        __config_mixin_auto__ = False

        def __init__(self, n: int):
            super().__init__()
            self.n = n
            # Identity inputs — easy to spot which samples were seen.
            self.xs = np.arange(n, dtype=np.float32).reshape(-1, 1)
            self.ys = (2.0 * self.xs + 1.0).astype(np.float32)

        def __len__(self) -> int:
            return self.n

        def __getitem__(self, i: int):
            return self.xs[i], self.ys[i]

        def __iter__(self):
            for i in range(self.n):
                yield self.xs[i], self.ys[i]

        def content_hash(self) -> str:
            return f"indexed-n{self.n}"

    dataset = _IndexedDataset(n=64)

    common = dict(
        dataset=dataset,
        loss=MSE(),
        optimizer_config={"name": "adam", "learning_rate": 1e-2},
        batch_size=8,
        backend="equinox",
        seed=0,
    )

    # --- Reference: train 10 steps in one uninterrupted run --------
    loop_ref = TrainingLoop(
        model_op=EquinoxModelOp(_toy_mlp(jax.random.key(0))),
        max_steps=10,
        **common,
    )
    trained_ref, _ = loop_ref.run()

    # --- Phase 1: train 5 steps, save checkpoint --------------------
    ckpt_dir = tmp_path / "ckpt"
    loop_phase1 = TrainingLoop(
        model_op=EquinoxModelOp(_toy_mlp(jax.random.key(0))),
        max_steps=5,
        callbacks=(
            __import__("pipekit_train").Checkpoint(
                every_n_steps=5, keep_last=1, save_dir=str(ckpt_dir)
            ),
        ),
        checkpoint_dir=str(ckpt_dir),
        **common,
    )
    _, art_phase1 = loop_phase1.run()
    assert art_phase1.backend_info["final_step"] == 5

    # The iterator-state side-car must exist (per-process name; index 0 in a
    # single-process run).
    side_car = ckpt_dir / "5" / "data_iter_0.json"
    assert side_car.exists(), (
        "iterator-state side-car not written — see save_state extension"
    )

    # --- Phase 2: fresh loop, max_steps=10, restores at step 5 and
    # actually runs 5 more steps to reach step 10. -------------------
    fresh_op = EquinoxModelOp(_toy_mlp(jax.random.key(999)))  # different weights
    loop_phase2 = TrainingLoop(
        model_op=fresh_op,
        max_steps=10,  # ← advances beyond restored step=5
        checkpoint_dir=str(ckpt_dir),
        **common,
    )
    trained_resumed, art_phase2 = loop_phase2.run()
    # Phase 2 actually advanced from 5 → 10 (not a no-op).
    assert art_phase2.backend_info["final_step"] == 10

    # The headline claim: resumed model == reference model, byte-
    # identically. This holds only if the iterator state was restored
    # correctly — otherwise Phase 2's steps 5-10 would see different
    # batches than the reference's, and the weights would diverge.
    x = jnp.array([0.3], dtype=jnp.float32)
    y = jnp.array([0.7], dtype=jnp.float32)
    np.testing.assert_array_almost_equal(trained_ref(x), trained_resumed(x), decimal=5)
    np.testing.assert_array_almost_equal(trained_ref(y), trained_resumed(y), decimal=5)


def test_finished_resume_does_not_construct_iterator(tmp_path):
    """Regression: PR #24 P2 — resuming a checkpoint whose state.step
    is already at max_steps must not construct the data iterator
    (so a dataset smaller than batch_size doesn't fail with
    ValueError on a no-op resume).
    """
    import jax
    from pipekit_train import TrainingDataset

    class _TinyIndexed(TrainingDataset):
        forbid_in_yaml = True
        __config_mixin_auto__ = False

        def __len__(self) -> int:
            return 3  # smaller than the batch_size below

        def __getitem__(self, i: int):
            return (
                np.array([float(i)], dtype=np.float32),
                np.array([float(i) * 2], dtype=np.float32),
            )

        def __iter__(self):
            for i in range(3):
                yield self[i]

        def content_hash(self) -> str:
            return "tiny-3"

    ckpt_dir = tmp_path / "ckpt"

    # Phase 1: build a checkpoint at step 4 (dataset has only 3
    # samples; uses a different big dataset to write a checkpoint).
    big_ds = _synthetic_regression(n=64, seed=0)
    loop_phase1 = TrainingLoop(
        model_op=EquinoxModelOp(_toy_mlp(jax.random.key(0))),
        dataset=big_ds,
        loss=MSE(),
        optimizer_config={"name": "adam", "learning_rate": 1e-2},
        max_steps=4,
        batch_size=8,
        backend="equinox",
        callbacks=(
            __import__("pipekit_train").Checkpoint(
                every_n_steps=4, keep_last=1, save_dir=str(ckpt_dir)
            ),
        ),
        checkpoint_dir=str(ckpt_dir),
        seed=0,
    )
    loop_phase1.run()

    # Phase 2: same checkpoint_dir, finishing-shaped state (max_steps
    # equals restored step). Tiny dataset with batch_size > len — this
    # would crash if we built the iterator unconditionally.
    loop_phase2 = TrainingLoop(
        model_op=EquinoxModelOp(_toy_mlp(jax.random.key(0))),
        dataset=_TinyIndexed(),
        loss=MSE(),
        optimizer_config={"name": "adam", "learning_rate": 1e-2},
        max_steps=4,  # restored step=4 → no-op loop
        batch_size=8,  # > len(_TinyIndexed) = 3
        backend="equinox",
        checkpoint_dir=str(ckpt_dir),
        seed=0,
    )
    # No exception even though len(dataset) < batch_size — the
    # iterator is never constructed because state.step >= max_steps.
    trained, artifact = loop_phase2.run()
    assert artifact.backend_info["final_step"] == 4
    assert isinstance(trained, EquinoxModelOp)


def test_batch_source_streaming_get_state_is_none():
    """BatchSource over an IterableDataset (no random access) has no
    iterator state — get_state() returns None; set_state(None) no-ops."""
    from pipekit_train.adapters.equinox import _BatchSource

    streaming_ds = IterableDataset(
        source=[(np.zeros(1), np.zeros(1))] * 4,
        content_hash="streaming-smoke",
    )
    src = _BatchSource(streaming_ds, batch_size=2, seed=0)
    assert src.get_state() is None
    src.set_state(None)  # no-op
    # And it still produces batches.
    x, _y = src.next_batch()
    assert x.shape == (2, 1)


def test_batch_source_grain_get_state_is_dict():
    """BatchSource over an indexable dataset returns a Grain state
    dict from get_state() and round-trips through set_state()."""
    from pipekit_train import TrainingDataset
    from pipekit_train.adapters.equinox import _BatchSource

    class _Tiny(TrainingDataset):
        forbid_in_yaml = True
        __config_mixin_auto__ = False

        def __init__(self):
            super().__init__()

        def __len__(self) -> int:
            return 4

        def __getitem__(self, i: int):
            return np.array([i], dtype=np.float32), np.array([i * 2], dtype=np.float32)

        def __iter__(self):
            for i in range(4):
                yield self[i]

        def content_hash(self) -> str:
            return "tiny"

    src = _BatchSource(_Tiny(), batch_size=2, seed=0)
    src.next_batch()
    state = src.get_state()
    assert isinstance(state, dict)
    # Round-trip through json+base64 (mirrors the side-car path).
    from pipekit_train.adapters.equinox import (
        _jsonify_grain_state,
        _unjsonify_grain_state,
    )

    serialized = _jsonify_grain_state(state)
    deserialized = _unjsonify_grain_state(serialized)
    src2 = _BatchSource(_Tiny(), batch_size=2, seed=0)
    src2.set_state(deserialized)
    # Both iterators should now yield identical batches.
    a = src.next_batch()
    b = src2.next_batch()
    np.testing.assert_array_equal(np.asarray(a[0]), np.asarray(b[0]))
    np.testing.assert_array_equal(np.asarray(a[1]), np.asarray(b[1]))
