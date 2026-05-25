---
status: draft
version: 0.1.0
---

# Example — Emulator (chemistry transport surrogate)

The bridge example. Train a neural emulator on simulations of an
expensive forward model, then drop the emulator back into a
`pipekit-cycle.Cycle` as a `NeuralForward`. Maps to master plan §2.2
+ §4.2.

```python
import pipekit as pk
import pipekit_train as pt
import pipekit_cycle as pc
import pipekit_experiment as pe
import pipekit_jax as pj
import equinox as eqx

# 1. The expensive physics. From an external library (e.g. plumax).
from plumax.adapters.pipekit import ChemistryForward
expensive_forward = ChemistryForward(species=["CH4", "NH3"], dt=3600.0)
# `expensive_forward` satisfies pipekit_cycle.ForwardModel.

# 2. The parameter / state prior — domain-specific.
class AtmosphericPrior(pk.Operator):
    """Samples initial atmospheric states from a climatology."""
    distribution: str
    def __call__(self) -> dict[str, jax.Array]: ...

prior = AtmosphericPrior(distribution="climatology")

# 3. SimulationDataset — yields (state, simulated_trajectory) pairs.
sim_ds = pt.SimulationDataset(
    forward_model=expensive_forward,
    prior=prior,
    n_samples=10_000,
    cycle=pc.Cycle(step_op=expensive_forward, n_steps=24),  # 24h rollout
    seed=0,
    split="train",
)

# 4. Disk-backed cache so we don't regenerate every epoch.
cached_ds = pt.CachedDataset(
    source=sim_ds,
    cache_dir="s3://cache/chemistry_sim/",
    format="zarr",
)
val_ds = pt.CachedDataset(
    source=sim_ds.with_split("val"),
    cache_dir="s3://cache/chemistry_sim/",
    format="zarr",
)

# 5. The neural emulator.
class EmulatorNet(eqx.Module):
    """A small MLP / CNN / Transformer that maps state_t → state_{t+24h}."""
    ...
model_op = pj.JaxModelOp(EmulatorNet(...))

# 6. Pipekit-experiment integration.
registry = pe.LocalModelRegistry(root="./models")

# 7. Training loop — MSE for emulators with deterministic output;
#    swap for NLL if the emulator predicts a distribution.
loop = pt.TrainingLoop(
    model_op=model_op,
    dataset=cached_ds,
    val_dataset=val_ds,
    loss=pt.MSE(),
    optimizer_config={"name": "adamw", "lr": 5e-4},
    max_steps=100_000,
    batch_size=64,
    eval_every_n_steps=1_000,
    backend="equinox",
    callbacks=(
        pt.Checkpoint(every_n_steps=5_000, save_dir="./ckpts/chemistry",
                      registry=registry),
    ),
    metric_writer=pt.JSONLWriter(path="./logs/chemistry.jsonl"),
)

# 8. Train. Returns a (model_op, artifact) pair.
emulator_op, artifact = loop.run()

# 9. Drop the emulator into a Cycle as a NeuralForward.
emulator_forward = pc.NeuralForward(model_op=emulator_op, dt=3600.0 * 24)

# 10. Use it for fast forecasting — same Cycle as before, faster forward.
forecast = pc.Cycle(step_op=emulator_forward, n_steps=30)
trajectory = forecast(initial_state)

# 11. Re-load the emulator months later for a regulatory replay.
artifact.save("./artifacts/chemistry_emulator_v3.json")
loaded = pe.TrainingArtifact.load("./artifacts/chemistry_emulator_v3.json")
reloaded_op = loaded.reload_model(registry)
# `pc.NeuralForward(model_op=reloaded_op, dt=3600.0 * 24)` is byte-identical.
```

**What this example demonstrates:**

- `SimulationDataset` wrapping a `ForwardModel` and producing
  training pairs without any user-side data-generation loop.
- The `cycle=` argument rolling out trajectories so the emulator
  trains on multi-step outputs, not just one-step.
- `CachedDataset` on Zarr — first epoch hits the expensive forward;
  subsequent epochs read from disk.
- The trained emulator wrapped via `NeuralForward` and substituting
  for the expensive `ChemistryForward` in any `Cycle` — the train→serve
  loop closes with a single operator swap.

**Notes on caching scale.**

100K plume / chemistry simulations is hours of compute. The
`CachedDataset` makes this a one-time cost. The cache is content-
addressed by `sim_ds.content_hash()`, which folds in the forward
model's `state_signature`, the prior config, `n_samples`, `seed`,
and `split`. If any of those change, a new cache is built; if none
change, the cache is reused across runs and across machines (when
`cache_dir` is an S3 / GCS URI).

For very large simulators, generation itself should run via
`pipekit.parallel.ProcessMap` over the seed space — that's separate
from the dataset and lives at the orchestrator layer.
