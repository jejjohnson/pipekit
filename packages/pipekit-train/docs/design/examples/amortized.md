---
status: draft
version: 0.1.0
---

# Example — Amortized inference (plume source attribution)

Simulation-based inference / neural posterior estimation. Train a
conditional density estimator on `(simulated_obs, source_params)`
pairs; at inference time, condition on a new observation and sample
posterior parameters. Maps to master plan §2.3 + §4.3.

## A recipe that runs

This toy has a closed-form answer to check against. The simulator adds two
parameters that have standard normal priors, so that the observation
$y = \theta_1 + \theta_2$ can't separate them. Each parameter's marginal posterior
is $\theta_i \mid y \sim \mathcal{N}(y / 2, 1 / 2)$. The density head belongs to the
user or a model library rather than pipekit-train, as `boundaries.md` says.

This recipe only targets those marginals. The head below is a diagonal Gaussian,
whereas the true joint posterior is perfectly anticorrelated, because
$\theta_1 + \theta_2 = y$ exactly. Don't draw both parameters from this head and
expect them to add up to $y$. A joint posterior needs a head with correlations,
such as the flow in the sketch further down.

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pipekit_ops as pk
import pipekit_train as pt
from pipekit_train.adapters.equinox import EquinoxModelOp


class Prior(pk.Operator):
    def _apply(self, seed=None):
        return np.random.default_rng(seed).standard_normal(2).astype(np.float32)


class AddParameters:
    dt = 1.0
    state_signature = None

    def step(self, state, dt):
        return np.array([state.sum()], dtype=np.float32)

    def get_config(self):
        return {}


class GaussianHead(eqx.Module):
    loc: jax.Array
    log_scale: jax.Array

    def log_prob(self, theta):
        z = (theta - self.loc) * jnp.exp(-self.log_scale)
        return (-0.5 * z**2 - self.log_scale - 0.5 * jnp.log(2 * jnp.pi)).sum(axis=-1)


class AmortizedPosterior(eqx.Module):
    net: eqx.nn.MLP

    def __call__(self, y):
        out = self.net(y)
        return GaussianHead(out[:2], out[2:])


dataset = pt.SimulationDataset(AddParameters(), Prior(), n_samples=4_000, invert=True)
network = eqx.nn.MLP(1, 4, 32, 2, key=jax.random.key(0))
model = EquinoxModelOp(AmortizedPosterior(network))

trained, artifact = pt.TrainingLoop(
    model_op=model,
    dataset=dataset,
    loss=pt.NLL(),
    optimizer_config={"name": "adam", "lr": 3e-3},
    max_steps=1_500,
    batch_size=64,
    backend="equinox",
).run()

# marginal calibration on held-out simulations: about 95 % of parameters within ±2σ
y, theta = (np.stack(a) for a in zip(*dataset.with_split("test")))
posterior = jax.vmap(trained.module)(jnp.asarray(y))
inside = jnp.abs(theta - posterior.loc) <= 2 * jnp.exp(posterior.log_scale)
print(float(inside.mean()))
```

`tests/test_amortized_recipe.py` runs this and checks the calibration and the
closed form.

## The same shape on a real problem

The sketch below hasn't been run, because it depends on plumax and pyrox.

```python
import pipekit_ops as pk
import pipekit_train as pt
import pipekit_cycle as pc
import pipekit_experiment as pe

# 1. The simulator — plume forward + observation operator.
from plumax.adapters.pipekit_ops import PlumeForward, ColumnObs

plume_forward = PlumeForward()
obs_op = ColumnObs(instrument="TROPOMI")
# plume_forward satisfies pipekit_cycle.ForwardModel.
# obs_op satisfies pipekit_cycle.ObservationOperator.


# 2. Prior over source parameters.
class SourcePrior(pk.Operator):
    """Prior over (lat, lon, strength, height)."""

    loc_bounds: tuple[float, float, float, float]
    strength_loguniform: tuple[float, float]
    height_uniform: tuple[float, float]

    def __call__(self) -> jax.Array: ...


prior = SourcePrior(
    loc_bounds=(-10, 10, 30, 40),
    strength_loguniform=(1.0, 1e4),
    height_uniform=(0.0, 500.0),
)

# 3. SimulationDataset for SBI — emit (observed_concentrations, params).
#    Crucially the obs_op is applied so the network learns to invert
#    *observed* concentrations, not the latent state.
sbi_ds = pt.SimulationDataset(
    forward_model=plume_forward,
    prior=prior,
    n_samples=100_000,
    obs_op=obs_op,  # ← key difference vs emulator training
    invert=True,  # ← inputs are observations, targets are params
    seed=42,
    split="train",
)
val_ds = sbi_ds.with_split("val")

# 4. The conditional density estimator — from pyrox.
from pyrox.adapters.pipekit_train import ConditionalNormalizingFlow

posterior_model = ConditionalNormalizingFlow(
    n_dim=4,  # 4 source params
    condition_dim=obs_op.output_dim,  # observation embedding size
    flow_type="masked_autoregressive",
    n_layers=8,
    hidden_dim=128,
)

# 5. Pipekit-experiment.
registry = pe.LocalModelRegistry(root="./models")
tracker = pe.adapters.mlflow.MLflowTracker(tracking_uri="http://mlflow:5000")

# 6. Train as a normalising-flow likelihood. NLL ⇒ the flow learns
#    log p(params | obs).
loop = pt.TrainingLoop(
    model_op=posterior_model,
    dataset=pt.CachedDataset(sbi_ds, cache_dir="./cache/plume_sbi/"),
    val_dataset=pt.CachedDataset(val_ds, cache_dir="./cache/plume_sbi/"),
    loss=pt.NLL(),  # flow's own log_prob
    optimizer_config={"name": "adamw", "lr": 1e-3},
    max_steps=200_000,
    batch_size=128,
    backend="equinox",  # JAX-traceable
    callbacks=(
        pt.EarlyStopping(metric="val_loss", patience=10),
        pt.Checkpoint(
            every_n_steps=5_000, save_dir="./ckpts/plume_sbi", registry=registry
        ),
        pt.LogToExperiment(tracker=tracker, run_name="plume_npe"),
    ),
    metric_writer=pt.JSONLWriter(path="./logs/plume_sbi.jsonl"),
    seed=42,
)

posterior_op, artifact = loop.run()

# 7. Use the amortized posterior at inference.
#    Given any new observation, sample parameters in milliseconds —
#    no simulator calls at inference time.
new_obs = load_real_tropomi_pixel("2026-04-12T13:00:00Z")
posterior_samples = posterior_op.sample(condition=new_obs, n_samples=10_000)
# posterior_samples: array of shape (10_000, 4) — lat, lon, strength, height.

# 8. The posterior_op is a pipekit_ops.Operator — drops into a full
#    attribution pipeline.
attribution_pipeline = pk.Sequential(
    [
        obs_op_preprocess,  # any obs-side preprocessing
        posterior_op,  # samples posterior
        posterior_summary_op,  # MAP / credible intervals / etc.
    ]
)
```

**What this example demonstrates:**

- `SimulationDataset(obs_op=..., invert=True)` — the dataset emits
  `(observed_concentrations, params)` pairs by composing the forward model
  with the observation operator. The same dataset as emulator training
  with its pair flipped; the loss and head differ too.
- Pyrox's `ConditionalNormalizingFlow` plugged in as a
  `pipekit_ops.Operator` via the `pyrox.adapters.pipekit_train`
  extra-gated adapter. Pipekit-train sees only an `Operator`; the
  algorithmic core lives in pyrox.
- `NLL` here is computed by the flow itself (each batch is scored
  by its own `log_prob`). The adapter's auto-synthesised TrainTask
  handles this — the user does not write a custom `train_step`.
- The trained posterior is *amortized*: one (potentially expensive)
  training; many fast queries at inference. The attribution pipeline
  at the bottom is composable just like any inference pipeline.

**Why pipekit-train matters here.**

Without pipekit-train, this workflow is three different codebases
glued together — a simulator script, a custom SBI training loop, a
custom inference script. With pipekit-train, the simulator is a
`SimulationDataset`, the SBI loop is a `TrainingLoop`, the inference
is a `Sequential` of `Operator`s. The artifact at the end is one
JSON file that reproduces all three byte-identically.
