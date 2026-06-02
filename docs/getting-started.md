# Getting Started

Five minutes from `git clone` to your first composed pipeline, first
data-assimilation cycle, and first registered model. Pick the section
that matches what you came for; you don't need to read them in
order.

## Install

The fastest path — clone the workspace and sync:

```bash
git clone https://github.com/jejjohnson/pipekit.git
cd pipekit
uv sync --all-groups
```

Once any of the packages publish to PyPI, you install only what you
need:

```bash
uv add pipekit                   # carrier-agnostic core only
uv add pipekit-cycle             # adds time-stepping / DA
uv add pipekit-train             # adds training pipelines
uv add 'pipekit-train[equinox]'  # plus the Equinox backend adapter
```

For the full per-package install matrix (every extra spelled out)
see [Installation](installation.md).

## Your first pipeline

A pipeline is a chain of `Operator`s. Compose them with `|`:

```python
from pipekit import Operator, Tap


class Scale(Operator):
    def __init__(self, factor: float) -> None:
        self.factor = factor

    def _apply(self, x):
        return x * self.factor


pipe = Scale(2.0) | Tap(print, name="log") | Scale(3.0)
result = pipe(5.0)
# Prints: 10.0
# result == 30.0
```

The `Tap` is identity-with-side-effect — the carrier passes through
unchanged, but `print` fires. Use it for logging, metrics, debug
breakpoints. Sister: `Sink` (same machinery, different intent —
persistence).

For non-linear flows use `Graph` with named nodes; see
[Concepts](concepts.md).

## Your first DA cycle (`pipekit-cycle`)

A forecast-only loop:

```python
import pipekit_cycle as pc


forecast = pc.Cycle(
    step_op=pc.CallableForward(my_step_fn, dt=3600.0),
    n_steps=24,
    save_history=True,
)
final_carrier, final_state = forecast(initial_carrier, initial_state)
trajectory = forecast.history  # list of (carrier, state) per saved step
```

A full data-assimilation cycle requires an algorithm library to
satisfy the `AnalysisStep` protocol — for example
`filterx.adapters.pipekit.EnKFAnalysis`. The pipekit-cycle side stays
algorithm-agnostic:

```python
from filterx.adapters.pipekit import EnKFAnalysis

da = pc.EnsembleDACycle(
    forward_model=my_forward_model,
    obs_op=pc.IdentityObs(),
    analysis_step=EnKFAnalysis(inflation=1.05),
    obs_source=my_obs_source_op,
    n_steps=24,
    n_members=40,
)
members, state = da(initial_members, pc.DAState(obs_err_cov=R))
```

The full walkthrough lives in [tutorials/da-cycle.md](tutorials/da-cycle.md).

## Your first trained model (`pipekit-train`)

Train a tiny MLP regressor on synthetic data, end-to-end:

```python
import jax
import equinox as eqx

from pipekit_train import IterableDataset, MSE, TrainingLoop
from pipekit_train.adapters.equinox import EquinoxModelOp


model_op = EquinoxModelOp(eqx.nn.MLP(
    in_size=1, out_size=1, width_size=16, depth=2, key=jax.random.key(0),
))

loop = TrainingLoop(
    model_op=model_op,
    dataset=my_dataset,                       # any TrainingDataset
    loss=MSE(),
    optimizer_config={"name": "adam", "learning_rate": 1e-2},
    max_steps=100,
    batch_size=16,
    backend="equinox",
    seed=42,
)

trained_op, artifact = loop.run()
```

`trained_op` is a `pipekit.Operator` that wraps the trained module;
it composes into any `Sequential` / `Graph` / `Cycle` for inference.
The `artifact` is a `TrainingArtifact` carrying the config hash,
dataset hash, and registry URI (when wired to a `ModelRegistry`).

The full walkthrough lives in [tutorials/train-emulator.md](tutorials/train-emulator.md).

## Your first registry round-trip (`pipekit-experiment`)

Store a trained operator by content hash, retrieve it by hash or by
tag:

```python
import pipekit_experiment as pe


registry = pe.LocalModelRegistry("/tmp/models")

h = registry.store(
    trained_op,
    name="methane_emulator_v3",
    tags={"family": "methane"},
)

# Load by hash, or by tag name:
op_by_hash = registry.load(h)
op_by_tag = registry.load("methane_emulator_v3")

# Atomic tag promotion:
registry.tag(h, "production", force=True)
```

`S3ModelRegistry` (under the `[s3]` extra) is a drop-in replacement
backed by `fsspec`.

## Closing the loop — train, register, deploy in a cycle

The reason the packages exist together: train a model in
`pipekit-train`, register it via `pipekit-experiment`, drop it into a
`pipekit-cycle` forecast as a `NeuralForward`, run a forecast that
calls your trained model every step. The handoff is one operator
swap — no rewrite.

```python
import pipekit_cycle as pc
import pipekit_experiment as pe


# Train (see above) → register
h = registry.store(trained_op, name="surrogate_v1")

# Reload at deploy time → drop into a Cycle
op = registry.load(h)
forecast = pc.Cycle(
    step_op=pc.NeuralForward(op, dt=3600.0),
    n_steps=24,
)
```

This is the same Operator interface end-to-end. The full
end-to-end notebook is at
[notebooks/pipekit_train_cycle_handoff.ipynb](notebooks/pipekit_train_cycle_handoff.ipynb).

## Where to go next

- **[Concepts](concepts.md)** — the 10-minute mental model. Read
  this if anything above felt magic.
- **[Installation](installation.md)** — per-package install matrix.
- **[Tutorials](tutorials/train-emulator.md)** — substantive
  end-to-end walkthroughs.
- **[API reference](api/pipekit.md)** — generated from docstrings.
