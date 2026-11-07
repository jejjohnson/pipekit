# Getting Started

Five minutes from `git clone` to your first composed pipeline, first data-assimilation cycle, and first registered model.
Pick the section that matches what you came for; you don't need to read them in order.

## Install

The fastest path is to clone the workspace and sync it:

```bash
git clone https://github.com/jejjohnson/pipekit.git
cd pipekit
make install
```

`make install` syncs every dependency group except `train-extras`.
The training examples below need the Equinox backend, so add it with `uv sync --group dev --group train-extras`.

Once the packages publish to PyPI, you install only what you need:

```bash
uv add pipekit-ops               # carrier-agnostic core only
uv add pipekit-cycle             # adds time-stepping / DA
uv add pipekit-train             # adds training pipelines
uv add 'pipekit-train[equinox]'  # plus the Equinox backend adapter
uv add pipekit-jax               # JaxModelOp, for registry round-trips
```

For the full per-package install matrix (every extra spelled out) see [Installation](installation.md).

## Your first pipeline

A pipeline is a chain of `Operator`s, composed with `|`:

```python
from pipekit_ops import Operator, Tap


class Scale(Operator):
    def __init__(self, factor: float) -> None:
        self.factor = factor

    def _apply(self, x):
        # y = a x
        return x * self.factor


# today (pipekit-ops)
pipe = Scale(2.0) | Tap(print, name="log") | Scale(3.0)
result = pipe(5.0)
# Prints: 10.0
# result == 30.0
```

The `Tap` is identity with a side effect: the carrier passes through unchanged, but `print` fires.
Use it for debug prints and breakpoints.
For a log record per call, `Log()` does the same job without a closure, so it stays in the pipeline's config.
Its sister is `Sink` (the same machinery, with persistence as the intent).

For non-linear flows use `Graph` with named nodes; see [Concepts](concepts.md).

## Your first DA cycle (`pipekit-cycle`)

A forecast-only loop, where `my_step_fn(state, dt)` is your model's step:

```python
import pipekit_cycle as pc

# today (pipekit-cycle)
forecast = pc.Cycle(
    step_op=pc.CallableForward(my_step_fn, dt=3600.0),
    n_steps=24,
    save_history=True,
)
final_carrier, final_state = forecast(initial_carrier, initial_state)
trajectory = forecast.history  # list of (carrier, state) per saved step
```

A full data-assimilation cycle needs an algorithm library to satisfy the `AnalysisStep` protocol; pipekit-cycle ships none.
By convention the adapter lives at `<libname>.adapters.pipekit.<Algorithm>Analysis`, for example `filterx.adapters.pipekit.EnKFAnalysis`.
The pipekit-cycle side stays algorithm-agnostic:

```python
# not pinned (filterx)
from filterx.adapters.pipekit import EnKFAnalysis

# today (pipekit-cycle)
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

Train a tiny MLP regressor on synthetic data, end to end:

```python
import equinox as eqx
import jax

from pipekit_train import MSE, TrainingLoop
from pipekit_train.adapters.equinox import EquinoxModelOp

# today (pipekit-train)
model_op = EquinoxModelOp(
    eqx.nn.MLP(
        in_size=1,
        out_size=1,
        width_size=16,
        depth=2,
        key=jax.random.key(0),
    )
)

# today (pipekit-train)
# minimise (1/n) ∑ ‖f(x) − y‖² with Adam
loop = TrainingLoop(
    model_op=model_op,
    dataset=my_dataset,  # any TrainingDataset, e.g. an IterableDataset
    loss=MSE(),
    optimizer_config={"name": "adam", "learning_rate": 1e-2},
    max_steps=100,
    batch_size=16,
    backend="equinox",
    seed=42,
)

trained_op, artifact = loop.run()
```

`trained_op` is a `pipekit_ops.Operator` that wraps the trained module; it composes into any `Sequential`, `Graph` or `Cycle` for inference.
The `artifact` is a `TrainingArtifact` carrying the training pipeline's YAML, the dataset hash, the trained model's hash and, when a callback stored the model in a `ModelRegistry`, its registry URI.

The full walkthrough lives in [tutorials/train-emulator.md](tutorials/train-emulator.md).

## Your first registry round-trip (`pipekit-experiment`)

The registry stores an operator's config and its weights under one content hash.
A model operator's config is a debug record rather than a faithful round-trip (it is flagged `forbid_in_yaml`), so `registry.load` refuses it.
We store the trained module as a `JaxModelOp` with its weights, and reload the weights into a skeleton of the same architecture:

```python
import pipekit_experiment as pe
from pipekit_jax import JaxModelOp

# today (pipekit-experiment)
registry = pe.LocalModelRegistry("/tmp/models")

# today (pipekit-jax)
model = JaxModelOp(trained_op.module)
h = registry.store(
    model,
    name="methane_emulator_v3",
    weights=model.serialize_weights(),
    tags={"family": "methane"},
)

# today (pipekit-jax)
# a skeleton of the same architecture receives the stored weights, by hash or by name
template = JaxModelOp(
    eqx.nn.MLP(in_size=1, out_size=1, width_size=16, depth=2, key=jax.random.key(1))
)
op_by_hash = template.from_registry(registry, h)
op_by_name = template.from_registry(registry, "methane_emulator_v3")

# today (pipekit-experiment)
# atomic tag promotion
registry.tag(h, "production", force=True)
```

Operators whose config is the whole story (a fitted scaler, a threshold) come back with `registry.load(h)` directly.
`S3ModelRegistry` (under the `[s3]` extra) is a drop-in replacement backed by `fsspec`.

## Closing the loop: train, register, deploy in a cycle

This is the reason the packages exist together.
Train a model in `pipekit-train`, register it through `pipekit-experiment`, and drop it into a `pipekit-cycle` forecast as a `NeuralForward`, which calls the trained model every step.
The handoff is one operator swap, with no rewrite.

```python
import pipekit_cycle as pc

# Train (see above), then register (see above): h is the stored hash

# today (pipekit-jax)
# reload at deploy time into the skeleton
op = template.from_registry(registry, h)
# today (pipekit-cycle)
# x_t ↦ x_{t+1} = f(x_t), 24 times
forecast = pc.Cycle(
    step_op=pc.NeuralForward(op, dt=3600.0),
    n_steps=24,
)
```

This is the same Operator interface end to end.
The full end-to-end notebook is at [notebooks/pipekit_train_cycle_handoff.ipynb](notebooks/pipekit_train_cycle_handoff.ipynb).

## Where to go next

- **[Concepts](concepts.md)**: the 10-minute mental model. Read this if anything above felt magic.
- **[Installation](installation.md)**: the per-package install matrix.
- **[Tutorials](tutorials/train-emulator.md)**: substantive end-to-end walkthroughs.
- **[API reference](api/pipekit-ops.md)**: generated from docstrings.
