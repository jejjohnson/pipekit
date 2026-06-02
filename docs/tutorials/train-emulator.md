# Tutorial — Train an emulator end-to-end

Walks through training a tiny MLP emulator on synthetic regression
data with `pipekit-train`'s Equinox backend, then dropping the
trained model into a `pipekit-cycle` forecast and persisting it via
`pipekit-experiment`'s `ModelRegistry`. This is the same loop that
real emulator workflows use, just scaled down.

**Time:** ~15 minutes if you read every word, ~5 minutes if you
copy-paste.

**Prerequisites:**

```bash
uv add 'pipekit-train[equinox,experiment,cycle]'
```

## Setup

```python
import jax
import equinox as eqx
import numpy as np

from pipekit_train import (
    IterableDataset, MSE, TrainingLoop, JSONLWriter,
)
from pipekit_train.adapters.equinox import EquinoxModelOp
```

We're training to fit `y = 2x + 1` — the world's simplest regression
target, but the loop is identical for any `(x, y)` pair stream.

## Step 1 — Build the dataset

`TrainingDataset` is a `pipekit.Operator` that yields `(input,
target)` pairs and exposes a content hash. The simplest concrete
class is `IterableDataset`, which wraps a list / iterator:

```python
def synthetic_pairs(n: int = 128, seed: int = 0):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(-1, 1, size=(n, 1)).astype(np.float32)
    ys = 2.0 * xs + 1.0 + 0.01 * rng.standard_normal((n, 1)).astype(np.float32)
    return list(zip(xs, ys, strict=True))


train_data = IterableDataset(
    source=synthetic_pairs(n=128),
    content_hash="synth-2x+1-n128-seed0",  # any stable string
)
```

The `content_hash` is your reproducibility anchor — the trained
model's artifact records it so the run is replayable.

## Step 2 — Wrap an Equinox module as an Operator

`EquinoxModelOp` adapts any `eqx.Module` (or `eqx.nn.MLP`,
`eqx.nn.Conv2d`, …) so it composes as a pipekit `Operator`:

```python
model_op = EquinoxModelOp(eqx.nn.MLP(
    in_size=1,
    out_size=1,
    width_size=16,
    depth=2,
    key=jax.random.key(0),
))
```

After training the loop returns *another* `EquinoxModelOp` — same
shape, different weights. That's the train→serve handoff: an
operator goes in, an operator comes out.

## Step 3 — Configure and run the loop

```python
loop = TrainingLoop(
    model_op=model_op,
    dataset=train_data,
    loss=MSE(),
    optimizer_config={"name": "adam", "learning_rate": 1e-2},
    max_steps=100,
    batch_size=16,
    backend="equinox",
    metric_writer=JSONLWriter("metrics.jsonl"),
    seed=42,
)

trained_op, artifact = loop.run()
```

`JSONLWriter` streams per-step metrics to a JSONL file you can tail
in another terminal. The default writer is `None` (no logging).

`artifact` is a `TrainingArtifact` — a JSON-roundtrippable record of
the config hash, dataset hash, registry URI (if wired), and tracker
run id (if wired). It's the receipt for "what produced this model."

## Step 4 — Persist via `ModelRegistry`

```python
import pipekit_experiment as pe


registry = pe.LocalModelRegistry("/tmp/models")
h = registry.store(
    trained_op,
    name="toy_emulator",
    tags={"family": "synthetic-regression"},
)
print(f"Stored under content hash {h}")
```

The hash is deterministic — same trained operator config = same
hash. That property is what makes `pipekit-experiment` a *content*-
addressed registry rather than a directory of YAML files.

To swap in cloud storage just change the registry class:

```python
import fsspec
registry = pe.S3ModelRegistry("s3://my-bucket/models", fs=fsspec.filesystem("s3"))
```

(Same `store` / `load` / `tag` API.)

## Step 5 — Reload and drop into a forecast

```python
import pipekit_cycle as pc


op = registry.load("toy_emulator")  # by tag name; same shape as the original

forecast = pc.Cycle(
    step_op=pc.NeuralForward(op, dt=1.0),
    n_steps=10,
    save_history=True,
)

x0 = np.array([0.0], dtype=np.float32)
final, _ = forecast(x0, None)  # state=None is fine for a forecast-only run
print(forecast.history)  # trajectory of (carrier, state) per step
```

`NeuralForward` is the `pipekit-cycle` operator that adapts a
trained model as a `ForwardModel`. The loop runs your model 10
times in succession, carrying its own state through.

That's the full handoff — train → register → reload → forecast,
all over the same `Operator` interface.

## What you actually learned

- A pipeline is a chain of `Operator`s; `Sequential` is just
  `op1 | op2 | op3` with extra plumbing.
- `TrainingLoop` is a `StatefulOperator` whose carry-state is the
  optimizer state + step count + metrics.
- The trained model is an `Operator`, deterministically content-
  hashable, and drops into `pipekit-cycle` without any glue code.
- The reproducibility story is the `TrainingArtifact` + the
  content hash. Same hash = same model. No magic.

## Where to go next

- **[DA cycle tutorial](da-cycle.md)** — bring in observations and
  an analysis step.
- **[pipekit-train design doc](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-train/docs/design)**
  — the full architecture, ADRs, and roadmap.
- **[Notebook: quickstart MLP regression](../notebooks/pipekit_train_quickstart.ipynb)**
  — the same code with mid-train loss plots.
- **[Notebook: cycle handoff](../notebooks/pipekit_train_cycle_handoff.ipynb)**
  — the train → forecast loop on a toy oscillator.
