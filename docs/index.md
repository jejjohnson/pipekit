# pipekit

**Composable scientific pipelines: train, register and deploy with one operator interface.**

`pipekit` is a uv workspace of seven Python packages (plus a code-free `pipekit` umbrella that installs them all).
Together they turn the pieces of a scientific ML workflow into composable `Operator`s: data loading, QC, training, data assimilation, the model registry and reproducibility.
The same operator that returns a trained emulator drops into a forecast `Cycle` as a step function, so there is one mental model and one composition surface from end to end.

The example below reloads a trained emulator by name and forecasts 24 steps with it as the forward model.
The weights come back into a skeleton of the same architecture (a model operator's config is not a faithful round-trip, so the registry keeps its weights as the data).
Here `x0` is the initial state, a vector of length 8.

```python
import equinox as eqx
import jax
from pipekit_cycle import Cycle, NeuralForward
from pipekit_experiment import LocalModelRegistry
from pipekit_jax import JaxModelOp

# Shapes: D = 8 state variables

# today (pipekit-experiment)
registry = LocalModelRegistry("/models")
# today (pipekit-jax)
# a skeleton of the trained architecture, f : ℝ^D → ℝ^D
template = JaxModelOp(
    eqx.nn.MLP(in_size=8, out_size=8, width_size=64, depth=2, key=jax.random.key(0))
)
# today (pipekit-jax)
emulator = template.from_registry(registry, "methane_emulator_v3")
# today (pipekit-cycle)
# x_t ↦ x_{t+1} = f(x_t), 24 times
forecast = Cycle(step_op=NeuralForward(emulator, dt=3600.0), n_steps=24)
final_state, _ = forecast(x0, None)                  # (D,) → (D,)
```

## Three doors

<div class="grid cards" markdown>

-   :material-rocket-launch: **[Getting Started](getting-started.md)**

    ---

    From `git clone` to your first pipeline, first DA cycle, and
    first registered model, in five minutes of copy and paste.

-   :material-book-open-page-variant: **[Concepts](concepts.md)**

    ---

    The 10-minute mental model. What's an `Operator`? Why
    carrier-agnostic? How does dispatch work?

-   :material-toolbox: **[Tutorials](tutorials/train-emulator.md)**

    ---

    End-to-end walkthroughs. Train an emulator. Run a DA cycle.
    Persist via the registry.

-   :material-package-variant: **[Installation](installation.md)**

    ---

    Per-package install matrix. Every extra spelled out.

-   :material-api: **[API Reference](api/pipekit-ops.md)**

    ---

    `mkdocstrings`-generated reference, kept in sync with source.

-   :material-notebook: **[Notebooks](notebooks/pipekit_train_quickstart.ipynb)**

    ---

    Executed Jupyter notebooks: an MLP quickstart, and the train → forecast
    handoff on a toy oscillator.

</div>

## The packages

| Package              | Status        | Purpose                                                              |
|----------------------|---------------|----------------------------------------------------------------------|
| `pipekit-ops`        | ✅ implemented | Core (`pipekit_ops`): `Operator`, `Sequential`, `Graph`, control and flow limits, dict and iterable operators, `Map`, `Cache` and stores, observe, QC. |
| `pipekit-array`      | 🚧 partial    | Array API dispatch, dim functions and their operators, `ApplyAlong`, `BatchAlong`, `DimArray`, `KernelOp`, `Estimator`; the array QC asserts, the array `Histogram` and `ModelOp` are still to come. |
| `pipekit-cycle`      | ✅ implemented | Time-stepping, DA cycles, observation/forward protocols.             |
| `pipekit-experiment` | ✅ implemented | Content-addressed model registry, tracker protocols, tool adapters.  |
| `pipekit-evaluate`   | 🚧 partial    | Benchmark ladder; `Unit × Lens × Stage` scorers; `EvaluationReport`. |
| `pipekit-train`      | ✅ implemented | Training pipelines: datasets, losses, the loop, Equinox, NumPyro and BlackJAX adapters. |
| `pipekit-jax`        | ✅ implemented | `JaxModelOp` (an `eqx.Module` wrapper with registry weight round-trip), `JaxOperator`, `ScanCycle`. |

## Links

- [GitHub repository](https://github.com/jejjohnson/pipekit)
- [Changelog](https://github.com/jejjohnson/pipekit/blob/main/CHANGELOG.md)
- [Contributing](contributing.md)
- Master plan: [`research_journal_v2/notes/geotoolz/master_plan/`](https://github.com/jejjohnson/research_journal_v2/tree/main/notes/geotoolz/master_plan)
