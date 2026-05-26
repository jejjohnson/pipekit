# pipekit

**Composable scientific pipelines. Train, register, deploy — one
operator interface end-to-end.**

`pipekit` is a uv workspace of seven Python packages that together
turn the pieces of a scientific ML workflow into composable
`Operator`s: data loading, QC, training, data assimilation, model
registry, reproducibility. The same operator that returns a trained
emulator drops into a forecast `Cycle` as a step function. One mental
model, one composition surface, end to end.

```python
from pipekit import Sequential
from pipekit_array import AssertNoNaN, MeanScalar
from pipekit_cycle import Cycle, NeuralForward
from pipekit_experiment import LocalModelRegistry

clean = AssertNoNaN() | MeanScalar(axis=-1)
out = clean(arr)

op = LocalModelRegistry("/models").load("methane_emulator_v3")
forecast = Cycle(step_op=NeuralForward(op, dt=3600.0), n_steps=24)
trajectory, _ = forecast(x0, state)
```

## Three doors

<div class="grid cards" markdown>

-   :material-rocket-launch: **[Getting Started](getting-started.md)**

    ---

    From `git clone` to your first pipeline, first DA cycle, and
    first registered model. 5 minutes of copy-paste.

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

-   :material-api: **[API Reference](api/pipekit.md)**

    ---

    `mkdocstrings`-generated reference, kept in sync with source.

-   :material-notebook: **[Notebooks](notebooks/pipekit_train_quickstart.ipynb)**

    ---

    Executed Jupyter notebooks — MLP quickstart, train → forecast
    handoff on a toy oscillator.

</div>

## The packages

| Package              | Status        | Purpose                                                              |
|----------------------|---------------|----------------------------------------------------------------------|
| `pipekit`            | ✅ implemented | Core: `Operator`, `Sequential`, `Graph`, control, observe, …         |
| `pipekit-array`      | 🚧 partial    | Array-API operators (Phase A landed; B-D pending).                   |
| `pipekit-cycle`      | ✅ implemented | Time-stepping, DA cycles, observation/forward protocols.             |
| `pipekit-experiment` | ✅ implemented | Content-addressed model registry, tracker protocols, tool adapters.  |
| `pipekit-evaluate`   | 📋 scaffolded | Evaluation metrics, lenses, units (planned).                         |
| `pipekit-train`      | ✅ implemented | Training pipelines — datasets, losses, loop, Equinox adapter.        |
| `pipekit-jax`        | ✅ implemented | `JaxModelOp` — `eqx.Module` wrapper with registry weight round-trip. |

## Links

- [GitHub repository](https://github.com/jejjohnson/pipekit)
- [Changelog](https://github.com/jejjohnson/pipekit/blob/main/CHANGELOG.md)
- [Contributing](contributing.md)
- Master plan: [`research_journal_v2/notes/geotoolz/master_plan/`](https://github.com/jejjohnson/research_journal_v2/tree/main/notes/geotoolz/master_plan)
