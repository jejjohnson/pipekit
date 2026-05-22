# pipekit

A carrier-agnostic operator-graph framework for composable scientific pipelines.

`pipekit` is a uv workspace of five Python packages. Three are
implemented; two are scaffolded for future work.

| Package              | Status        | Purpose                                                            |
|----------------------|---------------|--------------------------------------------------------------------|
| `pipekit`            | implemented   | Core: `Operator`, `Sequential`, `Graph`, control, observe, …       |
| `pipekit-array`      | scaffolded    | Array-API operators (numpy/JAX/torch backends).                    |
| `pipekit-cycle`      | implemented   | Time-stepping, DA cycles, observation/forward protocols.           |
| `pipekit-experiment` | implemented   | Content-addressed model registry, tracker protocols, tool adapters.|
| `pipekit-evaluate`   | scaffolded    | Evaluation metrics, lenses, units.                                 |

## Installation

The packages are not yet published to PyPI; install from the
workspace checkout:

```bash
git clone https://github.com/jejjohnson/pipekit.git
cd pipekit
uv sync --all-groups
```

Once published, each package will be installable independently:

```bash
uv add pipekit                            # core only
uv add pipekit-cycle                      # adds time-stepping / DA
uv add pipekit-experiment                 # adds registry + tracker protocols
uv add 'pipekit-experiment[hydra]'        # plus Hydra + hydra-zen adapter
uv add 'pipekit-experiment[dvc,metaflow]' # plus DVC and Metaflow adapters
```

## Quickstart

A pipeline is a chain of `Operator`s. Compose them with `|`:

```python
from pipekit import Operator, Sequential, ShapeTrace


class Scale(Operator):
    def __init__(self, factor: float) -> None:
        self.factor = factor

    def _apply(self, x):
        return x * self.factor


pipe = Scale(2.0) | ShapeTrace(mode="diff_only") | Scale(3.0)
result = pipe(5.0)  # → 30.0; ShapeTrace logs the carrier shape between Scales
```

Add data assimilation:

```python
import pipekit_cycle as pc

forecast = pc.Cycle(
    step_op=pc.CallableForward(my_model, dt=3600.0),
    n_steps=24,
    save_history=True,
)
final_carrier, final_state = forecast(initial_carrier, initial_state)
trajectory = forecast.history  # list of (carrier, state) per saved step
```

Register a trained model by content hash:

```python
import pipekit_experiment as pe

registry = pe.LocalModelRegistry("/tmp/models")
h = registry.store(trained_op, name="methane_emulator_v3")
loaded = registry.load("methane_emulator_v3")  # resolves the tag
```

## Links

- [pipekit API](api/pipekit.md)
- [pipekit-cycle API](api/pipekit-cycle.md)
- [pipekit-experiment API](api/pipekit-experiment.md)
- [Contributing](contributing.md)
- [GitHub](https://github.com/jejjohnson/pipekit)
- [Changelog](https://github.com/jejjohnson/pipekit/blob/main/CHANGELOG.md)
