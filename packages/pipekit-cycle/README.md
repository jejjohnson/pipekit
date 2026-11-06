<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/pipekit/main/docs/assets/packages/pipekit-cycle/logo-dark.svg">
    <img alt="pipekit-cycle" src="https://raw.githubusercontent.com/jejjohnson/pipekit/main/docs/assets/packages/pipekit-cycle/logo-light.svg" width="389">
  </picture>
</p>

# pipekit-cycle

Time-stepping, data-assimilation cycles, and the three structural
protocols (`ForwardModel`, `ObservationOperator`, `AnalysisStep`) that
algorithm libraries satisfy without importing pipekit-cycle itself.

Built on `pipekit_ops.state` (`StatefulOperator` + `CarryState` from
pipekit core).

## Install

```bash
uv add pipekit-cycle
```

## What's shipped (v0.0.1)

| Module      | Symbols                                                                  |
|-------------|--------------------------------------------------------------------------|
| `cycle`     | `Cycle`, `EnsembleCycle`, `WindowedCycle`, `Recurrence`                  |
| `protocols` | `ForwardModel`, `ObservationOperator`, `AnalysisStep` (runtime-checkable)|
| `da`        | `DACycle`, `EnsembleDACycle`, `SmootherCycle`                            |
| `obs`       | `IdentityObs`, `LinearObs`, `CallableObs`, `CompositeObs`                |
| `forward`   | `CallableForward`, `CompositeForward`, `NeuralForward`                   |
| `state`     | `DAState`, `IterationState`, `WindowState` (canonical `CarryState`s)     |

## Quickstart — forecast only

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

## Quickstart — DA cycle

```python
from filterx.adapters.pipekit_ops import EnKFAnalysis  # algorithm library

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

`obs_source` is indexed from 0 on every call by default. Pass `obs_clock="state"` to continue from `state.cycle_count`, so that calling a cycle again on its returned state reads the next observations and gives the same analysis as one uninterrupted run.

## Algorithm library integration

Algorithm libraries (filterx, vardax, plumax, …) plug in by exposing
adapter classes that satisfy the runtime-checkable Protocols in
`pipekit_cycle.protocols`. No pipekit-cycle import is required in the
algorithm core; the conventional location is
`<libname>.adapters.pipekit`.

## Design docs

- [Latent Data Assimilation](docs/design/latent.md) — adds `Encoder`,
  `Decoder`, `LatentMap`, `LatentForwardModel` protocols and a
  `LatentDACycle` orchestrator that supports strong-latent / hybrid /
  prior-only flavours. Status: draft v0.1.

## References

Master plan: [Report 10](https://github.com/jejjohnson/research_journal_v2/blob/main/notes/geotoolz/master_plan/toolz_8_pipekit_cycle.md).
API reference: [pipekit-cycle](https://github.com/jejjohnson/pipekit/blob/main/docs/api/pipekit-cycle.md).
