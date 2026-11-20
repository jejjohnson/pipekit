# pipekit API reference

pipekit is a uv workspace of seven packages. Each has one page here, generated from its docstrings. The guides, tutorials, notebooks and design pages are in the [main docs](../).

| Package | Import | What it holds |
|---|---|---|
| [pipekit-ops](pipekit-ops.md) | `pipekit_ops` | The core: carrier-agnostic operators and how they compose and run |
| [pipekit-array](pipekit-array.md) | `pipekit_array` | Array-API operators on top of pipekit-ops |
| [pipekit-cycle](pipekit-cycle.md) | `pipekit_cycle` | Time-stepping, data assimilation and observation operators |
| [pipekit-experiment](pipekit-experiment.md) | `pipekit_experiment` | The boundary between pipekit pipelines and external orchestration |
| [pipekit-evaluate](pipekit-evaluate.md) | `pipekit_evaluate` | Multidimensional evaluation, and the `benchmark` subpackage |
| [pipekit-train](pipekit-train.md) | `pipekit_train` | Training pipelines for emulators and amortized inference |
| [pipekit-jax](pipekit-jax.md) | `pipekit_jax` | JAX / Equinox carrier integration |
