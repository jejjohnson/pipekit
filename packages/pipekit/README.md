<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/pipekit/main/docs/assets/logo-dark.svg">
    <img alt="pipekit" src="https://raw.githubusercontent.com/jejjohnson/pipekit/main/docs/assets/logo-light.svg" width="308">
  </picture>
</p>

**pipekit is the whole pipekit family in one install: the two foundations, pipekit-ops and pipekit-array, and the five packages built on them.**

<p align="center">
  <a href="https://jejjohnson.github.io/pipekit/"><b>Docs</b></a> ·
  <a href="https://github.com/jejjohnson/pipekit"><b>Repository</b></a> ·
  <a href="https://github.com/jejjohnson/pipekit/blob/main/docs/installation.md"><b>Installation</b></a>
</p>

Version 0.2.0.

## What you get

```bash
uv add pipekit
```

installs the seven packages below.
**The umbrella holds no code**: `import pipekit` exports no names (it only explains the old imports), so you import each package under its own name.

```python
import pipekit_array as pa
import pipekit_cycle as pc
import pipekit_ops as pk
```

Install the umbrella for an application or a notebook that uses most of the family.
A library should depend on the package it imports instead, usually `pipekit-ops` (no third-party dependencies) or `pipekit-array`; the umbrella pulls in JAX and Equinox through `pipekit-jax`.

## Where it sits

```
              your pipeline
                    │
   pipekit-cycle       time steps, DA cycles
   pipekit-train       training loops
   pipekit-experiment  registry, trackers
   pipekit-evaluate    benchmarks, scorers
   pipekit-jax         JAX / Equinox models
                    │
                    ▼  build on
   ┌─────────────────────────────────┐
   │ pipekit-array                   │
   │ named dims, any Array API       │
   │ backend                         │
   ├─────────────────────────────────┤
   │ pipekit-ops                     │
   │ Operator, Sequential, Graph;    │
   │ pure Python, no dependencies    │
   └─────────────────────────────────┘
     two foundations, each usable
     on its own
```

## The packages

| Distribution | Import | Use it for | Version |
|---|---|---|---|
| [`pipekit-ops`](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-ops) | `pipekit_ops` | **Foundation.** `Operator`, `Sequential`, `Graph` and the ready-made operators | 0.2.1 |
| [`pipekit-array`](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-array) | `pipekit_array` | **Foundation.** Named-dim array operators on any Array API backend | 0.1.1 |
| [`pipekit-cycle`](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-cycle) | `pipekit_cycle` | Time stepping and data assimilation | 0.1.0 |
| [`pipekit-experiment`](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-experiment) | `pipekit_experiment` | Model registry and run tracking | 0.1.0 |
| [`pipekit-evaluate`](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-evaluate) | `pipekit_evaluate` | Benchmarks and scoring | 0.2.0 |
| [`pipekit-train`](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-train) | `pipekit_train` | Training pipelines | 0.2.0 |
| [`pipekit-jax`](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-jax) | `pipekit_jax` | Equinox models and diffrax forward models | 0.1.0 |

Extras stay with the package that owns them, so you add them by name: `uv add "pipekit-train[equinox]"`, `uv add "pipekit-array[xarray]"`.

## Moving from `pipekit` < 0.2

Before 0.2.0, `pipekit` was the core package.
It is now `pipekit-ops` (import `pipekit_ops`); `from pipekit import Operator` raises an `ImportError` that says where it went.

## License

MIT; see [LICENSE](https://github.com/jejjohnson/pipekit/blob/main/LICENSE).
