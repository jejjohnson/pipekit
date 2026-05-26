# Installation

The full per-package install matrix. `pipekit` is a uv workspace —
locally you sync the whole thing; once published, you install only
the packages you need.

## Workspace install (recommended for development)

```bash
git clone https://github.com/jejjohnson/pipekit.git
cd pipekit
uv sync --all-groups
```

This pulls every package, every dev tool, every dependency group.
Everything works out of the box. Tests, docs, type-check, pre-commit
hooks — all ready.

## Per-package install (production / end-user)

Once any of the packages publish to PyPI, you install only what you
need:

```bash
uv add pipekit                    # core
uv add pipekit-cycle              # time-stepping + DA
uv add pipekit-experiment         # registry + tracker protocols
uv add pipekit-train              # training pipelines
uv add pipekit-array              # Array-API operators
uv add pipekit-jax                # JaxModelOp (Equinox round-trip)
```

`pipekit-evaluate` is scaffolded; no v0.1 surface yet.

### `pipekit` core

```bash
uv add pipekit
```

Pure Python; zero third-party dependencies. The composition
primitives (`Operator`, `Sequential`, `Graph`, control flow,
observation, parallel maps, caching, QC, YAML round-trip).

### `pipekit-cycle`

```bash
uv add pipekit-cycle
```

Depends on `pipekit` only. Time-stepping (`Cycle`,
`EnsembleCycle`, `WindowedCycle`), DA cycles (`DACycle`,
`EnsembleDACycle`, `SmootherCycle`), the three runtime-checkable
protocols (`ForwardModel`, `ObservationOperator`, `AnalysisStep`)
that algorithm libraries satisfy.

### `pipekit-experiment`

```bash
uv add pipekit-experiment              # registry + protocols + artifacts
uv add 'pipekit-experiment[dvc]'       # plus DVC adapter
uv add 'pipekit-experiment[hydra]'     # plus Hydra + hydra-zen
uv add 'pipekit-experiment[metaflow]'  # plus Metaflow @step wrapper
uv add 'pipekit-experiment[s3]'        # plus fsspec-backed S3ModelRegistry
```

Combine extras as needed:

```bash
uv add 'pipekit-experiment[dvc,hydra,metaflow,s3]'
```

The core surface (registry, protocols, artifacts) is importable even
when no extras are installed. Each adapter raises a clean
`ImportError` with the install hint on first use.

### `pipekit-train`

```bash
uv add pipekit-train                    # core training surface (no backend)
uv add 'pipekit-train[equinox]'         # Equinox + Optax + Grain + Orbax adapter
uv add 'pipekit-train[lightning]'       # Lightning adapter (v0.2 — pending)
uv add 'pipekit-train[keras]'           # Keras 3 adapter (v0.3 — pending)
uv add 'pipekit-train[catalog]'         # geocatalog CatalogDataset support
uv add 'pipekit-train[cycle]'           # SimulationDataset over pipekit-cycle
uv add 'pipekit-train[experiment]'      # LogToExperiment callback
uv add 'pipekit-train[cache]'           # local-fs CachedDataset (zarr)
uv add 'pipekit-train[s3]'              # cloud-fs CachedDataset (zarr + fsspec)
uv add 'pipekit-train[gcs]'             # GCS variant
```

Typical combo for a JAX user shipping reproducible training:

```bash
uv add 'pipekit-train[equinox,experiment,cycle]'
```

### `pipekit-array`

```bash
uv add pipekit-array            # operators only (raises at call if no backend)
uv add 'pipekit-array[numpy]'   # numpy >= 2.0
uv add 'pipekit-array[jax]'     # jax >= 0.4.20
uv add 'pipekit-array[torch]'   # torch >= 2.0 (needs [compat])
uv add 'pipekit-array[cupy]'    # cupy >= 13
uv add 'pipekit-array[dask]'    # dask[array] >= 2024 (partial conformance)
uv add 'pipekit-array[compat]'  # array-api-compat (required for torch dispatch)
```

The default install brings nothing — operators raise `TypeError` at
call time if the input doesn't implement the Array API. Recommended
minimal install is `pipekit-array[numpy]`.

For PyTorch you need both extras:

```bash
uv add 'pipekit-array[torch,compat]'
```

`array-api-compat` is required because PyTorch tensors don't expose
`__array_namespace__` natively.

### `pipekit-jax`

```bash
uv add pipekit-jax
```

Carries `equinox`, `jax`, and `orbax-checkpoint` as hard
dependencies. Provides `JaxModelOp` — the `eqx.Module` wrapper that
round-trips weights byte-identically through
`pipekit-experiment.ModelRegistry`. The Equinox-specific upgrade
from pipekit-train's in-package `EquinoxModelOp`.

## Python version

All packages require **Python 3.12+**. Set by every `pyproject.toml`.

## Dev tooling

The workspace ships with:

| Concern              | Tool                                                                     |
|----------------------|--------------------------------------------------------------------------|
| Package manager      | [`uv`](https://github.com/astral-sh/uv) with workspace + lockfile        |
| Lint & format        | [`ruff`](https://github.com/astral-sh/ruff) (line-length 88, py312)      |
| Type checking        | [`ty`](https://github.com/astral-sh/ty)                                  |
| Tests                | `pytest` (+ `pytest-cov`)                                                |
| Docs                 | MkDocs + Material + `mkdocstrings` + `mkdocs-jupyter`                    |
| Pre-commit           | `.pre-commit-config.yaml` + `pre-commit-ci`                              |
| Releases             | [release-please](https://github.com/googleapis/release-please)           |

Sync individual groups for partial development environments:

```bash
uv sync --group dev          # runtime + tests
uv sync --group lint         # ruff
uv sync --group typecheck    # ty
uv sync --group docs         # mkdocs + jupyter
```

## Verifying the install

```bash
make test         # full pytest suite (~500 tests)
make typecheck    # ty check across all packages
make lint         # ruff check + format check
make docs-serve   # local MkDocs preview at http://localhost:8000
```

`make precommit` runs all four pre-commit hooks (the exact CI
checklist). See the Makefile for the full task surface (`make help`).
