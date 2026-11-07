# Installation

The full per-package install matrix.
`pipekit` is a uv workspace: locally you sync the whole thing, and once the packages are published you install only the ones you need.

## Workspace install (recommended for development)

```bash
git clone https://github.com/jejjohnson/pipekit.git
cd pipekit
make install
```

`make install` runs `uv sync --all-groups --no-group train-extras` and installs the pre-commit hooks.
This pulls every package, every dev tool and every dependency group except `train-extras` (pipekit-train's `[equinox]`, `[numpyro]`, `[blackjax]` and `[cache]` extras), which we leave out because ty reports new diagnostics once it can see those libraries.
Tests, docs, type checks and pre-commit hooks are then ready.
The pipekit-train adapter tests skip without `train-extras`; to run them:

```bash
uv sync --group dev --group train-extras
uv run --group train-extras pytest packages/pipekit-train/tests
```

## Per-package install (production / end-user)

Once the packages publish to PyPI, you install only what you need:

```bash
uv add pipekit-ops                # core
uv add pipekit                    # everything below, in one install
uv add pipekit-array              # Array API operators and dim functions
uv add pipekit-cycle              # time-stepping + DA
uv add pipekit-experiment         # registry + tracker protocols
uv add pipekit-evaluate           # benchmark ladder, scorers, reports
uv add pipekit-train              # training pipelines
uv add pipekit-jax                # JaxOperator, JaxModelOp, ScanCycle
```

Until then, install a package straight from the repository, as the README's quick start does:

```bash
uv add "pipekit-ops @ git+https://github.com/jejjohnson/pipekit.git#subdirectory=packages/pipekit-ops"
```

The versions on `main` are `pipekit-ops` 0.2.1, `pipekit-array` 0.1.1, `pipekit-evaluate` and `pipekit-train` 0.2.0, `pipekit-cycle`, `pipekit-experiment` and `pipekit-jax` 0.1.0, and the `pipekit` umbrella 0.2.0.

### `pipekit-ops` (the core)

```bash
uv add pipekit-ops
```

Pure Python, with zero third-party dependencies, imported as `pipekit_ops`.
It holds the composition primitives (`Operator`, `Sequential`, `Graph`), control flow and flow limits, dict and iterable operators, observers, `Map` and its parallel presets, `Cache` and its stores, QC, and the JSON round-trip (`dumps` / `loads`).
Until 0.2.0 this package was `pipekit`; `pipekit` is now the umbrella that installs every package below, JAX included.

### `pipekit-cycle`

```bash
uv add pipekit-cycle
uv add 'pipekit-cycle[array]'    # plus pipekit-array
```

Depends on `pipekit-ops` only.
It ships time-stepping (`Cycle`, `EnsembleCycle`, `WindowedCycle`), DA cycles (`DACycle`, `EnsembleDACycle`, `SmootherCycle`, `BFNCycle`) and the runtime-checkable protocols (`ForwardModel`, `ObservationOperator`, `AnalysisStep`, and the reduced-order, ensemble and inversion seams) that algorithm libraries satisfy.

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

The core surface (registry, protocols, artifacts) is importable even when no extras are installed.
Each adapter raises a clean `ImportError` with the install hint on first use.
The `[mlflow]` and `[wandb]` extras pin those libraries for your own `ExperimentTracker`; no adapter for them ships yet.

### `pipekit-evaluate`

```bash
uv add pipekit-evaluate                # benchmark ladder, scorers, reports
uv add 'pipekit-evaluate[cycle]'       # plus pipekit-cycle
uv add 'pipekit-evaluate[xskill]'      # plus xskillscore
uv add 'pipekit-evaluate[proper]'      # plus properscoring
```

Depends on `pipekit-ops`, NumPy and SciPy.
It is pre-alpha: it ships the benchmark ladder, the `Unit × Lens × Stage` scorer taxonomy and the evaluation report.

### `pipekit-train`

```bash
uv add pipekit-train                    # core training surface (no backend)
uv add 'pipekit-train[equinox]'         # Equinox + Optax + Grain + Orbax adapter
uv add 'pipekit-train[numpyro]'         # NumPyro SVI and MCMC adapters
uv add 'pipekit-train[blackjax]'        # BlackJAX adapter
uv add 'pipekit-train[keras]'           # Keras 3 adapter (v0.3, pending)
uv add 'pipekit-train[cycle]'           # SimulationDataset over pipekit-cycle
uv add 'pipekit-train[experiment]'      # LogToExperiment callback
uv add 'pipekit-train[cache]'           # local-fs CachedDataset (zarr)
uv add 'pipekit-train[s3]'              # cloud-fs CachedDataset (zarr + fsspec)
uv add 'pipekit-train[gcs]'             # GCS variant
```

A `[catalog]` extra for geocatalog `CatalogDataset` support is documented in the pipekit-train design, but it lands only once geocatalog publishes to PyPI; it is not declared in `pyproject.toml` today.

Typical combo for a JAX user shipping reproducible training:

```bash
uv add 'pipekit-train[equinox,experiment,cycle]'
```

### `pipekit-array`

```bash
uv add pipekit-array             # operators only (raises at call if no backend)
uv add 'pipekit-array[numpy]'    # numpy >= 2.0
uv add 'pipekit-array[jax]'      # jax >= 0.4.20
uv add 'pipekit-array[torch]'    # torch >= 2.0 (needs [compat])
uv add 'pipekit-array[cupy]'     # cupy >= 13
uv add 'pipekit-array[dask]'     # dask[array] >= 2024 (partial conformance)
uv add 'pipekit-array[compat]'   # array-api-compat (required for torch dispatch)
uv add 'pipekit-array[xarray]'   # the xr.DataArray adapter
uv add 'pipekit-array[sklearn]'  # Estimator (scikit-learn on any carrier)
uv add 'pipekit-array[einx]'     # pipekit_array.einx
```

The default install brings no backend, so operators raise `TypeError` at call time if the input does not implement the Array API.
The recommended minimal install is `pipekit-array[numpy]`.

For PyTorch you need both extras:

```bash
uv add 'pipekit-array[torch,compat]'
```

`array-api-compat` is required because PyTorch tensors do not expose `__array_namespace__` natively.

### `pipekit-jax`

```bash
uv add pipekit-jax
uv add 'pipekit-jax[diffrax]'   # plus DiffraxForwardModel
uv add 'pipekit-jax[array]'     # plus to_jax for pipekit-array's KernelOp
```

Depends on `pipekit-ops`, `pipekit-cycle`, `jax`, `equinox` and `jaxtyping`.
It provides `JaxOperator` (an operator that is also an `eqx.Module`), `ScanCycle` and `Rollout` for time stepping under `jit`, and `JaxModelOp`, the `eqx.Module` wrapper that round-trips weights byte-identically through any `pipekit_experiment.ModelRegistry`.
`JaxModelOp` is the Equinox-specific upgrade of pipekit-train's in-package `EquinoxModelOp`.

## Python version

All packages require **Python 3.12+**, as every `pyproject.toml` sets.

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
uv sync --group dev            # runtime + tests
uv sync --group lint           # ruff
uv sync --group typecheck      # ty
uv sync --group docs           # mkdocs + jupyter
uv sync --group train-extras   # the pipekit-train backends, for their tests
```

## Verifying the install

```bash
make test         # full pytest suite (over 2,000 tests)
make typecheck    # ty check across all packages
make lint         # ruff check . (the whole repo)
make docs-serve   # local MkDocs preview at http://localhost:8000
```

`make precommit` runs the pre-commit hooks: the file-hygiene checks (end of file, trailing whitespace, YAML, large files), `ruff` and `ruff-format`.
See the Makefile for the full task surface (`make help`).
