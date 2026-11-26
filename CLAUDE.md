# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`pipekit` is a uv workspace of seven Python packages, plus a code-free
`pipekit` umbrella that installs them all. Together they
provide a carrier-agnostic operator-graph framework for composable
scientific pipelines. Built with Python 3.12+, uv, pytest, mystmd and Zensical.

The packages, in dependency order:

| Package              | Status        | Purpose                                                                       |
|----------------------|---------------|-------------------------------------------------------------------------------|
| `pipekit-ops`        | implemented   | Carrier-agnostic core (`pipekit_ops`): `Operator`, `Sequential`, `Graph`, control, flow limits, dict and iterable operators, `Map`, stores and `Cache`, observe, QC. |
| `pipekit-array`      | partial       | Array-API dispatch, `MeanScalar`, `StackAlong` / `ConcatenateAlong`, `DimArray`, the dim functions and their operators, windows, `quantile` / `dot`, `KernelOp`, `Estimator`, `ApplyAlong`, `BatchAlong` (the array-shaped batched map), einx and xarray bridges; `StackAlong` and `ConcatenateAlong` take `dim=`. Pending from the v0.1 catalogue: the array QC asserts (`AssertClose`, `AssertValueRange`, `AssertNoNaN`, `AssertValidFraction`), the array `Histogram`, `ModelOp`. `Subsample` was retired in favour of `isel` with a stepped slice. |
| `pipekit-cycle`      | implemented   | Time-stepping, DA cycles, BFN, adjoint specs, observation/forward protocols.  |
| `pipekit-experiment` | implemented   | Content-addressed model registry, tracker protocols, DVC/Hydra/Metaflow adapters. |
| `pipekit-evaluate`   | partial       | Benchmark ladder (cube, protocols, reference rule), scorer taxonomy, `MetricScorer`, `EvaluationReport`. |
| `pipekit-train`      | implemented   | Training pipelines + Equinox, NumPyro and BlackJAX adapters. Design at `packages/pipekit-train/docs/design/`. |
| `pipekit-jax`        | implemented   | `JaxOperator`, `JaxModelOp` (weight round-trip through `ModelRegistry`), `ScanCycle`, `Rollout`. The diffrax forward model and the adjoint maths are in integrix. |

The master plan lives in
[`research_journal_v2/notes/geotoolz/master_plan/`](https://github.com/jejjohnson/research_journal_v2/tree/main/notes/geotoolz/master_plan):
report 2 covers pipekit core, 10 covers pipekit-cycle, 11 covers
pipekit-train, 12 covers pipekit-experiment.

## Common Commands

```bash
make install              # uv sync --all-groups except train-extras, then pre-commit hooks
make test                 # uv run pytest -v (all packages)
make format               # ruff format . && ruff check --fix .
make lint                 # ruff check .
make typecheck            # ty check on every package's src dir
make precommit            # pre-commit run --all-files
make docs-serve           # build the docs site and serve it locally
```

### Running the pipekit-train adapter tests

The Equinox and Bayesian adapter tests skip unless the `train-extras` group is
installed, as the CI test job does. `make install` leaves it out because ty
reports new diagnostics once it can see those libraries.

```bash
uv sync --group dev --group train-extras
uv run --group train-extras pytest packages/pipekit-train/tests
```

### Running a single test

```bash
uv run pytest packages/pipekit-ops/tests/test_observe.py::test_tap_passes_through -v
```

### Pre-commit checklist (all four must pass)

```bash
uv run pytest -q                                    # Tests
uv run --group lint ruff check .                    # Lint, the ENTIRE repo
uv run --group lint ruff format --check .           # Format, the ENTIRE repo
make typecheck                                      # ty on all eight src dirs
```

**Critical**: Always lint/format with `.` (repo root). CI runs
`ruff check .` which includes every package's `tests/` and `scripts/`.

## Architecture

### Workspace layout

```
packages/
├── pipekit/                  # Umbrella: no code, installs every package below
├── pipekit-ops/              # Core framework (no internal deps), import pipekit_ops
│   ├── src/pipekit_ops/
│   │   ├── carriers/         # NamedArrayAdapter and its registry, backend_of,
│   │   │                     # to_backend, Signature, guard, is_traced, on_import
│   │   ├── core/             # Operator, Sequential, Graph, StatefulOperator,
│   │   │                     # CarryState, Predictor, pipe/compose; nested_config,
│   │   │                     # require_operator, resolve_loaded, record_under,
│   │   │                     # record_moves for libraries that write operators
│   │   ├── saving/           # dumps, loads, loads_sandboxed, pipeline_hash,
│   │   │                     # stable_json, Store, MemoryStore, DirectoryStore
│   │   ├── operators/        # Identity…Sink, dicts, iterables, Fanout, Branch…Retry,
│   │   │                     # Timeout…Until, Cache, Memoize, Quarantine, Assert*
│   │   ├── execution/        # Map, ThreadMap, ProcessMap, AsyncMap, BatchedMap,
│   │   │                     # Prefetch, Fold, executor_scope, check_pickleable
│   │   ├── observe/          # Tap, Snapshot, ShapeTrace, Profile, Histogram, Log,
│   │   │                     # jit_report
│   │   └── testing/          # assert_round_trip, public_operators
│   │                         # an area imports only from the ones before it;
│   │                         # public classes are recorded under pipekit_ops (O17)
│   └── tests/
├── pipekit-array/            # Array-API operators and dim-level functions
│   ├── src/pipekit_array/
│   │   ├── arrays/           # plain arrays: floating_inputs, set_at, scan, checks,
│   │   │                     # safe roots, weighted reductions ([numpy,compat])
│   │   ├── named/            # named carriers: DimArray, isel, reduce, rename, pad,
│   │   │                     # window, stack, map_over, batch, to_rows
│   │   ├── kernels/          # KernelSpec, KernelOp, elementwise
│   │   ├── operators/        # the named functions as Operators (ISel, Reduce, …)
│   │   ├── bridges/          # xarray, einx, sklearn (one extra each)
│   │   └── testing/          # check_adapter, assert_backends_agree, …
│   │                         # an area imports only from the ones before it
│   └── tests/
├── pipekit-cycle/            # Time-stepping + DA on top of pipekit_ops.state
│   ├── src/pipekit_cycle/
│   │   ├── cycle.py          # Cycle, EnsembleCycle, WindowedCycle, Recurrence
│   │   ├── da.py             # DACycle, EnsembleDACycle, SmootherCycle
│   │   ├── bfn.py            # BFNCycle (back-and-forth nudging)
│   │   ├── adjoints.py       # DirectAdjoint, TruncatedAdjoint, … (adjoint specs)
│   │   ├── protocols.py      # ForwardModel, ObservationOperator, AnalysisStep,
│   │   │                     # + ensemble/inversion seams (ObservationNoise, …),
│   │   │                     # SupportsAdjoint
│   │   ├── obs.py            # IdentityObs, LinearObs, CallableObs, CompositeObs
│   │   ├── forward.py        # CallableForward, CompositeForward, NeuralForward
│   │   └── state.py          # DAState, IterationState, WindowState
│   └── tests/
├── pipekit-experiment/       # Registry + tracker protocols + tool adapters
│   ├── src/pipekit_experiment/
│   │   ├── protocols.py      # ExperimentTracker, ModelRegistry
│   │   ├── run.py            # Run, RunMetrics, RunArtifacts
│   │   ├── registry.py       # LocalModelRegistry, S3ModelRegistry
│   │   ├── artifacts.py      # TrainingArtifact, InferenceArtifact
│   │   └── adapters/         # dvc, hydra, metaflow (one module per tool)
│   └── tests/
├── pipekit-evaluate/         # Benchmark ladder (cube/protocols/reference/run),
│                             # taxonomy.py (Unit/Lens/Stage), scorers.py,
│                             # report.py (EvaluationReport, evaluate)
├── pipekit-jax/              # depends on pipekit-ops and pipekit-cycle
│   └── src/pipekit_jax/
│       ├── operator.py       # JaxOperator (an Operator that is an eqx.Module)
│       ├── model_op.py       # JaxModelOp
│       ├── weights.py        # WeightsMixin (weight-blob round trip)
│       ├── cycle.py          # ScanCycle, EnsembleScanCycle, Rollout
│       ├── kernel.py         # to_jax, JaxKernelOp ([array])
│       ├── pytree.py         # registers Sequential, Graph, JaxModelOp as pytrees
│       └── testing.py        # check_jax_operator
└── pipekit-train/            # Training loop + backends. Full design at
                              # packages/pipekit-train/docs/design/.
    └── src/pipekit_train/
        ├── data/             # TrainingDataset, IterableDataset, SimulationDataset,
        │                     # CachedDataset, Split, hash_config
        ├── losses/           # Loss Protocol, MSE, NLL, KL, Composite, call_loss
        ├── loop/             # TrainingLoop + backend dispatch, callbacks, writers
        ├── sweep/            # HyperSweep + ParameterGrid
        └── backends/         # one area per backend, behind its extra (D17)
            ├── equinox/      # reference backend (Equinox+Optax+Orbax+Grain),
            │                 # _model/_sharding/_state/_batches/_evaluation/_run
            ├── numpyro/      # run_svi, run_mcmc
            ├── blackjax/     # BlackjaxTask, run
            ├── keras/        # v0.3; raises NotImplementedError
            └── _bayes.py     # shared NumPyro/BlackJAX seam (NumpyroTask, …)
```

Each package's public API is re-exported through its
`src/<pkg>/__init__.py`. The workspace root ships no code; the
top-level `pyproject.toml` only configures `[tool.uv.workspace]`.

### Dependency rules

- `pipekit-ops` has no internal deps; pure Python, no third-party.
- `pipekit` (the umbrella) depends on every package and holds no code;
  no package may depend on it.
- `pipekit-array` depends on `pipekit-ops` only; backends and bridges
  are optional extras (`[numpy]`, `[jax]`, `[torch]`, `[cupy]`,
  `[dask]`, `[compat]`, `[xarray]`, `[sklearn]`, `[einx]`).
- `pipekit-cycle` depends on `pipekit-ops` only; `[array]` adds
  `pipekit-array`.
- `pipekit-experiment` depends on `pipekit-ops` only; tool integrations
  are gated behind optional extras (`[dvc]`, `[hydra]`, `[metaflow]`,
  `[s3]`, plus `[mlflow]` and `[wandb]`, which pin the libraries but
  ship no adapter yet).
- `pipekit-evaluate` depends on `pipekit-ops`, NumPy and SciPy;
  `[cycle]`, `[xskill]` and `[proper]` are optional.
- `pipekit-train` depends on `pipekit-ops` only; backend tools
  (Equinox+Optax, NumPyro, BlackJAX, Keras 3), pipekit-cycle,
  pipekit-experiment and the zarr caches are gated behind optional
  extras (`[equinox]`, `[numpyro]`, `[blackjax]`, `[keras]`,
  `[cycle]`, `[experiment]`, `[cache]`, `[s3]`, `[gcs]`).
- `pipekit-jax` depends on `pipekit-ops`, `pipekit-cycle`, JAX,
  Equinox and jaxtyping; `[array]` is optional. No package depends on
  diffrax, or on a library of the stack above pipekit (integrix, functax).
- Algorithm libraries (filterx, vardax, plumax, …) are *not* internal
  deps; they plug in by satisfying the runtime-checkable Protocols
  defined in `pipekit_cycle.protocols` and `pipekit_experiment.protocols`.

## Documentation Examples

Example notebooks live in `docs/notebooks/` as `jupytext` percent-format
`.py` files (or as pre-executed `.ipynb`). The workflow:

1. Write the `.py` source (jupytext percent format).
2. Convert and execute: `jupytext --to notebook foo.py` then
   `jupyter nbconvert --execute --inplace foo.ipynb`.
3. Delete the `.py`; the executed `.ipynb` is the committed source of
   truth.
4. List the notebook in the `toc` of `docs/myst.yml`; mystmd renders the
   committed outputs as they are, without running it.

Figures render inline via `plt.show()`; do **not** use `savefig` or
commit separate PNG files. The `.ipynb` cell outputs are the single
source of rendered figures.

## Coding Conventions

- Google-style docstrings.
- `dataclasses` for plain data carriers; `Protocol` for structural
  interfaces (`pipekit_cycle.protocols`, `pipekit_experiment.protocols`).
- Type hints on all public functions and methods.
- `from __future__ import annotations` in every module; ruff config
  pins `target-version = "py312"`.
- Pure functions where possible; side effects isolated and explicit.
- Surgical changes only; don't refactor adjacent code or add
  docstrings to unchanged code.
- Operators carrying user closures (`Lambda`, `Tap`, `Branch`, …)
  set `forbid_in_yaml: ClassVar[bool] = True`; the same flag doubles
  as the pre-deployment pickleability lint via
  `pipekit_ops.check_pickleable`.

## Plans

Plans and design documents go in `.plans/` (gitignored, never
committed). Track work via GitHub issues instead.

## PR Review Comments

When addressing PR review comments, always resolve each review thread
after fixing it via the GitHub GraphQL API (`resolveReviewThread`
mutation). Do not leave addressed comments unresolved. To obtain the
required `threadId`, first list the pull request's review threads via
the GitHub GraphQL API (see the "Pull Request Review Comments" section
in `AGENTS.md` for a minimal query and end-to-end workflow).

## Code Review

Follow the guidance in `/CODE_REVIEW.md` for all code review tasks.
