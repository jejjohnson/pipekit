# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`pipekit` is a uv workspace of seven Python packages that together
provide a carrier-agnostic operator-graph framework for composable
scientific pipelines. Built with Python 3.12+, uv, pytest, and MkDocs.

The packages, in dependency order:

| Package              | Status        | Purpose                                                                       |
|----------------------|---------------|-------------------------------------------------------------------------------|
| `pipekit`            | implemented   | Carrier-agnostic core: `Operator`, `Sequential`, `Graph`, control, observe, … |
| `pipekit-array`      | partial       | Array-API operators (Phase A: namespace dispatch + MeanScalar + Stack/Concat). Phases B-D pending. |
| `pipekit-cycle`      | implemented   | Time-stepping, DA cycles, observation/forward protocols.                      |
| `pipekit-experiment` | implemented   | Content-addressed model registry, tracker protocols, DVC/Hydra/Metaflow.      |
| `pipekit-evaluate`   | scaffolded    | Evaluation metrics, lenses, units. Planned, not implemented.                  |
| `pipekit-train`      | implemented   | Training pipelines + Equinox adapter. Design at `packages/pipekit-train/docs/design/`. |
| `pipekit-jax`        | implemented   | `JaxModelOp` — eqx.Module wrapper with weight-blob round-trip through `ModelRegistry`. |

The master plan lives in
[`research_journal_v2/notes/geotoolz/master_plan/`](https://github.com/jejjohnson/research_journal_v2/tree/main/notes/geotoolz/master_plan):
report 2 covers pipekit core, 10 covers pipekit-cycle, 11 covers
pipekit-train, 12 covers pipekit-experiment.

## Common Commands

```bash
make install              # uv sync --all-groups + pre-commit hooks
make test                 # uv run pytest -v (all packages)
make format               # ruff format . && ruff check --fix .
make lint                 # ruff check .
make typecheck            # ty check across all package src dirs
make precommit            # pre-commit run --all-files
make docs-serve           # local MkDocs preview
```

### Running a single test

```bash
uv run pytest packages/pipekit/tests/test_observe.py::test_tap_passes_through -v
```

### Pre-commit checklist (all four must pass)

```bash
uv run pytest -q                                    # Tests
uv run --group lint ruff check .                    # Lint — ENTIRE repo
uv run --group lint ruff format --check .           # Format — ENTIRE repo
uv run --group typecheck ty check packages/pipekit/src/pipekit \
    packages/pipekit-cycle/src/pipekit_cycle \
    packages/pipekit-experiment/src/pipekit_experiment   # Typecheck
```

**Critical**: Always lint/format with `.` (repo root). CI runs
`ruff check .` which includes every package's `tests/` and `scripts/`.

## Architecture

### Workspace layout

```
packages/
├── pipekit/                  # Core framework (no internal deps)
│   ├── src/pipekit/
│   │   ├── _base/            # Operator, Sequential, Graph (the foundations)
│   │   ├── blocks.py         # Identity, Const, Lambda, Sink
│   │   ├── compose.py        # pipe, compose, juxt, complement
│   │   ├── control.py        # Branch, Switch, Try, Coalesce, Retry
│   │   ├── observe.py        # Tap, Snapshot, ShapeTrace, Profile, Histogram
│   │   ├── combine.py        # Fanout
│   │   ├── cache.py          # Cache, Memoize
│   │   ├── qc.py             # Quarantine, AssertShape, AssertDType, …
│   │   ├── signature.py      # Signature (shape inference)
│   │   ├── parallel.py       # ThreadMap, ProcessMap, AsyncMap, BatchedMap
│   │   ├── serial.py         # dumps, loads, register, loads_sandboxed
│   │   └── state.py          # StatefulOperator, CarryState
│   └── tests/
├── pipekit-array/            # Array-API operators (Phase A landed; B-D pending)
├── pipekit-cycle/            # Time-stepping + DA on top of pipekit.state
│   ├── src/pipekit_cycle/
│   │   ├── cycle.py          # Cycle, EnsembleCycle, WindowedCycle, Recurrence
│   │   ├── da.py             # DACycle, EnsembleDACycle, SmootherCycle
│   │   ├── protocols.py      # ForwardModel, ObservationOperator, AnalysisStep
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
├── pipekit-evaluate/         # SCAFFOLD ONLY (no implementation yet)
└── pipekit-train/            # SCAFFOLD — Lightning/Keras adapter stubs +
                              # Protocols + JSONLWriter. Full design at
                              # packages/pipekit-train/docs/design/.
    └── src/pipekit_train/
        ├── loss.py           # Loss Protocol (carrier-agnostic)
        ├── callbacks.py      # Callback Protocol
        ├── writer.py         # MetricWriter Protocol + JSONLWriter
        └── adapters/
            ├── lightning.py  # v0.2 — raises NotImplementedError
            └── keras.py      # v0.3 — raises NotImplementedError
```

Each package's public API is re-exported through its
`src/<pkg>/__init__.py`. The workspace root ships no code — the
top-level `pyproject.toml` only configures `[tool.uv.workspace]`.

### Dependency rules

- `pipekit` has no internal deps; pure Python, no third-party.
- `pipekit-cycle` depends on `pipekit` only.
- `pipekit-experiment` depends on `pipekit` only; tool integrations
  are gated behind optional extras (`[dvc]`, `[hydra]`, `[metaflow]`,
  `[s3]`).
- `pipekit-train` depends on `pipekit` only; backend tools
  (Lightning, Equinox+Optax, Keras 3) and data sources (geocatalog,
  pipekit-cycle) are gated behind optional extras
  (`[equinox]`, `[lightning]`, `[keras]`, `[catalog]`, `[cycle]`,
  `[experiment]`).
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
4. `mkdocs-jupyter` renders pre-executed `.ipynb` with `execute: false`.

Figures render inline via `plt.show()` — do **not** use `savefig` or
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
- Surgical changes only — don't refactor adjacent code or add
  docstrings to unchanged code.
- Operators carrying user closures (`Lambda`, `Tap`, `Branch`, …)
  set `forbid_in_yaml: ClassVar[bool] = True`; the same flag doubles
  as the pre-deployment pickleability lint via
  `pipekit.check_pickleable`.

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
