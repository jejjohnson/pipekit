# pipekit

[![Tests](https://github.com/jejjohnson/pipekit/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/pipekit/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/pipekit/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/pipekit/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/pipekit/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/pipekit/actions/workflows/typecheck.yml)
[![Deploy Docs](https://github.com/jejjohnson/pipekit/actions/workflows/pages.yml/badge.svg)](https://github.com/jejjohnson/pipekit/actions/workflows/pages.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com/)

Author: J. Emmanuel Johnson
Repo: [https://github.com/jejjohnson/pipekit](https://github.com/jejjohnson/pipekit)

A carrier-agnostic operator-graph framework for composable scientific
pipelines. `pipekit` is a uv workspace of five Python packages —
three implemented, two scaffolded for future work — that together
cover the L0–L4 spectrum: from raw I/O through data assimilation to
trained-model deployment.

---

## 📦 Packages

| Package              | Status        | Purpose                                                             |
|----------------------|---------------|---------------------------------------------------------------------|
| `pipekit`            | implemented   | Core: `Operator`, `Sequential`, `Graph`, control, observe, …        |
| `pipekit-array`      | scaffolded    | Array-API operators (numpy/JAX/torch backends).                     |
| `pipekit-cycle`      | implemented   | Time-stepping, DA cycles, observation/forward protocols.            |
| `pipekit-experiment` | implemented   | Content-addressed model registry, tracker protocols, tool adapters. |
| `pipekit-evaluate`   | scaffolded    | Evaluation metrics, lenses, units.                                  |

Each package has its own README under [`packages/`](packages/) with a
module inventory and a quickstart.

The master plan lives in
[`research_journal_v2/notes/geotoolz/master_plan/`](https://github.com/jejjohnson/research_journal_v2/tree/main/notes/geotoolz/master_plan):
Report 2 covers pipekit core, Report 10 covers pipekit-cycle,
Report 12 covers pipekit-experiment.

---

## 🚀 Quick start

```bash
# Prerequisites: uv (https://github.com/astral-sh/uv)
git clone https://github.com/jejjohnson/pipekit.git
cd pipekit
make install      # uv sync --all-groups + pre-commit hooks
make test         # run the full pytest suite (~300 tests)
make docs-serve   # preview the MkDocs site locally
```

Once published to PyPI, packages will be installable independently:

```bash
uv add pipekit                            # core only
uv add pipekit-cycle                      # adds time-stepping / DA
uv add pipekit-experiment                 # adds registry + tracker protocols
uv add 'pipekit-experiment[hydra]'        # plus Hydra + hydra-zen adapter
uv add 'pipekit-experiment[dvc,metaflow]' # plus DVC and Metaflow adapters
```

---

## 📂 Repository layout

```
pipekit/
├── packages/
│   ├── pipekit/                  # Core framework (no internal deps)
│   ├── pipekit-array/            # Scaffold — array-API operators (planned)
│   ├── pipekit-cycle/            # Time-stepping + DA on top of pipekit.state
│   ├── pipekit-experiment/       # Registry + tracker protocols + tool adapters
│   └── pipekit-evaluate/         # Scaffold — evaluation metrics (planned)
├── docs/                         # MkDocs documentation source
├── notebooks/                    # Jupyter notebooks
├── .github/                      # Workflows, issue templates, dependabot, …
├── pyproject.toml                # Workspace config (uv workspace, ruff, pytest)
├── uv.lock                       # Reproducible lockfile
├── Makefile                      # Self-documenting task runner
├── mkdocs.yml                    # Documentation site configuration
├── AGENTS.md                     # Standing instructions for AI coding agents
├── CLAUDE.md                     # Claude-Code-specific architecture notes
└── CHANGELOG.md                  # Auto-generated changelog (release-please)
```

The workspace root `pyproject.toml` ships no Python code — it only
declares the `[tool.uv.workspace]` membership and shared tool config.

---

## 🧰 Tooling

`pipekit` uses a modern Python stack:

| Concern              | Tool                                                                     |
|----------------------|--------------------------------------------------------------------------|
| Package manager      | [`uv`](https://github.com/astral-sh/uv) with workspace + lockfile        |
| Lint & format        | [`ruff`](https://github.com/astral-sh/ruff) (line-length 88, py312)      |
| Type checking        | [`ty`](https://github.com/astral-sh/ty)                                  |
| Tests                | `pytest` (+ `pytest-cov`)                                                |
| Docs                 | MkDocs + Material + `mkdocstrings` + `mkdocs-jupyter`                    |
| Pre-commit           | `.pre-commit-config.yaml` + `pre-commit-ci`                              |
| Releases             | [release-please](https://github.com/googleapis/release-please) (conventional commits) |
| Security             | CodeQL                                                                   |
| Dependency updates   | Dependabot                                                               |

Every concern is wired into CI on every PR. See [`.github/workflows/`](.github/workflows/)
for the full pipeline.

---

## 🪪 Conventions

- **Conventional commits** — release-please reads commit history to
  generate `CHANGELOG.md` and version bumps.
- **Google-style docstrings** — enforced by `mkdocstrings` config.
- **Surgical changes** — see [`AGENTS.md`](AGENTS.md) for the
  Karpathy-style guardrails every agent (and human) should follow.
- **Per-package lint never sufficient** — `ruff check .` runs on the
  whole repo because some lint issues only show up across packages
  (test conftest collisions, …).

---

## 📖 Further reading

- [pipekit API reference](https://jejjohnson.github.io/pipekit/api/pipekit/)
- [pipekit-cycle API reference](https://jejjohnson.github.io/pipekit/api/pipekit-cycle/)
- [pipekit-experiment API reference](https://jejjohnson.github.io/pipekit/api/pipekit-experiment/)
- [CONTRIBUTING.md](CONTRIBUTING.md) — local dev workflow
- [CODE_REVIEW.md](CODE_REVIEW.md) — review standards
- [AGENTS.md](AGENTS.md) — guidance for AI coding agents
