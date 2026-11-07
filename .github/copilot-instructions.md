# Copilot Instructions

## Project Overview

- **Python**: 3.12+
- **Package Manager**: uv, as a workspace of eight packages
- **Layout**: `packages/<pkg>/src/<pkg_module>/` per package (for example `packages/pipekit-ops/src/pipekit_ops/`)
- **Testing**: pytest
- **Docs**: MkDocs + Material + mkdocstrings + mkdocs-jupyter

The packages are the core `pipekit-ops`, its six sister packages (`pipekit-array`, `pipekit-cycle`, `pipekit-experiment`, `pipekit-evaluate`, `pipekit-train`, `pipekit-jax`) and the code-free `pipekit` umbrella.
`CLAUDE.md` has the module layout and the dependency rules.

## Build & Test Commands

```bash
make install     # uv sync --all-groups except train-extras, then pre-commit hooks
make test        # Run tests (uv run pytest -v)
make lint        # Lint code (ruff check .)
make format      # Format code (ruff format + ruff check --fix)
make typecheck   # Type check (ty check on every package's src dir)
make precommit   # Run pre-commit on all files
make docs-serve  # Serve docs locally
```

## Before Every Commit: Mandatory Checklist

**All four checks must pass before any commit.** CI runs them on the entire repo (`ruff check .`), not just one package's `src/`, so always run the commands below from the repo root.

```bash
# 1. Tests: zero failures required
uv run pytest -v

# 2. Lint: run on the ENTIRE repo (includes tests/ and scripts/)
uv run --group lint ruff check .

# 3. Format check: run on the ENTIRE repo
uv run --group lint ruff format --check .

# 4. Type check: every package's src dir
make typecheck
```

> **Common pitfall**: Running `ruff check packages/pipekit-ops/src/` instead of `ruff check .` misses import-sorting errors in `tests/` and `scripts/`. The CI workflow runs `ruff check .`. Always use `.` (repo root), not a subdirectory.

## Key Directories

| Path | Purpose |
|------|---------|
| `packages/<pkg>/src/<pkg_module>/` | Source code of one package |
| `packages/<pkg>/tests/` | That package's test suite |
| `packages/<pkg>/docs/design/` | Design notes, for the larger packages |
| `docs/` | Documentation (MkDocs) |
| `notebooks/` | Jupyter notebooks |
| `scripts/` | Repo scripts (for example the README figures) |

## Behavioral Guidelines

### Do Not Nitpick
- Ignore style issues that linters/formatters catch (formatting, import order, quote style)
- Don't suggest changes to code you weren't asked to modify
- Match existing patterns even if you'd do it differently

### Always Propose Tests
When implementing features or fixing bugs:
1. Write a test that verifies the expected behavior
2. Implement the change
3. Verify the test passes

### Never Suggest Without a Proposal
Bad: "You should add validation here"
Good: "Add validation here. Proposed implementation:"
```python
if value < 0:
    raise ValueError("Value must be non-negative")
```

### Simplicity First
- No abstractions for single-use code
- No speculative features beyond what was asked
- If 200 lines could be 50, propose the simpler version

### Surgical Changes
- Only modify lines directly related to the request
- Don't refactor adjacent code
- Don't add docstrings/comments to code you didn't change
- Remove only imports/functions that YOUR changes made unused

## Plans

Plans and design documents go in `.plans/` (gitignored, never committed). Track work via GitHub issues, not committed plan files.

## PR Review Comments

When addressing PR review comments, always resolve each review thread after fixing it via the GitHub GraphQL API (`resolveReviewThread` mutation). Do not leave addressed comments unresolved. See the "Pull Request Review Comments" section in `AGENTS.md` for the exact GraphQL queries and workflow.

## Code Review

For all code review tasks, follow the guidance in `/CODE_REVIEW.md`.
