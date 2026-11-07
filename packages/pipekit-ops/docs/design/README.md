---
status: draft
version: 0.1.0
---

# pipekit-ops Design Doc: the missing operators and execution

**pipekit-ops adds the plumbing that real pipelines write as `Lambda` today: dict and list handling, flow limits, one execution story from serial to distributed, a disk cache, and serialisable observers and checks.**
**It never inspects a carrier and never takes a third-party dependency.**

This doc set covers what was missing from `pipekit_ops` after PR #199 moved the core into `packages/pipekit-ops/`.
It was written against main `903b0c3`, and every claim about the code at that commit names the source file it was checked against.
Phases P1 to P5 of the roadmap have since shipped, in #209, #210, #211, #215, #216 and #217 (pipekit-ops 0.2.1); only P6, the async-aware wrappers, is still deferred.
The pages now mark what ships as `today`, and where the build departed from the design, an "As built" note says what was built and why; [decisions.md](decisions.md) collects them.

## Structure

```
docs/design/
├── README.md          # This file: index, reading order, sources.
├── vision.md          # The decision, the boundaries, the running example end to end.
├── api/
│   ├── README.md      # The catalogue: one table per group
│   │                  # (Operator · Signature · Semantics · Status).
│   ├── operators.md   # Dict plumbing, iterables, flow, cache + store,
│   │                  # observe + QC: semantics and sketches.
│   └── execution.md   # The main section: Map + executors, ordering,
│                      # errors, backpressure, cancellation, chunking,
│                      # pickling, async, JAX tracers, Prefetch.
├── toolz.md           # Parity table: every toolz name → pipekit-ops status.
├── decisions.md       # ADRs O1 to O16: Recommended / Alternative,
│                      # then the as-built notes.
└── roadmap.md         # Testing gates with pass conditions, phases,
                       # risks, fallbacks and measured results.
```

## Reading order

1. **[vision.md](vision.md)**: what pipekit-ops adds, what it never does, and the Sentinel-2 example that runs through every page.
2. **[api/README.md](api/README.md)**: the whole catalogue at a glance.
3. **[api/execution.md](api/execution.md)**: the execution design, the largest change.
4. **[api/operators.md](api/operators.md)**: the other groups in detail.
5. **[toolz.md](toolz.md)**: where each toolz function lands, and where toolz changed the plan.
6. **[decisions.md](decisions.md)**: the commitments, each with its alternative.
7. **[roadmap.md](roadmap.md)**: the order of work, the tests that close each phase, and what they measured.

## Sources

- The pipekit-ops source at main `903b0c3`, in `packages/pipekit-ops/src/pipekit_ops/`, and, for the as-built notes, the source on main after #217 and the descriptions of the six implementing pull requests.
- The background page `docs/background/toolz.md`, the tour `docs/core-tour.md`, `docs/patterns.md` and `docs/concepts.md`.
- The master plan, Report 2 (pipekit core): Groups A to N, with Group K reserved for a disk-cache backend (see the module docstring of `packages/pipekit-ops/src/pipekit_ops/__init__.py`).
- The sibling design sets `packages/pipekit-array/docs/design/` (ADR A5 on `BatchedMap`) and `packages/pipekit-train/docs/design/`.
- [pytoolz/toolz](https://github.com/pytoolz/toolz) at commit `451af60dec590a6010e2babdbf391ea8f815122f` (2026-09-18, tag 1.2.0): `toolz/itertoolz.py`, `functoolz.py`, `dicttoolz.py`, `recipes.py`, `curried/__init__.py` and `sandbox/parallel.py`, read on 2026-11-06.
- The Python standard library: `concurrent.futures` and `asyncio`, checked on CPython 3.13.

Where the agreed gap list and toolz disagree, [decisions.md](decisions.md) records the choice.
