# pipekit-evaluate

> **Status — v0.0 / planning only.** This package is scaffolded for
> future work; the source directory currently contains no metrics.

Planned: a multidimensional evaluation framework with three orthogonal
axes:

- **Unit** (what's scored): Field / Statistic / Trajectory / Event / Budget
- **Lens** (kind of critique): Point-wise / Probabilistic / Spectral /
  Structural / Detection / Physical-constraint
- **Stage** (lifecycle): Training / Validation / Final / Monitoring

Implementation depends on `pipekit-array` (for the array reductions)
and on `xr_toolz.lagrangian` / `xr_toolz.events` (for the
trajectory- and event-flavoured units).

See [Report 11 — pipekit-evaluate](https://github.com/jejjohnson/research_journal_v2/blob/main/notes/geotoolz/master_plan/toolz_11_pipekit_evaluate.md).
