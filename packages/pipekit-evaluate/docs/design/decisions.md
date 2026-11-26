# Decisions

Each record states the decision, the alternatives it was weighed against, and why it wins.
[benchmark_ladder.md](benchmark_ladder.md) is the design these records refine.

## E1. Three areas, and the shared protocols at the bottom (#298)

**Decision (26 Nov 2026):** the package is `scoring`, `reports` and `benchmark`, and an area imports only from the ones before it.
`Estimate`, `CovarianceEstimate`, `Scorer` and `ReferenceSource` (with `as_sources`) move from the benchmark into `scoring`, because the scorers and the report use them as much as the benchmark does.
A name is imported from the package or from its one area.

**Alternatives considered:**

- **Keep `benchmark` as a subpackage and the rest as flat modules.** Rejected: `report` and `scorers` imported those four names from inside `benchmark`, and `benchmark.run` imported `report`, so the two halves of the package imported each other.
- **Re-export the moved names from `benchmark` too.** Rejected: a name with two homes is two names to keep in step; `from pipekit_evaluate import Estimate` keeps working.

**Rationale:** the scorer taxonomy is the base the benchmark builds on, as [benchmark_ladder.md](benchmark_ladder.md) §4 says; the layout now says it too.
The package defines no operator, so no saved state or `pipeline_hash` changes.
pipekit-ops (O17), pipekit-array (A30) and pipekit-train (D17) are areas too.
