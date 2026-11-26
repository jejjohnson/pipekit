# Decisions

Each record states the decision, the alternatives it was weighed against, and why it wins.
Master plan Report 12 is the design these records refine.

## X1. The tool integrations are bridges (#304)

**Decision (26 Nov 2026):** `pipekit_experiment.adapters` is `pipekit_experiment.bridges`, with one module per tool as before (`dvc`, `hydra`, `metaflow`).
The other modules (`protocols`, `run`, `registry`, `artifacts`) keep their paths.

**Alternatives considered:**

- **Keep `adapters`.** Rejected: in this workspace "adapter" names three things, a named-array adapter in pipekit-ops, a training backend in pipekit-train before D17, and a tool integration here. pipekit-array calls code that works with another library a bridge (A30), so this package uses that word.
- **Areas for the rest of the package as well.** Rejected: its four other modules already each have one job and import only `run` among themselves; areas would move them for no reader's gain.

**Rationale:** one word, one meaning across the workspace.
The package defines no operator, so no saved state or `pipeline_hash` changes.
