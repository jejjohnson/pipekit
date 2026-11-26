# Decisions

Each record states the decision, the alternatives it was weighed against, and why it wins.
[latent.md](latent.md) is the design of the latent-space seams.

## C1. Four areas, and a recorded module (#303)

**Decision (26 Nov 2026):** the package is `protocols`, `cycles`, `components` and `assimilation`, and an area imports only from the ones before it.
`protocols` keeps its import path, so `from pipekit_cycle.protocols import ForwardModel` is unchanged.
The adjoint specs and the carry states go with the cycles, the observation operators and forward models are the components a cycle is built from, and the DA cycles sit on top.
Each operator is recorded under `pipekit_cycle` with `record_under` (pipekit-ops O17), and so is the private `_ForwardModelAdapter`, which `Cycle` saves in its state when it wraps a plain `ForwardModel`.
`record_moves` lets a state saved under a flat module (`pipekit_cycle.da`, `pipekit_cycle.cycle`, …) load.

**Alternatives considered:**

- **Keep the flat modules and only fix the one wrong import.** Rejected: `forward` imported a private helper from `da`, so the forward models depended on data assimilation; moving that helper fixes the fault, and the areas say who each module is for, at a cost of a few import lines in vardax and filterax.
- **Record the operators under their files.** Rejected, as in O17: the module is part of `pipeline_hash`, so each later move would change every hash again.

**Rationale:** the consumers use different parts. vardax runs the assimilation cycles with the observation operators and the adjoint specs; sesmax uses `DAState` and `ForwardModel`, and filterax satisfies the protocols and its tests use `DAState`; somax, plumax and xtremax use only the protocols, which keep their path. The layout now says which part is which.
The helper that writes a component's config is in `components`, below both its users.
