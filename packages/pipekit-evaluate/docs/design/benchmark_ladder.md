# Design — The Benchmark Ladder

> **Status — draft v0.1 (design only, no code).** A proposal for the
> *orchestration* layer of `pipekit-evaluate`: how we turn the staggered
> simulation-based-inference (SBI) workflow into a reproducible,
> cross-domain benchmark. This document does **not** change the planned
> `Unit × Lens × Stage` scorer taxonomy in the package README — it sits
> on top of it.

---

## 1. Motivation

Across our favourite problems — methane plumes, ocean state, atmosphere,
land extremes — we keep re-deriving the same staircase. We have a
generative physical story, an expensive-but-exact way to invert it, and a
desire to replace that expensive inversion with something fast (an
emulator, then a direct predictor). Each rung is only trustworthy if it
reproduces the rung below it.

That staircase is the benchmark. The framework's job is to make *"rung
N's output is the ground truth for rung N+1"* a first-class,
reproducible, domain-agnostic object — and to score each rung with the
right critique (point error is not enough; we care about calibrated
posteriors and physical consistency).

The differentiator here is **not** any single model. It is the
**validation protocol and the framework** that make the comparison fair,
reproducible, and probabilistically honest.

---

## 2. Two axes that already exist in the ecosystem

### 2.1 The L0–L4 data hierarchy (the *vertical* axis)

```
L0  Unstructured obs      raw instrument output
L1  Structured obs        calibrated, georeferenced swaths
L2  Gap-filled obs        per-pixel geophysical retrievals (gaps allowed)
L3  Analysis              regular grid, gap-filled ("world right now")
L4  Reanalysis / forecast dynamical model output
```

This maps onto packages we already have: discovery/index at L0–L1
(`geocatalog`), locality at L1–L2 (`geopatcher`), gridded operators
(`xrtoolz`), DA cycles at L3–L4 (`pipekit-cycle`).

### 2.2 The inference ladder (the *horizontal* axis)

```
(1) Simple model            generative story; physics you can simulate
(2) Model-based inference    slow but exact; classical DA / inversion   <- ORACLE
(3) Model emulator           neural surrogate trained on (1)
(4) Emulator-based inference  (2) again, seconds not hours
(5) Amortized predictor       direct map observations -> posterior params
(6) Improve                   upgrade any rung, previous rung as truth
```

### 2.3 The benchmark is the product of the two

A benchmark *cell* is a `(data level, inference rung)` pair. A
**benchmark run** fills cells and scores each against its designated
reference. This `level × rung` matrix is the single mental model that
generalises across all four domains.

---

## 3. Relationship to the existing scorer taxonomy

`pipekit-evaluate` already declares three orthogonal scoring axes (see
the package README):

- **Unit** — Field / Statistic / Trajectory / Event / Budget
- **Lens** — Point-wise / Probabilistic / Spectral / Structural /
  Detection / Physical-constraint
- **Stage** — Training / Validation / Final / Monitoring

That taxonomy answers *"how do I score one prediction against one
reference?"*. The benchmark ladder is a **separate, higher layer** that
answers *"which predictions, against which references, across which
cells, reproducibly?"*. It **invokes** scorers; it does not replace them.

```
Benchmark (this doc)         orchestration: tasks, rungs, references, runs
   └── selects Scorers       Unit × Lens × Stage  (existing taxonomy)
          └── reductions     pipekit-array (planned) / numpy / scipy
```

Concretely: the ladder's "score rung 4 against the oracle's posterior"
is implemented as a **Probabilistic Lens** over a **Statistic Unit**;
"check the plume conserves mass" is a **Physical-constraint Lens** over a
**Budget Unit**. The ladder picks *which* `(Unit, Lens)` pairs are
mandatory for each cell.

---

## 4. Core protocols (proposed)

All runtime-checkable `Protocol`s, carrier-agnostic, in
`pipekit_evaluate.benchmark`. They deliberately mirror the style of
`pipekit_cycle.protocols` and reuse its symbols rather than redefining
them.

### 4.1 Reused from `pipekit-cycle` (do not redefine)

- `ForwardModel` — the rung-1 generative simulator (θ → state).
- `ObservationOperator` — state → observations (the L3/L4 → L1/L2 map).
- `AnalysisStep` — the inversion kernel that rung-2 oracles wrap
  (`filterx` / `vardax` / `plumax` adapters).

A `Task`'s simulator and observation operator are *exactly* these
protocols, so a benchmark task is assembled from the same parts a
`DACycle` is built from. No parallel hierarchy.

### 4.2 New in `pipekit-evaluate`

```python
@runtime_checkable
class Task(Protocol):
    """One scenario: prior + simulator + obs operator + reference data."""
    name: str
    domain: str                      # "plume" | "ocean" | "atmosphere" | ...

    def prior(self) -> Distribution: ...
    def simulator(self) -> ForwardModel: ...        # rung 1 (pipekit-cycle)
    def observation_operator(self) -> ObservationOperator: ...
    def datasets(self) -> Mapping[str, Any]: ...     # L0..L4 products by key

@runtime_checkable
class Estimator(Protocol):
    """Any rung that maps observations -> estimate/posterior."""
    rung: int                        # 2 | 4 | 5
    def fit(self, task: Task) -> "Estimator": ...    # no-op for rung 2
    def __call__(self, obs: Any) -> "Estimate": ...

class Estimate(Protocol):
    """A point and/or a posterior. Scorers branch on what's present."""
    def point(self) -> Any: ...
    def sample(self, n: int, *, key=None) -> Any: ...     # None if point-only
    def log_prob(self, theta: Any) -> Any: ...            # optional

@runtime_checkable
class Oracle(Protocol):
    """The designated reference Estimator for a task (usually rung 2)."""
    def reference_for(self, task: Task) -> Estimator: ...

@dataclass
class Benchmark:
    """Runs the level × rung matrix and scores each cell vs its reference."""
    task: Task
    estimators: Sequence[Estimator]
    scorers: Sequence["Scorer"]      # from the Unit × Lens taxonomy
    reference: Estimator             # default: task oracle
    # -> writes a pipekit_experiment.Run per cell
    def run(self) -> "BenchmarkReport": ...
```

### 4.3 The "previous rung is ground truth" rule

The single invariant that makes this a benchmark and not five scripts:

| Rung being scored | Reference (ground truth)                         |
|-------------------|--------------------------------------------------|
| (2) oracle        | known θ from the simulator (synthetic-truth)     |
| (3) emulator      | rung-1 simulator outputs                         |
| (4) emu-inference | **rung-2 oracle posterior**                      |
| (5) amortized     | **rung-2 oracle posterior** (and known θ)        |
| (6) improve       | the best validated rung below it                 |

This table is data, not code branches — it lives on the `Task` so each
domain can override it (e.g. a domain with no tractable oracle falls back
to known-θ coverage only).

---

## 5. Scoring: where the moat is

The ladder mandates three scorer families per cell. Each is one entry in
the existing **Lens** axis; we are specifying *which* are non-optional.

1. **Probabilistic / calibration (Probabilistic Lens).** The headline
   metric for rungs 4 and 5. Not "is the mean close?" but "is the
   posterior right?":
   - Simulation-based calibration (SBC) rank histograms; expected
     coverage vs nominal; PIT histograms.
   - Proper scores: CRPS, energy score (via the `proper` extra —
     `properscoring` — already declared in `pyproject.toml`).
   - **Posterior-distance to the oracle**: C2ST, MMD, sliced-Wasserstein
     between the rung-4/5 posterior and the rung-2 oracle posterior.
     This is the metric that says "the fast thing reproduces the slow
     thing", and almost nobody reports it.

2. **Physical consistency (Physical-constraint Lens over Budget Unit).**
   We can check conservation because the operators already exist:
   mass/flux budgets for plumes, geostrophic balance / streamfunction
   for ocean (`xrtoolz.ocn`), column-averaging-kernel consistency for
   CH₄ (`xrtoolz.atm.gas.ch4`). A prediction that fits the data but
   violates a conservation law should score badly — most ML benchmarks
   never check this.

3. **Point + spectral (Point-wise / Spectral Lens).** RMSE/bias plus
   power-spectral and structural metrics (`xrtoolz.geo` spectral/metrics)
   to catch over-smoothing — the classic failure mode of L3 gap-filling.

`xskillscore` (the `xskill` extra) covers the gridded deterministic +
ensemble verification surface; `properscoring` covers the proper scores;
the SBC/coverage/posterior-distance pieces are the genuinely new code.

---

## 6. Pilot vertical slice — plume / methane

The pilot is chosen because every rung already has an implementation in
`research_notebook/projects/plume_simulation` — we are wiring existing
parts into the protocol, not writing physics.

| Rung | Plume realisation (existing symbols)                                            |
|------|--------------------------------------------------------------------------------|
| (1)  | `gauss_plume.simulate_plume` / `gauss_puff.simulate_puff` (forward `ForwardModel`) |
| obs  | `radtran` SRF + matched-filter retrieval (L1 radiance → L2 enhancement)         |
| (2)  | `gauss_plume.infer_emission_rate` (NumPyro NUTS) — the **oracle** posterior over Q |
| (3)  | emulator of the forward model via `pipekit-train` + `pipekit-jax` (`JaxModelOp`) |
| (4)  | rung-2 inference with the emulator swapped in as `ForwardModel`                 |
| (5)  | amortized net: retrieved scene → posterior over (Q, stability class)           |
| (6)  | swap `gauss_puff` → `les_fvm` (higher-fidelity Eulerian truth) and re-score     |

Headline pilot question: *how well do rungs 4 and 5 recover the rung-2
NUTS posterior over emission rate Q (and the categorical stability
class), and do their plumes still conserve mass?*

Higher-fidelity truth is already available: `les_fvm` (finitevolX
Eulerian advection–diffusion) is the L2/L3 reference for the rung-6
"improve" step, and `gauss_puff` already has a cross-check notebook
against it.

---

## 7. Where code lives

| Concern                                   | Home                                            |
|-------------------------------------------|-------------------------------------------------|
| Protocols (`Task`/`Estimator`/`Oracle`/`Benchmark`) | `pipekit-evaluate` (`benchmark/`)     |
| Scorers (Unit × Lens × Stage)             | `pipekit-evaluate` (`metrics/`)                 |
| Simulator / obs / analysis protocols      | `pipekit-cycle` (reused, not duplicated)        |
| Runs, registry, reproducibility artifacts | `pipekit-experiment` (`Run`, `LocalModelRegistry`) |
| Domain physics + per-domain `Task` impls  | `xrtoolz` (`ocn`/`atm`/`rs`) + project packages |
| Pilot `Task` (plume) + run driver         | `research_notebook/projects/plume_simulation` (Hydra + DVC) |

Rationale: the framework stays carrier-agnostic and reusable; physics and
runnable experiments stay in their existing homes. A domain joins the
benchmark by shipping one `Task` implementation — nothing in
`pipekit-evaluate` changes.

---

## 8. Build order (vertical slice first)

Per `AGENTS.md` (Simplicity First, Goal-Driven): build the plume slice
end-to-end, then *extract* the abstraction. Do not design the protocols
in the abstract first.

```
1. Protocol stubs + a plume Task wrapping existing forward/oracle
   -> verify: Benchmark.run() executes rung 2 vs known-θ on a toy scene
2. Probabilistic scorers (SBC ranks, coverage, CRPS via properscoring)
   -> verify: SBC on the oracle is ~uniform on a well-specified toy
3. Posterior-distance scorers (C2ST / MMD / sliced-Wasserstein)
   -> verify: identical posteriors score ~0; shifted posteriors score >0
4. Emulator rung (3) + emulator-inference rung (4) via pipekit-train/jax
   -> verify: rung-4 posterior-distance to oracle below a set threshold
5. Physical-constraint scorer (mass budget)
   -> verify: a deliberately mass-violating field is flagged
6. Amortized rung (5) + the level × rung BenchmarkReport + DVC wiring
   -> verify: reproducible report regenerates byte-stable metrics
```

Only after rungs 1–6 work for plume do we add a second domain (ocean is
the natural next, given `xrtoolz.ocn` + `somax`) — and the only new code
should be its `Task`.

---

## 9. Open questions / decisions for review

1. **Posterior representation.** Standardise on samples (works for NUTS
   + most amortized nets) and treat density-returning estimators as
   optional? Proposed: yes — `Estimate.sample()` is the required surface,
   `log_prob()` optional.
2. **Where does `Distribution` (the prior type) come from?** Numic vs a
   thin protocol vs reuse NumPyro/`distrax`. Proposed: a minimal local
   `Prior` protocol (`sample`, optional `log_prob`) to avoid a hard SBI
   dependency in `pipekit-evaluate` core.
3. **Oracle-less domains.** Land extremes may have no tractable oracle.
   The reference table (§4.3) already allows a known-θ-only fallback —
   confirm that is acceptable, or do we require a designated oracle per
   domain?
4. **xskillscore vs hand-rolled reductions.** Lean on the `xskill` extra
   for gridded/ensemble verification, keep only SBC/coverage/posterior-
   distance in-house? Proposed: yes.
5. **`pipekit-array` dependency timing.** Scorer reductions ideally use
   `pipekit-array` (currently scaffolded). Until it lands, use numpy +
   the Array API directly and swap later.

---

## 10. References

- Package taxonomy: `packages/pipekit-evaluate/README.md`
  (Unit × Lens × Stage).
- Reused protocols: `packages/pipekit-cycle/README.md`
  (`ForwardModel`, `ObservationOperator`, `AnalysisStep`).
- Registry / runs: `packages/pipekit-experiment/README.md`.
- Pilot physics: `research_notebook/projects/plume_simulation/README.md`.
- Master plan: Report 11 — pipekit-evaluate.
