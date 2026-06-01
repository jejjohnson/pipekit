# Design — Ocean Pilot (a second `Task`)

> **Status — draft v0.1 (design only, no code).** A companion to
> [`benchmark_ladder.md`](benchmark_ladder.md). It works the same staggered
> ladder for a sea-surface ocean problem (SSH / SST / SSS / ocean colour)
> to show that **a new domain joins the benchmark by shipping one `Task`
> implementation** — every protocol, scorer, and runner is reused.

---

## 1. Why ocean is the right second domain

The plume pilot proved the ladder on a small, mostly-static inverse
problem. The ocean stresses the parts plume did not:

- **Multi-channel observables** — SSH, SST, SSS, ocean colour, each with
  its own observation operator, noise model, gap structure, and physical
  constraint. (Plume was essentially one channel.)
- **Strongly dynamical** — the "world right now" (L3) is the analysis of
  a chaotic dynamical system, so the forecast (L4) and its lead-time
  skill matter. This is where `somax` lives.
- **Discretization as a first-class axis** — fields live on Arakawa
  C-grids at a chosen resolution; a credible emulator must be
  resolution-aware. This adds a benchmark axis plume didn't have.

Crucially, almost none of this needs new framework code: the
`research_notebook/projects/assimilation` project already runs the ladder
for Lorenz systems with `vardax` (Strong-4DVar oracle, `AmortizedPosterior`,
`FourDVarNet`) and its own `benchmark.py` harness. The ocean `Task`
generalises that harness from Lorenz ODEs to `somax` PDE fields.

---

## 2. The ocean "world" — state → observables

The generative story (rung 1) is a `somax` dynamical run. The model state
(QG streamfunction ψ / PV q, or shallow-water (h, u, v)) is the latent
truth; the four observables are **observation operators** applied to it.

```
            somax state  (ψ / q  or  h, u, v;  + advected tracers)
                  │
   ┌──────────────┼───────────────┬───────────────┬───────────────┐
   ▼              ▼               ▼               ▼               ▼
  SSH η         SST T          SSS S        ocean colour C     (velocity)
 η = f0/g·ψ   advected       advected      advected +        geostrophic
              tracer         tracer        reaction, ≥0      diagnostic
   │              │               │               │
   ▼              ▼               ▼               ▼
 nadir +       IR / µwave     L-band         VIS radiometer
 SWOT swath    (cloud gaps)   (coarse,noisy) (heavy cloud gaps)
```

Each observation operator is a `pipekit_cycle.ObservationOperator`, and
several already exist as `xrtoolz.ocn` operators:

| Observable | Forward (state → obs) | Existing `xrtoolz.ocn` support |
|------------|------------------------|--------------------------------|
| SSH η      | ψ → η (η = f0/g·ψ), then sparse along-track / swath sampling | `Streamfunction`, `CalculateSSHAlongtrack`, `ValidateSSH` |
| surface velocity | η → (u, v) geostrophic | `GeostrophicVelocities`, `AgeostrophicVelocities` |
| SST T      | tracer advection by surface flow + cloud mask | `Advection`, `Frontogenesis` |
| SSS S      | tracer advection + coarse/noisy L-band sampling | `Advection` |
| ocean colour C | advection–reaction tracer, non-negative + cloud mask | `Advection` |

So the SSH/SST/SSS/OC observation operators are *compositions of operators
that already ship*; the `Task` wires them, it doesn't invent physics.

---

## 3. L0–L4, instantiated for the ocean

| Level | Ocean instantiation | Tooling |
|-------|---------------------|---------|
| L0 | raw radiometer / altimeter counts | (out of scope for pilot) |
| L1 | calibrated, geolocated swaths / along-track | `geocatalog` discovery, `xrtoolz.rs` |
| L2 | per-pixel retrievals **with gaps** — along-track SSH, cloudy SST/OC, coarse SSS | `CalculateSSHAlongtrack`, masks |
| L3 | **analysis**: gap-filled regular grid ("world right now") | `interpolation` project (GP) / `vardax` (DA) |
| L4 | **reanalysis / forecast**: dynamical run | `somax` QG / SWM via `somax-sim` |

The two arrows that the ladder scores hardest:

- **L2 → L3 (analysis / interpolation).** Sparse, multi-channel,
  gappy obs → dense gridded state. Two estimator families (§5).
- **L3 → L4 (forecast).** Initial analysed state → future state; scored
  by lead-time skill against a `somax` truth run (Lyapunov-clock view,
  exactly as the `assimilation` project already plots).

---

## 4. The six rungs, mapped to real symbols

| Rung | Ocean realisation | Package / symbol |
|------|-------------------|------------------|
| (1) simple model | QG / SWM forward run = the truth GCM-lite | `somax` (Barotropic/Baroclinic QG, SWM); `somax-sim run` |
| obs operator | ψ→SSH, advected SST/SSS/OC, geostrophic (u,v); add gaps + noise | `xrtoolz.ocn` ops as `ObservationOperator`s |
| (2) model-based inference (**oracle**) | Strong-4DVar with the `somax` model as the dynamical constraint; or exact GP for the static-prior analysis sub-task | `vardax` `Strong4DVar` via `pipekit_cycle.AnalysisStep`; `gaussx`/`pyrox` exact GP |
| (3) emulator | resolution-invariant neural surrogate of the `somax` forward | `geonnax.FNO` / `geonnax.SFNO`, trained via `pipekit-train` + `pipekit-jax` (`JaxModelOp`); wrap as `pipekit_cycle.NeuralForward` |
| (4) emulator-based inference | rung-2 4DVar with the emulator swapped in as `ForwardModel` | `pipekit_cycle.DACycle` + emulator |
| (5) amortized predictor | gappy multi-channel obs → analysed state (+ uncertainty), direct | `vardax.AmortizedPosterior` / `FourDVarNet`; `geonnax.UNet`/`FNO` backbone |
| (6) improve | barotropic QG → baroclinic multi-layer QG (higher fidelity), or QG → SWM; re-score | `somax` model swap; previous rung as truth |

The same `Estimator` protocol covers rungs 2/4/5; the only ocean-specific
code is the `Task` (state, observation operators, datasets) and the
per-variable scorer selection.

---

## 5. Two analysis flavours — both are `Estimator`s

The `interpolation` project already lays out the two stacks for L2→L3 SSH
mapping; in the benchmark they are simply two `Estimator` implementations
scored against the same reference:

- **Static-prior GP** (`gaussx` + `pyrox` pathwise sampler): linear
  inverse problem, closed-form posterior + Matheron sampling. Natural
  **oracle for the static-prior sub-task** (exact GP), and a fast
  approximate variant (RFF) is a separate `Estimator`.
- **Dynamical-prior variational DA** (`vardax`: 3DVar / 4DVar / DYMOST /
  BFN-QG / 4DVarNet): the prior is the `somax` QG dynamics. **Strong-4DVar
  is the oracle** for the dynamical sub-task; `AmortizedPosterior` and
  `FourDVarNet` are the rung-5 fast estimators scored against it.

This directly realises the §4.3 reference table from `benchmark_ladder.md`:
rung-5 estimators are scored against the rung-2 oracle posterior, not just
against the known `somax` truth.

---

## 6. Multi-channel scoring (where the ocean moat is)

Each observable gets its own scorers, drawn from the existing
`Unit × Lens` taxonomy. The ocean adds **physical-constraint** and
**detection** lenses that plume barely exercised:

| Lens (existing axis) | Ocean scorer | Existing support |
|----------------------|--------------|------------------|
| Point-wise (Field) | per-variable RMSE/bias for SSH/SST/SSS/OC | `xrtoolz.geo` metrics, `xskillscore` |
| Spectral (Statistic) | SSH / KE wavenumber spectra — catch over-smoothing | `xrtoolz.geo` spectral; `KineticEnergy`, `EddyKineticEnergy` |
| Probabilistic (Statistic) | coverage / CRPS / SBC; posterior-distance to Strong-4DVar oracle | `properscoring`, `vardax.simulation_based_calibration`, `vardax.assert_posterior_agreement` |
| **Physical-constraint (Budget)** | geostrophic balance residual; near-zero surface divergence; PV consistency; tracer mass conservation; ocean-colour non-negativity | `GeostrophicVelocities`, `Divergence`, `PotentialVorticityBarotropic`, `Advection` |
| **Detection (Event)** | eddy detection (Okubo–Weiss), front detection (frontogenesis) — hit/miss on coherent structures | `OkuboWeiss`, `Frontogenesis` |

The headline ocean question: *does a fast estimator (rung 4/5) reconstruct
SSH that (a) matches the oracle posterior, (b) still satisfies geostrophy
and conserves tracer mass, and (c) preserves the eddy field and KE
spectrum* — not merely a low gridded RMSE. The constraint and detection
lenses are precisely what a pure-RMSE leaderboard misses.

---

## 7. Ocean-specific benchmark axes

Beyond the `level × rung` matrix, the ocean `Task` exposes three sweep
axes that the report should stratify over:

1. **Discretization / resolution invariance.** Train the emulator (rung 3)
   at 64², evaluate at 128² (and vice-versa). `geonnax.FNO`/`SFNO` are
   resolution-invariant by construction, so this is a *measurable claim*,
   not an assumption — a first-class benchmark axis.
2. **Forecast lead-time skill.** Free-forecast from the analysed state and
   plot RMSE(t) over many Lyapunov times — the exact view the
   `assimilation` project already produces; reused verbatim.
3. **Gappy-obs robustness.** Sweep cloud-cover fraction / along-track
   density / SWOT-vs-nadir sampling and watch each rung degrade. The
   amortized rung's robustness here is its main selling point.

---

## 8. Where code lives (ocean specifics)

Same split as `benchmark_ladder.md §7`; the ocean-specific homes:

| Concern | Home |
|---------|------|
| Ocean `Task` (state, obs operators, datasets, reference table) | `research_notebook/projects/` (new `ocean_benchmark`, sibling to `assimilation`/`interpolation`) |
| Dynamics (rung 1) | `somax` |
| Observation operators + physical-constraint / detection lenses | `xrtoolz.ocn` (reused) |
| Oracle + amortized estimators | `vardax` adapters satisfying `pipekit_cycle.AnalysisStep` |
| Emulator (rung 3) | `geonnax` + `pipekit-train`/`pipekit-jax` |
| Static-prior analysis estimator | `gaussx` + `pyrox` |
| Protocols, scorers, runner | `pipekit-evaluate` (unchanged — this is the point) |

The `assimilation/benchmark.py` harness (`assim_batch`, `free_forecast`,
`assemble_full_trajectory`, `MethodResult` with `rmse_trace(t)`) is the
concrete prototype to generalise into `pipekit-evaluate`'s `Benchmark`.

---

## 9. Build order (ocean slice)

```
1. Ocean Task: somax barotropic-QG truth + SSH-only obs operator (along-track)
   -> verify: Benchmark.run() scores Strong-4DVar (oracle) vs somax truth
2. Add static-prior GP estimator (gaussx/pyrox) for L2->L3 SSH analysis
   -> verify: exact GP ~matches oracle on a dense-obs limit
3. Physical-constraint lenses: geostrophy residual + surface divergence
   -> verify: a non-geostrophic reconstruction is flagged
4. Emulator rung (geonnax FNO) + resolution-invariance axis
   -> verify: FNO trained at 64 deg scores within tolerance at 128
5. Amortized rung (vardax AmortizedPosterior) + posterior-distance + SBC
   -> verify: SBC ~uniform vs oracle; posterior-distance below threshold
6. Add SST/SSS/OC channels + detection lenses (Okubo-Weiss, frontogenesis)
   -> verify: multi-channel BenchmarkReport reproduces under DVC
```

Only step 1 and the channel/scorer wiring are new; steps 2–5 reuse
existing `gaussx`/`geonnax`/`vardax` code through the protocols.

---

## 10. Open questions (ocean-specific)

1. **State representation for the oracle.** 4DVar over ψ/q vs over (h,u,v)
   — pick per `somax` model, or standardise on ψ for QG? Proposed: follow
   the `somax` model's native state; the `Task` declares it.
2. **SSH ↔ ψ constant.** f0/g varies with latitude; for a β-plane QG box
   use a reference f0 (matches `Streamfunction(g=…, f0=…)`), or full
   variable-f? Proposed: reference f0 for the pilot, document the
   approximation.
3. **Ocean-colour reaction term.** Pure advection (conservative tracer) for
   the pilot, or a minimal NPZ-style source? Proposed: advection-only
   first; reaction is a rung-6 "improve" upgrade.
4. **Shared truth catalogue.** Should `somax` truth runs be registered once
   in `pipekit-experiment` and reused across rungs (content-addressed), so
   every estimator sees byte-identical truth? Proposed: yes.

---

## 11. References

- Sibling design: [`benchmark_ladder.md`](benchmark_ladder.md).
- Dynamics: `somax` README (QG / SWM model zoo, `somax-sim`).
- Observation operators + physical lenses: `xrtoolz/src/xrtoolz/ocn/operators.py`.
- Oracle + amortized + SBC: `research_notebook/projects/assimilation/README.md`
  (`vardax`, `benchmark.py`, `simulation_based_calibration`,
  `assert_posterior_agreement`).
- Static-prior analysis: `research_notebook/projects/interpolation/README.md`
  (`gaussx`/`pyrox` pathwise SSH).
- Emulators: `geonnax` README (`FNO`, `SFNO`, `UNet`).
