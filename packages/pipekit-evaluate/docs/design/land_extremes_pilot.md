# Design — Land Extremes Pilot (the statistics-first `Task`)

> **Status — draft v0.1 (design only, no code).** A companion to
> [`benchmark_ladder.md`](benchmark_ladder.md) and
> [`ocean_pilot.md`](ocean_pilot.md). It works the staggered ladder for a
> **land-extremes** problem (temperature / wind / surface pressure /
> precipitation), where the generative story is *statistical* — a marked
> spatio-temporal point process — rather than a physical PDE. This pilot
> deliberately stresses the parts of the framework the first two did not:
> the **Event** Unit, the **Detection** Lens, the **oracle-less /
> covariate-extrapolation** fallback, and **tail** (not mean) calibration.

---

## 1. Why land extremes is the contrasting third domain

| | Plume | Ocean | **Land extremes** |
|---|---|---|---|
| Generative story | physical PDE (advection–diffusion) | physical PDE (QG/SWM) | **statistical point process** |
| What we infer | emission rate Q | gridded state | **covariate effects on tail risk** |
| Oracle | NUTS posterior | Strong-4DVar | **full-Bayes MCMC** (synthetic) / *none* (real data) |
| Dominant Lens | Probabilistic | Physical-constraint | **Detection + tail-Probabilistic** |
| Distinctive axis | — | resolution invariance | **non-stationary extrapolation** |

The scientifically meaningful quantity here is *how the tail moves with a
covariate* — e.g. the change in 100-year return level per +1 °C of global
mean surface temperature (GMST). That makes two things first-class that
the physical pilots barely touched: **events** (threshold exceedances are
points, not fields) and **extrapolation** (the policy question is the
return level at a GMST we have never observed).

**Honesty up front (green-field):** unlike plume/ocean, there is no
existing EVT or point-process package in the ecosystem. `pyrox`,
`gaussx`, and `geonnax` supply the inference and basis building blocks,
but the EVT likelihoods, the thinning simulator, and the EVT/PP scorers
are **new code**. `xrtoolz.events` (which the `pipekit-evaluate` plan
relies on for the Event Unit) is **not yet in the tree** — this pilot is
what would drive it. Treat the symbols below as *proposed* unless a
package is named.

---

## 2. Two ladders that line up

You described a **modelling-fidelity** ladder; it maps cleanly onto the
**inference** ladder. The fidelity ladder is the rung-6 "improve" axis.

```
modelling fidelity (rung-6 "improve" axis)
  GWR  ──▶  spatial hierarchical EVT  ──▶  marked spatio-temporal
  (local                (GEV/GPD with         thinned point process
   varying-              covariate params      (LGCP intensity λ(s,t|x);
   coefficient           + spatial GP field)    marks ~ GPD; declustered)
   regression)
```

```
inference ladder (per fidelity level)
  (1) simulate points/marks from the model         <- generative story
  (2) full-Bayes MCMC recover params (oracle)       <- slow, exact
  (3) emulate the expensive piece (intensity ∫λ)    <- surrogate
  (4) MCMC/SVI with the cheap likelihood            <- fast
  (5) amortized: point pattern -> posterior params  <- NPE / SBI
  (6) climb the fidelity ladder, prev rung as truth
```

GWR is the **cheap baseline estimator to beat** (and a sanity oracle in
the dense-data limit); the spatial hierarchical Bayesian model is the
**oracle**; amortized neural posterior estimation is rung 5.

---

## 3. Variables as marks; covariates as inputs

Each weather variable contributes extremes as **marks** on an event
process; the occurrence process and the mark distribution are modelled
separately.

| Variable | Extreme convention | Mark distribution | Notes |
|----------|--------------------|-------------------|-------|
| Temperature | block maxima / POT | GEV / GPD | heatwaves cluster → extremal index θ |
| Wind | peaks-over-threshold | GPD | storm clustering, declustering needed |
| Surface pressure | low-pressure minima | GPD (negated) | doubles as a covariate (storm proxy) |
| Precipitation | POT, zero-inflated | GPD amount × Bernoulli occurrence | occurrence *is* the point process |

Covariates feed the intensity λ(s, t | **x**) and the GEV/GPD parameters:

- **GMST** — the non-stationarity driver; the headline coefficient.
- **space** (lon/lat, elevation) — via `geonnax` spatial bases:
  `SphericalHarmonicEncoder`, `SlepianEncoder`, `OrthogonalRandomFeatures`.
- **time / seasonality** — via `geonnax.CyclicEncoder`, `FourierNet`.
- the spatially-varying coefficient field itself — a GP (`gaussx`) or a
  basis field (`geonnax.vssgp` / RFF), which is precisely **GWR
  generalised to a Bayesian spatial prior**.

---

## 4. L0–L4, instantiated for land extremes

| Level | Land-extremes instantiation | Tooling |
|-------|-----------------------------|---------|
| L0 | raw station records / gridded reanalysis cells | `geocatalog` discovery |
| L1 | QC'd, gap-flagged station / grid series | `xrtoolz` validation |
| L2 | **declustered threshold exceedances** = the marked point pattern | *new* declustering op (→ `xrtoolz.events`) |
| L3 | **analysis**: covariate-conditioned return-level / intensity field | rung-2/4/5 estimators |
| L4 | **projection**: return-level field at a *future* GMST | the simulator at shifted covariates |

The two arrows the ladder scores hardest:

- **L2 → L3.** Marked point pattern → posterior over covariate effects
  and the intensity/return-level field.
- **L3 → L4 (extrapolation).** Push the fitted model to GMST + ΔT and
  score predicted return levels against the simulator's known future —
  the out-of-distribution test that *is* the point of the domain.

---

## 5. The six rungs, mapped to (proposed) symbols

| Rung | Land-extremes realisation | Package / symbol |
|------|---------------------------|------------------|
| (1) simple model | simulate a marked STPP: thinned (Lewis–Shedler) inhomogeneous Poisson/LGCP intensity λ(s,t\|x); marks ~ GPD | *new* `stpp` simulator (JAX); GP field via `gaussx`, basis via `geonnax` |
| obs operator | threshold + decluster the simulated series into an event pattern; apply station masks / missingness | *new* declustering op (→ `xrtoolz.events`) |
| (2) model-based inference (**oracle**) | full-Bayes NUTS over the hierarchical EVT/LGCP model | `pyrox` `PyroxModule`/`Parameterized` + NumPyro `NUTS` |
| baseline | GWR — local varying-coefficient fit (fast, frequentist) | *new* GWR on `gaussx` weighted least squares |
| (3) emulator | surrogate for the expensive intensity integral ∫λ ds dt, or a neural log-intensity field | `geonnax` (RFF/VSSGP/SIREN basis) + `pipekit-train`/`pipekit-jax` |
| (4) emulator-based inference | NUTS/SVI with the cheap likelihood | `pyrox` SVI `AutoGuide` + emulator |
| (5) amortized predictor | point pattern → posterior over (covariate coeffs, GPD params); DeepSets/set-encoder backbone | `geonnax` encoder + amortized head; SBI-style |
| (6) improve | GWR → spatial hierarchical → marked STPP; add space–time interaction; prev rung as truth | fidelity climb (§2) |

Only the protocol wiring is shared with the other pilots; the EVT/PP
*content* (rows marked *new*) is what this pilot builds.

---

## 6. Scoring — tails, events, and extrapolation (the moat)

Mean-field RMSE is almost meaningless for extremes. The mandatory scorer
families, drawn from the existing `Unit × Lens` taxonomy:

| Lens (existing axis) | Land-extremes scorer | Status |
|----------------------|----------------------|--------|
| **Probabilistic (Statistic)** — *tail* | return-level coverage at high quantiles; threshold-weighted CRPS (twCRPS); quantile loss at τ→1; SBC of the **GMST coefficient** vs the oracle | *new* (twCRPS extends `properscoring`) |
| **Detection (Event)** | exceedance hit/miss/false-alarm (POD / FAR / CSI / extremal dependence index) on the event pattern | *new* (→ `xrtoolz.events`) |
| **Point-process residuals (Event)** | time-rescaling theorem → inter-event uniformity (KS); spatial K-function / pair-correlation vs theoretical; Voronoi/intensity reliability | *new* |
| **Covariate attribution (Statistic)** | posterior recovery + coverage of the true β_GMST (Δ return level per °C) | *new* |
| **Extrapolation (Statistic)** | predicted return levels at GMST+ΔT vs simulator truth; degradation curve in ΔT | *new* |
| **Constraint (Budget)** — weak | monotonicity (return level ↑ with GMST), spatial smoothness of the coefficient field | *new* |

Headline land-extremes question: *does a fast estimator (rung 4/5)
recover the oracle's posterior over the GMST effect, keep its
return-level intervals calibrated in the tail, pass point-process
residual diagnostics, and still predict the right return levels at a
warming level it never saw* — none of which a point-RMSE leaderboard can
see.

---

## 7. The oracle-less reality (the §4.3 fallback, made concrete)

For **synthetic** data the simulator gives known θ and a tractable
full-Bayes oracle, so the standard reference table applies. For **real**
land-extremes data there is *no* known truth and the full posterior is
often the best estimate we have. The fallback:

1. **Synthetic protocol (primary).** Known-θ coverage + SBC; rungs 4/5
   scored against the rung-2 MCMC posterior. This is where calibration is
   *proven*.
2. **Real-data protocol.** No oracle. Use held-out predictive scoring
   (twCRPS / quantile loss on withheld blocks), spatial/temporal
   block-bootstrap for uncertainty, and treat the slow full-Bayes fit as
   the reference the fast amortized estimator must reproduce. Report
   *relative* agreement, never absolute "truth".

The `Task` declares which protocol is active; the `reference` table from
`benchmark_ladder.md §4.3` already supports the known-θ-only fallback.

---

## 8. Where code lives

| Concern | Home |
|---------|------|
| Land-extremes `Task` (simulator, declustering, datasets, reference table) | `research_notebook/projects/` (new `land_extremes`, sibling to `assimilation`) |
| Marked-STPP simulator + EVT likelihoods (GEV/GPD/LGCP) | *new* `stpp`/`evtx` module — start inside the project, promote to a package if it grows |
| Covariate encoders + basis intensity fields | `geonnax` (reused) |
| Spatially-varying-coefficient prior (GWR generalised) | `gaussx` GP (reused) |
| MCMC oracle + SVI fast inference + amortized head | `pyrox` (reused) |
| Declustering + event Detection scorers | *new* → land in `xrtoolz.events` |
| EVT/PP probabilistic scorers (return-level coverage, twCRPS, residuals) | *new* → `pipekit-evaluate` `metrics/` (Event + tail-Probabilistic) |
| Protocols + runner | `pipekit-evaluate` (unchanged) |

This pilot is the natural driver to finally populate `xrtoolz.events` and
the Event-unit corner of `pipekit-evaluate` that the package README
already anticipates.

---

## 9. Build order (land-extremes slice)

```
1. 1-D non-stationary GPD simulator (Q exceedances, scale = a + b·GMST)
   + NUTS oracle in pyrox
   -> verify: oracle recovers (a, b); SBC of b ~ uniform on synthetic
2. GWR baseline estimator + return-level coverage + twCRPS scorers
   -> verify: tail coverage ~ nominal for the oracle; GWR is worse (sanity)
3. Spatial field: varying-coefficient GP prior (gaussx) + geonnax space/time bases
   -> verify: recovered coefficient field tracks the simulated field
4. Detection + point-process residual scorers (POD/FAR; time-rescaling KS)
   -> verify: a mis-specified intensity fails the residual test
5. Amortized rung (geonnax DeepSets head) + posterior-distance to oracle
   -> verify: posterior over b within tolerance of NUTS; seconds not hours
6. Marked STPP + extrapolation axis (score return levels at GMST+ΔT)
   -> verify: extrapolation degradation curve reproduces under DVC
```

Steps 1–2 are a tractable 1-D slice that proves the tail-calibration
machinery before any spatial complexity; the spatial point process only
enters at step 3+.

---

## 10. Open questions (land-extremes-specific)

1. **POT vs block maxima** as the default convention? Proposed: POT/GPD
   (it *is* a point process, so it unifies with the STPP story); offer
   block-maxima/GEV as an alternative mark model.
2. **Declustering method** (runs declustering vs intervals estimator vs
   model the extremal index θ directly). Proposed: model θ — it is itself
   a benchmark target, not a preprocessing nuisance.
3. **How much point-process machinery to build now.** Proposed: start
   with a thinned *inhomogeneous Poisson* (no GP field) at step 1, add the
   LGCP/GP intensity at step 3 — avoid the integral-emulator (rung 3)
   until the likelihood is actually the bottleneck.
4. **Where the EVT/PP code lives long-term.** Inside `land_extremes`
   first; promote to a standalone `evtx`/`stppx` package only if a second
   consumer appears (per the ecosystem's "no abstraction for single use"
   rule). Flag for decision.
5. **Real-data inclusion in the pilot.** Synthetic-only first (clean
   calibration story), or wire one real dataset (e.g. station temperature)
   to exercise the oracle-less protocol? Proposed: synthetic-only for v0.1.

---

## 11. References

- Sibling designs: [`benchmark_ladder.md`](benchmark_ladder.md),
  [`ocean_pilot.md`](ocean_pilot.md).
- Inference engine: `pyrox` README (`PyroxModule`, `Parameterized`,
  NumPyro NUTS/SVI bridge).
- Covariate bases / encoders: `geonnax` README (`CyclicEncoder`,
  `SphericalHarmonicEncoder`, `SlepianEncoder`, `OrthogonalRandomFeatures`,
  `vssgp`, `FourierNet`).
- Spatial GP / GWR prior: `gaussx`.
- Event Unit target: `pipekit-evaluate` README (Unit × Lens × Stage;
  Event depends on the not-yet-present `xr_toolz.events`).
