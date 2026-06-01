# Design — Land Extremes Pilot (the statistics-first `Task`)

> **Status — draft v0.1 (design only, no code).** A companion to
> [`benchmark_ladder.md`](benchmark_ladder.md) and
> [`ocean_pilot.md`](ocean_pilot.md). It works the staggered ladder for a
> **land-extremes** problem (temperature / wind / surface pressure /
> precipitation), where the generative story can be *statistical* — a
> marked spatio-temporal point process — or *physical* (§4). This pilot
> deliberately stresses the parts of the framework the first two did not:
> the **Event** Unit, the **Detection** Lens, the **oracle-less /
> covariate-extrapolation** fallback, and **tail** (not mean) calibration.

---

## 1. Why land extremes is the contrasting third domain

| | Plume | Ocean | **Land extremes** |
|---|---|---|---|
| Generative story | physical PDE (advection–diffusion) | physical PDE (QG/SWM) | **statistical point process** *or* physical (§4) |
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
and `somax`/`finitevolX` supply the physical generators of §4, but the
EVT likelihoods, the thinning simulator, and the EVT/PP scorers are **new
code**. `xrtoolz.events` (which the `pipekit-evaluate` plan relies on for
the Event Unit) is **not yet in the tree** — this pilot is what would
drive it. Treat the symbols below as *proposed* unless a package is named.

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

> The four variables are not statistically independent marks — they share
> a physical basis (§4) that both supplies a physically-grounded rung-1
> generator and couples them through balance/budget constraints.

---

## 4. Physical basis for the measured variables

The statistical framing treats exceedances as marks on a point process.
But every surface variable has a **physical basis** that is standard in
meteorology/hydrology yet rarely wired into ML extremes benchmarks. Each
physical model can play up to three roles in the ladder:

- **(G) generative** — an alternative, physically-grounded rung-1
  simulator (benchmark question: *does the statistical model recover
  physical-generator truth?*);
- **(P) prior** — a physically-informed prior on a parameter the
  benchmark then updates (the Clausius–Clapeyron case is the star);
- **(C) constraint** — a physical-constraint Lens in scoring (§7).

### 4.1 Per-variable physical models

| Variable | Physical model / relation | Role | Builds on |
|----------|---------------------------|------|-----------|
| **Temperature** | **Surface energy balance** Rₙ = H + LE + G — near-surface T from radiation/flux partitioning | G, C | `xrtoolz.atm`, `finitevolX` |
| | **Force-restore / bucket land-surface model** + **soil-moisture–temperature feedback** (Seneviratne et al. 2010) — the heatwave amplifier (dry soil → less LE → hotter) | G, C | new LSM op; `finitevolX` |
| | **Monin–Obukhov surface-layer similarity** — 2 m T from a model level + stability | obs operator, C | `xrtoolz.atm` |
| | diurnal-cycle model (harmonic + nocturnal exponential decay) | detrend before EVT | `geonnax.CyclicEncoder` |
| **Wind** | **Geostrophic + Ekman balance** — surface wind = reduced & rotated geostrophic wind (boundary-layer friction); ties wind ↔ pressure | G, C | reuse `xrtoolz.ocn.GeostrophicVelocities` |
| | **Gradient / cyclostrophic balance** — curved flow around lows → storm-wind extremes | G, C | `xrtoolz` kinematics |
| | **Logarithmic wind profile** (Monin–Obukhov, roughness z₀) — surface wind/gust from upper wind | obs operator | `xrtoolz.atm` |
| | **Weibull / Rayleigh** speed law from bivariate-Gaussian components | semi-physical mark | EVT module |
| **Surface pressure** | **Hydrostatic balance** dp/dz = −ρg + **hypsometric/barometric eqn** — pressure ↔ column temperature/altitude; sea-level reduction | obs operator, C | reuse `xrtoolz.ocn.LapseRate`/`BruntVaisalaFrequency` |
| | **QG height-tendency / omega equation** — cyclogenesis → pressure-minima extremes | G | `somax` QG (barotropic/baroclinic) |
| **Precipitation** | **Clausius–Clapeyron** — saturation vapour pressure ≈ +7 %/°C → physical scaling of precip extremes with T/GMST (super-CC up to ~2× for convective) | **P**, C | physically-informed prior on β_GMST |
| | **Moisture budget** P − E = −∇·(q **u**) — convergence-driven precip | G, C | `finitevolX`, `xrtoolz` |
| | **CAPE / CIN + convective quasi-equilibrium** — thermodynamic basis for convective extremes | G, C | `xrtoolz.atm` |
| | **Linear orographic precipitation** (Smith & Barstad 2004) — terrain forcing as a physics-based spatial covariate | G, covariate | `finitevolX`, `geonnax` |
| | **Self-organized criticality** (Peters & Neelin 2006) — precip onset as a critical phenomenon → physical basis for power-law/heavy tails and for the thresholded point-process framing | justifies EVT/PP | — |

### 4.2 Cross-variable couplings (compound extremes)

The same physics makes the variables *jointly* constrained — the basis
for compound events that independent marks cannot represent:

- **Soil-moisture–temperature–precipitation feedback** / land–atmosphere
  coupling hotspots (Koster et al. 2004) — compound hot-and-dry extremes.
- **Pressure ↔ wind** (geostrophic/Ekman) and **temperature ↔ pressure**
  (hydrostatic) — a *multivariate* physical-constraint Lens (§7).

### 4.3 Physically-motivated point processes (the bridge to "marked STPP")

Your "marked spatio-temporal thinned point process" already has
physics-flavored, operational incarnations — adopt their structure rather
than a generic LGCP:

- **Neyman–Scott Rectangular Pulse (NSRP)** & **Bartlett–Lewis
  Rectangular Pulse (BLRP)** rainfall models (Rodriguez-Iturbe, Cox &
  Isham 1987/88) — storm origins as a Poisson process, each spawning a
  *cluster* of rain cells (rectangular pulses) with random
  intensity/duration. This **is** a physically interpretable marked
  cluster point process — the canonical rung-1 generator for precipitation.
- **Stochastic weather generators** (Richardson/WGEN) — Markov-chain
  occurrence + gamma/exponential amounts with covariate hooks; a
  ready-made occurrence-process + mark generator.
- **Poisson-process limit of EVT** (Pickands/Smith point-process
  representation of GPD exceedances) — the theoretical reason POT *is* a
  point process; where the statistical and physical framings meet.

### 4.4 What this changes in the pilot

- **Rung 1 gains a physical option** alongside the statistical STPP:
  NSRP/BLRP for precip, `somax` QG for the pressure/wind synoptic driver,
  surface-energy-balance + force-restore for temperature. Cross-scoring a
  statistical estimator against a *physical* generator (and vice-versa) is
  a benchmark cell in its own right.
- **A physically-informed prior** on the precip GMST coefficient
  (Clausius–Clapeyron ≈ 7 %/°C); the benchmark scores whether data update
  it sensibly and whether the recovered scaling stays physical.
- **New constraint Lenses** (added to §7): CC-scaling monotonicity,
  geostrophic/hydrostatic balance residuals, moisture-budget closure,
  soil-moisture–temperature consistency.
- **Heavy reuse**: `somax` (synoptic driver), `finitevolX` (moisture
  transport, orographic, LSM PDEs), `xrtoolz.ocn`/`atm` operators
  (geostrophic balance, lapse rate, stratification — already shipped),
  `geonnax` (orographic / spatial covariate bases).

---

## 5. L0–L4, instantiated for land extremes

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

## 6. The six rungs, mapped to (proposed) symbols

| Rung | Land-extremes realisation | Package / symbol |
|------|---------------------------|------------------|
| (1) simple model | simulate a marked STPP (thinned LGCP intensity λ(s,t\|x); marks ~ GPD) **or** a physical generator (§4: NSRP/BLRP rainfall, `somax` QG synoptic driver, SEB + force-restore) | *new* `stpp` simulator (JAX); `somax`; `finitevolX`; GP field via `gaussx`; basis via `geonnax` |
| obs operator | threshold + decluster into an event pattern; station masks / missingness | *new* declustering op (→ `xrtoolz.events`) |
| (2) model-based inference (**oracle**) | full-Bayes NUTS over the hierarchical EVT/LGCP model | `pyrox` `PyroxModule`/`Parameterized` + NumPyro `NUTS` |
| baseline | GWR — local varying-coefficient fit (fast, frequentist) | *new* GWR on `gaussx` weighted least squares |
| (3) emulator | surrogate for the intensity integral ∫λ ds dt, or a neural log-intensity field | `geonnax` (RFF/VSSGP/SIREN basis) + `pipekit-train`/`pipekit-jax` |
| (4) emulator-based inference | NUTS/SVI with the cheap likelihood | `pyrox` SVI `AutoGuide` + emulator |
| (5) amortized predictor | point pattern → posterior over (covariate coeffs, GPD params); DeepSets/set-encoder backbone | `geonnax` encoder + amortized head; SBI-style |
| (6) improve | GWR → spatial hierarchical → marked STPP; statistical ↔ physical generator; add space–time interaction; prev rung as truth | fidelity climb (§2, §4) |

Only the protocol wiring is shared with the other pilots; the EVT/PP
*content* (rows marked *new*) is what this pilot builds.

---

## 7. Scoring — tails, events, and extrapolation (the moat)

Mean-field RMSE is almost meaningless for extremes. The mandatory scorer
families, drawn from the existing `Unit × Lens` taxonomy:

| Lens (existing axis) | Land-extremes scorer | Status |
|----------------------|----------------------|--------|
| **Probabilistic (Statistic)** — *tail* | return-level coverage at high quantiles; threshold-weighted CRPS (twCRPS); quantile loss at τ→1; SBC of the **GMST coefficient** vs the oracle | *new* (twCRPS extends `properscoring`) |
| **Detection (Event)** | exceedance hit/miss/false-alarm (POD / FAR / CSI / extremal dependence index) on the event pattern | *new* (→ `xrtoolz.events`) |
| **Point-process residuals (Event)** | time-rescaling theorem → inter-event uniformity (KS); spatial K-function / pair-correlation vs theoretical; Voronoi/intensity reliability | *new* |
| **Physical-constraint (Budget)** | CC-scaling monotonicity (precip tail ↑ ~7 %/°C); geostrophic/hydrostatic balance residuals; moisture-budget closure; soil-moisture–temperature consistency (§4) | *new*; reuse `xrtoolz` balance ops |
| **Covariate attribution (Statistic)** | posterior recovery + coverage of the true β_GMST (Δ return level per °C) | *new* |
| **Extrapolation (Statistic)** | predicted return levels at GMST+ΔT vs simulator truth; degradation curve in ΔT | *new* |

Headline land-extremes question: *does a fast estimator (rung 4/5)
recover the oracle's posterior over the GMST effect, keep its
return-level intervals calibrated in the tail, pass point-process
residual diagnostics, respect the physical balances/scalings of §4, and
still predict the right return levels at a warming level it never saw* —
none of which a point-RMSE leaderboard can see.

---

## 8. The oracle-less reality (the §4.3 fallback, made concrete)

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
   *relative* agreement, never absolute "truth". Physical-constraint
   Lenses (§4, §7) still apply with no oracle — a real advantage of the
   physically-grounded models.

The `Task` declares which protocol is active; the `reference` table from
`benchmark_ladder.md §4.3` already supports the known-θ-only fallback.

---

## 9. Where code lives

| Concern | Home |
|---------|------|
| Land-extremes `Task` (simulator, declustering, datasets, reference table) | `research_notebook/projects/` (new `land_extremes`, sibling to `assimilation`) |
| Marked-STPP simulator + EVT likelihoods (GEV/GPD/LGCP) | *new* `stpp`/`evtx` module — start inside the project, promote to a package if it grows |
| Physical generators (§4): NSRP/BLRP, SEB + force-restore LSM, moisture/orographic | *new* ops on `finitevolX`; synoptic driver from `somax` |
| Covariate encoders + basis intensity fields | `geonnax` (reused) |
| Spatially-varying-coefficient prior (GWR generalised) | `gaussx` GP (reused) |
| MCMC oracle + SVI fast inference + amortized head | `pyrox` (reused) |
| Physical-balance constraint scorers (geostrophic, hydrostatic, lapse) | reuse `xrtoolz.ocn`/`atm` operators |
| Declustering + event Detection scorers | *new* → land in `xrtoolz.events` |
| EVT/PP probabilistic scorers (return-level coverage, twCRPS, residuals) | *new* → `pipekit-evaluate` `metrics/` (Event + tail-Probabilistic) |
| Protocols + runner | `pipekit-evaluate` (unchanged) |

This pilot is the natural driver to finally populate `xrtoolz.events` and
the Event-unit corner of `pipekit-evaluate` that the package README
already anticipates.

---

## 10. Build order (land-extremes slice)

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
5. Physical generator + constraint Lens: Clausius-Clapeyron precip scaling
   -> verify: CC-informed prior recovered; a non-physical scaling is flagged
6. Amortized rung (geonnax DeepSets head) + posterior-distance to oracle
   -> verify: posterior over b within tolerance of NUTS; seconds not hours
7. Marked STPP (NSRP/BLRP) + extrapolation axis (return levels at GMST+ΔT)
   -> verify: extrapolation degradation curve reproduces under DVC
```

Steps 1–2 are a tractable 1-D slice that proves the tail-calibration
machinery before any spatial or physical complexity; the point process
and physical generators only enter at step 5+.

---

## 11. Open questions (land-extremes-specific)

1. **POT vs block maxima** as the default convention? Proposed: POT/GPD
   (it *is* a point process, so it unifies with the STPP story); offer
   block-maxima/GEV as an alternative mark model.
2. **Declustering method** (runs declustering vs intervals estimator vs
   model the extremal index θ directly). Proposed: model θ — it is itself
   a benchmark target, not a preprocessing nuisance.
3. **How much point-process machinery to build now.** Proposed: start
   with a thinned *inhomogeneous Poisson* (no GP field) at step 1, add the
   LGCP/GP intensity at step 3, the physical NSRP/BLRP cluster process at
   step 7 — avoid the integral-emulator (rung 3) until the likelihood is
   actually the bottleneck.
4. **Statistical vs physical generator as the default rung-1 truth.**
   Proposed: statistical (clean known-θ) for v0.1; add the §4 physical
   generators as a second truth source for cross-model robustness once the
   scorers are trusted.
5. **Where the EVT/PP code lives long-term.** Inside `land_extremes`
   first; promote to a standalone `evtx`/`stppx` package only if a second
   consumer appears (per the ecosystem's "no abstraction for single use"
   rule). Flag for decision.
6. **Real-data inclusion in the pilot.** Synthetic-only first (clean
   calibration story), or wire one real dataset (e.g. station temperature)
   to exercise the oracle-less protocol? Proposed: synthetic-only for v0.1.

---

## 12. References

- Sibling designs: [`benchmark_ladder.md`](benchmark_ladder.md),
  [`ocean_pilot.md`](ocean_pilot.md).
- Inference engine: `pyrox` README (`PyroxModule`, `Parameterized`,
  NumPyro NUTS/SVI bridge).
- Covariate bases / encoders: `geonnax` README (`CyclicEncoder`,
  `SphericalHarmonicEncoder`, `SlepianEncoder`, `OrthogonalRandomFeatures`,
  `vssgp`, `FourierNet`).
- Spatial GP / GWR prior: `gaussx`.
- Physical generators / balances: `somax` (QG/SWM), `finitevolX`
  (PDE operators), `xrtoolz.ocn`/`atm` (geostrophic, lapse rate,
  stratification operators).
- Event Unit target: `pipekit-evaluate` README (Unit × Lens × Stage;
  Event depends on the not-yet-present `xr_toolz.events`).

### Physical-model literature (for the §4 generators / constraints)

- Clausius–Clapeyron scaling of precipitation extremes — Trenberth et al.
  (2003); super-CC convective scaling — Lenderink & van Meijgaard (2008).
- Soil-moisture–temperature/heatwave feedback — Seneviratne et al. (2010,
  *Earth-Sci. Rev.*); land–atmosphere coupling hotspots — Koster et al.
  (2004, *Science*).
- Cluster point-process rainfall models (NSRP / BLRP) — Rodriguez-Iturbe,
  Cox & Isham (1987, 1988, *Proc. R. Soc. A*).
- Stochastic weather generators — Richardson (1981); WGEN.
- Self-organized criticality of precipitation — Peters & Neelin (2006,
  *Nature Physics*).
- Linear theory of orographic precipitation — Smith & Barstad (2004,
  *J. Atmos. Sci.*).
- Point-process representation of extremes — Pickands (1971); Smith
  (1989); Coles (2001, *An Introduction to Statistical Modeling of
  Extreme Values*).
