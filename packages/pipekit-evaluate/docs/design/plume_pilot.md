# Design — Plume Pilot (the canonical first `Task`)

> **Status — draft v0.2 (design only, no code).** A companion to
> [`benchmark_ladder.md`](benchmark_ladder.md) (the domain-independent
> framework), alongside [`ocean_pilot.md`](ocean_pilot.md),
> [`land_extremes_pilot.md`](land_extremes_pilot.md), and
> [`fusion_pilot.md`](fusion_pilot.md). It works the staggered ladder for a
> methane-plume problem and follows it the whole way — **from the methane
> retrieval (L1 radiance → column enhancement) through per-event emission
> inference to the population layer and operational persistency forecasts**.
> This is the recommended *first* pilot: every rung, and most of the
> complexity axis, already has an implementation in
> [`plumax`](https://github.com/jejjohnson/plumax), so the framework is
> wired from existing parts rather than written from scratch.

---

## 1. Why plume is the canonical first pilot

- **A complete vertical already exists.** `plumax` ships the retrieval
  stack (`hapi_lut`, `radtran`, `matched_filter`), the per-event forward
  models and inversions across four fidelity tiers
  (`gauss_plume`, `gauss_puff`, `lagrangian`, `les_fvm`, `coupled`), and
  the population/forecasting layer (`population` + the standalone
  `methane_pod`). The pilot wires existing symbols into the benchmark
  protocols; it does not write physics.
- **It exercises *both* benchmark axes.** The staged ladder (model →
  model-based inference → emulator → … → amortized) is the per-event
  inverse problem; the complexity axis (§3) is `plumax`'s **tier ladder**
  for the forward model *and* the inference kernel ladder
  (MLE → MAP → Laplace → NUTS). One pilot fills a real slab of the
  `level × rung × complexity` cube.
- **Cleanly extracts the framework.** Per `AGENTS.md` (Simplicity First,
  Goal-Driven), we build one vertical slice end-to-end at the *simplest*
  complexity step and *then* promote the shapes into `pipekit-evaluate`.

---

## 2. The plume "world" — retrieval → state → source → population

The latent of interest at the per-event scale is the **emission rate Q**
(with a categorical **stability class**); at the population scale it is the
**event intensity λ(t)** and **size distribution f(Q)**. The full chain,
which the pilot follows end to end:

```
   λ(t), f(Q), P_d(·)                            <- population params  (Tier V)
        │  thinned marked point process (emission events over time)
        ▼
   Q, stability, wind                            <- per-event params θ
        │  dispersion (advection–diffusion; tier-dependent fidelity, §3)
        ▼
   concentration / column field  ΔΩ(x, y)        <- state
        │  observation operator: AK / Beer–Lambert + RT + SRF
        ▼
   radiance  ──matched filter──▶ retrieved ΔXCH₄ <- observation (L1 → L2)
```

The pilot runs this chain in **both directions**:

- **Forward (generative):** population → events → plume field → radiance →
  retrieved enhancement. This is what the simulator produces for synthetic
  benchmarks.
- **Inverse (the benchmark target):** retrieved ΔXCH₄ → posterior over Q
  per event (Tiers I–IV) → catalog of per-event posteriors → posterior
  over (λ, f, P_d) (Tier V) → **persistency forecasts** (wait time,
  occurrence probability, …).

"Methane retrieval → persistency" *is* this full traversal, and each link
is a place where the benchmark scores a fast/rich estimator against a
slow/simple reference.

---

## 3. The complexity axis, instantiated for the plume

This is the §2.3 axis of [`benchmark_ladder.md`](benchmark_ladder.md), made
concrete. **Each rung carries its own ladder**; climbing right is a richer
method in the *same* role, scored against the simpler step it replaces.

### 3.1 Rung (1) — the forward model (the `plumax` tier ladder)

| Step | Forward model | `plumax` home | Inference it unlocks |
|------|---------------|---------------|----------------------|
| analytical | steady-state Gaussian plume (Briggs σ) | `gauss_plume` (`simulate_plume`, `plume_concentration`) | linear-in-Q closed form, MAP, NUTS |
| analytical (time) | Gaussian puff (Pasquill–Gifford), diffrax wind | `gauss_puff` | as above, time-resolved |
| stochastic-ODE | Lagrangian Langevin particles + footprint | `lagrangian` (Markov-1, Hanna turbulence) | closed-form Gaussian/lognormal source inversion |
| PDE | Eulerian finite-volume advection–diffusion | `les_fvm` | strong-constraint 4D-Var (adjoint via diffrax) |
| coupled multi-physics | transport + averaging-kernel / RTM, multi-instrument | `coupled` (`CoupledForward`, `Instrument`, `coupled.rtm`) | closed-form joint posterior over (Q, per-instrument bias) |

The `plumax` index calls this the **data-driven modeling cycle** repeated
at each tier; here it is one axis of the benchmark cube. **Higher fidelity
is the ground truth for lower fidelity** on the cases where it is the
better physics (§4.3 of the framework doc): `gauss_puff` is cross-checked
against `les_fvm`; the rung-6 "improve" step swaps Gaussian → Eulerian and
re-scores.

### 3.2 Rung (2) — the inference kernel ladder

| Step | Kernel | `plumax` / tool home |
|------|--------|----------------------|
| MLE / analytic | IME, cross-sectional flux, Gaussian-plume mass balance (§4.3) | *new* ops over `gauss_plume` |
| closed form | linear-Gaussian / lognormal BLUE with Matérn-3/2 prior | `lagrangian.inversion` (`linear_gaussian_inversion`, `lognormal_inversion`) |
| MAP / Laplace | L-BFGS MAP + Gauss-Newton Laplace covariance | `les_fvm.fourdvar` (`posterior_covariance`, `laplace_sample`, Hessian via `gaussx`) |
| MCMC | NumPyro **NUTS** over (Q, stability) — the **oracle** | `gauss_plume.infer_emission_rate` (`gaussian_plume_model`) |
| closed-form fusion | joint posterior over (Q, bias) across satellites | `coupled.fuse_observations` → `FusionPosterior` |

### 3.3 Rungs (3)–(5) — emulator and amortized ladders

- **(3) emulator:** linear/POD → CNN → **FNO** neural operator on the
  `les_fvm` field, trained via `pipekit-train` and wrapped as a
  `JaxModelOp` (`pipekit-jax`); served back through the fixed
  `forward(params, met) → observations` interface so the inference loop
  doesn't notice the swap.
- **(4) emulator-based inference:** the rung-2 ladder again (MAP → Laplace
  → NUTS), now driven by the surrogate forward — "PDE-free 4D-Var".
- **(5) amortized predictor:** point Q regressor → mixture-density net →
  **normalizing flow** over the joint (Q, stability class) posterior,
  mapping a retrieved scene directly to a posterior.

Mandate for the first slice: pin every rung at its **simplest** step
(Gaussian forward, MAP kernel, low-capacity emulator, point amortized
head). Climbing the complexity axis is a separate pass, each step scored
against the simpler one (§4.3 of the framework doc).

---

## 4. Physical basis for the measured variables

The measured variable is the **column methane enhancement** ΔXCH₄
retrieved from radiance; the inferred quantities are the per-event
**emission rate Q** and the population **intensity λ(t)**. Each physical
model can play up to three roles: **(G)** generative rung-1, **(P)**
physically-informed prior, **(C)** physical-constraint Lens (§8).

### 4.1 Dispersion physics (state generation)

| Model | Role | Status / home |
|-------|------|---------------|
| **Advection–diffusion PDE** — governing equation; Gaussian plume/puff are closed-form solutions under constant wind + K-theory | G, C | `plumax.gauss_plume`, `gauss_puff` (shipped); `les_fvm` for the full PDE |
| **Taylor's theory of turbulent dispersion** — σ_y, σ_z grow with travel time; basis for the Briggs / Pasquill–Gifford coefficients | basis | underpins shipped Gaussian dispersion |
| **Monin–Obukhov surface-layer similarity** — sets the stability class (A–F) and the wind profile driving dispersion | P (stability prior) | `plumax` prerequisites; links to the stability-class latent |
| **Lagrangian Langevin particles (Markov-1 + Hanna turbulence)** — wind-realistic stochastic transport + backward footprint | G (higher fidelity) | `plumax.lagrangian` (shipped) |
| **Eulerian finite-volume / LES (Smagorinsky SGS)** — resolved advection–diffusion on a C-grid | G (highest fidelity) | `plumax.les_fvm` (shipped) |

### 4.2 Retrieval physics (observation operator)

| Model | Role | Status / home |
|-------|------|---------------|
| **Beer–Lambert law** — gas absorption → radiance, per-pixel forward; HITRAN Voigt LUTs | obs op, C | `plumax.hapi_lut` (shipped) |
| **Band-integrated radiative transfer + spectral response function** — SWIR two-way path | obs op | `plumax.radtran` (`forward_nonlinear` / `forward_taylor`, `ForwardResult`; shipped) |
| **Air mass factor + column averaging kernel** — true column ↔ retrieved column (geometry, vertical sensitivity) | obs op, C | `plumax.assimilation.obs_operator`; `xrtoolz.atm.gas.ch4` |
| **Matched filter** (hyperspectral) — statistical-physical enhancement retrieval; low-rank background covariance | obs op | `plumax.matched_filter` (`matched_filter_snr`, `detection_threshold`) + `gaussx.LowRankUpdate` |

### 4.3 Physical inversion methods (fast baseline estimators)

These are *fast, physically interpretable estimators to beat* — additional
rungs scored against the same oracle (the MLE end of the §3.2 ladder):

- **Integrated Mass Enhancement (IME)** (Varon et al. 2018) —
  Q = IME · U_eff / L. The standard single-image emission estimate.
  **(C: the IME integral is a mass budget.)**
- **Cross-sectional flux method** — Q = ∫ ΔΩ · U across a downwind transect.
- **Gaussian-plume mass-balance inversion** — analytic Q from the
  closed-form plume (the frequentist sibling of the NUTS oracle).

### 4.4 What this changes in the pilot

- **Adds physical fast estimators** (IME, cross-sectional flux) as the
  cheapest step of the rung-2 complexity ladder — directly comparable to
  the amortized net (*how much does the learned predictor beat IME, and
  where?*).
- **Physical-constraint Lenses** (§8): mass conservation (IME budget),
  plume continuity, non-negativity of enhancement.
- **Physically-informed priors**: wind-speed (U_eff) uncertainty dominates
  IME error — a physical wind prior propagates into Q uncertainty and is
  itself a benchmark target.

---

## 5. L0–L4, instantiated for the plume

| Level | Plume instantiation | Tooling |
|-------|---------------------|---------|
| L0 | raw at-sensor spectra | `plumax` L1/L2 ingest (prereq; out of scope for pilot) |
| L1 | calibrated, geolocated radiance | `geocatalog`, `xrtoolz.rs`, `plumax.radtran` |
| L2 | per-pixel column enhancement ΔXCH₄ (matched filter) | `plumax.matched_filter`, `hapi_lut` |
| L3 | denoised / masked enhancement field over the plume | `geopatcher` + masks |
| L4 | inferred emission rate Q (+ forecast plume); population intensity λ(t) | Tier I–V estimators |

---

## 6. The six rungs, mapped to existing `plumax` symbols

The simulator and observation operator come straight from `plumax.adapters`
(which structurally satisfy `pipekit_cycle`'s `ForwardModel` /
`ObservationOperator`), so the plume `Task` is assembled from the same
parts a `DACycle` is — no parallel hierarchy.

| Rung | Plume realisation (existing symbols) |
|------|--------------------------------------|
| (1) simple model | `gauss_plume.simulate_plume` / `gauss_puff` forward (fidelity per §3.1) |
| obs operator | `radtran` SRF + `matched_filter` retrieval (L1 radiance → L2 ΔXCH₄) |
| (2) model-based inference (**oracle**) | `gauss_plume.infer_emission_rate` (NumPyro NUTS) — posterior over Q |
| physical baseline | IME / cross-sectional flux (§4.3) — fast physical Q estimate |
| (3) emulator | FNO of the `les_fvm` forward via `pipekit-train` + `pipekit-jax` (`JaxModelOp`) |
| (4) emulator-based inference | rung-2 ladder with the emulator swapped in as `ForwardModel` |
| (5) amortized predictor | net: retrieved scene → posterior over (Q, stability class) |
| (6) improve | climb a tier (Gaussian → `les_fvm`) *or* a kernel (MAP → NUTS) and re-score |

Higher-fidelity truth is available in-package: `les_fvm` (Eulerian
advection–diffusion + strong-constraint 4D-Var, `les_fvm.fourdvar`) is the
L2/L3 reference for the rung-6 "improve" step, and `gauss_puff` already has
a cross-check against it.

Headline per-event question: *how well do rungs 4 and 5 recover the rung-2
NUTS posterior over Q (and the categorical stability class), do their
plumes conserve mass, and do they beat the IME physical baseline?*

---

## 7. From per-event Q to population & persistency (Tier V)

This is the new top of the vertical — the arc that takes the pilot **to
persistency**. It reuses the per-event posteriors above as *soft
observations* of the true emission marks.

### 7.1 Catalog — collect per-event posteriors

Each per-event estimate (a `GaussianPosterior` from `lagrangian`, a
Laplace draw from `les_fvm.fourdvar`, or a `FusionPosterior` from
`coupled.fuse_observations`) is materialised into a cross-tier catalog
entry via `plumax.population.catalog.event_from_posterior`. The catalog is
tier-agnostic: a Tier-I NUTS posterior and a Tier-IV fusion posterior enter
the same `EmissionCatalog`.

### 7.2 Population inference (Tier V.B) — the second staged ladder

The point-process layer runs its **own** staged ladder, with its **own**
complexity axis:

| Step | Population estimator | home |
|------|----------------------|------|
| simple model | thinned marked temporal point process: λ(t) + f(Q) + POD P_d(·) | `methane_pod` (intensity, pod_functions, paradox) |
| closed form | Gamma–Poisson rate; log-linear inhomogeneous intensity | `population.point_process` (`fit_poisson_rate`, `fit_inhomogeneous_intensity`) |
| hierarchical | lognormal size-distribution fit w/ per-event uncertainty propagation | `population.size_distribution` (`fit_lognormal_size_distribution`) |
| MCMC (**oracle**) | NumPyro NUTS over (λ-params, mark-params, POD-params) | `methane_pod.fitting` |
| amortized | (basin tile, history window) → posterior over (λ, f, total mass, next-event time) | future work |

The **importance correction** is mandatory: each per-event posterior must
carry its per-event prior log-density so the population fit re-weights
`f / π_per-event` and does not double-count the prior (see `plumax`
Tier V design — "Importance correction is mandatory").

### 7.3 Persistency (Tier V.C) — the operational forecasts

The operational layer an LDAR crew / satellite-tasking dispatcher consumes,
a thin posterior-aware wrapper over `methane_pod.intensity` (proposed
`plumax.population.persistency`). Four metrics, each returning **posterior
samples** of the metric — full UQ, no point estimates:

| Metric | Question | proposed symbol |
|--------|----------|-----------------|
| Expected wait time 𝔼[Δt ∣ t₀] | how long until the next event? | `expected_wait_time` |
| Occurrence probability ℙ(N(t₁,t₂) ≥ 1) | chance of an event in a window? | `occurrence_probability` |
| Conditional intensity λ(t ∣ t_prev) | does a recent detection raise the rate? (Hawkes) | (intensity callable) |
| Cumulative count 𝔼[N(0,T)] + bounds | how many events this year, with 95% CI? | `cumulative_count`, `next_event_quantile` |

This closes the loop: a **retrieval** at L1/L2 propagates all the way to a
**dispatch decision** with calibrated uncertainty — and every link is a
benchmark cell with a designated reference.

---

## 8. Scoring

The three mandatory families from the `Unit × Lens` taxonomy, plus the
physical baseline comparison and the new population/persistency families:

1. **Probabilistic (per event)** — SBC, coverage, CRPS on Q;
   **posterior-distance** (C2ST / MMD / sliced-Wasserstein) of rung-4/5
   posteriors to the rung-2 NUTS oracle.
2. **Physical-constraint** — mass conservation (IME/Budget Lens),
   non-negativity, plume continuity (§4).
3. **Point + spectral** — enhancement-field RMSE and spatial structure
   (catch over-smoothed plumes).
4. **Baseline gap** — skill of each rung *relative to IME* (a fast physical
   estimator), not just absolute error.
5. **Population (Tier V.B)** — population SBC over (λ, f, P_d);
   importance-weight ESS diagnostic (flags events whose per-event posterior
   is far from the population mark distribution); per-event-prior swap-out
   invariance; missing-mass / POD-corrected total vs published basin
   inventories (Permian).
6. **Persistency (Tier V.C)** — posterior coverage of the wait-time and
   occurrence-probability credible intervals against a synthetic source
   with known λ_true(t); homogeneous-limit closed-form agreement; diurnal
   sanity (𝔼[Δt ∣ 14:00] ≪ 𝔼[Δt ∣ 02:00]).

---

## 9. Where code lives

| Concern | Home |
|---------|------|
| Plume `Task` (forward, obs operator, datasets, reference table) | project repo wiring `plumax` (Hydra + DVC) |
| Dispersion + retrieval physics | `plumax` (`gauss_plume`, `gauss_puff`, `lagrangian`, `les_fvm`, `hapi_lut`, `radtran`, `matched_filter`); `xrtoolz.atm.gas.ch4` |
| Forward / obs adapters (pipekit-cycle protocols) | `plumax.adapters`, `plumax.operators` |
| Physical baselines (IME, cross-sectional flux) | *new* ops over `plumax.gauss_plume` |
| Emulator (rung 3) | `geonnax` + `pipekit-train` / `pipekit-jax` |
| Oracle + amortized inference | `plumax` NumPyro models (`gauss_plume.infer_emission_rate`); `pyrox` |
| Multi-instrument fusion (Tier IV) | `plumax.coupled` (`fuse_observations`, `coupled.rtm`) |
| Matched-filter covariance | `gaussx` (`LowRankUpdate`, Woodbury solve) |
| Population + persistency (Tier V) | `plumax.population` + standalone `methane_pod` |
| Protocols, scorers, runner | `pipekit-evaluate` (unchanged) |

---

## 10. Build order (plume slice)

```
1. Protocol stubs + a plume Task wrapping plumax.gauss_plume forward + NUTS oracle
   -> verify: Benchmark.run() executes rung 2 vs known-θ on a toy scene
2. Probabilistic scorers (SBC ranks, coverage, CRPS via properscoring)
   -> verify: SBC on the oracle is ~uniform on a well-specified toy
3. Posterior-distance scorers (C2ST / MMD / sliced-Wasserstein)
   -> verify: identical posteriors score ~0; shifted posteriors score >0
4. Physical baseline (IME) + mass-conservation constraint Lens
   -> verify: IME recovers Q on a clean synthetic; a mass-violating field is flagged
5. Emulator rung (3) + emulator-inference rung (4)
   -> verify: rung-4 posterior-distance to oracle below a set threshold
6. Population layer: catalog per-event posteriors -> point_process fit (Tier V.B)
   -> verify: population SBC ~uniform; importance-weight ESS healthy
7. Persistency metrics (Tier V.C) + amortized rung (5)
   -> verify: wait-time CI coverage ~nominal; homogeneous-limit closed form matches
8. level × rung × complexity BenchmarkReport + DVC wiring
   -> verify: reproducible report regenerates byte-stable metrics
```

Steps 1–5 are the per-event slice at the **simplest** complexity step;
steps 6–7 add the population → persistency arc; a *later* pass climbs the
complexity axis (Gaussian → `les_fvm`, MAP → NUTS, regressor → flow),
scoring each step against the simpler one (§4.3 of the framework doc).

---

## 11. Open questions (plume-specific)

1. **Stability class** as a categorical latent in the oracle vs
   marginalised out? Proposed: keep it explicit — recovering it is a
   benchmark target.
2. **Wind treatment.** Fixed U_eff vs a wind prior propagated into Q? IME
   error is wind-dominated, so proposed: a wind prior, and score its
   contribution to Q uncertainty.
3. **Retrieval in the loop.** Score from retrieved L2 enhancement (full
   chain) or from synthetic enhancement directly? Proposed: both — the gap
   between them isolates retrieval error.
4. **Per-event posterior payload.** Confirm the cross-tier catalog carries
   `per_event_prior_logpdf` (required for the importance correction) for
   *every* tier's posterior, including the closed-form `FusionPosterior`.
5. **Persistency intensity model.** Poisson default with a Hawkes opt-in
   for clustering super-emitters? The wait-time integral is closed form for
   Poisson and needs quadrature for Hawkes.

---

## 12. References

- Framework: [`benchmark_ladder.md`](benchmark_ladder.md). Siblings:
  [`ocean_pilot.md`](ocean_pilot.md),
  [`land_extremes_pilot.md`](land_extremes_pilot.md),
  [`fusion_pilot.md`](fusion_pilot.md).
- Pilot physics & tier ladder: `plumax` design docs —
  [tier I Gaussian](https://github.com/jejjohnson/plumax/blob/main/docs/design/01_tier1_gaussian.md),
  [tier II Lagrangian](https://github.com/jejjohnson/plumax/blob/main/docs/design/02_tier2_lagrangian.md),
  [tier III Eulerian FV](https://github.com/jejjohnson/plumax/blob/main/docs/design/03_tier3_eulerian.md),
  [RTM stack](https://github.com/jejjohnson/plumax/blob/main/docs/design/04_rtm_stack.md),
  [tier IV coupled](https://github.com/jejjohnson/plumax/blob/main/docs/design/05_tier4_coupled.md),
  [tier V population](https://github.com/jejjohnson/plumax/blob/main/docs/design/06_tier5_population.md),
  [persistency](https://github.com/jejjohnson/plumax/blob/main/docs/design/06c_persistency.md).
- Column physics: `xrtoolz.atm.gas.ch4` (air mass factor, averaging kernel).
- Matched-filter algebra: `gaussx` (`LowRankUpdate`, Woodbury solve).
- Population library: `methane_pod` (intensity, POD, paradox, NUTS fitter).

### Physical-model literature (for the §4 generators / constraints)

- Single-image emission retrieval (IME) — Varon et al. (2018,
  *Atmos. Meas. Tech.*).
- Gaussian-plume derivation — Stockie (2011, *SIAM Review*).
- Turbulent dispersion — Taylor (1921); Pasquill–Gifford / Briggs
  dispersion-coefficient schemes.
- OU sub-grid turbulence calibrated on LES — Gorroño et al. (2023).
- Point-process foundations — Daley & Vere-Jones (2003, 2008).
