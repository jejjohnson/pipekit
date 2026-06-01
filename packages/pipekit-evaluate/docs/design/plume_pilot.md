# Design — Plume Pilot (the canonical first `Task`)

> **Status — draft v0.1 (design only, no code).** A companion to
> [`benchmark_ladder.md`](benchmark_ladder.md) (the domain-independent
> framework), alongside [`ocean_pilot.md`](ocean_pilot.md),
> [`land_extremes_pilot.md`](land_extremes_pilot.md), and
> [`fusion_pilot.md`](fusion_pilot.md). It works the staggered ladder for a
> methane-plume emission-rate inverse problem. This is the recommended
> *first* pilot: every rung already has an implementation in
> `research_notebook/projects/plume_simulation`, so the framework is wired
> from existing parts rather than written from scratch.

---

## 1. Why plume is the canonical first pilot

- **Mostly-static, low-dimensional inverse problem** — infer an emission
  rate Q (plus a stability class) from a single retrieved scene. The
  smallest end-to-end instance of the ladder.
- **Every rung already exists.** `plume_simulation` ships the forward
  models, the retrieval, and a NumPyro NUTS inference that *is* the rung-2
  oracle. The pilot wires existing symbols into the protocol; it does not
  write physics.
- **Cleanly extracts the framework.** Per `AGENTS.md` (Simplicity First,
  Goal-Driven), we build this one vertical slice end-to-end and *then*
  promote the shapes into `pipekit-evaluate`.

---

## 2. The plume "world" — state → observable

The latent of interest is the **emission rate Q** (with a categorical
**stability class**). The generative chain is:

```
   Q, stability, wind                          <- parameters θ
        │  dispersion (advection–diffusion)
        ▼
   concentration / column field  ΔΩ(x, y)      <- state
        │  retrieval forward model (Beer–Lambert + RT + SRF)
        ▼
   radiance  ->  retrieved column enhancement  <- observation (L1 -> L2)
```

Inference inverts the chain: retrieved enhancement → posterior over Q.

---

## 3. Physical basis for the measured variables

The measured variable is the **column methane enhancement** ΔXCH₄
retrieved from radiance; the inferred quantity is the **emission rate Q**.
Each physical model can play up to three roles: **(G)** generative rung-1,
**(P)** physically-informed prior, **(C)** physical-constraint Lens (§7).

### 3.1 Dispersion physics (state generation)

| Model | Role | Status / home |
|-------|------|---------------|
| **Advection–diffusion PDE** — governing equation; Gaussian plume/puff are closed-form solutions under constant wind + K-theory | G, C | `plume_simulation.gauss_plume`, `gauss_puff` (shipped) |
| **Taylor's theory of turbulent dispersion** — σ_y, σ_z grow with travel time; the physical basis for the Briggs / Pasquill–Gifford coefficients | basis | underpins shipped dispersion |
| **Briggs plume rise** — buoyant/momentum rise sets effective source height for hot stacks | G (augment) | *new* (extends `gauss_plume`) |
| **Monin–Obukhov surface-layer similarity** — sets the stability class (A–F) and the wind profile driving dispersion | P (stability prior) | links to the shipped stability-class latent |
| **Ornstein–Uhlenbeck meander / LES (Smagorinsky SGS)** — turbulent realism on puff centres / resolved flow | G (higher fidelity) | OU shipped; LES is the stated future + `les_fvm` Eulerian L2/L3 |

### 3.2 Retrieval physics (observation operator)

| Model | Role | Status / home |
|-------|------|---------------|
| **Beer–Lambert law** — gas absorption → radiance, per-pixel forward model | obs op, C | `plume_simulation.hapi_lut` (shipped) |
| **Band-integrated radiative transfer + spectral response function** | obs op | `plume_simulation.radtran` (shipped) |
| **Air mass factor + column averaging kernel** — true column ↔ retrieved column (geometry, vertical sensitivity) | obs op, C | `xrtoolz.atm.gas.ch4` |
| **Matched filter** (hyperspectral) — statistical-physical enhancement retrieval; low-rank background covariance | obs op | `radtran` + `gaussx.LowRankUpdate` (shipped) |

### 3.3 Physical inversion methods (fast baseline estimators)

Like GWR in the land pilot, these are *fast, physically interpretable
estimators to beat* — additional rungs scored against the same oracle:

- **Integrated Mass Enhancement (IME)** (Varon et al. 2018) —
  Q = IME · U_eff / L, where IME integrates column enhancement over the
  plume mask, U_eff is an effective wind, L a plume length scale. The
  standard single-image emission estimate. **(C: the IME integral is a
  mass budget.)**
- **Cross-sectional flux method** — Q = ∫ ΔΩ · U across a downwind
  transect.
- **Gaussian-plume mass-balance inversion** — analytic Q from fitting the
  closed-form plume (the frequentist sibling of the NUTS oracle).
- **Source-pixel / flux-divergence methods** — for area/regional sources.

### 3.4 What this changes in the pilot

- **Adds physical fast estimators** (IME, cross-sectional flux) as baseline
  rungs — directly comparable to the amortized net (benchmark question:
  *how much does the learned predictor beat IME, and where?*).
- **Physical-constraint Lenses** (§7): mass conservation (IME budget),
  plume continuity, non-negativity of enhancement.
- **Physically-informed priors**: wind-speed (U_eff) uncertainty dominates
  IME error — a physical wind prior propagates into Q uncertainty and is
  itself a benchmark target.

---

## 4. L0–L4, instantiated for the plume

| Level | Plume instantiation | Tooling |
|-------|---------------------|---------|
| L0 | raw at-sensor spectra | (out of scope for pilot) |
| L1 | calibrated, geolocated radiance | `geocatalog`, `xrtoolz.rs` |
| L2 | per-pixel column enhancement ΔXCH₄ (matched filter) | `radtran`, `hapi_lut` |
| L3 | denoised / masked enhancement field over the plume | `geopatcher` + masks |
| L4 | inferred emission rate Q (+ forecast plume) | rung-2/4/5 estimators |

---

## 5. The six rungs, mapped to existing symbols

| Rung | Plume realisation (existing symbols) |
|------|--------------------------------------|
| (1) simple model | `gauss_plume.simulate_plume` / `gauss_puff.simulate_puff` (forward `ForwardModel`) |
| obs operator | `radtran` SRF + matched-filter retrieval (L1 radiance → L2 enhancement) |
| (2) model-based inference (**oracle**) | `gauss_plume.infer_emission_rate` (NumPyro NUTS) — posterior over Q |
| physical baseline | IME / cross-sectional flux (§3.3) — fast physical Q estimate |
| (3) emulator | emulator of the forward model via `pipekit-train` + `pipekit-jax` (`JaxModelOp`) |
| (4) emulator-based inference | rung-2 inference with the emulator swapped in as `ForwardModel` |
| (5) amortized predictor | net: retrieved scene → posterior over (Q, stability class) |
| (6) improve | swap `gauss_puff` → `les_fvm` (higher-fidelity Eulerian truth) and re-score |

Headline plume question: *how well do rungs 4 and 5 recover the rung-2
NUTS posterior over Q (and the categorical stability class), do their
plumes conserve mass, and do they beat the IME physical baseline?*

Higher-fidelity truth is available: `les_fvm` (finitevolX Eulerian
advection–diffusion) is the L2/L3 reference for the rung-6 "improve" step,
and `gauss_puff` already has a cross-check notebook against it.

---

## 6. Scoring

The three mandatory families from the `Unit × Lens` taxonomy, plus the
physical baseline comparison:

1. **Probabilistic** — SBC, coverage, CRPS on Q; **posterior-distance**
   (C2ST / MMD / sliced-Wasserstein) of rung-4/5 posteriors to the rung-2
   oracle.
2. **Physical-constraint** — mass conservation (the IME/Budget Lens),
   non-negativity, plume continuity (§3).
3. **Point + spectral** — enhancement-field RMSE and spatial structure
   (catch over-smoothed plumes).
4. **Baseline gap** — skill of each rung *relative to IME* (a fast
   physical estimator), not just absolute error.

---

## 7. Where code lives

| Concern | Home |
|---------|------|
| Plume `Task` (forward, obs operator, datasets, reference table) | `research_notebook/projects/plume_simulation` (Hydra + DVC) |
| Dispersion + retrieval physics | `plume_simulation` (shipped); `xrtoolz.atm.gas.ch4` |
| Physical baselines (IME, cross-sectional flux) | *new* ops in `plume_simulation` |
| Emulator (rung 3) | `geonnax` + `pipekit-train` / `pipekit-jax` |
| Oracle + amortized inference | `plume_simulation` NumPyro models; `pyrox` |
| Matched-filter covariance | `gaussx` |
| Protocols, scorers, runner | `pipekit-evaluate` (unchanged) |

---

## 8. Build order (plume slice)

```
1. Protocol stubs + a plume Task wrapping existing forward + oracle
   -> verify: Benchmark.run() executes rung 2 vs known-θ on a toy scene
2. Probabilistic scorers (SBC ranks, coverage, CRPS via properscoring)
   -> verify: SBC on the oracle is ~uniform on a well-specified toy
3. Posterior-distance scorers (C2ST / MMD / sliced-Wasserstein)
   -> verify: identical posteriors score ~0; shifted posteriors score >0
4. Physical baseline (IME) + mass-conservation constraint Lens
   -> verify: IME recovers Q on a clean synthetic; a mass-violating field is flagged
5. Emulator rung (3) + emulator-inference rung (4)
   -> verify: rung-4 posterior-distance to oracle below a set threshold
6. Amortized rung (5) + level × rung BenchmarkReport + DVC wiring
   -> verify: reproducible report regenerates byte-stable metrics
```

---

## 9. Open questions (plume-specific)

1. **Stability class** as a categorical latent in the oracle vs marginalised
   out? Proposed: keep it explicit — recovering it is a benchmark target.
2. **Wind treatment.** Fixed U_eff vs a wind prior propagated into Q? IME
   error is wind-dominated, so proposed: a wind prior, and score its
   contribution to Q uncertainty.
3. **Retrieval in the loop.** Score from retrieved L2 enhancement (full
   chain) or from synthetic enhancement directly? Proposed: both — the gap
   between them isolates retrieval error.

---

## 10. References

- Framework: [`benchmark_ladder.md`](benchmark_ladder.md). Siblings:
  [`ocean_pilot.md`](ocean_pilot.md),
  [`land_extremes_pilot.md`](land_extremes_pilot.md),
  [`fusion_pilot.md`](fusion_pilot.md).
- Pilot physics: `research_notebook/projects/plume_simulation/README.md`
  (`gauss_plume`, `gauss_puff`, `les_fvm`, `hapi_lut`, `radtran`).
- Column physics: `xrtoolz.atm.gas.ch4` (air mass factor, averaging kernel).
- Matched-filter algebra: `gaussx` (`LowRankUpdate`, Woodbury solve).

### Physical-model literature (for the §3 generators / constraints)

- Single-image emission retrieval (IME) — Varon et al. (2018,
  *Atmos. Meas. Tech.*).
- Gaussian-plume derivation — Stockie (2011, *SIAM Review*).
- Turbulent dispersion — Taylor (1921); Pasquill–Gifford / Briggs
  dispersion-coefficient schemes.
- OU sub-grid turbulence calibrated on LES — Gorroño et al. (2023).
