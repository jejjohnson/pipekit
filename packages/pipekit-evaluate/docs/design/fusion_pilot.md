# Design — Fusion Pilot (heterogeneous observation fusion)

> **Status — draft v0.1 (design only, no code).** A companion to
> [`benchmark_ladder.md`](benchmark_ladder.md), alongside
> [`plume_pilot.md`](plume_pilot.md), [`ocean_pilot.md`](ocean_pilot.md),
> and [`land_extremes_pilot.md`](land_extremes_pilot.md). It works the
> staggered ladder for **merging heterogeneous observations** —
> geostationary (GEO) + low-Earth-orbit (LEO) satellites + in-situ — into a
> single coherent L3 analysis. Unlike the other pilots (which infer a
> *parameter* or a *state*), this pilot's product *is the observation
> layer the other pilots consume* — so it links to all of them.

---

## 1. Why fusion is the connective pilot

The other three pilots each *consume* observations and infer something.
This pilot *produces* the best observations. It is the L1/L2 → L3 step
factored out and benchmarked on its own:

- **Heterogeneous in every dimension.** GEO (coarse space, ~10-min
  cadence, fixed disk), LEO (fine space, sparse revisit, polar swaths),
  in-situ (point support, accurate, sparse) — different resolutions,
  footprints, error models, biases, and *support* (point vs area).
- **It is "I have no idea how" made tractable** by one observation: every
  sensor is an `ObservationOperator` on a *shared latent field*. Fusion is
  then just **joint inference of that latent** from all sensors at once —
  the same BLUE / OI / GP / DA machinery the ocean pilot already uses,
  with a *bank* of observation operators instead of one.
- **It links to every other pilot two ways.** (a) The per-sensor operators
  here *are* the other pilots' observation operators (the LEO altimeter is
  ocean's `CalculateSSHAlongtrack`; a plume sensor is `radtran`). (b) The
  fused L3 feeds back as better input to those pilots — so the headline
  metric is **downstream transfer** (§8).

This is operationally standard (blended SST: OSTIA / OISST / CMC;
multi-satellite precip: IMERG; everything: ERA5) — the contribution is the
*honest, calibrated, cross-domain benchmark* of it.

---

## 2. The unifying idea — every sensor is an `ObservationOperator`

```
                 shared latent field  x(s, t)        <- the "truth"
                         │
   ┌─────────────────────┼──────────────────────┬───────────────────┐
   ▼                     ▼                      ▼                   ▼
 H_geo                 H_leo                  H_insitu            H_leo2
 coarse PSF,           fine PSF,              point support,      L-band,
 ~10-min cadence,      sparse swaths,         sparse, accurate,   coarse,
 fixed-disk geometry   nadir geometry         bias-anchored       noisy
   │                     │                      │                   │
   ▼                     ▼                      ▼                   ▼
       y_geo,  y_leo,  y_insitu, ...   = heterogeneous observations
                         │
                         ▼   fusion = infer x | {y_k, H_k, R_k, bias_k}
                  fused analysis  x̂(s, t) + uncertainty   (L3)
```

The bank `{H_k}` composes into a single `pipekit_cycle.CompositeObs`; each
`H_k` carries its own error covariance `R_k` and (optionally) a bias term.
Fusion is one inverse problem, not a pipeline of ad-hoc merges. That is
the whole trick — and it is exactly the shape the framework already has.

---

## 3. Heterogeneous sources (concrete)

| Class | Examples | Strength | Weakness |
|-------|----------|----------|----------|
| **GEO** | GOES-R/ABI, Himawari/AHI, MSG·MTG/SEVIRI·FCI | temporal (~10 min), full-disk | coarse space; edge-of-disk geometry |
| **LEO (optical/IR)** | Sentinel-2/3, MODIS, VIIRS, Landsat | fine space, multi-spectral | 1–2/day revisit; clouds |
| **LEO (microwave/altimetry)** | SMOS/SMAP (L-band), SWOT, nadir altimeters | all-weather / SSH | coarse footprint; swath gaps |
| **In-situ** | buoys, Argo, tide gauges, GHCN/ASOS stations, ships | accurate, bias anchor | point support; very sparse |

The benchmark's job is to turn this table into measurable trade-offs: GEO
gives the **time** axis, LEO the **space** axis, in-situ the **accuracy /
bias anchor** — and fusion should provably inherit all three.

---

## 4. Observation-physics basis (what makes sensors disagree)

Fusion is only honest if the per-sensor *forward physics* is modelled, not
hand-waved. Each item is both a piece of the observation operator **(H)**
and a potential **constraint / diagnostic (C)**. The *geophysical-variable*
physics (what generates SSH/SST/precip/CH₄) is inherited from whichever
domain pilot the fusion feeds (§3 of those docs).

| Effect | What it does | Role | Home |
|--------|--------------|------|------|
| **Point spread function / spatial response** | each sensor integrates the true field over its footprint; point vs area mismatch | H, C | *new* PSF op; `geopatcher` windows |
| **Spectral response function** | band integration over wavelength | H | reuse `radtran.SpectralResponseFunction` |
| **Viewing & illumination geometry / BRDF** | directional/anisotropic effects; GEO edge-of-disk vs LEO nadir | H, C | *new* |
| **Atmospheric correction** | TOA radiance ↔ surface variable | H | `xrtoolz.rs`, `radtran` |
| **Temporal sampling / aliasing** | GEO continuous vs LEO snapshots; diurnal-cycle aliasing (ties to SST warm layer) | H, C | *new* sampler |
| **Change of support (COSP)** | combining point (in-situ) with areal (pixel) data — block / area-to-point kriging | C | *new*; `gaussx` |
| **Inter-sensor bias** | calibration drift, spectral mismatch; needs cross-cal | C, estimator | VarBC (§7) |

These are the genuinely *fusion-specific* physics. Modelling them is what
separates a real fusion benchmark from "regrid-and-average".

---

## 5. L0–L4, instantiated for fusion

| Level | Fusion instantiation | Tooling |
|-------|----------------------|---------|
| L0 | per-sensor raw counts | — |
| L1 | per-sensor calibrated, geolocated obs (GEO, LEO, in-situ) | `geocatalog` discovery |
| L2 | **collocated / matched-up** multi-sensor obs at common space-time | `geocatalog` matchups (`band_matchup`) |
| L3 | **fused analysis** x̂(s,t) + uncertainty — the deliverable | rung-2/4/5 estimators |
| L4 | (optional) dynamical fused reanalysis | `vardax` DA + `somax` prior |

The hard arrow is **L2 → L3**: heterogeneous matched-up obs → a single
calibrated, gap-filled, uncertainty-bearing field. `geocatalog` already
does the collocation/matchup that builds L2 — a real head start.

---

## 6. The six rungs, mapped to symbols

| Rung | Fusion realisation | Package / symbol |
|------|--------------------|------------------|
| (1) simple model | a latent field + a *bank* of observation operators (GEO/LEO/in-situ samplers with PSF, geometry, noise, bias) → heterogeneous obs | `somax` latent (or GP field) + `pipekit_cycle.CompositeObs`; *new* sensor samplers |
| collocation | match obs across sensors to common space-time | `geocatalog` matchups |
| (2) model-based inference (**oracle**) | optimal fusion: exact GP / BLUE / 4DVar combining all sensors with correct H_k, R_k, bias | `gaussx`/`pyrox` exact GP; `vardax` 4DVar via `pipekit_cycle.AnalysisStep` |
| baseline | single-best-sensor; naïve regridded average; plain OI | OI on `gaussx` |
| (3) emulator | surrogate of the expensive fusion posterior | `geonnax` + `pipekit-train`/`pipekit-jax` |
| (4) emulator-based inference | fast fusion with the surrogate | `pipekit_cycle.DACycle` + emulator |
| (5) amortized predictor | all sensors → fused field + uncertainty, direct (deep kriging / multimodal / graph / neural-process) | `geonnax` (`UNet`/`FNO` for grids, DeepSets/graph for in-situ; `vssgp`/`randfeat` = amortized GP) |
| (6) improve | add sensors; better bias/error models; dynamical prior; couple to a downstream pilot | model swap; previous rung as truth |

The same `Estimator`/`Oracle` protocols apply; the only fusion-specific
new code is the heterogeneous **sensor samplers** (rung 1), bias/error
modelling, and the fusion scorers (§8).

---

## 7. Fusion method families — all `Estimator`s

Scored against the same reference, so the benchmark answers "which fusion
strategy, when?":

- **Estimation-theoretic** — Optimal Interpolation / BLUE; kriging,
  **co-kriging**, regression kriging; multi-output GP via the **Linear
  Model of Coregionalization**; **multi-fidelity GP** (Kennedy–O'Hagan
  AR1) for GEO↔LEO resolution fusion. Engine: `gaussx` + `pyrox`.
- **Data assimilation** — OI / 3D-/4D-Var / EnKF with a multi-sensor
  `CompositeObs` and per-sensor `R_k`, plus **Variational Bias Correction
  (VarBC)** to estimate and remove inter-sensor biases online. Engine:
  `vardax` + `pipekit-cycle`.
- **Learned** — deep kriging, multimodal fusion nets, **graph neural
  networks** over the irregular in-situ network, GEO-temporal + LEO-spatial
  **super-resolution/downscaling**, and **neural processes** (amortized
  GP). Engine: `geonnax`.

VarBC and triple collocation (§8) are the two pieces most teams skip and
the ones that make a fusion product trustworthy.

---

## 8. Scoring — the fusion moat

| Lens (existing axis) | Fusion scorer | Status |
|----------------------|---------------|--------|
| Point-wise (Field) | **held-out-sensor cross-validation** — withhold in-situ stations / one satellite, predict, score | *new* |
| Probabilistic (Statistic) | uncertainty coverage / CRPS, esp. **in gaps**; posterior-distance to the optimal-fusion oracle | `properscoring` |
| **Error attribution (Statistic)** | **triple collocation** — per-sensor error variance from 3 co-located datasets *without truth* (works on real data) | *new* |
| **Change-of-support (Budget)** | point↔area consistency: does the fused field, integrated over a pixel, match the areal obs and the point obs simultaneously? | *new*; `gaussx` |
| Spectral (Statistic) | **effective-resolution gain** — does fusion actually add LEO high-wavenumber + GEO high-frequency content vs each input? | `xrtoolz.geo` spectral |
| Detection (Event) | bias/inter-calibration removal; drift detection vs the in-situ anchor | *new* (VarBC diagnostics) |
| **Transfer (cross-pilot)** | plug the fused L3 into another pilot's inference; measure the **downstream skill gain** (e.g. plume Q error, ocean forecast skill, land return-level coverage) | reuse the other pilots |

The headline fusion question: *does fusing GEO+LEO+in-situ produce an L3
that is unbiased against in-situ, calibrated in its uncertainty
(especially in gaps), gains genuine spatio-temporal resolution, and —
above all — measurably improves the downstream pilots' inference* vs the
best single sensor. The **transfer** row is the one that justifies the
whole pilot existing.

---

## 9. Where code lives

| Concern | Home |
|---------|------|
| Fusion `Task` (latent, sensor bank, datasets, reference table) | `research_notebook/projects/` (new `fusion_benchmark`) |
| Latent-field generator (rung 1) | `somax` (dynamical) or `gaussx` (GP field) |
| Heterogeneous sensor samplers (PSF, geometry, sampling, bias) | *new* ops; reuse `radtran.SpectralResponseFunction`, `geopatcher` windows |
| Multi-sensor observation operator | reuse `pipekit_cycle.CompositeObs` |
| Collocation / matchup (L2) | `geocatalog` matchups (`band_matchup`) |
| Estimation-theoretic fusion | `gaussx` + `pyrox` |
| DA fusion + VarBC | `vardax` + `pipekit-cycle` |
| Learned fusion | `geonnax` + `pipekit-train`/`pipekit-jax` |
| Triple collocation / COSP / effective-resolution / transfer scorers | *new* → `pipekit-evaluate` `metrics/` |
| Protocols, runner | `pipekit-evaluate` (unchanged) |

`geocatalog`'s matchup machinery is the standout existing asset — the L2
collocation that every fusion method needs is already built.

---

## 10. Build order (fusion slice)

```
1. Fusion Task: a GP latent + two sensor samplers (one "GEO" coarse/frequent,
   one "LEO" fine/sparse) via CompositeObs; exact-GP oracle
   -> verify: Benchmark.run() recovers the latent; fused beats each sensor alone
2. Held-out-sensor CV + uncertainty-coverage scorers
   -> verify: coverage ~ nominal for the oracle; degrades gracefully in gaps
3. Add in-situ point sensor + change-of-support scorer
   -> verify: point↔area consistency holds for the oracle; naive averaging fails it
4. Triple collocation + inter-sensor bias (VarBC)
   -> verify: recovered per-sensor error variances ~ injected; injected bias removed
5. Effective-resolution scorer + a learned amortized fusion (geonnax)
   -> verify: fusion adds high-wavenumber content; amortized ~ oracle, fast
6. Transfer: feed the fused L3 into the ocean (or plume) pilot
   -> verify: downstream skill gain vs best single sensor, reproducible under DVC
```

Steps 1–2 are a tractable two-sensor GP slice that proves the joint-inverse
idea before any real sensor zoo; in-situ/COSP, bias, and the cross-pilot
transfer enter at steps 3–6.

---

## 11. Open questions (fusion-specific)

1. **Static field vs dynamical latent.** Start with a static GP latent
   (clean COSP/triple-collocation story) or go straight to a `somax`
   dynamical latent (enables DA + temporal fusion)? Proposed: static first.
2. **Bias model.** Per-sensor constant offset vs state-/airmass-dependent
   (VarBC predictors)? Proposed: constant offset first, then VarBC
   predictors as an "improve" step.
3. **Which downstream pilot to link first** for the transfer metric?
   Proposed: ocean (SSH altimetry GEO/LEO/in-situ fusion → forecast skill)
   — it reuses the most existing machinery.
4. **Real vs synthetic.** Synthetic sensor bank first (known truth +
   injected bias/error → clean calibration), one real matchup later
   (`geocatalog`)? Proposed: synthetic for v0.1; the triple-collocation and
   held-out-sensor scorers are exactly what make the real-data, oracle-less
   case tractable afterwards.
5. **Promotion.** If the sensor-sampler + fusion-scorer code grows, does it
   earn a package (`fusex`?) or stay in the project? Defer per the
   "no abstraction for single use" rule.

---

## 12. References

- Framework: [`benchmark_ladder.md`](benchmark_ladder.md). Siblings:
  [`plume_pilot.md`](plume_pilot.md), [`ocean_pilot.md`](ocean_pilot.md),
  [`land_extremes_pilot.md`](land_extremes_pilot.md).
- Collocation / matchups: `geocatalog` README (`band_matchup`, bundles).
- Multi-sensor observation operator: `pipekit-cycle` (`CompositeObs`).
- Estimation-theoretic fusion: `gaussx`, `pyrox`.
- DA fusion + bias correction: `vardax`, `pipekit-cycle`.
- Learned fusion: `geonnax` (`UNet`, `FNO`, `vssgp`, `randfeat`).

### Method literature (for §4 / §7 / §8)

- Optimal interpolation / BLUE — Gandin (1965); Bretherton et al. (1976).
- Co-kriging & change-of-support — Wackernagel (2003); Gotway & Young
  (2002, area-to-point).
- Linear model of coregionalization / multi-output GP — Álvarez et al.
  (2012).
- Multi-fidelity fusion — Kennedy & O'Hagan (2000).
- Triple collocation — Stoffelen (1998).
- Variational bias correction — Dee (2004); Auligné et al. (2007).
- Blended products (context) — OSTIA (Donlon et al. 2012); IMERG
  (Huffman et al. 2015).
