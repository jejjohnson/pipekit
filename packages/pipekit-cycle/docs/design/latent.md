---
status: draft
version: 0.1.0
---

# pipekit-cycle × Latent Data Assimilation

**Subject:** New substrate-neutral protocols and a `LatentDACycle` that let
algorithm libraries (vardax, filterax, plumax, …) perform data
assimilation in a learned low-dimensional latent space, without any
pipekit-cycle import in the algorithm core.

**Date:** 2026-05-28

**Decision anchor:** *(to be filed)* `D-pkc-latent` — *Encoder /
Decoder / LatentMap protocols belong in pipekit-cycle alongside
`ForwardModel`, `ObservationOperator`, `AnalysisStep`*.

---

## 1  Motivation

Modern data-assimilation systems increasingly run all or part of the
forecast-analysis loop in a **learned latent space** $\mathcal{Z}
\subset \mathbb{R}^{N_z}$ obtained from an autoencoder

$$
\varphi: \mathbb{R}^{N_x} \to \mathbb{R}^{N_z}, \qquad
\psi: \mathbb{R}^{N_z} \to \mathbb{R}^{N_x}, \qquad N_z \ll N_x.
$$

Three families of methods motivate first-class support:

1. **Latent 4DVar / latent EnKF (Peyron et al. 2021, Cheng et al. 2023)** —
   replace the high-dim state $x \in \mathcal{X}$ with a latent code
   $z \in \mathcal{Z}$ throughout the cycle; learn both the AE and a
   latent dynamics $M_z: \mathcal{Z} \to \mathcal{Z}$.

2. **AE-regularised variational DA (4DVarNet, Fablet et al. 2021)** —
   keep the state in $\mathcal{X}$ but use $\psi \circ \varphi$ as a
   reconstruction-based prior in the cost. Already in use inside
   `vardax`'s `BilinAEPrior` / `MLPAEPrior` / `ConvAEPrior` but with no
   shared protocol exposed to the rest of the ecosystem.

3. **Hybrid latent assimilation (Brajard et al. 2021, methane / SWOT
   pipelines)** — forecast in $\mathcal{X}$ with a physics model, encode
   at analysis time, perform the gain in $\mathcal{Z}$, decode back.

All three need the same minimal vocabulary: an encoder, a decoder, an
optional latent forward model, and an observation operator that knows
how to compose with the decoder. Today `pipekit_cycle.protocols`
provides `ForwardModel`, `ObservationOperator`, `AnalysisStep` — none of
them mention $\mathcal{Z}$. Algorithm libraries either bury the
encode/decode plumbing inside ad-hoc classes (vardax) or leave it to
users (filterax). The result is duplicated abstractions and no
substrate-neutral way to assemble a latent DA cycle.

This document adds **three protocols** (`Encoder`, `Decoder`,
`LatentMap`), **two operator helpers** (`LiftedObservationOperator`,
`EncodedForwardModel`), one carry state (`LatentDAState`), and one
orchestrator (`LatentDACycle`). Together they support all three
flavours above with a single state machine.

---

## 2  Notation

| Symbol | Meaning |
|---|---|
| $x \in \mathbb{R}^{N_x}$ | full-state vector |
| $z \in \mathbb{R}^{N_z}$ | latent code |
| $\varphi(x) = z$ | encoder, satisfies `Encoder` |
| $\psi(z) = \hat{x}$ | decoder, satisfies `Decoder` |
| $M_x: \mathcal{X} \to \mathcal{X}$ | forward model in state space (`ForwardModel`) |
| $M_z: \mathcal{Z} \to \mathcal{Z}$ | forward model in latent space (`LatentForwardModel`) |
| $H: \mathcal{X} \to \mathcal{Y}$ | observation operator (`ObservationOperator`) |
| $\tilde{H} = H \circ \psi$ | **lifted** obs operator: $z \mapsto y$ (`LiftedObservationOperator`) |
| $\mathbf{B}_z$ | background error covariance in $\mathcal{Z}$ |
| $\mathbf{R}$ | observation error covariance |

A reconstruction triple $(\varphi, \psi)$ is **consistent** if
$\psi(\varphi(x)) \approx x$ on the data manifold. The protocol
purposefully does *not* require $\psi \circ \varphi = \mathrm{id}$ —
generic encoders trade reconstruction for regularisation.

---

## 3  The three flavours, side by side

```
                STRONG-LATENT           PRIOR-ONLY LATENT        HYBRID
                (everything in z)       (current FourDVarNet)    (physics in x,
                                                                  update in z)
                ┌──────────────┐         ┌──────────────┐        ┌──────────────┐
   forecast :    z ──Mz──> z              x ──Mx──> x             x ──Mx──> x
                                                                       │
                                                                       ↓ φ
   analysis :    z ──update(y, Hψz)──> z  x ──update(y, Hx)+λ‖x−ψφx‖² z ──update(y, Hψz)
                                                                       │
                                                                       ↓ ψ
                ┌──────────────┐         ┌──────────────┐        ┌──────────────┐
   output   :    ψ(z*) = x_hat          x*                       ψ(z*) = x_hat
```

All three are obtained from a single `LatentDACycle` by toggling two
fields:

```python
LatentDACycle(
    forward_model: ForwardModel | LatentForwardModel,
    latent_map:    LatentMap,
    obs_op:        ObservationOperator,        # in x-space; lifted internally
    analysis_step: AnalysisStep,
    forecast_space: Literal["x", "z"] = "z",   # strong=z, hybrid=x
    update_space:   Literal["x", "z"] = "z",   # prior-only=x, strong=z, hybrid=z
    re_encode_every: int = 1,
    ...
)
```

| Flavour | `forecast_space` | `update_space` | `re_encode_every` |
|---|---|---|---|
| Strong-latent | `z` | `z` | `inf` (z is canonical) |
| Prior-only | `x` | `x` | `0` (encode used inside cost only) |
| Hybrid | `x` | `z` | `1` (encode after every forecast) |

The orchestrator owns the bookkeeping; algorithm libraries supply
analysis steps that consume whichever space they are configured for.

---

## 4  Protocols

All four are `runtime_checkable` and live in a new module
`pipekit_cycle/latent.py`. Algorithm libraries (vardax, filterax,
plumax) satisfy them structurally — no inheritance, no pipekit import
required.

```python
# pipekit_cycle/latent.py
from __future__ import annotations
from typing import Any, Literal, Protocol, runtime_checkable


@runtime_checkable
class Encoder(Protocol):
    """Map full state to latent code.  φ: x ↦ z."""

    def __call__(self, state: Any) -> Any: ...

    @property
    def latent_dim(self) -> int | None: ...


@runtime_checkable
class Decoder(Protocol):
    """Map latent code to full state.  ψ: z ↦ x_hat."""

    def __call__(self, latent: Any) -> Any: ...

    @property
    def state_signature(self) -> Any: ...


@runtime_checkable
class LatentMap(Protocol):
    """Bundle of (Encoder, Decoder) — the autoencoder side of latent DA.

    The two halves are separated by Protocol so an algorithm can declare
    "I only need to decode" (e.g. when the analysis ensemble already
    lives in z and we never re-encode).  ``LatentMap`` is the union when
    both halves are needed.
    """

    def encode(self, state: Any) -> Any: ...
    def decode(self, latent: Any) -> Any: ...

    @property
    def latent_dim(self) -> int | None: ...

    @property
    def state_signature(self) -> Any: ...


@runtime_checkable
class LatentForwardModel(Protocol):
    """Forward dynamics in latent space.  M_z: z ↦ z.

    Marker protocol — structurally identical to ``ForwardModel`` but
    with the semantic claim that ``state`` is a latent code, not a full
    physical state.  Used so ``LatentDACycle`` can refuse a plain
    ``ForwardModel`` in ``forecast_space="z"`` mode and vice versa.
    """

    def step(self, latent: Any, dt: float) -> Any: ...

    @property
    def dt(self) -> float: ...

    @property
    def latent_signature(self) -> Any: ...
```

### Why three protocols rather than one?

* `Encoder` alone is enough for **observation-side** code paths (e.g.,
  amortised posteriors that map $y \to z$ via $\varphi$).
* `Decoder` alone is enough for **lifted observation operators**
  $\tilde{H} = H \circ \psi$ and for the final read-out step.
* `LatentMap` is the conjunction; required only by orchestration code
  that needs to round-trip ($\varphi \to z \to \psi$).

Splitting keeps each algorithm library free to declare the minimum
contract it actually consumes.

---

## 5  Operator helpers

Two concrete `Operator` subclasses do the heavy lifting that every
latent DA pipeline repeats. They live in `pipekit_cycle/latent.py`
alongside the protocols.

### 5.1  `LiftedObservationOperator`

```python
class LiftedObservationOperator(Operator):
    """Compose a Decoder with an x-space ObservationOperator.

    Produces  z ↦ y  via  z ─ψ→ x_hat ─H→ y.

    Satisfies ``pipekit_cycle.ObservationOperator``, so it drops into
    any existing ``DACycle`` / ``AnalysisStep`` without changes.

    The ``linearize`` method returns ``H'(ψ(z)) · ψ'(z)`` when both
    halves expose Jacobians (chain rule); otherwise raises
    ``NotImplementedError``.
    """

    decoder: Decoder
    inner: ObservationOperator

    def __call__(self, latent):
        return self.inner(self.decoder(latent))

    def linearize(self, latent):
        x_hat = self.decoder(latent)
        H_lin = self.inner.linearize(x_hat)         # x-space tangent
        psi_lin = _jacobian_of(self.decoder, latent)
        return _compose_linops(H_lin, psi_lin)
```

The `_jacobian_of` helper is library-specific — for an `eqx.Module`
decoder it is `jax.jacfwd` / `jax.jacrev` wrapped as a
`lineax.AbstractLinearOperator`; for symbolic decoders it can be a
materialised matrix. Implementations live in `pipekit-jax` or
algorithm libraries, not in `pipekit-cycle`.

### 5.2  `EncodedForwardModel`

```python
class EncodedForwardModel(Operator):
    """Lift an x-space ForwardModel into z-space via the AE round-trip.

    Produces  z ─ψ→ x ─M_x→ x' ─φ→ z'.

    Satisfies ``LatentForwardModel``.  Used in the *strong-latent*
    flavour when the only forward model the user has is in x-space —
    the cost is one extra round-trip per step but no learned M_z is
    required.
    """

    latent_map: LatentMap
    inner: ForwardModel

    def step(self, latent, dt):
        x = self.latent_map.decode(latent)
        x_next = self.inner.step(x, dt)
        return self.latent_map.encode(x_next)

    @property
    def dt(self):
        return self.inner.dt
```

`EncodedForwardModel` is the bridge that lets users opt into the
strong-latent flavour **without** training a separate $M_z$. The
trade-off (extra encode/decode work, accumulated reconstruction error)
is explicit and documented; users with a learned $M_z$ skip this
helper and supply a `LatentForwardModel` directly.

---

## 6  Carry state — `LatentDAState`

The existing `DAState` tracks model time, cycle count, and an
`obs_err_cov`. Latent cycling needs three more fields, encapsulated in
a subclass that satisfies `CarryState`:

```python
@dataclass(frozen=True)
class LatentDAState(DAState):
    """DAState + latent bookkeeping.

    Fields:
        latent_state: current z (None until first encode).
        last_encoded_t: model time at which latent_state was last
            synchronised with the x-space state.  Used by the
            re-encode policy.
        latent_signature: shape / dtype hint for the latent space.
    """

    latent_state: Any = None
    last_encoded_t: float | None = None
    latent_signature: Any = None
```

Subclassing `DAState` (not replacing it) keeps every existing
`AnalysisStep` consumer compatible — they read only the parent fields.

---

## 7  `LatentDACycle`

The orchestrator. Builds on `DACycle`; differs in three places:

1. **Forecast dispatch**: chooses x-space step or latent step based on
   `forecast_space`.
2. **Update dispatch**: hands either $(x, y)$ or $(z, y)$ to the
   `AnalysisStep`. In the latter case the obs operator passed to the
   analysis step is `LiftedObservationOperator(decoder, obs_op)`.
3. **Re-encode policy**: decides when to call $\varphi$ to resynchronise
   $z$ from $x$ (hybrid mode), or $\psi$ to resynchronise $x$ from $z$
   (strong mode).

```python
class LatentDACycle(StatefulOperator):
    forward_model: ForwardModel | LatentForwardModel
    latent_map: LatentMap
    obs_op: ObservationOperator             # x-space, always
    analysis_step: AnalysisStep
    obs_source: Callable | None = None

    forecast_space: Literal["x", "z"] = "z"
    update_space: Literal["x", "z"] = "z"
    re_encode_every: int = 1                # 0 = never; inf = strong
    n_steps: int = 1
    save_history: bool = False

    def _apply(self, carrier, state: LatentDAState):
        # 1. Forecast
        if self.forecast_space == "z":
            z = state.latent_state
            z = self.forward_model.step(z, state.t)  # LatentForwardModel
            x = None  # lazily decoded if needed by obs_source / save
        else:
            x = self.forward_model.step(carrier, state.t)  # ForwardModel
            z = None

        # 2. Get observations (in y-space, never lifted)
        y = self.obs_source(state) if self.obs_source else None

        # 3. Analysis
        if y is not None:
            if self.update_space == "z":
                z = z if z is not None else self.latent_map.encode(x)
                lifted_H = LiftedObservationOperator(
                    decoder=self.latent_map, inner=self.obs_op
                )
                z = self.analysis_step(
                    forecast=z, obs=y,
                    obs_op=lifted_H,
                    obs_err_cov=state.obs_err_cov,
                )
            else:  # update_space == "x"
                x = x if x is not None else self.latent_map.decode(z)
                x = self.analysis_step(
                    forecast=x, obs=y,
                    obs_op=self.obs_op,
                    obs_err_cov=state.obs_err_cov,
                )

        # 4. Re-encode policy (resync z and x)
        new_state = self._resync(state, x, z)

        carrier_out = self._materialise(x, z, new_state)
        return carrier_out, new_state
```

The `_resync` helper is the only place where the three flavours
diverge. Pseudocode:

```
match (forecast_space, update_space, re_encode_every):
    case ("z", "z", _):           # strong: z is canonical
        keep z; x lazy
    case ("x", "x", 0):           # prior-only: x is canonical
        keep x; z unused
    case ("x", "z", k):           # hybrid: resync every k cycles
        if cycle_count % k == 0: z = encode(x)
        else: x = decode(z)       # carry both for next cycle
```

---

## 8  End-to-end usage

### 8.1  Strong-latent EnKF (filterax)

```python
import pipekit_cycle as pc
import filterax as flx
import equinox as eqx

class AE(eqx.Module):
    enc: eqx.nn.MLP
    dec: eqx.nn.MLP
    latent_dim: int = 8
    state_signature = None
    def encode(self, x): return self.enc(x)
    def decode(self, z): return self.dec(z)

ae = AE(...)  # pretrained
assert isinstance(ae, pc.LatentMap)        # structural check passes

# Either a learned latent M_z ...
latent_mz = MyLearnedDynamics(...)
assert isinstance(latent_mz, pc.LatentForwardModel)

# ... or wrap a physical M_x via encode/decode
latent_mz = pc.EncodedForwardModel(latent_map=ae, inner=physics_model)

cycle = pc.LatentDACycle(
    forward_model=latent_mz,
    latent_map=ae,
    obs_op=H_x,                            # untouched x-space H
    analysis_step=flx.LatentETKF(ae=ae),   # filterax wrapper, see below
    obs_source=satellite_obs,
    forecast_space="z", update_space="z",
    re_encode_every=10**9,                 # never re-encode
    n_steps=24,
)

initial = pc.LatentDAState(
    t=0.0, cycle_count=0,
    obs_err_cov=R, latent_state=ae.encode(x0),
)
analyses, state = cycle(x0, initial)
```

### 8.2  Hybrid 4DVar with physics forecast (vardax)

```python
import vardax as vdx

cycle = pc.LatentDACycle(
    forward_model=somax_model,                 # x-space physics
    latent_map=ae,
    obs_op=H_x,
    analysis_step=vdx.LatentStrongFourDVar(    # vardax model, see below
        forward=somax_model,
        latent_map=ae,
        obs_op=H_x,
        B_z_op=B_z,
    ),
    obs_source=L4_obs,
    forecast_space="x", update_space="z",
    re_encode_every=1,                         # encode once per window
    n_steps=8,
)
```

### 8.3  Prior-only (drop-in upgrade of existing FourDVarNet)

```python
cycle = pc.LatentDACycle(
    forward_model=physics_model,
    latent_map=ae,                              # only used inside cost
    obs_op=H_x,
    analysis_step=vdx.FourDVarNet1D(            # existing class, unchanged
        prior=vdx.BilinAEPrior1D(...),          # now satisfies LatentMap
    ),
    forecast_space="x", update_space="x",
    re_encode_every=0,
)
```

The third example is **opt-in only**: existing `DACycle` usage keeps
working. The point is that any AE prior that exposes `.encode` and
`.decode` now satisfies `LatentMap`, which means downstream tooling
(logging, registry, dataset adapters) can introspect them uniformly.

---

## 9  Composition with the rest of pipekit

* `LatentMap` instances participate in the `pipekit-experiment`
  registry exactly like any other `Operator` — content-addressed by
  the encoder/decoder pytrees plus shape signature. A trained AE
  becomes a first-class artefact.
* `pipekit-train.SimulationDataset` gains an optional
  `latent_map: LatentMap | None` argument. When set, the dataset emits
  `(params, encoded_obs)` pairs and the network is trained against
  $\varphi(y)$, which is the canonical way to wire latent amortised
  inference.
* `pipekit-jax.JaxModelOp` already weights-serialises arbitrary
  `eqx.Module`s, so AE weights round-trip with no new machinery.

No changes are required to `pipekit` core (`Operator`, `Sequential`,
`Graph`) or to `pipekit-experiment`.

---

## 10  Memory and compute considerations

| Quantity | x-space | strong-latent | hybrid | prior-only |
|---|---|---|---|---|
| State carried per cycle | $N_x$ | $N_z$ | $N_x + N_z$ | $N_x$ |
| Forecast cost | $C_{M_x}$ | $C_{M_z}$ (or $C_{M_x} + C_\varphi + C_\psi$ via `EncodedForwardModel`) | $C_{M_x}$ | $C_{M_x}$ |
| Analysis cost (gain solve) | $\mathcal{O}(N_y^3)$ | $\mathcal{O}(\min(N_y, N_z)^3)$ | same as strong | $\mathcal{O}(N_y^3)$ |
| Background storage $\mathbf{B}$ | $N_x \times N_x$ | $N_z \times N_z$ | $N_z \times N_z$ | $N_x \times N_x$ |

The headline win is the analysis cost: when $N_z \ll N_y$ the gain
solve drops by an order of magnitude. For variational methods (vardax)
the inner CG / Gauss-Newton iteration also shrinks; for ensemble
methods (filterax) the ensemble dimension $N_e$ can frequently be cut
because $N_z$ already constrains the manifold.

Reconstruction error $\| x - \psi(\varphi(x)) \|$ is the main
correctness risk for the strong-latent flavour; the hybrid flavour
bounds it by re-decoding every cycle and using the physics forecast
as ground truth.

---

## 11  Open questions

1. **Is the `latent_dim` property mandatory?** Setting `latent_dim ∈
   {int, None}` lets unstructured encoders (e.g., variable-rank PODs)
   participate. Keep optional; algorithm libraries that need it raise
   loudly when `None`.
2. **Should `LatentMap` enforce `psi(phi(x)) ≈ x`?** No — that's a
   property of the trained weights, not the protocol. We document the
   assumption and provide a `latent_map_reconstruction_error` helper
   in tests/.
3. **What about variational autoencoders?** The protocol covers
   deterministic AEs. For VAEs the `encode` method should return the
   mean of $q(z|x)$ by default; stochastic samples live behind a
   separate `sample_encode` method that we may add in v0.2.
4. **Adjoint through the encoder.** Vardax's `LatentStrongFourDVar`
   needs $\nabla_z \tilde{H}$; this falls out of JAX autodiff for
   `eqx.Module` encoders. For numerical (non-JAX) encoders the user
   must provide `linearize` manually.

---

## 12  Acceptance criteria for v0.1

* `pipekit_cycle.latent` exports `Encoder`, `Decoder`, `LatentMap`,
  `LatentForwardModel`, `LiftedObservationOperator`,
  `EncodedForwardModel`, `LatentDAState`, `LatentDACycle`.
* Each protocol is `runtime_checkable`; isinstance tests pass on
  `eqx.Module`-based AEs from vardax.
* `LatentDACycle` runs in all three modes against a synthetic Lorenz-96
  fixture and an identity-AE fixture (`phi = psi = id` ⇒ output equals
  baseline `DACycle`).
* Documentation example shows all three flavours composing with
  filterax and vardax adapters.
* No new mandatory dependencies; `eqx.Module` decoders are tested only
  in optional extras.

---

## 13  References

1. Peyron, M. et al. (2021). *Latent space data assimilation by using
   deep learning.* Q. J. R. Meteorol. Soc.
2. Cheng, S. et al. (2023). *Generalised latent assimilation in
   heterogeneous reduced spaces with machine learning surrogates.*
   J. Sci. Comput.
3. Fablet, R. et al. (2021). *Learning variational data assimilation
   models and solvers (4DVarNet).* J. Adv. Model. Earth Syst.
4. Brajard, J. et al. (2021). *Combining data assimilation and machine
   learning to infer unresolved scale parametrization.* Phil. Trans.
   R. Soc. A.
5. pipekit-cycle protocols — `protocols.py`, master plan Report 10,
   §2.3.
