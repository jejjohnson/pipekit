<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/pipekit/main/docs/assets/packages/pipekit-jax/logo-dark.svg">
    <img alt="pipekit-jax" src="https://raw.githubusercontent.com/jejjohnson/pipekit/main/docs/assets/packages/pipekit-jax/logo-light.svg" width="339">
  </picture>
</p>

**JAX and Equinox models as pipekit operators: weights that round-trip through a model registry, cycles that compile into one `jax.lax.scan`, and differentiable diffrax forward models.**

<p align="center">
  <a href="https://github.com/jejjohnson/pipekit/actions/workflows/ci.yml"><img alt="Tests" src="https://github.com/jejjohnson/pipekit/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Version 0.1.0" src="https://img.shields.io/badge/version-0.1.0-blue">
  <img alt="Python 3.12+" src="https://img.shields.io/badge/python-3.12%2B-blue">
  <a href="https://opensource.org/licenses/MIT"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-yellow.svg"></a>
</p>

<p align="center">
  <a href="https://jejjohnson.github.io/pipekit/"><b>Docs</b></a> ·
  <a href="https://jejjohnson.github.io/pipekit/api/pipekit-jax/"><b>API</b></a> ·
  <a href="https://jejjohnson.github.io/pipekit/tutorials/pipekit-under-jit/"><b>Under jit</b></a>
</p>

## Why it exists

Consider the plume emulator from [pipekit-train](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-train): an MLP that maps the emission rate $Q$ and wind speed $u$ to the enhancement downwind.
After training we want three things from it.
We want to store it and get back exactly the weights we validated, months later; we want to step it thousands of times inside a forecast without paying Python overhead per step; and we want gradients through the whole rollout, so the emulator or the initial state can be fitted to observations.

A pipekit pipeline round-trips through JSON, but an `eqx.Module` is a pytree of arrays plus Python objects (activations, sizes) that JSON cannot carry.
pipekit-jax closes that gap.
`JaxModelOp` wraps a module as an operator whose weights travel as a separate, signed blob through any `ModelRegistry`.
`ScanCycle` is the jit-native counterpart of `pipekit_cycle.Cycle`, and `DiffraxForwardModel` is a `ForwardModel` backed by an ODE or SDE solve, with the gradient strategy chosen by a pipekit-cycle adjoint spec.

## Where it sits

pipekit-jax is built on pipekit-ops and pipekit-cycle; it uses pipekit-array through the `[array]` extra.
pipekit-ops and pipekit-array are libraries in their own right, which you can build on directly.

```
   your JAX models, emulators, differentiable DA
                     │
          ┌──────────┴──────────┐
          │     pipekit-jax     │  ◀ this package
          │ JaxModelOp,         │
          │ ScanCycle, diffrax, │
          │ adjoint interpreters│
          └──────────┬──────────┘
        ┌────────────┼──────────────┐
        │            │              │
   pipekit-cycle     │         pipekit-array
   (dependency:      │         ([array] extra:
    ForwardModel,    │          to_jax for a
    adjoint specs)   │          KernelOp)
        │            │
        └─────┬──────┘
              │
         pipekit-ops
         (dependency: Operator, Sequential, Graph)

   also: JAX, Equinox, jaxtyping (dependencies), diffrax ([diffrax] extra)
```

- **pipekit-ops** is a required dependency.
  `JaxOperator` is both a `pipekit_ops.Operator` and an `eqx.Module`, and importing `pipekit_jax` registers `Sequential` and `Graph` as JAX pytrees, so `jax.grad` reaches the learnable leaves of operators inside a pipeline.
- **pipekit-cycle** is a required dependency: `ScanCycle` takes a `ForwardModel`, `DiffraxForwardModel` is one, and the adjoint interpreters read pipekit-cycle's specs.
- **pipekit-array** is optional: with the `[array]` extra, `to_jax` turns a `pipekit_array.KernelOp` into a `JaxKernelOp` whose array parameters are pytree leaves.

## Install

```bash
uv add pipekit-jax
uv add 'pipekit-jax[diffrax,array]'
```

| Extra | What it adds | When you need it |
| --- | --- | --- |
| *(none)* | `pipekit-ops`, `pipekit-cycle`, `jax`, `equinox`, `jaxtyping` | `JaxOperator`, `JaxModelOp`, `ScanCycle`, `Rollout`, and the deprecated `truncated_scan` |
| `diffrax` | `diffrax`, `integrix` | `DiffraxForwardModel`, `integrix.adjoint`, and the deprecated `to_diffrax_adjoint` |
| `array` | `pipekit-array` | `to_jax` and `JaxKernelOp` |

`DiffraxForwardModel`, `to_jax` and `JaxKernelOp` are imported lazily, so `import pipekit_jax` works without either extra.

## Quickstart

### Store and reload a trained model

The emulator maps $\mathbf{x} = (Q, u) \in \mathbb{R}^{D_x}$ to $\mathbf{y} \in \mathbb{R}^{D_y}$ with weights $\boldsymbol{\theta}$; $D_x = 2$ and $D_y = 4$.
This block also needs pipekit-experiment for the registry.

```python
import tempfile

import equinox as eqx
import jax
import jax.numpy as jnp
from pipekit_experiment import LocalModelRegistry
from pipekit_jax import JaxModelOp

# Shapes: D_x = 2 inputs (Q, u); D_y = 4 downwind distances


# user
# the same architecture, built twice: g_emu(x; θ) with θ trained, and a skeleton
def plume_mlp(seed):
    key = jax.random.key(seed)
    return eqx.nn.MLP(in_size=2, out_size=4, width_size=32, depth=2, key=key)


trained = JaxModelOp(plume_mlp(0))           # user: the output of a TrainingLoop
x = jnp.array([1.2, 3.0])                    # (D_x,)

# today (pipekit-jax, pipekit-experiment)
registry = LocalModelRegistry(tempfile.mkdtemp())
h = registry.store(trained, name="plume-emulator", weights=trained.serialize_weights())

# today (pipekit-experiment)
try:
    registry.load(h)                         # the config holds no weights
except RuntimeError:
    print("refused: forbid_in_yaml")         # refused: forbid_in_yaml

# today (pipekit-jax)
skeleton = JaxModelOp(plume_mlp(1))          # same structure, other weights
reloaded = skeleton.from_registry(registry, "plume-emulator")
print(bool(jnp.all(reloaded(x) == trained(x))))      # True
print(reloaded.serialize_weights() == trained.serialize_weights())  # True```

### A differentiable forecast

Exponential decay, $\mathrm{d}\mathbf{x} / \mathrm{d}t = -k \mathbf{x}$ with $k = 0.1$ and $\mathbf{x} \in \mathbb{R}^{N}$, solved by diffrax and stepped 24 times inside one `jax.lax.scan`.
For the cost $\mathcal{J}(\mathbf{x}_0) = \sum_i x_{24,i}$ the gradient is $\nabla_{\mathbf{x}_0} \mathcal{J} = e^{-2.4} \mathbf{1}_N$, which the block checks.
It needs the `[diffrax]` extra.

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import pipekit_cycle as pc
from integrix.adjoint import to_diffrax_adjoint
from pipekit_jax import DiffraxForwardModel, ScanCycle

# Shapes: N = 3 state variables


# user
# dx/dt = −k x, k = 0.1
def decay(t, x, args):
    return -0.1 * x                                       # (N,) → (N,)


# today (pipekit-jax[diffrax])
# x_t ↦ x_{t+1}: a Tsit5 solve over [0, Δt], Δt = 1, checkpointed adjoint
model = DiffraxForwardModel(
    vector_field=decay, dt=1.0, adjoint=pc.RecursiveCheckpointAdjoint()
)
print(isinstance(model, pc.ForwardModel))                 # True
print(isinstance(model, pc.SupportsAdjoint))              # True

# today (pipekit-jax)
# x_0 ↦ x_24 inside one jax.lax.scan
forecast = ScanCycle(step_op=model, n_steps=24)


# user
# 𝒥(x_0) = ∑ᵢ x_{24,i};  ∇_{x_0} 𝒥 = e^{−2.4} 1 ≈ 0.0907 1
def final_sum(x0):
    x24, _ = forecast(x0, None)                           # (N,) → (N,)
    return jnp.sum(x24)                                   # → ()


grad = eqx.filter_jit(jax.grad(final_sum))(jnp.ones(3))   # (N,)
print(jnp.allclose(grad, jnp.exp(-2.4), rtol=1e-4))       # True

# today (integrix, installed by pipekit-jax[diffrax])
# swapping the gradient strategy is configuration
backsolve = model.with_adjoint(pc.BacksolveAdjoint())
print(type(to_diffrax_adjoint(backsolve.adjoint)).__name__)  # BacksolveAdjoint
try:
    to_diffrax_adjoint(pc.TruncatedAdjoint(k=1))
except ValueError:
    print("truncation belongs to the rollout")            # use integrix.adjoint.truncated_scan there
```

## What ships

Version 0.1.0.
Every name below is exported from `pipekit_jax`, except `check_jax_operator`, which lives in `pipekit_jax.testing`.

| Module | Public names | Purpose |
| --- | --- | --- |
| `operator` | `JaxOperator` | Base class for an operator that is also an `eqx.Module`: JSON config, pytree leaves, weight round trip |
| `model_op` | `JaxModelOp` | Wrap any `eqx.Module` as an operator, with `serialize_weights`, `with_weights` and `from_registry` |
| `cycle` | `ScanCycle`, `EnsembleScanCycle`, `Rollout` | Time stepping in one `jax.lax.scan`: per state, `vmap`-ed over members, or a carrier-only trajectory |
| `diffrax_model` | `DiffraxForwardModel` | A `ForwardModel` and `SupportsAdjoint` backed by `diffrax.diffeqsolve` (`[diffrax]`) |
| `adjoints` | `to_diffrax_adjoint`, `truncated_scan` | Deprecated forwarders to `integrix.adjoint`, kept for one release |
| `kernel` | `to_jax`, `JaxKernelOp` | A pipekit-array `KernelOp` with its array parameters as leaves (`[array]`) |
| `pytree` | (on import) | Registers `Sequential`, `Graph` and `JaxModelOp` as JAX pytrees |
| `testing` | `check_jax_operator` | Assert the `JaxOperator` contract in a downstream test suite |

### Not built yet

- **Interpreters for the inner-solve and cycle layers.** `integrix.adjoint.to_diffrax_adjoint` covers the dynamics layer, and `integrix.adjoint.truncated_scan` and `ScanCycle(truncate=k)` cover truncation; checkpointed and implicit adjoints for a variational solve live in the algorithm libraries (vardax, filterax).

## How it works

### Why `registry.load` refuses a `JaxModelOp`

`JaxModelOp` is marked `forbid_in_yaml`: its config records only the module's class name, because an `eqx.Module` carries Python objects (activation functions, layer sizes, dtypes) that JSON cannot rebuild.
`pipekit_experiment`'s `registry.load` rebuilds operators from their config with `Operator.from_state`, so it raises rather than return a model with made-up weights.
The weights are still stored, verified against the content hash and readable with `load_weights`.
The round trip is therefore:

1. store with `registry.store(op, weights=op.serialize_weights())`;
2. build a **skeleton** in code, the same architecture with any weights;
3. call `skeleton.from_registry(registry, ref)`, which reads `load_weights(ref)` and returns a copy of the skeleton holding the stored weights, byte for byte.

`pipekit_train.Checkpoint(registry=...)` stores the weight blob this way whenever the trained model has `serialize_weights`, and the Equinox backend of `TrainingLoop` returns a `JaxModelOp` when you give it one, so a trained model goes into the registry and comes back without extra code.
`JaxModelOp` replaces pipekit-train's in-package `EquinoxModelOp`, with the same constructor; `EquinoxModelOp` writes weights but cannot read them back.

### The weight blob is signed

`serialize_weights` writes a header before the `equinox.tree_serialise_leaves` payload: the pytree's node types and the key path, shape and dtype of every array leaf.
Equinox loads leaves by position, so without the header a skeleton with the same number of leaves in another order, or another class with the same field names, would silently take weights it was never trained with.
`with_weights` compares the header with the skeleton and raises `ValueError` on any difference; only array leaves are stored, and every other value comes from the skeleton.
Blobs written before the header existed still load, by position.

### `JaxOperator`: one object, two contracts

Subclass `JaxOperator` and declare fields as on any `eqx.Module`.
The config is derived from the generated `__init__` and coerced to JSON builtins, so a `JaxOperator` with plain fields round-trips through `dumps` / `loads`, and the same object is a pytree whose inexact-array fields are differentiable.
Mark config-only fields `eqx.field(static=True)`, and set `forbid_in_yaml = True` when a field holds something JSON cannot carry; `check_jax_operator(op, *inputs)` checks eager against `filter_jit`, `vmap`, that every learnable field is a leaf with a gradient, and the config round trip, in one call.

### Cycles under `jit`

`ScanCycle(step_op, n_steps)` returns `(carrier, state)` exactly like `pipekit_cycle.Cycle`, but compiles to one `jax.lax.scan`, so compile time is flat in the number of steps; `run` also returns the history, rather than storing it on the object.
With `checkpoint=True` the backward pass keeps carries only every $\lceil \sqrt{n} \rceil$ steps and recomputes the rest, so `grad` stores $O(\sqrt{n})$ carries instead of $O(n)$.
With `truncate=k`, gradients flow through the carry of the last `k` steps only, as in `integrix.adjoint.truncated_scan`.
`EnsembleScanCycle` `vmap`s a cycle over a leading member axis, with one shared read-only state or one state per member; `Rollout` steps a carrier and returns the trajectory, keeping one carry per saved point with `checkpoint=True`.
The [pipekit under jit](https://jejjohnson.github.io/pipekit/tutorials/pipekit-under-jit/) tutorial walks through all three.

### Adjoint specs, interpreted by integrix

pipekit-cycle's adjoint specs carry no behaviour, and the adjoint maths lives in integrix: pipekit-jax composes the rollout and integrix computes the backward sweep.
`DiffraxForwardModel` hands its spec to `integrix.adjoint.to_diffrax_adjoint`, which maps `RecursiveCheckpointAdjoint` (the default), `DirectAdjoint`, `BacksolveAdjoint` and `ImplicitAdjoint` onto the diffrax classes of the same name, and refuses `TruncatedAdjoint`, because truncation belongs to an outer rollout, where `integrix.adjoint.truncated_scan` does it.
`pipekit_jax.to_diffrax_adjoint` and `pipekit_jax.truncated_scan` stay for one release and emit a `DeprecationWarning`; `truncated_scan` keeps its own copy, so the base install runs it without integrix.
Specs are matched by class name and fields, so a structurally identical spec from a library that does not depend on pipekit works too.
**Note**: `BacksolveAdjoint` (continuous adjoints) diverges for chaotic or stiff dynamics; never use it as a default.

## Learn more

- API reference: [pipekit-jax](https://jejjohnson.github.io/pipekit/api/pipekit-jax/).
- Tutorials: [Run pipelines under jit](https://jejjohnson.github.io/pipekit/tutorials/pipekit-under-jit/), [Train, register, forecast](https://jejjohnson.github.io/pipekit/tutorials/train-register-forecast/).
- Design: the reproducibility caveat this package closes is in pipekit-train's [`boundaries.md`, §13.1](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-train/docs/design/boundaries.md); master plan [Report 11](https://github.com/jejjohnson/research_journal_v2/blob/main/notes/geotoolz/master_plan/toolz_9_pipekit_train.md).
- The foundations: [pipekit-ops](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-ops) and [pipekit-array](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-array).
- Siblings: [pipekit-cycle](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-cycle) (the protocols and adjoint specs), [pipekit-experiment](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-experiment) (the registry), [pipekit-train](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-train) (the training loop).

## License

MIT. See [LICENSE](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-jax/LICENSE).
