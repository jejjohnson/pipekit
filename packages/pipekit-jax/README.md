<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/pipekit/main/docs/assets/packages/pipekit-jax/logo-dark.svg">
  <img alt="pipekit-jax" src="https://raw.githubusercontent.com/jejjohnson/pipekit/main/docs/assets/packages/pipekit-jax/logo-light.svg" width="339">
</picture>

**pipekit operators and cycles as JAX and Equinox objects: steps whose fields are pytree leaves, cycles in one `jax.lax.scan`, and weights that round-trip through a model registry, on pipekit-ops, pipekit-cycle, JAX and Equinox, with pipekit-array through an extra.**

[![Tests](https://github.com/jejjohnson/pipekit/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/pipekit/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/pipekit/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/pipekit/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/pipekit/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/pipekit/actions/workflows/typecheck.yml)
[![Docs](https://github.com/jejjohnson/pipekit/actions/workflows/pages.yml/badge.svg)](https://jejjohnson.github.io/pipekit/)
[![codecov](https://codecov.io/gh/jejjohnson/pipekit/branch/main/graph/badge.svg)](https://codecov.io/gh/jejjohnson/pipekit)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-jax/LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

📖 [Documentation](https://jejjohnson.github.io/pipekit/) ·
🚀 [Quickstart](https://jejjohnson.github.io/pipekit/quickstart-jax/) ·
📚 [API reference](https://jejjohnson.github.io/pipekit/reference/pipekit-jax/)

</div>

---

Most of what we want from a trained emulator or a discretised PDE needs JAX to see inside it. We want the gradient of a forecast with respect to its initial state and its parameters, thousands of steps without Python overhead per step, an ensemble in one compiled call, and the trained weights back from storage exactly as we validated them. A pipekit operator is a plain Python object with a JSON config, which JAX treats as one opaque leaf, and pipekit-cycle's `Cycle` steps in a Python `for` loop that unrolls under `jit`. pipekit-jax makes the operator an Equinox module as well, so that one step of a model,

```math
\mathbf{x}_{t+1} = f(\mathbf{x}_t; \boldsymbol{\theta})
```

where $`\mathbf{x}_t \in \mathbb{R}^{D_x}`$ is the state after $`t`$ steps, $`f`$ is the operator's call and $`\boldsymbol{\theta} \in \mathbb{R}^{D_\theta}`$ holds its array fields, becomes a pytree whose leaves are $`\boldsymbol{\theta}`$. Applied $`n`$ times inside one `jax.lax.scan`, the final state $`\mathbf{x}_n`$ is then a differentiable function of $`\mathbf{x}_0`$ and $`\boldsymbol{\theta}`$, and $`\boldsymbol{\theta}`$ can be written to bytes and read back into a fresh copy of the model.

```python
# Shapes: D_x state elements; n steps
# (Decay, final_sum, initial and rate as in the quickstart below; trained and
# skeleton as in the docs quickstart)

# today (pipekit-jax)
# x_0 ↦ x_n, f applied n times in one jax.lax.scan
forecast = ScanCycle(step_op=Decay(rate), n_steps=24)
# today (jax)
# ∇_{x_0} 𝒥 and ∂𝒥/∂λ, through every step
gradients = jax.grad(final_sum, argnums=(0, 1))(initial, rate)
# today (pipekit-jax)
# θ as a signed blob, read back into a model of the same structure
reloaded = skeleton.with_weights(trained.serialize_weights())
```

None of it is a new framework. A `JaxOperator` is a [pipekit-ops](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-ops) `Operator` and an [Equinox](https://github.com/patrick-kidger/equinox) `eqx.Module` at once. pipekit reads its config off the `__init__` that Equinox generates, so `|`, `Sequential`, `Graph` and the JSON round trip keep working, and its array fields are leaves for `eqx.filter_jit`, `eqx.filter_grad` and `jax.vmap`. The cycles take a [pipekit-cycle](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-cycle) `ForwardModel` or any stateful operator and keep `Cycle`'s `(carrier, state)` contract, but step inside one [JAX](https://github.com/jax-ml/jax) `jax.lax.scan`. The weights travel as Equinox's `tree_serialise_leaves` bytes behind a signature header, and with the `array` extra `to_jax` gives a [pipekit-array](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-array) `KernelOp` the same treatment.

<p align="center"><img src="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pipekit_jax_overview/pipekit_jax_overview.png" alt="pipekit-jax in one picture: a step that is both a pipekit operator and an Equinox module, and the five things done with it: scan, roll out, take a gradient, store and convert" width="100%"></p>

## 🌟 Highlights

- 🧩 **Operators that are modules**: subclass `JaxOperator` and declare fields as on any `eqx.Module`; the config is derived from those fields and coerced to JSON builtins, and the inexact-array fields are differentiable leaves.
- 🔗 **Pipelines JAX can see into**: importing `pipekit_jax` registers `Sequential`, `Graph` and `JaxModelOp` as JAX pytrees, so `eqx.filter_grad` reaches the leaves of every operator inside a pipeline.
- 🔁 **Cycles in one scan**: `ScanCycle` steps a stateful operator or a `ForwardModel` and returns the history from `run`, `Rollout` returns a carrier-only trajectory, and `EnsembleScanCycle` `vmap`s a cycle over ensemble members. On a 2-vCPU VM, the [Pipelines under jit](https://jejjohnson.github.io/pipekit/pipekit-under-jit/) tutorial measured 0.13 s to compile a 1000-step `ScanCycle` against 19.0 s for `Cycle`.
- 🧮 **Control over the backward pass**: `checkpoint=True` keeps a carry only at chunk boundaries and recomputes the rest, which the API page measured as 4.4 MiB against 62.9 MiB for the gradient of a 1000-step call. `truncate=m` lets gradients through the last $`m`$ steps only.
- 💾 **Weights that come back exactly**: `JaxModelOp.serialize_weights` writes a blob headed by the node types and the key path, shape and dtype of every array leaf. `with_weights` refuses a skeleton that differs from that header, and `from_registry` reads the blob from any `ModelRegistry`.
- ✅ **One contract check**: `check_jax_operator(op, *inputs)` asserts that eager equals `eqx.filter_jit`, that `vmap` batches, that every learnable field is a leaf with a gradient, and that the config round-trips with its leaves still learnable.
- 🧱 **KernelOps as leaves**: with the `array` extra, `to_jax` turns a pipekit-array `KernelOp` into a `JaxKernelOp` whose array parameters are leaves, with the same dispatch and output signature.

## 📐 One operator, two contracts

The one object is an operator that is also a pytree, and each thing we want from a model is a contract on one side or the other.

| What we want | Maths | Names | Module |
|---|---|---|---|
| A step JAX can see into | $`\nabla_{\boldsymbol{\theta}} \mathcal{J}`$ for every array field | `JaxOperator`, `JaxModelOp` | `operator`, `model_op` |
| A pipeline JAX can see into | $`f = f_2 \circ f_1`$, with the leaves of both | `Sequential`, `Graph` (registered on import) | `pytree` |
| $`n`$ steps in one compiled loop | $`\mathbf{x}_{t+1} = f(\mathbf{x}_t; \boldsymbol{\theta})`$, $`t = 0, \ldots, n-1`$ | `ScanCycle`, `Rollout` | `cycle` |
| An ensemble in one call | $`\mathbf{x}^{(e)}_{t+1} = f(\mathbf{x}^{(e)}_t; \boldsymbol{\theta})`$, $`e \in \mathcal{E}`$ | `EnsembleScanCycle` | `cycle` |
| The trained weights back | $`\boldsymbol{\theta}`$ to bytes and back into a skeleton | `serialize_weights`, `with_weights`, `from_registry` | `model_op`, on every `JaxOperator` |
| A named-array kernel on JAX | the kernel's array parameters as leaves | `to_jax`, `JaxKernelOp` | `kernel` (`array` extra) |

For a scalar cost $`\mathcal{J}(\mathbf{x}_n)`$ of the final state, reverse mode through the scan multiplies the transposed Jacobians of the steps,

```math
\nabla_{\mathbf{x}_0} \mathcal{J} = \mathbf{J}_1^\top \mathbf{J}_2^\top \ldots \mathbf{J}_n^\top \nabla_{\mathbf{x}_n} \mathcal{J}
```

where $`\mathbf{J}_t = \partial \mathbf{x}_t / \partial \mathbf{x}_{t-1} \in \mathbb{R}^{D_x \times D_x}`$ is the Jacobian of step $`t`$, and the gradient with respect to $`\boldsymbol{\theta}`$ adds one term per step. Each factor needs the carry $`\mathbf{x}_{t-1}`$, so plain reverse mode stores $`n`$ carries. With `checkpoint=True` a `ScanCycle` call keeps one carry every $`\lfloor \sqrt{n} \rfloor`$ steps and recomputes the steps in between, which stores $`O(\sqrt{n})`$ carries for one extra forward pass, and the result is the same gradient. With `truncate=m` the gradient stops at $`\mathbf{x}_{n-m}`$: only the last $`m`$ factors survive, the parameter gradient keeps the terms of the last $`m`$ steps only, and the result is an approximation of the full gradient. The [docs quickstart](https://jejjohnson.github.io/pipekit/quickstart-jax/) runs both.

## 📥 Install

pipekit-jax is not on PyPI, so install it from the GitHub repository. It brings in pipekit-ops, pipekit-cycle, JAX, Equinox and jaxtyping, and needs Python 3.12 or newer.

```bash
pip install \
  "pipekit-ops @ git+https://github.com/jejjohnson/pipekit.git#subdirectory=packages/pipekit-ops" \
  "pipekit-cycle @ git+https://github.com/jejjohnson/pipekit.git#subdirectory=packages/pipekit-cycle" \
  "pipekit-jax @ git+https://github.com/jejjohnson/pipekit.git#subdirectory=packages/pipekit-jax"
```

The one extra, `array`, adds pipekit-array for `to_jax` and `JaxKernelOp`; both are imported lazily, so `import pipekit_jax` works without it.

```bash
pip install \
  "pipekit-ops @ git+https://github.com/jejjohnson/pipekit.git#subdirectory=packages/pipekit-ops" \
  "pipekit-cycle @ git+https://github.com/jejjohnson/pipekit.git#subdirectory=packages/pipekit-cycle" \
  "pipekit-array @ git+https://github.com/jejjohnson/pipekit.git#subdirectory=packages/pipekit-array" \
  "pipekit-jax[array] @ git+https://github.com/jejjohnson/pipekit.git#subdirectory=packages/pipekit-jax"
```

pipekit-jax names pipekit-ops and pipekit-cycle as ordinary requirements, and neither is on PyPI either, so a project that depends on pipekit-jax tells uv where each pipekit package lives. Pin every pipekit package to the same release tag: all the packages of one release share one commit, and two different tags for the same package make uv report conflicting URLs.

```toml
[project]
dependencies = ["pipekit-ops", "pipekit-cycle", "pipekit-array", "pipekit-jax[array]"]

[tool.uv.sources]
pipekit-ops = { git = "https://github.com/jejjohnson/pipekit", subdirectory = "packages/pipekit-ops", tag = "pipekit-array-v0.3.1" }
pipekit-cycle = { git = "https://github.com/jejjohnson/pipekit", subdirectory = "packages/pipekit-cycle", tag = "pipekit-array-v0.3.1" }
pipekit-jax = { git = "https://github.com/jejjohnson/pipekit", subdirectory = "packages/pipekit-jax", tag = "pipekit-array-v0.3.1" }
pipekit-array = { git = "https://github.com/jejjohnson/pipekit", subdirectory = "packages/pipekit-array", tag = "pipekit-array-v0.3.1" }
```

uv applies `[tool.uv.sources]` only to a project's direct dependencies, so every pipekit package named there is listed in `dependencies` too.

The `pipekit-array` line is needed only with the `array` extra. The `pipekit` umbrella distribution installs pipekit-jax with the other six packages.

## 🌀 Quickstart: a differentiable forecast

We step exponential decay, $`\mathrm{d}\mathbf{x} / \mathrm{d}t = -\lambda \mathbf{x}`$ with rate $`\lambda = 0.1`$ and $`\mathbf{x} \in \mathbb{R}^{D_x}`$, $`n = 24`$ times with a fourth-order Runge–Kutta model of step $`\Delta t = 1`$. So $`\boldsymbol{\theta} = \lambda`$, and $`k_1, \ldots, k_4`$ are the four stage tendencies of one step. The model is a plain `eqx.Module` with `step` and `dt`, which is all a pipekit-cycle `ForwardModel` asks for. For the cost $`\mathcal{J}(\mathbf{x}_0, \lambda) = \sum_i x_{n,i}`$, the exact solution gives $`\nabla_{\mathbf{x}_0} \mathcal{J} = e^{-n \lambda \Delta t} \mathbf{1}`$ and $`\partial \mathcal{J} / \partial \lambda = -n \Delta t D_x e^{-n \lambda \Delta t}`$, which the scheme matches to its truncation error.

```python
import equinox as eqx
import jax
import jax.numpy as jnp
from pipekit_jax import ScanCycle


# Shapes: D_x = 3 state elements; n = 24 steps


# user
# dx/dt = −λ x, stepped by fourth-order Runge–Kutta: x_t ↦ x_{t+1}
class Decay(eqx.Module):
    rate: jax.Array
    dt: float = 1.0
    # no shape contract on the state
    state_signature = None

    def tendency(self, state: jax.Array) -> jax.Array:
        # f(x) = −λ x
        # (D_x,) → (D_x,)
        return -self.rate * state

    def step(self, state: jax.Array, dt: float) -> jax.Array:
        k1 = self.tendency(state)
        k2 = self.tendency(state + dt / 2 * k1)
        k3 = self.tendency(state + dt / 2 * k2)
        k4 = self.tendency(state + dt * k3)
        # x_{t+1} = x_t + Δt (k₁ + 2 k₂ + 2 k₃ + k₄) / 6
        # (D_x,) → (D_x,)
        return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


# user
# 𝒥(x_0, λ) = ∑ᵢ x_{n,i}
def final_sum(initial: jax.Array, rate: jax.Array) -> jax.Array:
    # today (pipekit-jax)
    # x_0 ↦ x_n in one jax.lax.scan, with no state to carry
    forecast = ScanCycle(step_op=Decay(rate), n_steps=24)
    final, _ = forecast(initial, None)
    # (D_x,) → ()
    return jnp.sum(final)


# x_0 = 1 and λ = 0.1
# → (D_x,), ()
initial = jnp.ones(3)
rate = jnp.asarray(0.1)

# today (jax)
# ∇_{x_0} 𝒥 ≈ e^{−n λ Δt} 1 and ∂𝒥/∂λ ≈ −n Δt D_x e^{−n λ Δt}, through all n steps
gradients = eqx.filter_jit(jax.grad(final_sum, argnums=(0, 1)))(initial, rate)
# → (D_x,), ()
grad_initial, grad_rate = gradients

print(grad_initial.round(4), round(float(grad_rate), 3))
```

This prints `[0.0907 0.0907 0.0907] -6.532`: each element of $`\mathbf{x}_0`$ reaches the final sum with weight $`e^{-2.4} = 0.0907`$, and the sum falls by 6.532 per unit increase of $`\lambda`$, against $`-72 e^{-2.4} = -6.532`$ for the exact solution. Both gradients went back through all 24 Runge–Kutta steps inside one compiled scan. Nothing here is specific to decay, so any `ForwardModel` (an object with `step`, `dt` and `state_signature`, such as a finite-volume step) scans and differentiates the same way under `ScanCycle`, which also takes a `StatefulOperator`. A carrier-to-carrier operator, such as a learned emulator in a `JaxModelOp` or a `Graph` of `JaxOperator`s, steps through `Rollout` instead, since `ScanCycle` refuses it with a `TypeError`.

The [docs quickstart](https://jejjohnson.github.io/pipekit/quickstart-jax/) carries the rest: storing a trained emulator and reloading it from a registry, the signed weight blob, writing and checking a `JaxOperator`, gradients through a pipeline, the history, checkpointing and truncation of a cycle, `to_jax`, and how each of them works.

## 🔬 What else is in the box

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pipekit_jax_roles/pipekit_jax_roles-dark.svg">
    <img alt="pipekit-jax on the GeoStack role ladder: the scan cycles and Rollout as recipes, JaxOperator, JaxModelOp and the conversions as steps, with JAX and Equinox on the rungs below" src="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pipekit_jax_roles/pipekit_jax_roles-light.svg" width="100%">
  </picture>
</p>

Every name below is exported from `pipekit_jax`, except `check_jax_operator`, which is imported from `pipekit_jax.testing`. The steps (`JaxOperator`, `JaxModelOp`, `JaxKernelOp`) are one transformation each with their own config, ready to chain, and the cycles (`ScanCycle`, `EnsembleScanCycle`, `Rollout`) are recipes that run a step $`n`$ times as one job. `to_jax` builds a step, and `check_jax_operator` is a test helper beside the ladder.

| Module | Symbols | Notes |
|---|---|---|
| `pipekit_jax.operator` | `JaxOperator` | Base class for an operator that is also an `eqx.Module`: JSON config, pytree leaves and the weight round trip |
| `pipekit_jax.model_op` | `JaxModelOp` | Any `eqx.Module` as an operator, with `serialize_weights`, `with_weights` and `from_registry` |
| `pipekit_jax.cycle` | `ScanCycle`, `EnsembleScanCycle`, `Rollout` | Time stepping in one `jax.lax.scan`: carrier and state, `vmap`-ed over members, or a carrier-only trajectory |
| `pipekit_jax.kernel` | `to_jax`, `JaxKernelOp` | A pipekit-array `KernelOp` with its array parameters as leaves (`array` extra) |
| `pipekit_jax.pytree` | none (on import) | Registers `Sequential`, `Graph` and `JaxModelOp` as JAX pytrees |
| `pipekit_jax.testing` | `check_jax_operator` | Asserts the `JaxOperator` contract in a downstream test suite |

## 📓 Tutorials

| # | Tutorial | What it shows |
|---|---|---|
| 1 | [Pipelines under jit](https://jejjohnson.github.io/pipekit/pipekit-under-jit/) | a PDE step as a `Graph` of `JaxOperator`s, rolled out with `Rollout`, its two parameters recovered by gradient descent, and a 32-member `EnsembleScanCycle`, with compile times and memory measured |
| 2 | [Train, register, forecast](https://jejjohnson.github.io/pipekit/train-register-forecast/) | an emulator trained with pipekit-train, stored with its preprocessing, and reloaded through `JaxModelOp` for a forecast |

## 📖 Documentation

- **[Quickstart](https://jejjohnson.github.io/pipekit/quickstart-jax/)**: the registry round trip, the signed blob, `JaxOperator` and its contract check, pipelines as pytrees, the cycle options, `to_jax`, and how each works.
- **[API reference](https://jejjohnson.github.io/pipekit/reference/pipekit-jax/)**: every public symbol, generated from the docstrings, with the measured compile times and gradient memory of the cycles.
- **[Documentation](https://jejjohnson.github.io/pipekit/)**: the whole workspace, with the concepts, the other packages and their tutorials.
- **[pipekit-train boundaries, §13.1](https://jejjohnson.github.io/pipekit/boundaries-1/)**: the reproducibility caveat of pipekit-train's own `EquinoxModelOp`, which `JaxModelOp` closes.

## 🧩 Ecosystem

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pipekit_jax_ecosystem/pipekit_jax_ecosystem-dark.svg">
    <img alt="The packages that import pipekit-jax on their main branches, each with the modules it imports: gridax through its pipekit extra, sesmax-core and somax" src="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pipekit_jax_ecosystem/pipekit_jax_ecosystem-light.svg" width="100%">
  </picture>
</p>

pipekit-jax is one package of the pipekit [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/), at tier 0 of the [GeoStack](https://github.com/jejjohnson/geostack). It owns the bridge from pipekit's operators to JAX's transformations, and composes the rest in:

| Concern | Owner |
|---|---|
| `Operator`, `StatefulOperator`, `Sequential`, `Graph` and the JSON state record (`JaxOperator` extends them) | [`pipekit-ops`](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-ops) |
| The `ForwardModel` protocol and the adjoint specs (`ScanCycle` and `Rollout` step a `ForwardModel`) | [`pipekit-cycle`](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-cycle) |
| Modules as pytrees, filtered transformations and leaf serialisation | [`equinox`](https://github.com/patrick-kidger/equinox) |
| `jax.lax.scan`, `jax.checkpoint`, `jax.vmap` and `jax.grad` | [`jax`](https://github.com/jax-ml/jax) |
| `KernelOp` and its dispatch (the `array` extra) | [`pipekit-array`](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-array) |

Dependencies point one way: pipekit-ops, pipekit-cycle and pipekit-array never import pipekit-jax, and pipekit-jax reads a registry by duck typing, so pipekit-experiment is not a dependency either. Outside the workspace, gridax imports `JaxOperator` and `check_jax_operator` through its `pipekit` extra for its C-grid operators, and sesmax-core and somax import `JaxOperator` and `Rollout` for their rollout training and import `pipekit_jax` for its pytree registration. A forward model backed by a diffrax solve is [integrix](https://github.com/jejjohnson/integrix)'s `integrix.march.DiffraxForwardModel`, and integrix also interprets pipekit-cycle's adjoint specs, without importing any pipekit package.

## 🌳 Repository layout

```
packages/pipekit-jax/
├── src/pipekit_jax/
│   ├── __init__.py      ← the public names; to_jax and JaxKernelOp load lazily
│   ├── operator.py      ← JaxOperator
│   ├── model_op.py      ← JaxModelOp
│   ├── weights.py       ← the signed weight blob, shared by both
│   ├── cycle.py         ← ScanCycle, EnsembleScanCycle, Rollout
│   ├── kernel.py        ← to_jax, JaxKernelOp (array extra)
│   ├── pytree.py        ← registers Sequential, Graph and JaxModelOp on import
│   └── testing.py       ← check_jax_operator
└── tests/               ← the contract, the cycles against Cycle, the weight round trip, the tutorial under jit
```

## ⚙️ Development

The package is developed inside the workspace, so clone the repository and run the tasks from its root.

```bash
git clone https://github.com/jejjohnson/pipekit.git
cd pipekit
make install                              # every dependency group + the pre-commit hooks
uv run pytest packages/pipekit-jax/tests  # this package's tests
make lint                                 # ruff check on the whole repo
make typecheck                            # ty check on every package
make docs                                 # the MyST prose and the Zensical API reference, with a link check
```

`make help` lists every task. [`AGENTS.md`](https://github.com/jejjohnson/pipekit/blob/main/AGENTS.md) holds the rules for contributors and coding agents.

## 📄 License

MIT. See [LICENSE](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-jax/LICENSE).
