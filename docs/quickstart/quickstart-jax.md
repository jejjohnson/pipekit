# Quickstart: pipekit-jax

The [pipekit-jax README](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-jax) differentiates a 24-step forecast through one `jax.lax.scan`. This page carries the rest: why the package exists, where it sits, storing and reloading a trained model, the signed weight blob, writing and checking a `JaxOperator`, gradients through a pipeline, the options of the cycles, `to_jax`, and how each works. It uses the README's notation: $\mathbf{x}_t \in \mathbb{R}^{D_x}$ is the state after $t$ steps, $f(\mathbf{x}_t; \boldsymbol{\theta})$ is one step of a model with array fields $\boldsymbol{\theta} \in \mathbb{R}^{D_\theta}$, $n$ is the number of steps and $\mathcal{J}$ is a scalar cost.

Every block on this page runs as written, each on its own, with pipekit-jax installed; the first also needs pipekit-experiment for the registry, and the last needs the `array` extra.

## Why it exists

Consider the plume emulator from [pipekit-train](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-train): an MLP $g_\text{emu}(\mathbf{x}; \boldsymbol{\theta})$ that maps the emission rate $Q$ and the wind speed $u$, stacked as $\mathbf{x} = (Q, u) \in \mathbb{R}^{D_x}$, to the enhancement $\mathbf{y} \in \mathbb{R}^{D_y}$ at $D_y$ distances downwind. After training we want three things from it. We want to store it and get back exactly the weights we validated, months later; we want to step it thousands of times inside a forecast without paying Python overhead per step; and we want gradients through the whole rollout, so the emulator or the initial state can be fitted to observations.

A pipekit pipeline round-trips through JSON, but an `eqx.Module` is a pytree of arrays plus Python objects (activations, sizes) that JSON cannot carry. pipekit-jax closes that gap. `JaxModelOp` wraps a module as an operator whose weights travel as a separate, signed blob through any `ModelRegistry`, and `ScanCycle` is the jit-native counterpart of `pipekit_cycle.Cycle`: it steps any `ForwardModel`, and `jax.grad` goes through the whole rollout.

## Where it sits

pipekit-jax is built on pipekit-ops and pipekit-cycle, and it uses pipekit-array through the `array` extra. pipekit-ops and pipekit-array are libraries in their own right, which you can build on directly.

| Package | Kind of edge | What pipekit-jax takes from it |
|---|---|---|
| pipekit-ops | dependency | `Operator`, `StatefulOperator`, `Sequential`, `Graph` and the JSON state record |
| pipekit-cycle | dependency | the `ForwardModel` protocol that `ScanCycle` and `Rollout` step, and the adjoint specs |
| pipekit-array | `array` extra | `KernelOp`, which `to_jax` converts |
| JAX, Equinox | dependencies | the transformations, and modules as pytrees |

`JaxOperator` is both a `pipekit_ops.Operator` and an `eqx.Module`, and importing `pipekit_jax` registers `Sequential` and `Graph` as JAX pytrees, so `jax.grad` reaches the learnable leaves of operators inside a pipeline. `ScanCycle` takes a `ForwardModel`. With the `array` extra, `to_jax` turns a `pipekit_array.KernelOp` into a `JaxKernelOp` whose array parameters are pytree leaves. jaxtyping is a declared dependency as well, though no module of the package imports it.

| Extra | What it adds | When you need it |
|---|---|---|
| none | `pipekit-ops`, `pipekit-cycle`, `jax`, `equinox`, `jaxtyping` | `JaxOperator`, `JaxModelOp`, `ScanCycle`, `EnsembleScanCycle`, `Rollout` |
| `array` | `pipekit-array` | `to_jax` and `JaxKernelOp` |

`to_jax` and `JaxKernelOp` are imported lazily, so `import pipekit_jax` works without the extra, and calling `to_jax` without it raises an `ImportError` that names the extra.

## Store and reload a trained model

The emulator maps $\mathbf{x} = (Q, u)$ to $\mathbf{y}$ with weights $\boldsymbol{\theta}$, with $D_x = 2$ and $D_y = 4$. Here an untrained MLP stands in for the output of a `TrainingLoop`, and a second MLP of the same architecture with other weights is the skeleton that receives the stored ones.

```python
import tempfile

import equinox as eqx
import jax
import jax.numpy as jnp
from pipekit_experiment import LocalModelRegistry
from pipekit_jax import JaxModelOp


# Shapes: D_x = 2 inputs (Q, u); D_y = 4 downwind distances


# user
# g(x; θ), the same architecture built from any seed
def plume_mlp(seed: int) -> eqx.nn.MLP:
    key = jax.random.key(seed)
    return eqx.nn.MLP(in_size=2, out_size=4, width_size=32, depth=2, key=key)


# today (pipekit-jax)
# the trained emulator, here a stand-in for the output of a TrainingLoop
trained = JaxModelOp(plume_mlp(0))
# x = (Q, u), → (D_x,)
inputs = jnp.array([1.2, 3.0])

# today (pipekit-experiment)
registry = LocalModelRegistry(tempfile.mkdtemp())
# today (pipekit-jax)
# θ as a signed blob of bytes
weights = trained.serialize_weights()
# today (pipekit-experiment)
# the config and the blob, stored under one content hash
ref = registry.store(trained, name="plume-emulator", weights=weights)

# today (pipekit-experiment)
# the config alone holds no weights, so the registry refuses to rebuild the model
try:
    registry.load(ref)
    refused = False
except RuntimeError:
    refused = True

# today (pipekit-jax)
# a skeleton: the same structure with other weights
skeleton = JaxModelOp(plume_mlp(1))
# today (pipekit-jax)
# the skeleton with θ read back from the registry, byte for byte
reloaded = skeleton.from_registry(registry, "plume-emulator")
# g(x; θ) before and after the round trip, (D_x,) → (D_y,)
same_output = bool(jnp.all(reloaded(inputs) == trained(inputs)))
same_bytes = reloaded.serialize_weights() == weights

print(refused, same_output, same_bytes)
```

This prints `True True True`: the registry refused to rebuild the model from its config, and the skeleton came back with outputs and weight bytes identical to the trained model's.

`JaxModelOp` is marked `forbid_in_yaml`, because its config records only the module's class name: an `eqx.Module` carries Python objects (activation functions, layer sizes, dtypes) that JSON cannot rebuild. `pipekit_experiment`'s `registry.load` rebuilds operators from their config with `Operator.from_state`, so it raises a `RuntimeError` rather than return a model with made-up weights. The weights are still stored, verified against the content hash and readable with `load_weights`. The round trip is therefore:

1. store with `registry.store(op, weights=op.serialize_weights())`;
2. build a skeleton in code, the same architecture with any weights;
3. call `skeleton.from_registry(registry, ref)`, which reads `load_weights(ref)` and returns a copy of the skeleton holding the stored weights, byte for byte.

`from_registry` raises a `TypeError` for a registry without `load_weights` and a `ValueError` when the model was stored without a weight blob.

pipekit-train does this without extra code. Its `Checkpoint(registry=...)` callback stores the weight blob whenever the trained model has `serialize_weights`, and the Equinox backend of `TrainingLoop` returns the trained model in the wrapper it was given, so a `JaxModelOp` goes in and a `JaxModelOp` comes out. pipekit-train's own `EquinoxModelOp` is a deliberate twin with the same constructor, kept so that pipekit-train depends only on pipekit-ops: it writes its weights with `serialize_weights` but cannot read them back, so pass a `JaxModelOp` to `TrainingLoop` when the model has to be reloaded. The reproducibility caveat this closes is logged in pipekit-train's [boundaries page, §13.1](https://jejjohnson.github.io/pipekit/boundaries/). The [Train, register, forecast](../tutorials/train-register-forecast.md) tutorial runs the whole path.

## The weight blob is signed

`serialize_weights` writes a header before the `equinox.tree_serialise_leaves` payload: the pytree's node types and the key path, shape and dtype of every array leaf. Equinox loads leaves by position, so without the header a skeleton with the same number of leaves in another order, or another class with the same field names, would silently take weights it was never trained with. `with_weights` compares the header with the skeleton and raises a `ValueError` on any difference. Here the skeleton has hidden layers of width 16, and the blob was written by one of width 32.

```python
import equinox as eqx
import jax
from pipekit_jax import JaxModelOp


# today (pipekit-jax)
# weights of an MLP with hidden width 32
blob = JaxModelOp(
    eqx.nn.MLP(in_size=2, out_size=4, width_size=32, depth=2, key=jax.random.key(0))
).serialize_weights()

# today (pipekit-jax)
# a skeleton of another width: its key paths match, its shapes do not
wrong = JaxModelOp(
    eqx.nn.MLP(in_size=2, out_size=4, width_size=16, depth=2, key=jax.random.key(0))
)
try:
    wrong.with_weights(blob)
except ValueError as error:
    print(error)
```

This prints the error, which names the first leaf that differs:

```text
JaxModelOp.with_weights: the blob's tree structure does not match the skeleton (saved ['.layers[0].weight', [32, 2], 'float32'], skeleton ['.layers[0].weight', [16, 2], 'float32']). Build the skeleton with the same structure, field order and shapes as the operator that was serialised.
```

Only array leaves are stored, and every other value (a Python-float temperature, a flag, the static config) comes from the skeleton. A blob with more array leaves than the skeleton asks for also raises, because a partial restore would drop weights silently. Blobs written before the header existed, such as those of pipekit-train's `EquinoxModelOp`, still load, by position. `JaxModelOp` persists its wrapped module, and every `JaxOperator` persists itself, so any operator built on the base class has the same three methods.

## Write a `JaxOperator`

Subclass `JaxOperator` and declare fields as on any `eqx.Module`. The config is derived from the generated `__init__` and coerced to JSON builtins, so a `JaxOperator` with plain fields round-trips through `Operator.from_state`, and the same object is a pytree whose inexact-array fields are differentiable. The block writes one explicit diffusion step $u_i \mapsto u_i + \kappa (u_{i+1} - 2 u_i + u_{i-1}) / \Delta x^2$ on a periodic grid of $N$ cells, with spacing $\Delta x$ as configuration and diffusivity $\kappa$ as a learnable leaf.

```python
import equinox as eqx
import jax
import jax.numpy as jnp
from pipekit_jax import JaxOperator
from pipekit_jax.testing import check_jax_operator
from pipekit_ops import Operator


# Shapes: N = 64 grid cells


# user
# u ↦ u + κ ∇²u on a periodic grid of spacing Δx, with κ learnable
class Diffuse(JaxOperator):
    # Δx, configuration and not a leaf
    dx: float = eqx.field(static=True)
    # κ, a leaf that stays an array after a JSON round trip
    kappa: jax.Array = eqx.field(converter=jnp.asarray)

    def _apply(self, field: jax.Array) -> jax.Array:
        # ∇²u ≈ (u_{i+1} − 2 uᵢ + u_{i−1}) / Δx², (N,) → (N,)
        laplacian = (jnp.roll(field, -1) - 2 * field + jnp.roll(field, 1)) / self.dx**2
        # (N,) → (N,)
        return field + self.kappa * laplacian


# uᵢ = sin(2π i / N) on cells of unit width, → (N,)
field = jnp.sin(2 * jnp.pi * jnp.arange(64) / 64)
step = Diffuse(dx=1.0, kappa=0.25)

# today (pipekit-jax)
# eager ≡ filter_jit, vmap, a gradient for every learnable leaf, the config round trip
check_jax_operator(step, field)

# today (pipekit-ops)
# the JSON record and the operator rebuilt from it
rebuilt = Operator.from_state(step.state)
print(step.state["config"], eqx.is_inexact_array(rebuilt.kappa))
```

This prints `{'dx': 1.0, 'kappa': 0.25} True`: the JSON config of the step, and the rebuilt diffusivity is still an inexact array, so `eqx.filter_grad` still sees it.

Three rules make a `JaxOperator` behave:

- **Mark config-only fields `eqx.field(static=True)`**, so they are not leaves.
- **Give learnable fields `converter=jnp.asarray`.** A JSON reload passes a Python float back into the constructor, and a Python float is not an inexact array, so without the converter the reloaded parameter is silently a constant to `eqx.filter_grad`. `check_jax_operator` catches this. JSON also drops dtypes, so `op.state` raises a `TypeError` for an array field whose converter would not restore its dtype (a `float16` field needs a casting converter), and for complex values.
- **Set `forbid_in_yaml = True` when a field holds something JSON cannot carry** (a callable, a module without a JSON form), and persist its arrays with `serialize_weights`. The state record of such an operator is still JSON-encodable, with a debug form of each such value, so a registry can store it next to the blob.

`check_jax_operator(op, *inputs)` checks, in one call, that the operator is a pipekit `Operator` and an `eqx.Module`, that eager equals `eqx.filter_jit` with the operator passed as an argument, that `jax.vmap` over stacked inputs matches the stacked eager outputs (skip it with `vmap=False`), that every inexact-array field is a leaf with a gradient, and, unless the operator is `forbid_in_yaml`, that the config round trip keeps the same leaves and the same outputs. Call it from a library's test suite on every operator it ships.

Compile an operator with `eqx.filter_jit(op)`, or with `jax.jit` on a function that takes the operator as an argument. `jax.jit(op)` hashes the callable, and an `eqx.Module` with array fields is unhashable.

## Gradients through a pipeline

Importing `pipekit_jax` registers `pipekit_ops.Sequential`, `pipekit_ops.Graph` and `JaxModelOp` with `jax.tree_util`, so JAX transformations see through a pipeline to the leaves of the operators it holds. The block chains two scalings, $\mathbf{x} \mapsto a_2 a_1 \mathbf{x}$ with $\mathbf{x} \in \mathbb{R}^{N}$, and differentiates $\mathcal{L}(a_1, a_2) = \sum_i a_2 a_1 x_i$ with respect to both factors.

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import pipekit_jax  # noqa: F401  (registers Sequential and Graph as pytrees)
from pipekit_jax import JaxOperator
from pipekit_ops import Sequential


# Shapes: N = 3 values


# user
# x ↦ a x, with a learnable
class Scale(JaxOperator):
    factor: jax.Array = eqx.field(converter=jnp.asarray)

    def _apply(self, values: jax.Array) -> jax.Array:
        # (N,) → (N,)
        return self.factor * values


# today (pipekit-ops)
# x ↦ a₂ a₁ x, a pipeline of two operators
pipeline = Sequential([Scale(2.0), Scale(3.0)])
# → (N,)
values = jnp.ones(3)


# user
# 𝓛(a₁, a₂) = ∑ᵢ a₂ a₁ xᵢ
def total(pipe: Sequential, values: jax.Array) -> jax.Array:
    # (N,) → ()
    return jnp.sum(pipe(values))


# today (equinox)
# ∂𝓛/∂a₁ = a₂ ∑ᵢ xᵢ and ∂𝓛/∂a₂ = a₁ ∑ᵢ xᵢ, one per leaf inside the pipeline
grads = eqx.filter_grad(total)(pipeline, values)
print(len(jax.tree.leaves(pipeline)), [float(leaf) for leaf in jax.tree.leaves(grads)])
```

This prints `2 [9.0, 6.0]`: the pipeline has two leaves, and with $a_1 = 2$, $a_2 = 3$ and $\sum_i x_i = 3$ the gradients are $\partial \mathcal{L} / \partial a_1 = 9$ and $\partial \mathcal{L} / \partial a_2 = 6$.

`Sequential`'s children are its operators, in order. `Graph`'s children are its node operators in topological order, and the topology (inputs, node wiring, outputs) is static aux data that holds no operators or parameters. An operator that is itself a pytree (a `JaxOperator`, or a `JaxModelOp` through its module) contributes its own leaves, and a plain `pipekit_ops.Operator` is one opaque leaf: fine under `eqx.filter_jit` and `eqx.filter_grad`, where non-array leaves are static, but not as a direct argument to `jax.jit`. Without the registration a container is one opaque leaf, and a learnable coefficient inside it is a constant to `jax.grad`. pipekit-ops stays JAX-free, because the registration is an import side effect of this package, like setting a `jax.config` flag.

:::{note}
Registration is by exact type, so subclasses of `Sequential` and `Graph` are not registered. Operator sharing is not preserved: an operator used by two graph nodes flattens to two children, and after an unflatten with new leaves they are two independent objects, as with Equinox modules. `Fanout`, `Branch`, `Switch`, `Cache`, `Try` and the parallel maps are not registered yet.
:::

## Cycles under `jit`

`pipekit_cycle.Cycle` steps in a Python `for` loop. Under `jax.jit` that loop unrolls, so the traced program, the compile time and the memory all grow with the number of steps, and `save_history` appends to `self.history` from inside traced code. `ScanCycle(step_op, n_steps)` keeps `Cycle`'s contract, $(\mathbf{c}_{t+1}, \mathbf{s}_{t+1}) = \mathrm{step}(\mathbf{c}_t, \mathbf{s}_t)$ for a carrier $\mathbf{c}_t$ and a state $\mathbf{s}_t$, and returns $(\mathbf{c}_n, \mathbf{s}_n)$ exactly like `Cycle`, but compiles to one `jax.lax.scan` whatever $n$ is. The step is a `StatefulOperator`, or a `ForwardModel`, which is wrapped so that the state passes through unchanged. `run` also returns the history, recorded with `Cycle`'s convention (after step $i + 1$ for every $i$ divisible by the stride $s$, so $\lceil n / s \rceil$ entries), rather than storing it on the object.

The block uses one explicit Euler step of the README's decay, $\mathbf{x}_{t+1} = (1 - \lambda \Delta t) \mathbf{x}_t$ with $\lambda = 0.1$, $\Delta t = 0.5$, $n = 24$ and $s = 4$, for which $\partial \mathcal{J} / \partial \lambda = -n \Delta t D_x (1 - \lambda \Delta t)^{n-1}$ with $\mathcal{J} = \sum_i x_{n,i}$. Each `today (jax)` line differentiates a `ScanCycle` with other settings.

```python
import equinox as eqx
import jax
import jax.numpy as jnp
from pipekit_jax import ScanCycle


# Shapes: N = 3 state variables; n = 24 steps; n / s = 6 records


# user
# x_{t+1} = (1 − k Δt) x_t, one explicit Euler step of dx/dt = −k x
class Decay(eqx.Module):
    rate: jax.Array
    dt: float = 0.5
    state_signature = None

    def step(self, state: jax.Array, dt: float) -> jax.Array:
        # (N,) → (N,)
        return state - dt * self.rate * state


# today (pipekit-jax)
# x_0 ↦ x_n, recording every s = 4 steps
cycle = ScanCycle(step_op=Decay(jnp.asarray(0.1)), n_steps=24, history_stride=4)
# → (N,)
initial = jnp.ones(3)
# today (pipekit-jax)
# x_n and the records x_1, x_5, …, x_21, → (N,), (n / s, N)
final, _, (records, _) = eqx.filter_jit(cycle.run)(initial, None)


# user
# 𝒥(k) = ∑ᵢ x_{n,i} for a cycle with the given settings
def final_sum(rate: jax.Array, **settings: object) -> jax.Array:
    # today (pipekit-jax)
    forecast = ScanCycle(step_op=Decay(rate), n_steps=24, **settings)
    final, _ = forecast(initial, None)
    # (N,) → ()
    return jnp.sum(final)


rate = jnp.asarray(0.1)
# today (jax)
# ∂𝒥/∂k through all n steps, keeping every carry
full = jax.grad(final_sum)(rate)
# today (jax)
# the same gradient, keeping a carry every ⌊√n⌋ = 4 steps and recomputing the rest
remat = jax.grad(lambda k: final_sum(k, checkpoint=True))(rate)
# today (jax)
# ∂𝒥/∂k through the last 6 steps only: 6 / 24 of the full gradient here
cut = jax.grad(lambda k: final_sum(k, truncate=6))(rate)

# → ()
ratio = cut / full
print(records.shape, f"{full:.4f} {remat:.4f} {ratio:.2f}")
```

This prints `(6, 3) -11.0648 -11.0648 0.25`: six records of the three-element state, the full gradient (against $-36 \times 0.95^{23} = -11.065$), the same gradient with checkpointing, and the truncated gradient as a quarter of the full one. With truncation the carry entering the last $m$ steps is a constant, so for this linear decay only $m$ of the $n$ equal terms of $\partial \mathcal{J} / \partial \lambda$ remain, and $m / n = 6 / 24$.

- **`checkpoint=True`** rematerialises on the backward pass. The forward pass keeps the carry only at chunk boundaries (every $\lfloor \sqrt{n} \rfloor$ steps when the cycle is called, every $s$ steps in `run`), and reverse mode recomputes the steps inside a chunk. Memory for `grad` drops from $O(n)$ carries to $O(\sqrt{n})$, or $O(n / s + s)$ in `run`, for one extra forward pass. The [API page](https://jejjohnson.github.io/pipekit/reference/pipekit-jax/) has the measured memory.
- **`truncate=m`** lets gradients flow through the carry of the last $m$ steps only: the carry leaving step $i$ is `stop_gradient`-ed when $i < n - m$, as `integrix.adjoint.truncated_scan` does. History entries recorded by `run` keep the gradient of their own step, as scan outputs do, and forward values are unchanged.
- **`EnsembleScanCycle(cycle, state_axis)`** `vmap`s a cycle over a leading member axis of size $N_\mathcal{E}$ on every leaf of the carrier. With `state_axis=None` (the default) one state is shared by all members and is read-only: every step of every member reads the incoming state, the state a step returns is discarded, and the incoming state is returned unchanged. With `state_axis=0` each member threads its own state, stacked on a leading axis like the carrier. It is not a drop-in for `pipekit_cycle.EnsembleCycle` when the step writes a shared state, because `EnsembleCycle` threads one state through the members in turn, which a `vmap` cannot reproduce; use `state_axis=0` or `EnsembleCycle` for that.
- **`Rollout(step, n_steps, save_every)`** is the carrier-only form: it steps an operator or a `ForwardModel`, $\mathbf{c}_{t+1} = \mathrm{step}(\mathbf{c}_t)$, and returns the trajectory $(\mathbf{c}_0, \mathbf{c}_k, \mathbf{c}_{2k}, \ldots, \mathbf{c}_n)$ for $k$ = `save_every`, stacked along a new leading axis, with $\mathbf{c}_0$ left out when `include_initial=False`. `n_steps` must be a multiple of `save_every`, a stateful step raises a `TypeError` (use `ScanCycle`), and `checkpoint=True` keeps one carry per saved point, $O(n / k + k)$.

All three handle any pytree carrier (dicts, `eqx.Module` states, custom carriers) by stacking with `jax.tree.map`, and all three are `JaxOperator`s, so the learnable leaves of the step are reachable by `eqx.filter_grad`. The [Pipelines under jit](../tutorials/pipekit-under-jit.md) tutorial walks through `Rollout`, `ScanCycle` and `EnsembleScanCycle` on a diffusion model, with measured compile times.

## KernelOps on JAX

A pipekit-array `KernelOp` that holds arrays (a fitted background mean and covariance, a learned target) needs them as pytree leaves under `eqx.filter_jit` and `eqx.filter_grad`. `to_jax` generates that shell from the `KernelOp` instead of a hand-written operator per kernel: every parameter that is an array or holds arrays (a `DimArray` of statistics) becomes a leaf, converted with `jnp.asarray`, and the kernel, its spec, the core-dim map, the geometry functions and every other parameter are static. The block shifts $N$ values by an offset $b$ and differentiates $\mathcal{L}(b) = \sum_i (x_i + b)$.

```python
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from pipekit_array import KernelOp
from pipekit_jax import to_jax


# Shapes: N = 4 values


# user
# x ↦ x + b, elementwise
def shift(values: np.ndarray, offset: np.ndarray) -> np.ndarray:
    # (N,) → (N,)
    return values + offset


# today (pipekit-array)
# a kernel with one array parameter b
kernel_op = KernelOp(shift, "()->()", core={}, offset=np.float32(1.0))
# today (pipekit-jax)
# the same kernel with b as a pytree leaf
jax_op = to_jax(kernel_op)


# user
# 𝓛(b) = ∑ᵢ (xᵢ + b)
def total(op: object, values: jnp.ndarray) -> jnp.ndarray:
    # (N,) → ()
    return jnp.sum(op(values))


# today (equinox)
# ∂𝓛/∂b = N
grads = eqx.filter_grad(total)(jax_op, jnp.arange(4.0))
print(sorted(jax_op.params), float(grads.params["offset"]))
```

This prints `['offset'] 4.0`: the one array parameter became a leaf, and its gradient is $N = 4$. The call is the same dispatch as the `KernelOp`'s (`pipekit_array.kernels.run_kernel`), so the same carriers work, and `compute_output_signature` returns the `KernelOp`'s output signature. A non-array parameter must be hashable, since it becomes a static field, or `to_jax` raises a `TypeError`. `JaxKernelOp` holds the kernel and the geometry functions, so its config is a debug form only (`forbid_in_yaml`); persist learned arrays with `serialize_weights`. `to_jax` lives in pipekit-jax rather than on `KernelOp` because pipekit-jax already reaches pipekit-array, through pipekit-cycle's `array` extra, and an edge from pipekit-array to pipekit-jax would close a cycle.

## Adjoint specs and diffrax models, in integrix

pipekit-cycle's adjoint specs carry no behaviour, and the adjoint maths lives in [integrix](https://github.com/jejjohnson/integrix). `integrix.adjoint.to_diffrax_adjoint` maps `RecursiveCheckpointAdjoint`, `DirectAdjoint`, `BacksolveAdjoint` and `ImplicitAdjoint` onto the diffrax classes of the same name and raises for `TruncatedAdjoint`, and `integrix.adjoint.truncated_scan` does what `TruncatedAdjoint` says for a rollout. A forward model that a diffrax solve computes, of an ODE or an SDE, is `integrix.march.DiffraxForwardModel`, and it steps in a `ScanCycle` like any other `ForwardModel`. Specs are matched by class name and fields, so integrix imports no pipekit package, and pipekit imports no library above it. Both used to live in pipekit-jax.

## Not built yet

- **Interpreters for the inner-solve and cycle layers.** `integrix.adjoint` covers the dynamics layer and `ScanCycle(truncate=m)` covers truncation; checkpointed and implicit adjoints for a variational solve live in the algorithm libraries (vardax, filterax).
- **Pytree registration for the other containers** (`Fanout`, `Branch`, `Switch`, `Cache`, `Try`, the parallel maps).

## Further reading

- API reference: [pipekit-jax](https://jejjohnson.github.io/pipekit/reference/pipekit-jax/).
- Tutorials: [Pipelines under jit](../tutorials/pipekit-under-jit.md) and [Train, register, forecast](../tutorials/train-register-forecast.md).
- Design: the reproducibility caveat this package closes is in pipekit-train's [boundaries page, §13.1](https://jejjohnson.github.io/pipekit/boundaries/), and the master plan is [Report 11](https://github.com/jejjohnson/research_journal_v2/blob/main/notes/geotoolz/master_plan/toolz_9_pipekit_train.md).
- The foundations: [pipekit-ops](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-ops) and [pipekit-array](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-array).
- Siblings: [pipekit-cycle](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-cycle) (the protocols and adjoint specs), [pipekit-experiment](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-experiment) (the registry) and [pipekit-train](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-train) (the training loop).
