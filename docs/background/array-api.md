# Background: the array API standard

*What the standard gives pipekit-array, and what pipekit-array adds to it.*

pipekit-array is written against the [Python array API standard](https://data-apis.org/array-api/) rather than against NumPy or JAX. This page says what the standard is, what it covers, and which gaps pipekit-array fills with its own code. It is background: for the names that ship, read the [tour](../tour-array.md), and for the rules of mixing libraries, the [carriers and backends guide](../guides/carriers-and-backends.md).

:::{note} About this page
:class: dropdown

| | |
|---|---|
| **Reading time** | about 10 minutes |
| **Audience** | anyone who wants to know why pipekit-array calls `xp.concat` and writes its own `set_at` |
| **Companions** | [Vision, section 5](../design-array/vision.md) (the bet), [ADR A23, A25 and A27](../design-array/decisions.md) (the host rule and the fillers), [Writing a library on pipekit-array](../guides/library-on-pipekit-array.md) (the fillers at work) |
| **Checked against** | pipekit-array 0.3.1, NumPy 2.4, JAX 0.10 and array-api-compat 1.14; every output below is from a real run |
:::

## What the standard is

The standard is a specification that array libraries follow, so that one function works on any of them. Every conforming array implements `__array_namespace__()`, which returns a module-like object with `mean`, `sum`, `reshape`, `where` and the rest as functions ([vision, section 5](../design-array/vision.md)). A function that takes the namespace from its input, and calls only what is in it, runs on NumPy, JAX, CuPy, PyTorch and dask, the five backends the vision page sets out to cover. [array-api-compat](https://github.com/data-apis/array-api-compat) gives the namespace of a library that has none of its own (PyTorch), and it is the base of the plain-array functions of `pipekit_array.arrays`; this is why PyTorch needs the `compat` extra.

```python
import array_api_compat
import array_api_strict as xs
import jax.numpy as jnp
import numpy as np
from pipekit_array.arrays import namespace_of


# Shapes: N = 3 points


# today (pipekit-array)
# (N,), on three libraries: the namespace each array reports itself
arrays = {"numpy": np.ones(3), "jax": jnp.ones(3), "strict": xs.ones(3)}
reported = {name: x.__array_namespace__().__name__ for name, x in arrays.items()}
# today (pipekit-array)
# the namespace that array-api-compat and pipekit-array find from the same arrays
found = {name: array_api_compat.array_namespace(x).__name__ for name, x in arrays.items()}
chosen = {name: namespace_of(x).__name__ for name, x in arrays.items()}
print(reported)
print(found)
print(chosen)
```

This prints

```text
{'numpy': 'numpy', 'jax': 'jax.numpy', 'strict': 'array_api_strict'}
{'numpy': 'array_api_compat.numpy', 'jax': 'jax.numpy', 'strict': 'array_api_strict'}
{'numpy': 'array_api_compat.numpy', 'jax': 'jax.numpy', 'strict': 'array_api_strict'}
```

that is, the namespace of each array as the array reports it, as array-api-compat finds it (a wrapped NumPy, which adds what NumPy lacks) and as `namespace_of` returns it, which is array-api-compat's.

## What it covers

The standard is well defined for the operations that pipekit-array needs: basic maths, reductions, reshapes, indexing and concatenation. The names are the standard's, and they are not always the older NumPy names: the code calls `xp.concat`, `xp.permute_dims`, `xp.astype` and `xp.isdtype`. A library that tests on `array_api_strict` finds the calls that only work on NumPy: radiomax's contract is that only functions of the standard are called on the namespace, and its tests run on `array_api_strict` to refuse the rest.

```python
# today (pipekit-array)
# what the strict namespace has, and what only NumPy has
wanted = ["concat", "permute_dims", "astype", "isdtype", "mean", "where", "reshape"]
extras = ["concatenate", "transpose", "einsum", "bincount", "pad", "nanmean"]
print({name: hasattr(xs, name) for name in wanted})
print({name: hasattr(xs, name) for name in extras})
```

This prints

```text
{'concat': True, 'permute_dims': True, 'astype': True, 'isdtype': True, 'mean': True, 'where': True, 'reshape': True}
{'concatenate': False, 'transpose': False, 'einsum': False, 'bincount': False, 'pad': False, 'nanmean': False}
```

that is, the names that the code calls on a namespace are on the strict one, and the names that NumPy has and the standard has not are not: no `concatenate`, `transpose`, `einsum`, `bincount`, `pad` or `nanmean`.

## What it leaves out

[Vision, section 5](../design-array/vision.md) lists where the standard is not enough. Each item below says what pipekit-array does about it.

| Gap | What pipekit-array does |
|---|---|
| Indexed updates, loops, padding | `set_at`, `add_at`, `multiply_at`, `scan`, `vmap`, `pad` ([ADR A27](../design-array/decisions.md)) |
| Binning, `einsum` | `Groups` in place of `bincount` ([ADR A25](../design-array/decisions.md)); `dot` uses `einsum` where the namespace has it |
| NaN semantics | named, never hidden ([ADR A6](../design-array/decisions.md)) |
| dask laziness | never computed for you ([ADR A7](../design-array/decisions.md)) |
| Mutability | every function is pure ([ADR A2](../design-array/decisions.md)) |
| Device and dtype of host values | the host rule ([ADR A23](../design-array/decisions.md)) |
| Named dims | the named layer: `DimArray` and the dim functions |
| FFTs, advanced linear algebra | nothing yet: an extension that libraries implement inconsistently, which would need a per-library fallback ([boundaries, Q1](../design-array/boundaries.md)) |

### Updates, loops and padding

The standard has no `x[i] = v` that works everywhere, and JAX arrays are immutable. `set_at`, `add_at` and `multiply_at` return an updated copy: `x.at[i]` on JAX data, and a copy with an assignment elsewhere. `scan` and `vmap` are loops that JAX compiles and that run as Python loops elsewhere, and `pad` puts zeros around an array. On JAX data they are JAX's own calls, with the same jaxpr, and the tests compare the two; a library that stays on JAX loses nothing. A repeated index adds each time, where NumPy's `y[[0, 0]] += 1` adds once.

```python
import jax
from pipekit_array.arrays import add_at, pad, set_at


# Shapes: N = 3 points


# today (pipekit-array)
# the update of the library's own call, as a jaxpr: x₁ = 5
on_jax = jnp.zeros(3)
same = str(jax.make_jaxpr(lambda x: set_at(x, 1, 5.0))(on_jax)) == str(
    jax.make_jaxpr(lambda x: x.at[1].set(5.0))(on_jax)
)
# today (pipekit-array)
# y₀ += 1 twice: NumPy's y[[0, 0]] += 1 adds once, add_at adds each time
y = np.zeros(3)
y[[0, 0]] += 1.0
twice = add_at(np.zeros(3), np.asarray([0, 0]), 1.0)
# today (pipekit-array)
# on the strict namespace, which has neither `at` nor `pad`: (N + 2,)
padded = pad(set_at(xs.ones(3), 0, 9.0), ((1, 1),))
print(same, y.tolist(), twice.tolist(), np.asarray(padded).tolist())
```

This prints `True [1.0, 0.0, 0.0] [2.0, 0.0, 0.0] [0.0, 9.0, 1.0, 1.0, 0.0]`: the same jaxpr as JAX's own update, one addition for NumPy's `+=` and two for `add_at`, and an array that was updated and padded on a library that has neither.

### The device and the dtype of host values

A formula takes a constant or a table from NumPy next to data on another library, and a user expects them to join. The standard says nothing about this, so [ADR A23](../design-array/decisions.md) sets a host rule: Python numbers, sequences and NumPy arrays join the library and the device of the other arrays, and an array of the call's library is never rebuilt. A Python number takes the dtype of the arrays, so float32 stays float32. A list or a NumPy array counts with its own dtype, so a float64 one widens a float32 field to float64, which a guard reports.

```python
from pipekit_array.arrays import floating_inputs, like


# Shapes: N = 2 points


# today (pipekit-array)
# (N,), float32, on a second device
device = xs.Device("device1")
field = xs.asarray([1.0, 2.0], dtype=xs.float32, device=device)
# today (pipekit-array)
# a Python float takes the field's dtype, and a list its own; both go to the field's device
_, (_, number) = floating_inputs(field, 2.0)
_, (_, listed) = floating_inputs(field, [3.0, 4.0])
# today (pipekit-array)
# a constant in the field's dtype, on its device
constant = like(0.5, field, xs)
print(number.dtype, number.device == device, listed.dtype, listed.device == device, constant.dtype)
```

This prints `array_api_strict.float32 True array_api_strict.float64 True array_api_strict.float32`: the number in the field's float32 and the list in float64, both on the field's second device, and a constant in float32.

### Named dims

The standard has axes by position only, and no dim names, labels or coordinates. This is the one gap that is not a missing function: the named layer of pipekit-array exists to fill it, with `DimArray` (a raw array, one name per axis and static labels) and the dim functions that take names in place of axes, which run on the standard's namespace underneath. The [tour](../tour-array.md) shows them.

## Where next

- [Vision, section 5](../design-array/vision.md): the same list as the design page states it.
- [Carriers and backends](../guides/carriers-and-backends.md): what joins and what raises.
- [Writing a library on pipekit-array](../guides/library-on-pipekit-array.md): the fillers in a formula library.
