# Writing a library on pipekit-array

A formula library is a set of functions of plain arrays: they take pressures, radiances or angles and return the same kind of array, with no dim names. [geophysix](https://github.com/jejjohnson/geophysix) and [radiomax](https://github.com/jejjohnson/radiomax) are written this way on pipekit-array. This page builds one small module of the same shape, from the formula to the tests, on a running example: the density of dry air from pressure and temperature, $\rho = p / (R_d T)$, its speed of sound, and the mass of air in a column of layers.

The blocks run in order, as one script, so a later block uses the names of an earlier one. They need the `numpy`, `jax`, `compat`, `xarray` and `equinox` packages. A library that only ships formulas depends on `pipekit-array[numpy,compat]` and leaves JAX and xarray to its users, as radiomax does: the base install must import and pass its tests with neither.

## The formulas

Each function opens with `floating_inputs`, which returns the namespace of the call and the inputs in one floating dtype: Python numbers and NumPy arrays join the library of the other arrays, float32 stays float32, and integers become the namespace's default float ([ADR A23](../design-array/decisions.md)). The body calls the namespace `xp` only, so it runs on NumPy, JAX, dask and any other library of the array API standard. Three more helpers keep it honest:

- `like(value, ref, xp)` puts a constant in the dtype and on the device of a reference array, so a float32 field is not widened by a Python float.
- `check_values(array, bad, message)` raises on concrete arrays and under `jit` through Equinox, so a rule such as $T > 0$ holds on every path.
- `safe_sqrt` is 0 with a zero gradient at $x \le 0$, and NaN at NaN ([ADR A26](../design-array/decisions.md)), so the gradient of the other cells stays finite.

```python
import jax
import numpy as np
from pipekit_array.arrays import check_values, floating_inputs, like, safe_sqrt


# the library's conftest.py turns this on (see the test kit below); here it is first,
# so that JAX arrays are float64 for the whole page
jax.config.update("jax_enable_x64", True)


# Shapes: N = 3 stations


# user
# R_d: the specific gas constant of dry air, J kg⁻¹ K⁻¹
R_DRY = 287.05


# user
# ρ = p / (R_d T), in kg m⁻³
def air_density(pressure, temperature, gas_constant=R_DRY):
    # today (pipekit-array)
    # the namespace of the call, and both inputs in one floating dtype
    xp, (p, t) = floating_inputs(pressure, temperature)
    # today (pipekit-array)
    # T > 0 K on every path, also under jit
    t = check_values(t, xp.any(t <= 0), "'temperature' must be > 0 K")
    # today (pipekit-array)
    # R_d in p's dtype, on its device
    r = like(gas_constant, p, xp)
    # (N,)
    return p / (r * t)


# user
# c = √(γ R_d T), in m s⁻¹
def sound_speed(temperature, heat_ratio=1.4, gas_constant=R_DRY):
    xp, (t,) = floating_inputs(temperature)
    gamma, r = like(heat_ratio, t, xp), like(gas_constant, t, xp)
    # today (pipekit-array)
    # 0 with a zero gradient where γ R T ≤ 0
    return safe_sqrt(gamma * r * t, xp)


pressure = np.asarray([100_000.0, 90_000.0, 80_000.0])
temperature = np.asarray([288.0, 281.0, 275.0])
print(air_density(pressure, temperature).round(4).tolist(), air_density(pressure, 288.0).dtype)
try:
    air_density(pressure, -temperature)
except ValueError as error:
    print(type(error).__name__, str(error).split(",")[0])
```

This prints

```text
[1.2096, 1.1158, 1.0134] float64
ValueError 'temperature' must be > 0 K
```

that is, the density of three stations, a Python-number temperature that joins a NumPy pressure, and the check that refuses a negative temperature.

## Formulas that take carriers

A function of plain arrays has no dim names. `elementwise` lets it take named carriers as well: a `DimArray`, an `xr.DataArray` or any carrier with a registered adapter. Arguments align by dim name, the function is called on the data, and the result has the dims of all the inputs together. A DataArray result has no name and no attributes, because the quantity and its unit have changed, and keeps its coordinates. An array with dims but no names next to a labelled one raises `ValueError`, and a DataArray next to another kind of carrier raises `BackendError`.

```python
import jax.numpy as jnp
import pipekit_array as pa
from pipekit_array.kernels import elementwise


# Shapes: S = 2 sites; L = 3 levels


# today (pipekit-array)
# the same formula, now taking carriers as well
air_density = elementwise()(air_density)
sound_speed = elementwise()(sound_speed)

# today (pipekit-array)
# (S, L) on NumPy, and the temperature on JAX
p2 = pa.DimArray(np.asarray([[100_000.0, 90_000.0, 80_000.0], [101_000.0, 91_000.0, 81_000.0]]), ("site", "level"))
t2 = pa.DimArray(jnp.asarray([[288.0, 281.0, 275.0], [290.0, 283.0, 277.0]]), ("site", "level"))
# (S, L) on JAX: NumPy joins JAX
rho = air_density(p2, t2)
# (S, L) as a DataArray, from the same data
rho_xr = air_density(p2.to_xarray(), t2.to_xarray())
print(type(rho).__name__, rho.dims, type(rho.data).__name__, type(rho_xr).__name__, rho_xr.name)
```

This prints `DimArray ('site', 'level') ArrayImpl DataArray None`: a `DimArray` on JAX from NumPy and JAX inputs, and a nameless DataArray from DataArrays.

## A function that consumes a dim

The mass of air in a column is a sum over the levels, $m = \sum_l \rho_l \Delta z_l$, in kg m⁻². That function consumes a dim, so it cannot be element-wise. Write it for arrays with the dim last, and describe it with a generalised-ufunc signature: `"(l),(l)->()"` reads as two inputs with a core dim `l` each and an output with none. `KernelOp` wraps it once, and `core` says which carrier dim `l` stands for, so the same function serves `"level"` on one carrier and `"layer"` on another. Every output has the loop dims of all the inputs, in the first input's order.

```python
from pipekit_array.kernels import KernelOp, kernel_output_dims


# Shapes: S = 2 sites; L = 3 levels


# user
# m = ∑ₗ ρₗ Δzₗ: (L,), (L,) → ()
def column_mass(density, thickness):
    xp, (rho, dz) = floating_inputs(density, thickness)
    # (S, L) → (S,)
    return xp.sum(rho * dz, axis=-1)


# today (pipekit-array)
# "l" is the level dim of the carrier
mass = KernelOp(column_mass, "(l),(l)->()", core={"l": "level"})
# (S, L): layers of 1000 m
dz2 = pa.DimArray(np.full((2, 3), 1000.0), ("site", "level"))
# (S, L), (S, L) → (S,)
column = mass(rho, dz2)
print(column.dims, np.asarray(column.data).round(1).tolist(), kernel_output_dims(mass.spec, {"l": "level"}, [("site", "level")] * 2))
```

This prints `('site',) [3338.8, 3352.2] (('site',),)`: the level dim consumed, the mass of each column in kg m⁻², and the output dims known from the input dims alone. A `KernelOp` holds a user function, so its config is a debug record (`forbid_in_yaml`) and it does not save to JSON.

## A named operator that saves

A named operator is a pipekit-ops `Operator` that builds the `KernelOp` and calls it, so what it saves is its own arguments, not a function. Write `_apply` and `get_config`; the config holds JSON values only. Build the `KernelOp` inside `_apply`, because `pipekit_ops.testing.assert_round_trip` refuses an operator that holds a `KernelOp` as an attribute, and then check that it saves and loads, and that the loaded copy has the same `pipeline_hash`.

```python
import json

import pipekit_ops as pk
from pipekit_ops.testing import assert_round_trip


# Shapes: S = 2 sites; L = 3 levels


class ColumnAirMass(pk.Operator):
    """m = ∑ₗ (p / (R_d T))ₗ Δzₗ, in kg m⁻², over the dim ``dim``."""

    def __init__(self, dim="level", gas_constant=R_DRY):
        self.dim = dim
        self.gas_constant = gas_constant

    # today (pipekit-array)
    # (S, L), (S, L), (S, L) → (S,)
    def _apply(self, pressure, temperature, thickness):
        kernel = KernelOp(column_mass, "(l),(l)->()", core={"l": self.dim})
        return kernel(air_density(pressure, temperature, self.gas_constant), thickness)

    def get_config(self):
        return {"dim": self.dim, "gas_constant": self.gas_constant}


op = ColumnAirMass()
# today (pipekit-ops)
# the operator, its JSON and the operator back from it
state = json.loads(pk.dumps(op))
back = assert_round_trip(op, p2, t2, dz2)
print(state["class"], state["config"])
print(type(back).__name__, pk.pipeline_hash(back) == pk.pipeline_hash(op), np.asarray(back(p2, t2, dz2).data).round(1).tolist())
```

This prints

```text
ColumnAirMass {'dim': 'level', 'gas_constant': 287.05}
ColumnAirMass True [3338.8, 3352.2]
```

that is, a JSON record of the class and its two arguments (and of the module it is imported from), and a copy with the same hash that gives the same masses. In a library the `module` is the library's own, and `record_under("mylib.operators", ColumnAirMass)` records the class under the module you import it from, so a saved pipeline and its hash survive moving the code between files.

## The test kit

`pipekit_array.testing` is what a library tests its functions with. A function gets four tests, as radiomax lays out: an oracle (`assert_matches_oracle`), a property, a gradient check (`assert_finite_gradient` or `assert_grad_matches_fd`) and a backend-parity test (`assert_backends_agree` and `assert_traces`). The checks compare float64 results, so the suite turns on JAX's 64-bit mode once, in `conftest.py`:

```python
# the conftest.py of the library: float64 comparisons need JAX's 64-bit mode
try:
    import jax
except ImportError:
    pass
else:
    jax.config.update("jax_enable_x64", True)
```

A library usually wraps the checks once in `tests/_backends.py`, as radiomax does with `functools.partial(testing.assert_backends_agree, labelled=True)` because all its functions are element-wise. The tests of the running example:

```python
from pipekit_array.testing import (
    assert_backends_agree,
    assert_finite_gradient,
    assert_matches_oracle,
    assert_traces,
    check_kernel_op,
)


# Shapes: S = 2 sites; L = 3 levels


p, t = np.asarray(p2.data), np.asarray(t2.data)
# today (pipekit-array)
# NumPy's numbers on every installed kind of array, float32 and float64, under jit, and on DataArrays
assert_backends_agree(air_density, p, t, labelled=True)
# today (pipekit-array)
# jit, vmap and an eager loop agree over a batch of 2 calls: (2, L) each
assert_traces(sound_speed, t)
# today (pipekit-array)
# the formula against its closed form, to 1e-15 relative
assert_matches_oracle(air_density(p, t), p / (287.05 * t), rtol=1e-15)
# today (pipekit-array)
# the derivative is finite at T = 0 K and below, where the root is cut off
assert_finite_gradient(sound_speed, np.asarray([0.0, 288.0, -5.0]))
# today (pipekit-array)
# the kernel gives one answer on NumPy, JAX and DataArray carriers, with the same dims
dz = np.full((2, 3), 1000.0)
carriers = [
    (pa.DimArray(p, ("site", "level")), pa.DimArray(dz, ("site", "level"))),
    (pa.DimArray(jnp.asarray(p), ("site", "level")), pa.DimArray(jnp.asarray(dz), ("site", "level"))),
    (pa.DimArray(p, ("site", "level")).to_xarray(), pa.DimArray(dz, ("site", "level")).to_xarray()),
]
check_kernel_op(mass, carriers)


# user
# the same root without the guard: the gradient is infinite at 0
def unsafe_speed(x):
    return jnp.sqrt(1.4 * 287.05 * x)


failing = {
    "assert_finite_gradient": lambda: assert_finite_gradient(unsafe_speed, np.asarray([0.0, 288.0])),
    "assert_matches_oracle": lambda: assert_matches_oracle(
        air_density(p, t) * 1.01, p / (287.05 * t), rtol=1e-6
    ),
}
for name, check in failing.items():
    try:
        check()
    except AssertionError as error:
        print(name, "|", str(error).split(";")[0])
```

This prints

```text
assert_finite_gradient | 1 derivatives are not finite
assert_matches_oracle | 6 of 6 compared entries out of tolerance
```

The five passing checks raise nothing, and the two failing ones show what a failure says: an infinite gradient where a plain root meets 0, and a result that is 1 % off its oracle.

The rest of the kit: `assert_adjoint` (the dot-product test of an operator and its adjoint), `assert_grad_matches_fd`, `assert_refuses_dataarrays` for a function that should not take labelled arrays, `check_adapter` for a carrier of your own, and `assert_mean`, `assert_variance` and `assert_uniform` for random draws. The [tour](../tour-array.md) lists them all, and the [carriers and backends guide](carriers-and-backends.md) says what each mix of arrays does.
