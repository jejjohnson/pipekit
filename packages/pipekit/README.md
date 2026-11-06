<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/pipekit/main/docs/assets/logo-dark.svg">
    <img alt="pipekit" src="https://raw.githubusercontent.com/jejjohnson/pipekit/main/docs/assets/logo-light.svg" width="308">
  </picture>
</p>

# pipekit

> The [pipekit monorepo](https://github.com/jejjohnson/pipekit) in one install.

`pip install pipekit` installs the seven packages of the workspace. The umbrella holds no code; import each package under its own name:

| Distribution | Import | Use it for |
|---|---|---|
| `pipekit-ops` | `pipekit_ops` | `Operator`, `Sequential`, `Graph` and the ready-made operators |
| `pipekit-array` | `pipekit_array` | Named-array and Array API operators |
| `pipekit-cycle` | `pipekit_cycle` | Time stepping and data assimilation |
| `pipekit-experiment` | `pipekit_experiment` | Model registry and run tracking |
| `pipekit-evaluate` | `pipekit_evaluate` | Benchmarks and scoring |
| `pipekit-train` | `pipekit_train` | Training pipelines |
| `pipekit-jax` | `pipekit_jax` | Equinox models and diffrax forward models |

```python
import pipekit_ops as pk
import pipekit_cycle as pc
```

Extras stay with the package that owns them: `pip install 'pipekit-train[equinox]'`, `pip install 'pipekit-array[xarray]'`.

Before 0.2.0, `pipekit` was the core package. It is now `pipekit-ops` (import `pipekit_ops`); `from pipekit import Operator` raises an `ImportError` that says where it went.
A library that only needs the core should depend on `pipekit-ops`, which has no third-party dependencies; this umbrella installs JAX through `pipekit-jax`.
