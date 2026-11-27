<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/pipekit/main/docs/assets/logo-dark.svg">
  <img alt="pipekit" src="https://raw.githubusercontent.com/jejjohnson/pipekit/main/docs/assets/logo-light.svg" width="308">
</picture>

**The whole pipekit family in one install: the two foundations, pipekit-ops and pipekit-array, and the five packages built on them.**

📖 [Documentation](https://jejjohnson.github.io/pipekit/) ·
🗂️ [Repository](https://github.com/jejjohnson/pipekit) ·
📥 [Installation](https://jejjohnson.github.io/pipekit/installation/)

</div>

---

`pipekit` holds no code. It depends on the seven packages of the workspace, so one install gives an application or a notebook all of them, and you import each package under its own name. `import pipekit` exports no names; it only explains where the imports of `pipekit` before 0.2 went.

```python
import pipekit_array as pa
import pipekit_cycle as pc
import pipekit_ops as pk
```

A library should depend on the package it imports instead, usually `pipekit-ops` (no third-party dependencies) or `pipekit-array`, because the umbrella also pulls in JAX and Equinox through `pipekit-jax`.

## 📥 Install

The packages are not on PyPI, so install the umbrella from the repository. It needs Python 3.12 or newer.

```bash
pip install "pipekit @ git+https://github.com/jejjohnson/pipekit.git#subdirectory=packages/pipekit"
```

The umbrella asks for each package by name, and pip and uv cannot find those names on PyPI, so name every package with the same tag in your `[tool.uv.sources]`, as the [workspace README](https://github.com/jejjohnson/pipekit#-install) shows. Extras stay with the package that owns them, so you add them by name, for example `pipekit-train[equinox]` or `pipekit-array[xarray]`.

## 🔬 The packages

| Distribution | Import | Use it for |
|---|---|---|
| [`pipekit-ops`](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-ops) | `pipekit_ops` | **Foundation.** `Operator`, `Sequential`, `Graph` and the ready-made operators |
| [`pipekit-array`](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-array) | `pipekit_array` | **Foundation.** Named-dim array operators on any Array API backend |
| [`pipekit-cycle`](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-cycle) | `pipekit_cycle` | Time stepping and data assimilation |
| [`pipekit-train`](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-train) | `pipekit_train` | Training loops and sweeps |
| [`pipekit-evaluate`](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-evaluate) | `pipekit_evaluate` | Scoring and benchmarks |
| [`pipekit-experiment`](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-experiment) | `pipekit_experiment` | Model registries and run tracking |
| [`pipekit-jax`](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-jax) | `pipekit_jax` | Equinox operators and cycles under `jit` |

Each package's README is its front door, with its extras and a quickstart.

## 🔁 Moving from `pipekit` before 0.2

Before 0.2.0, `pipekit` was the core package. It is now `pipekit-ops` (import `pipekit_ops`), and `from pipekit import Operator` raises an `ImportError` that says where it went.

## 📄 License

MIT. See [LICENSE](https://github.com/jejjohnson/pipekit/blob/main/LICENSE).
