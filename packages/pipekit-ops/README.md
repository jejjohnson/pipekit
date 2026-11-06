<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/pipekit/main/docs/assets/packages/pipekit-ops/logo-dark.svg">
    <img alt="pipekit-ops" src="https://raw.githubusercontent.com/jejjohnson/pipekit/main/docs/assets/packages/pipekit-ops/logo-light.svg" width="357">
  </picture>
</p>

# pipekit-ops (`pipekit_ops`)

> Part of the [pipekit monorepo](https://github.com/jejjohnson/pipekit): ships as the `pipekit-ops` distribution, imported as `pipekit_ops`. Until 0.2.0 this package was `pipekit`.

The pipekit core: the composition primitives (`Operator`, `Sequential`, `Graph`) that every sister package builds on, and the ready-made operators (control flow, observers, caching, QC, parallel maps).
Pure Python; no third-party dependencies.

## Install

From the workspace checkout:

```bash
uv sync --all-groups
```

Once published:

```bash
uv add pipekit-ops
```

## What's shipped (v0.0.1)

| Module        | Symbols                                                                              |
|---------------|--------------------------------------------------------------------------------------|
| `_base`       | `Operator`, `ConfigMixin`, `Carrier`, `Sequential`, `Graph`, `Input`, `Node`         |
| `compose`     | `pipe`, `compose`, `compose_left`, `complement`, `juxt`                              |
| `blocks`      | `Identity`, `Const`, `Lambda`, `Sink`                                                |
| `control`     | `Branch`, `Switch`, `Try`, `Coalesce`, `Retry`                                       |
| `observe`     | `Tap`, `Snapshot`, `ShapeTrace`, `Profile`, `Histogram`                              |
| `combine`     | `Fanout`                                                                             |
| `cache`       | `Cache`, `Memoize`                                                                   |
| `qc`          | `Quarantine`, `AssertShape`, `AssertDType`, `AssertHasAttribute`, `AssertCallable`   |
| `signature`   | `Signature` (named-dimension shape inference)                                        |
| `parallel`    | `ThreadMap`, `ProcessMap`, `AsyncMap`, `BatchedMap`, `check_pickleable`              |
| `serial`      | `dumps`, `loads`, `register`, `loads_sandboxed`                                      |
| `state`       | `StatefulOperator`, `CarryState` (the substrate for `pipekit-cycle`)                 |

## Quickstart

```python
from pipekit_ops import Operator, Sequential, Tap


class Scale(Operator):
    def __init__(self, factor: float) -> None:
        self.factor = factor

    def _apply(self, x):
        return x * self.factor


pipe = Scale(2.0) | Tap(print, name="log") | Scale(3.0)
assert pipe(5.0) == 30.0
```

## References

Master plan: [Report 2](https://github.com/jejjohnson/research_journal_v2/blob/main/notes/geotoolz/master_plan/toolz_2_pipekit.md).
API reference: [pipekit-ops](https://github.com/jejjohnson/pipekit/blob/main/docs/api/pipekit-ops.md).

## Moving from `pipekit` < 0.2

Change the import; every name and argument is unchanged.

```python
# before
from pipekit import Operator, Retry, Sequential

# after
from pipekit_ops import Operator, Retry, Sequential
```

Libraries should depend on `pipekit-ops`, not on `pipekit`: since 0.2.0 `pipekit` is the umbrella that installs the whole workspace, JAX included.
States saved before 0.2.0 (`dumps` output, registry entries) still load: a module saved as `pipekit.<name>` resolves to `pipekit_ops.<name>`.
Their `pipeline_hash` changes, because the hash covers each operator's module.
