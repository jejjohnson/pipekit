# pipekit

Carrier-agnostic operator-graph framework. The composition primitives
— `Operator`, `Sequential`, `Graph` — that every sister package
builds on. Pure Python; no third-party dependencies.

## Install

From the workspace checkout:

```bash
uv sync --all-groups
```

Once published:

```bash
uv add pipekit
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
from pipekit import Operator, Sequential, Tap


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
API reference: [pipekit](https://github.com/jejjohnson/pipekit/blob/main/docs/api/pipekit.md).
