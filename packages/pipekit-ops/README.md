<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/pipekit/main/docs/assets/packages/pipekit-ops/logo-dark.svg">
    <img alt="pipekit-ops" src="https://raw.githubusercontent.com/jejjohnson/pipekit/main/docs/assets/packages/pipekit-ops/logo-light.svg" width="357">
  </picture>
</p>

# pipekit-ops (`pipekit_ops`)

> Part of the [pipekit monorepo](https://github.com/jejjohnson/pipekit): ships as the `pipekit-ops` distribution, imported as `pipekit_ops`. Until 0.2.0 this package was `pipekit`.

The pipekit core: the composition primitives (`Operator`, `Sequential`, `Graph`) that every sister package builds on, and the ready-made operators.
Pure Python; no third-party dependencies.

There are five features we would like to highlight, all added in 0.2.1:

1. **Dict plumbing** (`dicts`): `Get`, `GetIn`, `Pick`, `Rename` and friends reshape the dicts that a `Graph`, a `Fanout` or a STAC item carries, without a `Lambda`, so the pipeline still round-trips through `dumps` / `loads`.
2. **Iterables** (`seq`): `Filter`, `GroupBy`, `CountBy`, `Reduce`, `Take` and friends work on a list of items inside the pipeline; predicates and keys are operators too.
3. **Execution** (`execution`): one engine, `Map`, runs an operator over many items serially, on threads, on processes, on an event loop, or on any `concurrent.futures.Executor` you bind with `executor_scope` (a dask or Ray one included).
   It pulls its input lazily, holds at most `max_in_flight` items, and fails fast, collects or logs errors; `ThreadMap`, `ProcessMap`, `AsyncMap` and `BatchedMap` are presets of it.
4. **Flow limits** (`flow`): `Timeout`, `RateLimit` and `Until` bound how long a call may hang, how often a service is hit, and how many times a step repeats.
5. **Stores** (`store`): `Cache(op, store="scenes", key=Get("id"))` keeps its entries in a named store bound with `store_scope`, such as a `DirectoryStore` that survives a restart and is shared by processes on one file system.

The design behind them, with the decisions and the measured test gates, is in [docs/design/](docs/design/README.md).

## Install

From the workspace checkout:

```bash
uv sync --all-groups
```

Once published:

```bash
uv add pipekit-ops
```

## What's shipped (0.2.1)

| Module | Symbols |
|---|---|
| `_base` | `Operator`, `ConfigMixin`, `Carrier`, `Sequential`, `Graph`, `Input`, `Node` |
| `compose` | `pipe`, `compose`, `compose_left`, `complement`, `juxt` |
| `blocks` | `Identity`, `Const`, `Lambda`, `Sink` |
| `control` | `Branch`, `Switch`, `Try`, `Coalesce`, `Retry` |
| `flow` | `Timeout`, `RateLimit`, `Until` |
| `observe` | `Tap`, `Snapshot`, `ShapeTrace`, `Profile`, `Histogram`, `Log` |
| `combine` | `Fanout` |
| `dicts` | `Get`, `GetIn`, `Pick`, `Dissoc`, `Rename`, `Merge`, `Assoc`, `UpdateIn`, `ValMap` |
| `seq` | `Not`, `Filter`, `Remove`, `Reduce`, `PartitionAll`, `Concat`, `Zip`, `GroupBy`, `CountBy`, `Count`, `Unique`, `Take`, `SortBy` |
| `cache` | `Cache` (in memory or in a named store), `Memoize` |
| `store` | `Store`, `MemoryStore`, `DirectoryStore`, `store_scope` |
| `qc` | `Quarantine`, `AssertShape`, `AssertDType`, `AssertHasAttribute`, `AssertCallable`, `AssertThat`, `AssertSignature`, `QCError` |
| `signature` | `Signature` (named-dimension shape inference), `compute_output_signature` |
| `adapters` | `NamedArrayAdapter`, `register_adapter`, `signature_of`, `backend_of`, … (named dims and array backends of a carrier) |
| `execution` | `Map`, `executor_scope`, `MapError`, `Failed`, `Prefetch`, `Fold` |
| `parallel` | `ThreadMap`, `ProcessMap`, `AsyncMap`, `BatchedMap` (presets of `Map`), `check_pickleable` |
| `serial` | `dumps`, `loads`, `register`, `unregister`, `allowed`, `loads_sandboxed`, `from_state_sandboxed`, `upgrade_state` |
| `hashing` | `pipeline_hash` |
| `state` | `StatefulOperator`, `CarryState` (the substrate for `pipekit-cycle`) |
| `protocols` | `Predictor`, `FittableTransformer` |
| `imports` | `on_import` |
| `testing` | `assert_round_trip` (imported as `pipekit_ops.testing`) |

## Quickstart

A step is a subclass of `Operator` with a constructor and an `_apply`; `|` chains steps into a `Sequential`.

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

A list of item dicts, screened, counted and mapped on four threads, with no `Lambda` anywhere:

```python
from pipekit_ops import CountBy, Filter, Get, Map, Not

items = [{"id": "a", "tile": "T1", "bad": False}, {"id": "b", "tile": "T1", "bad": True}]
good = Filter(Not(Get("bad")))
assert good(items) == [items[0]]
assert CountBy(Get("tile"))(items) == {"T1": 2}
assert Map(Get("id"), executor="thread", n_workers=4)(items) == ["a", "b"]
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
