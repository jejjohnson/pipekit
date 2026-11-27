<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/pipekit/main/docs/assets/packages/pipekit-ops/logo-dark.svg">
  <img alt="pipekit-ops" src="https://raw.githubusercontent.com/jejjohnson/pipekit/main/docs/assets/packages/pipekit-ops/logo-light.svg" width="357">
</picture>

**Operators and operator graphs for any carrier, in pure Python with no dependencies: one `Operator` class, composed with `|` and `Graph`, wrapped, mapped, observed and saved as JSON, with the toolz vocabulary as typed classes.**

[![Tests](https://github.com/jejjohnson/pipekit/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/pipekit/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/pipekit/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/pipekit/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/pipekit/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/pipekit/actions/workflows/typecheck.yml)
[![Docs](https://github.com/jejjohnson/pipekit/actions/workflows/pages.yml/badge.svg)](https://jejjohnson.github.io/pipekit/)
[![codecov](https://codecov.io/gh/jejjohnson/pipekit/branch/main/graph/badge.svg)](https://codecov.io/gh/jejjohnson/pipekit)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/jejjohnson/pipekit/blob/main/LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

📖 [Documentation](https://jejjohnson.github.io/pipekit/) ·
🚀 [Quickstart](https://jejjohnson.github.io/pipekit/quickstart-ops/) ·
📚 [API reference](https://jejjohnson.github.io/pipekit/reference/pipekit-ops/) ·
🧭 [Design](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-ops/docs/design)

</div>

---

Consider a season of Sentinel-2 scenes over one tile. A STAC search gives a list of scene records, so we drop the cloudy ones, read each scene from object storage (which fails now and then), cache what we read, composite, check the shapes and write the result. In practice this ends up as a script of loops, `try` blocks and a hand-rolled cache, which works once but is hard to rerun on another tile, to parallelise, or to describe in a config file. pipekit-ops makes every one of those steps an `Operator`, and every way of combining operators (a chain, a graph, a retry, a cache, a parallel map, an observer) is itself an `Operator`, so the whole job is one object,

```math
p = o_K \circ \ldots \circ o_2 \circ o_1
```

where $`o_k : \mathcal{X}_{k-1} \to \mathcal{X}_k`$ is the $`k`$-th of $`K`$ operators, $`\mathcal{X}_k`$ is the set of carriers it returns (a NumPy array, an xarray Dataset, a list of scene records or your own class) and $`p : \mathcal{X}_0 \to \mathcal{X}_K`$ is the pipeline. pipekit-ops never looks inside a carrier, so the sets $`\mathcal{X}_k`$ have no shape it knows of; the domain packages narrow them. And since $`p`$ is an operator too, it can be wrapped, mapped over many items, called, inspected, hashed and written to JSON like any one of its steps.

```python
# Shapes: N scene records; K ≤ N kept; T tiles
# (CloudBelow, ReadScene and scenes as in the quickstart below)

# today (pipekit-ops)
# p = o_3 ∘ o_2 ∘ o_1, one Sequential
pipeline = (
    pk.Filter(CloudBelow(20.0)) | pk.Map(ReadScene()) | pk.CountBy(pk.Get("tile"))
)
# today (pipekit-ops)
# (N records) → {tile: count}, T entries
per_tile = pipeline(scenes)
# today (pipekit-ops)
# p rebuilt from its JSON, with the same content hash
again = pk.loads(pk.dumps(pipeline))
```

None of it needs a third-party library. pipekit-ops declares `dependencies = []`, and `import pipekit_ops` loads only the standard library, so a library can build on it without pulling in NumPy, JAX or anything else. Its execution runs on [`concurrent.futures`](https://docs.python.org/3/library/concurrent.futures.html), [`asyncio`](https://docs.python.org/3/library/asyncio.html) and [`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html), so a dask.distributed, Ray or loky executor plugs into `Map` through the standard `Executor` protocol without pipekit-ops importing it. Its vocabulary comes from [toolz](https://github.com/pytoolz/toolz): `Filter`, `CountBy`, `GetIn` and `PartitionAll` keep their toolz names and meanings, but each one is a class configured at construction (the constructor plays the part of `toolz.curry`) whose arguments are JSON values or other operators, so a pipeline built from them round-trips and hashes ([Background: toolz](https://jejjohnson.github.io/pipekit/toolz/)).

<p align="center"><img src="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pipekit_ops_overview/pipekit_ops_overview.png" alt="pipekit-ops in one picture: one Operator and the pipeline p = o_K ∘ … ∘ o_1, with each step's type and the content hash h(p); five areas act on it (compose in core, wrap in operators, map in execution, observe, save in saving); carriers and testing underneath, on the Python standard library" width="100%"></p>

## 🌟 Highlights

- 🧱 **One class for any carrier**: an `Operator` has one method, `_apply`, and never inspects what it is given, so the same machinery runs on a NumPy array, an xarray Dataset, a dict of scene records or your own class.
- 🔗 **Composition that stays an operator**: `a | b | c` is one flattened `Sequential`, and a `Graph` built by calling operators on `Input`s wires named inputs to named outputs, rejects cycles and, through `summary()`, gives a table of each node's input and output `Signature` before anything runs.
- 🛟 **Control, limits and checks as wrappers**: `Retry` and `Try` with the exception list they always require, `Timeout`, `RateLimit` and `Until`, `Cache` with a named `MemoryStore` or `DirectoryStore`, and the `Assert*` checks and `Quarantine`, each taking an operator and returning one.
- 🧵 **One engine for many items**: `Map` runs an operator serially, on threads, on spawned processes, on asyncio or on an executor bound with `executor_scope`, holds at most `max_in_flight` items, and fails fast, collects the failures in a `MapError` or logs and continues; `ThreadMap`, `ProcessMap`, `AsyncMap` and `BatchedMap` are its presets.
- 🧾 **Configuration is data**: `dumps` and `loads` write and rebuild a whole pipeline as JSON, `pipeline_hash` is its content hash, and `loads_sandboxed` builds only the classes you `register`; an operator that holds a function (`Lambda`, `Tap`, `Branch`) is flagged, `loads` refuses it, and `check_pickleable` finds it before a process pool does.
- 👀 **Observers that change nothing**: `Tap`, `Log` and `ShapeTrace` pass the carrier through, as do the operators that `Snapshot`, `Profile` and `Histogram` place in a pipeline, and `jit_report` says which steps of a pipeline trace under `jax.jit` and why the others do not.
- 🪶 **Carriers read through adapters**: `signature_of` and `backend_of` read a carrier's named dims and array backend through a registered `NamedArrayAdapter`, `guard` reports each move of data between NumPy and JAX, and `on_import` runs a registration once a library such as JAX is imported, without importing it.

## 📐 Every way of combining operators is an operator

The package is seven areas, each with one job. An area imports only from the areas above it in this table, and `testing` from all of them.

| Area | Its job | Gives | Main names |
|---|---|---|---|
| `carriers` | reads a carrier's dims and backend | a `Signature`, a backend | `signature_of`, `backend_of`, `guard` |
| `core` | chains and wires operators | $`p = o_2 \circ o_1`$, or a DAG | `Operator`, `Sequential`, `Graph` |
| `saving` | saves and names a pipeline | JSON, and $`h(p)`$ | `dumps`, `loads`, `pipeline_hash` |
| `operators` | wraps an operator, or is one ready-made | another operator | `Retry`, `Cache`, `Filter`, `Get` |
| `execution` | runs an operator over many items | $`[o(x_1), \ldots, o(x_N)]`$ | `Map`, `Prefetch`, `Fold` |
| `observe` | looks without changing anything | the carrier, unchanged | `Tap`, `Log`, `jit_report` |
| `testing` | checks an operator's round trip | a passing test | `assert_round_trip` |

Here $`x_1, \ldots, x_N`$ are $`N`$ items of one carrier type. The hash in the `saving` row is what makes a pipeline something you can name,

```math
h(p) = \mathrm{SHA256}(\mathrm{json}(s_p))
```

where $`s_p`$ is the state of $`p`$ (its class, its module and its constructor arguments, with every nested operator as its own record and, for a `Graph`, every node and its wiring) and $`\mathrm{json}`$ writes it with sorted keys, so $`h(p)`$ is a 64-character hex digest. Two pipelines share $`h`$ exactly when they are configured the same, so $`h`$ is the key the pipekit-experiment registries store models under, and a `Cache` folds $`h`$ of its inner operator into every entry's key. Executors and stores are bound at run time with `executor_scope` and `store_scope` and are never part of $`s_p`$, so a pipeline hashes the same on a laptop and on a cluster. The [quickstart](https://jejjohnson.github.io/pipekit/quickstart-ops/) has the operator model, the wrappers and the execution engine in full.

## 📥 Install

pipekit-ops is not on PyPI, so install it from the repository. It brings in nothing beyond itself and needs Python 3.12 or newer.

```bash
pip install "pipekit-ops @ git+https://github.com/jejjohnson/pipekit.git#subdirectory=packages/pipekit-ops"
```

There are no extras, and a project that depends on pipekit-ops should pin a tag of the workspace rather than `main`. Every other pipekit package asks for pipekit-ops by name, and uv cannot find that name on PyPI, so a project that also uses one of them names pipekit-ops in its `[tool.uv.sources]`, with every pipekit package on the same tag:

```toml
[project]
dependencies = ["pipekit-ops", "pipekit-array"]

[tool.uv.sources]
pipekit-ops = { git = "https://github.com/jejjohnson/pipekit", subdirectory = "packages/pipekit-ops", tag = "pipekit-array-v0.3.1" }
pipekit-array = { git = "https://github.com/jejjohnson/pipekit", subdirectory = "packages/pipekit-array", tag = "pipekit-array-v0.3.1" }
```

The [quickstart](https://jejjohnson.github.io/pipekit/quickstart-ops/) also says how to move code written for `pipekit` before it was split into packages, and for the flat module layout before the seven areas.

## 🛰️ Quickstart: screen a month of scenes

The carrier is a list of $`N`$ scene records $`x_n`$, plain dicts as a STAC search returns them, each with an id, an MGRS tile and its cloud cover $`c_n`$ in percent. We keep the $`K`$ scenes with $`c_n < c_\text{max}`$ for $`c_\text{max} = 20`$, read each kept scene on four threads with a retry and a cache, and count the kept scenes in each of the $`T`$ tiles. `CloudBelow` and `ReadScene` are ours; everything else is pipekit-ops, and nothing outside the standard library is imported.

```python
import pipekit_ops as pk


# Shapes: N = 4 scene records; K ≤ N kept; T = 2 tiles


# user
class CloudBelow(pk.Operator):
    """True when a scene's cloud cover is under a limit, in percent."""

    def __init__(self, limit: float) -> None:
        # c_max
        self.limit = limit

    def _apply(self, scene: dict) -> bool:
        # c_n < c_max
        return scene["cloud"] < self.limit


# user
class ReadScene(pk.Operator):
    """Stand-in reader: a real one opens the scene's COG from object storage."""

    def _apply(self, scene: dict) -> dict:
        return {**scene, "path": f"s3://scenes/{scene['id']}.tif"}


# x_n = {id, tile, c_n}
# → N records
scenes = [
    {"id": "S2A_0601", "tile": "30SYJ", "cloud": 4.0},
    {"id": "S2B_0604", "tile": "30SYJ", "cloud": 63.0},
    {"id": "S2A_0611", "tile": "30SYJ", "cloud": 11.0},
    {"id": "S2B_0614", "tile": "30SXJ", "cloud": 2.0},
]

# today (pipekit-ops)
# o_1(x) = [x_n : c_n < c_max], so (N records) → (K records)
cloud_free = pk.Filter(CloudBelow(20.0))
# today (pipekit-ops)
# on OSError, read again, up to 3 attempts
read = pk.Retry(ReadScene(), attempts=3, base_delay=0.5, on=(OSError,))
# today (pipekit-ops)
# one entry per scene id, in the store bound as "scenes"
cached = pk.Cache(read, store="scenes", key=pk.Get("id"))
# today (pipekit-ops)
# p = o_4 ∘ o_3 ∘ o_2 ∘ o_1, one Sequential
pipeline = (
    cloud_free
    | pk.Map(cached, executor="thread", n_workers=4)
    | pk.Log("read {type} of scenes", level="INFO")
    | pk.CountBy(pk.Get("tile"))
)

# today (pipekit-ops)
store = pk.MemoryStore()
# today (pipekit-ops)
with pk.store_scope(scenes=store):
    # (N records) → {tile: count}, T entries
    per_tile = pipeline(scenes)

# today (pipekit-ops)
# p rebuilt from its JSON
again = pk.loads(pk.dumps(pipeline))
# today (pipekit-ops)
# h(p) of the rebuilt pipeline against h(p) of the original
same_hash = pk.pipeline_hash(again) == pk.pipeline_hash(pipeline)

print(per_tile, len(store), same_hash)
```

This prints `{'30SYJ': 2, '30SXJ': 1} 3 True`: two cloud-free scenes in tile 30SYJ and one in 30SXJ, three entries in the cache's store (one per scene read), and the same content hash for the pipeline rebuilt from its JSON. Every step in that chain round-trips, so `pipeline_hash` gives one key for "this pipeline, configured this way". We used `Log` rather than `Tap` because `Tap` holds a function, which `loads` refuses to rebuild. Nothing here is specific to scene records, so a real reader in place of `ReadScene` leaves the rest of the pipeline as it is, and `executor="scope"` inside an `executor_scope` sends the reads to a Dask or Ray cluster instead of four threads.

The [quickstart](https://jejjohnson.github.io/pipekit/quickstart-ops/) takes this further: the same composition on NumPy arrays with a `Graph` and an observer, then the operator model, every wrapper, the execution engine, saving and hashing, and what pipekit-ops leaves to other tools.

## 🔬 What else is in the box

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pipekit_ops_roles/pipekit_ops_roles-dark.svg">
    <img alt="pipekit-ops on the GeoStack role ladder: Sequential, Graph, the wrappers, Map and the observers as recipes, Operator and the ready-made operators as steps, jit_report as an algorithm, no models or strategies of its own, saving, hashing and carrier functions as primitives, and signatures, adapters, stores and results as structures, with run-time context and setup beside the ladder" src="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pipekit_ops_roles/pipekit_ops_roles-light.svg" width="100%">
  </picture>
</p>

Each public name sits on one rung of the GeoStack role ladder, by the job it does. Most of pipekit-ops is the upper rungs: `Operator` and the ready-made operators (`Get`, `Filter`, `Identity`) are ⑤ steps, one transformation with its config, while `Sequential`, `Graph`, the wrappers (`Retry`, `Cache`, `Branch`) and `Map` compose steps into ⑥ recipes, and `dumps`, `loads` and `pipeline_hash` are ① primitives, one call each, that treat a recipe as data. Underneath sit the ⓪ structures the steps pass around and read: `Signature`, the named-array adapters and the stores.

| Area | Symbols | Notes |
|---|---|---|
| `pipekit_ops.carriers` | `NamedArrayAdapter`, `LabelledAdapter`, `ConvertingAdapter`, `MissingValueAdapter`, `register_adapter`, `register_lazy_adapter`, `unregister_adapter`, `get_adapter`, `MissingAdapterError` | how a carrier with named dims is read |
| | `Signature`, `signature_of`, `compute_output_signature` | shapes before anything runs |
| | `backend_of`, `backend_name`, `to_backend`, `guard`, `guard_level`, `report_move`, `BackendError`, `BackendWarning` | which array library a value is on, and moves between them |
| | `is_traced`, `is_concrete`, `on_import`, `import_optional` | JAX tracers, and optional libraries |
| `pipekit_ops.core` | `Operator`, `ConfigMixin`, `Carrier`, `Sequential`, `Graph`, `Input`, `Node` | write a step, chain steps, wire a graph |
| | `StatefulOperator`, `CarryState`, `Predictor`, `FittableTransformer` | state carried between calls, and model seams |
| | `pipe`, `compose`, `compose_left`, `juxt`, `complement` | the toolz function surface over operators |
| | `nested_config`, `require_operator`, `resolve_loaded`, `record_under`, `record_moves` | helpers for libraries that write operators |
| `pipekit_ops.saving` | `dumps`, `loads`, `loads_sandboxed`, `from_state_sandboxed`, `register`, `unregister`, `allowed`, `upgrade_state` | configuration as JSON, and an allowlist |
| | `pipeline_hash`, `stable_json`, `stable_repr`, `sha256_hex` | the content hash and its parts |
| | `Store`, `MemoryStore`, `DirectoryStore`, `store_scope` | where a `Cache` keeps its entries |
| `pipekit_ops.operators` | `Identity`, `Const`, `Lambda`, `Sink` | blocks |
| | `Get`, `GetIn`, `Pick`, `Dissoc`, `Rename`, `Merge`, `Assoc`, `UpdateIn`, `ValMap` | dict plumbing |
| | `Not`, `Filter`, `Remove`, `Reduce`, `PartitionAll`, `Concat`, `Zip`, `GroupBy`, `CountBy`, `Count`, `Unique`, `Take`, `SortBy` | lists of items |
| | `Fanout` | one input, named outputs |
| | `Branch`, `Switch`, `Try`, `Coalesce`, `Retry` | decisions and failures |
| | `Timeout`, `RateLimit`, `Until` | flow limits |
| | `Cache`, `Memoize` | the same call twice |
| | `Quarantine`, `AssertShape`, `AssertDType`, `AssertHasAttribute`, `AssertCallable`, `AssertThat`, `AssertSignature`, `QCError` | quality control |
| `pipekit_ops.execution` | `Map`, `executor_scope`, `MapError`, `Failed`, `Prefetch`, `Fold`, `check_pickleable` | one operator over many items |
| | `ThreadMap`, `ProcessMap`, `AsyncMap`, `BatchedMap` | presets of `Map` |
| `pipekit_ops.observe` | `Tap`, `Log`, `ShapeTrace`, `Snapshot`, `Profile`, `Histogram` | look without changing the flow |
| | `jit_report`, `JitReport`, `JitStep` | which steps run under `jax.jit` |
| `pipekit_ops.testing` | `assert_round_trip`, `public_operators` | contract checks for other test suites |

**Where to import from.** `pipekit_ops` re-exports every name of the first six areas except `nested_config`, `require_operator`, `resolve_loaded`, `stable_json`, `stable_repr` and `sha256_hex`, which are imported from their area; `pipekit_ops.testing` is imported on its own. Import a name from the package or from its area, never from a file inside an area.

**Two `Reduce`s.** `pipekit_ops.Reduce` is a left fold over a list of items, whereas `pipekit_array.Reduce` reduces an array over named dims, so import the packages under their own names (`pk.Reduce`, `pa.Reduce`) when you use both.

## 📓 Tutorials

| # | Tutorial | What it shows |
|---|---|---|
| 1 | [Build a bigger graph](https://jejjohnson.github.io/pipekit/anomaly-graph/) | an anomaly detector as one `Graph`: parallel reads with retries and a cache, checks, a `Branch` and a `Fanout` |
| 2 | [ETL over many regions](https://jejjohnson.github.io/pipekit/etl-regions/) | one pipeline run over many regions |
| 3 | [Run inference](https://jejjohnson.github.io/pipekit/inference/) | sensor records in, cleaned, scored by a registered model, decisions out |
| 4 | [Pipelines under jit](https://jejjohnson.github.io/pipekit/pipekit-under-jit/) | a `Graph` traced under `jax.jit`, with pipekit-jax for gradients and scans |

## 📖 Documentation

- **[Quickstart](https://jejjohnson.github.io/pipekit/quickstart-ops/)**: a second example on arrays, the operator model, the wrappers, the execution engine, saving and hashing, the zero-dependency guarantee, what is not built, and how to move from older versions.
- **[Core tour](https://jejjohnson.github.io/pipekit/core-tour/)**: every group of names, with a run output for each.
- **[Concepts](https://jejjohnson.github.io/pipekit/concepts/)** and **[Protocols](https://jejjohnson.github.io/pipekit/protocols/)**: the five ideas behind the workspace, and the runtime-checkable seams a library satisfies without importing pipekit.
- **[Patterns](https://jejjohnson.github.io/pipekit/patterns/)** and **[Guards](https://jejjohnson.github.io/pipekit/guards/)**: batch jobs, services, orchestrators and audits; and `guard`, which reports data moving between NumPy and JAX, with `jit_report`, which says which steps are jittable.
- **[Background: toolz](https://jejjohnson.github.io/pipekit/toolz/)**: why operators are classes and not functions.
- **[Design docs](https://jejjohnson.github.io/pipekit/readme/)**: the vision, the operator catalogue, the execution design, toolz parity, the decisions (O1 to O17) and the roadmap with its measured gates.
- **[API reference](https://jejjohnson.github.io/pipekit/reference/pipekit-ops/)**: every public symbol, generated from the docstrings.

## 🧩 Ecosystem

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pipekit_ops_ecosystem/pipekit_ops_ecosystem-dark.svg">
    <img alt="The 16 packages that import pipekit-ops on their main branches, by layer, each with the areas it imports: the rest of the pipekit workspace, geotoolz's four packages, gridax and diagnostix, vardax, and plumax, radtrax, sesmax-core and sesmax-experiment; pipekit-ops builds on the standard library alone" src="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pipekit_ops_ecosystem/pipekit_ops_ecosystem-light.svg" width="100%">
  </picture>
</p>

pipekit-ops is the foundation of the pipekit [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/), in the Core layer of the [GeoStack](https://github.com/jejjohnson/geostack). It extends no third-party library; what it does not write itself comes from the standard library:

| Concern | Owner |
|---|---|
| Thread and process pools, and borrowed executors | [`concurrent.futures`](https://docs.python.org/3/library/concurrent.futures.html), [`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html) |
| The async executor | [`asyncio`](https://docs.python.org/3/library/asyncio.html) |
| Scoped executors, stores and guard levels | [`contextvars`](https://docs.python.org/3/library/contextvars.html) |
| `Log`, `Map` progress and warnings | [`logging`](https://docs.python.org/3/library/logging.html) |
| Configs, content hashes and the disk store | [`json`](https://docs.python.org/3/library/json.html), [`hashlib`](https://docs.python.org/3/library/hashlib.html), [`pickle`](https://docs.python.org/3/library/pickle.html) |
| The names and meanings of the dict, list and function operators | [toolz](https://github.com/pytoolz/toolz), as ideas only |

Dependencies point one way. The other pipekit packages and the packages below import pipekit-ops on their `main` branches, each from the areas listed, and pipekit-ops imports none of them.

| Package | Areas it imports |
|---|---|
| [pipekit-array](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-array) | `carriers`, `core` |
| [pipekit-cycle](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-cycle), [pipekit-jax](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-jax) | `core` |
| [pipekit-train](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-train) | `core`, `operators`, `saving` |
| [pipekit-experiment](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-experiment) | `core`, `saving` |
| [pipekit-evaluate](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-evaluate) | `saving` |
| [geotoolz](https://github.com/jejjohnson/geotoolz): geotoolz-ops | `carriers`, `core`, `saving`, `operators`, `observe` |
| geotoolz: xrtoolz-ops | `carriers`, `core` |
| geotoolz: geotoolz-array | `carriers` |
| geotoolz: geotoolz-patcher, through its `pipekit` extra | `core` |
| [gridax](https://github.com/jejjohnson/gridax), [plumax](https://github.com/jejjohnson/plumax) | `carriers`, `core` |
| [radtrax](https://github.com/jejjohnson/radtrax), through its `geo` extra | `core` |
| [sesmax](https://github.com/jejjohnson/sesmax): sesmax-core | `core` |
| sesmax: sesmax-experiment | `core`, `saving`, `operators`, `observe` |
| [vardax](https://github.com/jejjohnson/vardax) | `core` |
| [diagnostix](https://github.com/jejjohnson/diagnostix) | `carriers` |

## 🌳 Repository layout

```
packages/pipekit-ops/
├── src/pipekit_ops/
│   ├── __init__.py     ← re-exports the first six areas
│   ├── carriers/       ← adapters, backends, signatures, guards, optional imports
│   ├── core/           ← Operator, Sequential, Graph, state, protocols, compose
│   ├── saving/         ← dumps and loads, the sandbox, pipeline_hash, stores
│   ├── operators/      ← blocks, dicts, lists, Fanout, control, flow, Cache, QC
│   ├── execution/      ← Map and its presets, Prefetch, Fold
│   ├── observe/        ← the observers and jit_report
│   └── testing/        ← assert_round_trip and public_operators
├── tests/              ← one file per group, the dependency gates and the area rules
└── docs/design/        ← vision, catalogue, execution, toolz parity, decisions, roadmap
```

## ⚙️ Development

The package is developed inside the workspace, so clone the repository and run the tasks from its root.

```bash
git clone https://github.com/jejjohnson/pipekit.git
cd pipekit
make install                              # uv sync of every group but train-extras + pre-commit hooks
uv run pytest packages/pipekit-ops/tests  # this package's fast tests
make lint                                 # ruff check on the whole repo
make typecheck                            # ty check on every package
make docs                                 # the MyST prose and the Zensical API, checked for links
```

`make help` lists every task. The root [`AGENTS.md`](https://github.com/jejjohnson/pipekit/blob/main/AGENTS.md) holds the rules for the whole workspace: the checks to run before every commit, the review workflow and the house style.

## 📄 License

MIT. See [LICENSE](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-ops/LICENSE).
