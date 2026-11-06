---
status: draft
version: 0.1.0
---

# Vision

## 1. The decision

pipekit-ops gains about thirty small operators and one execution engine.
They fill the gaps that pipelines currently fill with `Lambda`: reshaping the dicts that `Graph` and `Fanout` return, working on lists of items, limiting how fast and how long a step may run, and running one operator over many items on any executor.

Two rules hold for every one of them.

1. **No carrier inspection.**
   An operator in pipekit-ops never looks inside the carrier.
   It may index a container (`d[key]`, `len(xs)`), call another operator, or read a `Signature` through the adapter registry (`signature_of` in `packages/pipekit-ops/src/pipekit_ops/adapters.py`).
   It never reads `.shape`, `.crs` or pixel values itself; that belongs to pipekit-array, geoops and xrops.
2. **No third-party dependencies.**
   `packages/pipekit-ops/pyproject.toml` has `dependencies = []`, and it stays that way.
   Execution builds on `concurrent.futures`, `asyncio`, `multiprocessing` and `contextvars`.
   dask.distributed, Ray, loky and MPI plug in through the stdlib `concurrent.futures.Executor` protocol, without pipekit-ops importing them.

A third rule follows from the existing design: **callables enter only as operators.**
New operators take predicates, keys and folds as `Operator` instances or as plain keys (a `str` or `int`, used for item lookup the way toolz's `getter` does), never as bare functions.
A new operator then has a faithful config whenever its children do, and `forbid_in_yaml` stays on `Lambda` where `check_pickleable` already finds it.

## 2. What pipekit-ops never does

- It does not schedule a graph across machines.
  `Graph` evaluates in one process (`packages/pipekit-ops/src/pipekit_ops/_base/graph.py`); dask, Ray and orchestrators schedule pipekit operators, and pipekit keeps them pickleable so they can (see "Deliberately left out" in `docs/core-tour.md`).
- It does not parallelise implicitly.
  `Sequential` stays sequential, and nothing runs on a pool unless a `Map` asks for one.
- It does not ship carrier-specific writers.
  The disk cache in core pickles values; a Zarr or COG store lives in geoops or xrops and plugs into the same `Store` protocol.
- It does not show progress bars.
  Progress goes to `logging`; a notebook can attach tqdm to the log records itself.
- It does not curry.
  An operator's constructor is its partial application, as `docs/background/toolz.md` already says.

## 3. The running example

Every page uses one job: **read and screen a month of Sentinel-2 L2A scenes from a STAC API, and summarise them into a dict.**

The region covers 25 MGRS tiles, and a month holds about 12 scenes per tile, so the search returns $N \approx 300$ STAC items.
Each item is a plain dict (the STAC JSON): its cloud cover sits at `item["properties"]["eo:cloud_cover"]`, its tile ID at `item["properties"]["s2:mgrs_tile"]`, and each band's URL at `item["assets"][band]["href"]`.

The job has six needs, and each one maps to a group of this design.

| Need | What goes wrong today | Group |
|---|---|---|
| Pull fields out of nested item dicts | every access is a `Lambda`, so the pipeline cannot round-trip | dict plumbing |
| Drop cloudy items, count what is left | a `Lambda` over the list, or Python outside the pipeline | iterables |
| Respect the API's rate limit, retry, and give up on a hung read | `Retry` exists; there is no `RateLimit` or `Timeout` | flow |
| Read $K$ kept scenes in parallel | `ThreadMap` works, but has no backpressure, no error collection and no way to use a dask cluster | execution |
| Do not re-read a scene after a crash | `Cache` is in memory only, and its docstring promises a disk backend for v0.2 | cache and store |
| Check each scene and log what happened | `AssertCallable` and `Tap` hold closures | observe and QC |

Here is the whole job in the proposed vocabulary.
$N$ is the number of items the search returns, $K \le N$ the number kept after the cloud screen, and $B = 4$ the bands read per scene.
`StacSearch`, `MaxCloud`, `ReadBands` and `ValidFraction` are user operators with plain configs; everything else is pipekit-ops.

```python
import pipekit_ops as pk

# Shapes: N ≈ 300 items; K ≤ N kept; B = 4 bands; T = 25 MGRS tiles

# user
search = StacSearch(collection="sentinel-2-l2a", month="2026-10")
# proposed (P1)
cloud_cover = pk.GetIn(("properties", "eo:cloud_cover"))
# user
is_clear = MaxCloud(40.0)
# today
clear_enough = cloud_cover | is_clear
# proposed (P1)
# (N items) → (K items)
keep = pk.Filter(clear_enough)

# user
read = ReadBands(bands=("B02", "B03", "B04", "B08"))
# proposed (P3)
# give up on one attempt after 60 s
bounded = pk.Timeout(read, seconds=60.0)
# proposed (P3)
# at most R = 10 attempt starts in any window of T_w = 1 s; retries count too
limited = pk.RateLimit(bounded, calls=10, period=1.0)
# today
retried = pk.Retry(limited, attempts=4, base_delay=1.0, on=(OSError, TimeoutError))
# proposed (P4)
scene_id = pk.Get("id")
# proposed (P4)
cached = pk.Cache(retried, store="scenes", key=scene_id)
# proposed (P5)
checked = cached | pk.AssertSignature("band y x")
# proposed (P2)
# (K items) → (K scenes), at most W = 16 in flight
read_all = pk.Map(checked, executor="scope", max_in_flight=16, on_error="collect")

# proposed (P1)
tile_id = pk.GetIn(("properties", "s2:mgrs_tile"))
# proposed (P1)
# (K items) → dict of ≤ T counts
per_tile = pk.CountBy(tile_id)
```

The summary is a `Graph`, because it has one input and several named outputs.

```python
# today
items = pk.Input("items")
# user
found = search(items)
# today
kept = keep(found)
# today
scenes = read_all(kept)
# proposed (P1)
n_found = pk.Count()(found)
# proposed (P1)
n_kept = pk.Count()(kept)
# today
tiles = per_tile(kept)
# today
job = pk.Graph(
    inputs={"items": items},
    outputs={"n_found": n_found, "n_kept": n_kept, "per_tile": tiles, "scenes": scenes},
)
```

`job` contains no `Lambda`, so `pk.loads(pk.dumps(job))` rebuilds it and `pk.pipeline_hash(job)` names it, exactly as the round-trip contract in `packages/pipekit-ops/src/pipekit_ops/testing.py` requires.
The order of the wrappers matters: `RateLimit` sits inside `Retry`, so every retry also waits for the limiter, and `Cache` sits outside, so a cached scene costs no request at all.
Where the scenes are cached and which executor reads them are deployment choices, set around the call and not stored in the config; see [api/execution.md](api/execution.md) and [api/operators.md](api/operators.md).

```python
from concurrent.futures import ThreadPoolExecutor

# Shapes: as above

# proposed (P4)
scratch = pk.DirectoryStore("/scratch/s2")
pool = ThreadPoolExecutor(max_workers=8)
# proposed (P2)
on_threads = pk.executor_scope(pool)
# proposed (P4)
in_scratch = pk.store_scope(scenes=scratch)
# laptop: eight threads
with on_threads, in_scratch:
    summary = job(items=query)

# cluster: the same job on dask workers, no change to job
cluster_pool = client.get_executor()
# proposed (P2)
on_cluster = pk.executor_scope(cluster_pool)
with on_cluster, in_scratch:
    summary = job(items=query)
```

`client` is a `dask.distributed.Client`; its `get_executor()` returns a `concurrent.futures.Executor` (not pinned here: checked against the dask docs, not run).

## 4. Scope in numbers

| Group | Operators | New code (estimate) |
|---|---|---|
| Dict plumbing | 9 | 150 lines |
| Iterables | 12 (plus `Map`) | 250 lines |
| Flow | 3 | 200 lines |
| Execution | `Map`, `Prefetch`, `executor_scope`, presets rebuilt | 450 lines |
| Cache and store | `Store`, `MemoryStore`, `DirectoryStore`, `store_scope` | 200 lines |
| Observe and QC | 3 plus a `progress` argument on `Map` | 150 lines |

The estimates are for source only; tests roughly double them.
