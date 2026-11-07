---
status: draft
version: 0.1.0
---

# Operators

The groups other than execution, in the order of [README.md](README.md).
Each section says what is wrong today, what the operators do, and how the Sentinel-2 job uses them.

## 1. Dict plumbing

### Today

`Graph` returns a dict of its outputs (`packages/pipekit-ops/src/pipekit_ops/_base/graph.py`), `Fanout` returns a dict of its branches (`packages/pipekit-ops/src/pipekit_ops/combine.py`), and STAC items are nested dicts.
Every reshape is a `Lambda`, which is flagged `forbid_in_yaml` (`packages/pipekit-ops/src/pipekit_ops/blocks.py`), so one `Lambda(lambda d: d["id"])` stops the whole pipeline from round-tripping through `loads(dumps(...))`.

### Semantics

| Operator | Input $x$ | Output | Missing key |
|---|---|---|---|
| `Get(key)` | mapping or sequence | `x[key]`; with a tuple `(k0, k1)`, the tuple `(x[k0], x[k1])` | `KeyError` / `IndexError`, or `default` if given |
| `GetIn(keys)` | nested mappings | `x[k0][k1]…` | `KeyError`, or `default` if given |
| `Pick(keys)` | mapping | `{k: x[k] for k in keys}` in the order of `keys` | `KeyError` |
| `Dissoc(keys)` | mapping | $x$ without `keys` | ignored, as toolz `dissoc` |
| `Rename(mapping)` | mapping | keys renamed, order kept | ignored; a rename that collides with a kept key raises `ValueError` |
| `Merge()` | $m \ge 1$ mappings, or one sequence of them | union; for a key in several inputs, the last wins | n/a |
| `Assoc(key, op)` | mapping | a copy with `key: op(x)` | n/a |
| `UpdateIn(keys, op)` | nested mappings | a copy with the value at the path replaced by `op(value)` | `KeyError` unless `default` is given, in which case `op(default)` is stored (toolz `update_in`) |
| `ValMap(op)` | mapping | `{k: op(v)}` | n/a |

Inputs are never mutated: every output that is a dict is a new dict; the values are not copied.
A `Get` key that is not a `str`, `int` or tuple of them is refused at construction, so the config stays JSON.
`Get` and `GetIn` take a sequence of keys in different ways on purpose: `Get(("a", "b"))` picks two keys, `GetIn(("a", "b"))` walks a path; this is toolz's split (`get` vs `get_in`).

`UpdateIn` replaces the planned `OnKey(key, op)`: `OnKey("scene", op)` is `UpdateIn(("scene",), op)`, and toolz's name says what it does with nested keys too.
`Assoc` covers the case `OnKey` did not, adding a value computed from the whole dict.

### Running example

The summary graph returns `{"n_found", "n_kept", "per_tile", "scenes"}`.
The report sent to a dashboard needs everything except the scenes, with shorter key names.

```python
import pipekit_ops as pk

# Shapes: T ≤ 25 tiles

# proposed (P1)
drop_pixels = pk.Dissoc(("scenes",))
# proposed (P1)
shorter = pk.Rename({"n_found": "found", "n_kept": "kept"})
# today
to_report = drop_pixels | shorter
# dict of 4 → dict of 3
report = to_report(summary)
```

## 2. Iterables

### Today

Only the parallel maps take a list, and they all do the same thing: one operator per item.
Filtering, grouping and counting are `Lambda`s, or Python outside the pipeline.

### Semantics

Let $n$ be the number of items.

| Operator | Output | Notes |
|---|---|---|
| `Filter(pred)` | items with `pred(item)` truthy, in order | `pred` is an operator, often a `Sequential` ending in a comparison operator |
| `Remove(pred)` | items with `pred(item)` falsy | the toolz name; easier to read than `Filter(Not(pred))` |
| `Reduce(op, initial)` | $a_n$, where $a_0$ is `initial`, $a_{i+1} = f(a_i, x_i)$, $x_i$ is item $i$ and $f$ is `op` called on the tuple `(a_i, x_i)` | `op` takes one tuple, because an operator takes one carrier; no `initial` and $n = 0$ raises `TypeError` |
| `Fold(op, initial, chunksize, combine, executor)` | the same value as `Reduce` when `op` is associative and `initial` is its identity | chunks of `chunksize` are reduced in parallel through `Map`, then the partial results are reduced with `combine` (default `op`); toolz `sandbox.fold` |
| `PartitionAll(n)` | tuples of `n` consecutive items, the last shorter | replaces the planned `Chunk`, see [../toolz.md](../toolz.md) |
| `Concat()` | one flat list | flattens exactly one level; replaces the planned `Flatten` |
| `Zip(strict=True)` | list of tuples | unequal lengths raise `ValueError` unless `strict=False` |
| `GroupBy(key)` | `dict[group, list]` | groups in first-seen order; lists keep input order |
| `CountBy(key)` | `dict[group, int]` | |
| `Count()` | `int` | `len(x)` if defined, else consumes the iterable |
| `Unique(key=None)` | first occurrence of each item | needs hashable items or keys |
| `Take(n)` | the first `n` items | stops pulling after `n`, so a lazy input is read no further |
| `SortBy(key, reverse=False)` | sorted list | stable |

`Reduce` takes a tuple instead of two arguments because `Operator.__call__` maps positional arguments to `_apply`, and a `Sequential` passes one carrier between steps (`packages/pipekit-ops/src/pipekit_ops/_base/sequential.py`).
An operator written for `Reduce` reads `acc, item = pair`.

### Running example

```python
# Shapes: N ≈ 300 items; K kept; T ≤ 25 tiles

# proposed (P1)
cloud_cover = pk.GetIn(("properties", "eo:cloud_cover"))
# user
is_clear = MaxCloud(40.0)
# today
clear_enough = cloud_cover | is_clear
# proposed (P1)
keep = pk.Filter(clear_enough)
# (N items) → (K items)
kept = keep(found)

# proposed (P1)
tile_id = pk.GetIn(("properties", "s2:mgrs_tile"))
# proposed (P1)
per_tile = pk.CountBy(tile_id)
# (K items) → dict of ≤ T ints
counts = per_tile(kept)
```

## 3. Flow

### Timeout

`Timeout(op, seconds)` runs `op(x)` in a worker thread and waits at most `seconds` for it.
If `op` has not returned, `Timeout` raises the builtin `TimeoutError`, which is what `Retry(…, on=(TimeoutError,))` catches.

The worker thread is abandoned, not killed: Python cannot stop a thread safely.
It keeps running until `op` returns, and its result is dropped.
That is acceptable for the case this exists for, a network read with no timeout of its own, and it is documented as such.
To bound the damage, `Timeout` uses one small shared pool per process, and logs a warning when more than its `max_abandoned` (default 8) threads are still running.

Alternatives considered: `signal.alarm` works only in the main thread on POSIX, so it fails inside `Map(executor="thread")`; a process per call kills cleanly but costs a process start and pickling per item.
A process-based timeout is possible later as `Timeout(op, seconds, isolate="process")`.

### RateLimit

`RateLimit(op, calls, period)` allows at most `calls` starts of `op` in any window of `period` seconds.
It is a sliding-window limiter: it keeps the start times of the last `calls` calls, and a new call waits until the oldest is more than `period` seconds old.
The state is shared by all threads in the process, behind a lock; the sleep happens outside the lock.

The state does not cross processes.
On a pool of $p$ processes each worker has its own limiter, so the API sees up to $p$ times `calls` per `period`.
Two ways to keep a global limit: divide `calls` by $p$, or keep the rate-limited read on threads and send only CPU work to processes.
A cross-process limiter needs shared state (a manager or a file lock) and is out of scope for core.

Placement matters, and the docs show it: `Retry(RateLimit(read))` makes every retry wait for a slot, while `RateLimit(Retry(read))` lets one slot cover several attempts.

### Until

Let $f$ be `op`.
`Until(op, pred, max_iter, on_max="raise")` sets $x_0 = x$ and $x_{i+1} = f(x_i)$, and returns the first $x_i$ for which `pred(x_i)` is truthy, checking $x_0$ first.
If $i$ reaches `max_iter` first, it raises `RuntimeError` (`on_max="raise"`) or returns $x_m$ (`on_max="return"`), where $m$ is `max_iter`.
`max_iter` is required, so a pipeline cannot loop forever.

It is the stateless sibling of `pipekit_cycle.Recurrence(step_op, condition_op, max_iters)` (`packages/pipekit-cycle/src/pipekit_cycle/cycle.py`), which threads a carry state and lives in pipekit-cycle.
In the running example, `Until(NextPage(), IsLastPage(), max_iter=50)` walks STAC search pages; toolz has no name for it (`iterate` is the infinite version).

## 4. Cache and store

### Today

`Cache(inner)` keeps a private dict, keyed by `sha256(input key ‖ 0x00 ‖ stable_json(inner.get_config()))` (`packages/pipekit-ops/src/pipekit_ops/cache.py`).
The module docstring says a disk backend lands in v0.2, and the master plan reserves Group K for it.
Three problems besides the missing backend:

- The cache holds a `threading.Lock`, so a pipeline with a `Cache` cannot go to a process pool, and `check_pickleable` does not say so.
- An input without `tobytes` is keyed by `repr`, which for a STAC item dict is long and, for objects with a default `repr`, includes a memory address, so the key changes on every run.
- There is no way to key by a field, the way toolz `memoize(key=…)` allows.

### Store

```python
class Store(Protocol):
    def __contains__(self, key: str) -> bool: ...
    def load(self, key: str) -> Any: ...
    def save(self, key: str, value: Any) -> None: ...
```

Keys are 64-character hex digests, so a store never has to sanitise them.
Two stores ship in core:

| Store | Where | Concurrency |
|---|---|---|
| `MemoryStore()` | a dict | a lock; threads in one process |
| `DirectoryStore(root, codec="pickle")` | `root/ab/abcdef….pkl`, two-level fan-out | writes go to a temporary file and are renamed into place, so readers never see a partial value; processes on one file system can share it; the last writer wins |

`codec="pickle"` is the only codec in core; pickle loads run code, so a `DirectoryStore` is for directories the user controls, and the docs say so.
Carrier-specific stores (Zarr for xarray, COG for geoarrays, an object store) live in xrops and geoops and only implement the three methods.

### Cache, extended

`Cache(inner, store=None, key=None)`:

- `store=None` keeps today's behaviour exactly: a private `MemoryStore`, the same config `{"inner": …}` and the same `pipeline_hash`.
- `store="scenes"` names a store that `store_scope(scenes=...)` binds at run time; the name is in the config, the directory is not, for the same reason the executor is not (decision O5).
  An unbound name raises `LookupError` naming the store.
- `key=op` computes the input key from `op(x)` instead of from `x`, so `Cache(read, key=Get("id"))` keys a scene by its STAC ID.
- The config part of the key becomes `pipeline_hash(inner)`, which today equals `sha256(stable_json(inner.state))`: it covers nested operators and the inner class's module and name, which `get_config()` alone does not (two different classes with the same config collide today).
  For in-memory caches this only changes keys inside one process, which nobody can observe.
- `__getstate__` drops the lock and the private entries, so a `Cache` pickles; `hits` and `misses` stay per process.

### Why no `Checkpoint`

The gap list proposed `Checkpoint(store)`.
With a named store, `Cache(op, store="scenes")` already is that: it saves each result under a stable key and skips the work on a rerun.
A second class with the same behaviour would split the docs, and `pipekit_train.Checkpoint` (`packages/pipekit-train/src/pipekit_train/callbacks.py`) already means "save model state every $N$ steps".
A pass-through writer that persists without reading back is a `Sink`.

### Running example

```python
# Shapes: K kept items

# proposed (P1)
scene_id = pk.Get("id")
# proposed (P4)
cached = pk.Cache(retried, store="scenes", key=scene_id)
# proposed (P4)
scratch = pk.DirectoryStore("/scratch/s2")
# proposed (P4)
in_scratch = pk.store_scope(scenes=scratch)
with in_scratch:
    # 1 item → 1 scene
    scene = cached(item)
```

After a crash, a rerun loads every scene already in `/scratch/s2` and reads only the rest.

## 5. Observe and QC

### Today

`Tap`, `ShapeTrace`, `Quarantine` and `AssertCallable` hold closures and set `forbid_in_yaml` (`packages/pipekit-ops/src/pipekit_ops/observe.py`, `packages/pipekit-ops/src/pipekit_ops/qc.py`).
They are right for notebooks and they block a production pipeline from round-tripping.
`AssertShape` and `AssertDType` read `.shape` and `.dtype` with `getattr`; no check uses the named-dim `Signature` that the adapter registry already provides.

### Log

`Log(message="call {n}: {type} {signature}", level="INFO", logger="pipekit_ops")` passes its input through and writes one `logging` record per call.
The record's message is formatted with three fields: `{n}` (calls so far in this process), `{type}` (the input's class name) and `{signature}` (`signature_of(x).format()` when an adapter is registered, else `?`).
Everything else is left to `logging`: handlers, levels, JSON formatters.
`Log` is serialisable because it holds only strings.

### Progress, through Map

A progress bar needs to know $n$ and the count done so far; both live in `Map`, not in a pass-through step.
`Map(…, progress=every)` logs `done`, `failed`, `in_flight`, `n` (when the input has a length) and items per second every `every` completed items, at INFO on `pipekit_ops.parallel`.
A standalone `Progress` operator is not added (decision O11).

### AssertThat and AssertSignature

`AssertThat(pred, message)` is `AssertCallable` with an operator instead of a closure: it raises `QCError` (`packages/pipekit-ops/src/pipekit_ops/qc.py`) when `pred(x)` is falsy and passes $x$ through otherwise.
`AssertCallable` stays for notebooks.

`AssertSignature(pattern, dtype=None, missing="raise")` reads `signature_of(x)` and raises `QCError` unless `Signature.matches(pattern)` is true (`packages/pipekit-ops/src/pipekit_ops/signature.py`) and, if `dtype` is set, the signature's dtype equals it.
If no adapter is registered for $x$, `missing="raise"` raises `QCError`, and `missing="skip"` passes $x$ through.
Its `compute_output_signature` checks the input signature against the pattern at build time, so `Sequential.summary()` reports a mismatch before any data runs.
It never reads the carrier itself, only the adapter, so it keeps rule 1 of [../vision.md](../vision.md).

### Running example

```python
# Shapes: B = 4 bands; y, x pixels

# user
valid = ValidFraction()
# user
at_least_half = AtLeast(0.5)
# today
mostly_valid = valid | at_least_half
# today
check_valid = pk.AssertThat(mostly_valid, message="under 50 % valid pixels")
# today
check_dims = pk.AssertSignature("band y x", dtype="uint16")
# today
note = pk.Log("read {n}: {signature}", level="DEBUG")
# today
screen = check_dims | check_valid | note
```
