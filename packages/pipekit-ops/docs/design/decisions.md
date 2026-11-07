---
status: draft
version: 0.1.0
---

# Decisions

IDs O1 to O16.
Each row states the recommendation, the alternative it was weighed against, and why the recommendation wins.
Phases P1 to P5 have shipped (see [roadmap.md](roadmap.md)), so every decision except O15 is built as recommended; O15 is built in part.
The table keeps each decision as it was argued.
Where the build departed from it, the section "As built" after the table says what was built and why, under the same ID.

| ID | Topic | Recommended | Alternative | Why |
|---|---|---|---|---|
| O1 | Scope of core | Recommended: pipekit-ops adds container plumbing, flow limits, execution, a store protocol and serialisable checks, and never inspects a carrier or adds a dependency. | Alternative: a separate `pipekit-flow` package for the new operators. | The operators are small, stdlib-only and used by every sister package; a new package adds an install and a version to pin for about 1,400 lines. |
| O2 | Callables | Recommended: new operators take predicates, keys and folds as `Operator` instances or plain keys; a bare callable raises `TypeError` pointing to `Lambda`. | Alternative: accept callables, as toolz does, and set `forbid_in_yaml` on every new class. | With operators, a pipeline built from new ops round-trips and hashes; with callables, half the new catalogue would be flagged and `check_pickleable` would grow noisier. |
| O3 | Output type | Recommended: iterable ops return lists (dicts for the group-by family); `Map.iter` and `Prefetch` are the only lazy outputs. | Alternative: lazy iterators throughout, like toolz. | A `Graph` node may feed several consumers, and a `Cache` must hash its input; both break on a one-shot iterator. |
| O4 | One engine | Recommended: a single `Map(op, executor=…)` implements ordering, errors, backpressure, chunking and cancellation once. | Alternative: keep four classes and add the missing features to each. | Four copies of the same loop drift; today the classes already disagree on fail fast (0.20 s vs 1.01 s in the same test). |
| O5 | Where the executor lives | Recommended: `executor` is a name in the config; a live executor is bound with `executor_scope` and is not hashed. | Alternative: executor objects as config-carrying operators (`ThreadExecutor(8)`, `DaskExecutor(address)`). | The same pipeline must have the same `pipeline_hash` on a laptop and on a cluster; any `concurrent.futures.Executor` (dask, Ray, loky) then works with no code in pipekit. |
| O6 | The four presets | Recommended: keep `ThreadMap`, `ProcessMap`, `AsyncMap`, `BatchedMap` as subclasses of `Map` with unchanged constructors, configs and hashes; document them as presets. | Alternative: deprecate them in favour of `Map` and remove them in 0.3. | They appear in docs, tutorials and stored configs; their hashes are registry keys, and a deprecation would buy only a shorter list of names. |
| O7 | Default error policy | Recommended: `on_error="raise"` fails fast and cancels pending work; `"collect"` raises a `MapError` (an `ExceptionGroup`) with partial results; `"log_and_continue"` keeps today's `None` slots. | Alternative: make `"collect"` the default, as batch tools such as Prefect do. | Fail fast matches `ThreadMap` and `AsyncMap` today and Python's own `Executor.map`; collect is one argument away. |
| O8 | Backpressure | Recommended: `Map` pulls input lazily and holds at most `max_in_flight` items, default $2 n_w$ for owned pools and 32 for borrowed ones; in `Map.iter(ordered=True)` and `Prefetch`, finished results not yet yielded count as held. | Alternative: `Executor.map(…, buffersize=…)`, added in Python 3.14. | pipekit supports 3.12 and 3.13; and a hand-written loop over `submit` and `wait` is also what gives per-item error records and cancellation. |
| O9 | Process start method | Recommended: `start_method="spawn"` by default for owned process pools, with `"fork"` and `"forkserver"` as opt-outs. | Alternative: the platform default (fork on Linux up to 3.13). | fork deadlocks after JAX, BLAS or a thread pool has started threads, and makes pickling bugs platform-dependent. |
| O10 | Pickling check | Recommended: before the first remote submit, run `check_pickleable(op)` and then `pickle.dumps(op)` once; `Cache` and `RateLimit` define `__getstate__`. | Alternative: trust the `forbid_in_yaml` flag, as today. | The flag misses unflagged state: `ProcessMap(Cache(op))` fails today with a lock pickling error that `check_pickleable` does not report. |
| O11 | Progress | Recommended: `Map(…, progress=every)` logs counts and throughput through `logging`; `Log` is the pass-through observer. | Alternative: a `Progress` operator, or a tqdm extra. | A pass-through step cannot know $n$ or what is in flight; `Map` can. Core takes no dependency, and a tqdm logging handler is a user choice. |
| O12 | Disk cache | Recommended: a three-method `Store` protocol, `MemoryStore` and `DirectoryStore` in core, `Cache(inner, store=name, key=op)`, stores bound by `store_scope`; no `Checkpoint` class. | Alternative: `Cache(inner, path="/scratch")` with the path in the config, plus a separate `Checkpoint(store)` operator. | A path in the config makes the hash depend on the machine; `Checkpoint` would duplicate `Cache` with a store and clash with `pipekit_train.Checkpoint`. |
| O13 | Names from toolz | Recommended: use toolz names where semantics match (`PartitionAll`, `Concat`, `UpdateIn`, `Assoc`, `Dissoc`, `Get`, `GetIn`, `CountBy`, `Count`). | Alternative: the planned names (`Chunk`, `Flatten`, `OnKey`, `GetItem`). | The background page calls pipekit "toolz with each function replaced by a typed class"; matching names make that literal and avoid reusing `Count` for two meanings. |
| O14 | Two `BatchedMap`s | Recommended: the core class keeps `BatchedMap`; the planned pipekit-array class becomes `BatchAlong(inner, batch_size, dim)`, as the sibling proposal says. | Alternative: ADR A5 of pipekit-array: keep both names and let the import path disambiguate. | Hydra YAML and `from pipekit_ops import *` next to pipekit-array imports both see the bare class name; the core name has shipped since v0.1, the array one has not. |
| O15 | Async composition | Recommended: sync wrappers (`Retry`, `Try`, `Cache`, `Timeout`, `RateLimit`, `Until`, the iterable ops) refuse a child with an `async def _apply` at construction; async-aware versions come later through an optional `_apply_async`. | Alternative: make every wrapper detect and await coroutines. | Today `Retry(async_op)` returns an un-awaited coroutine and retries nothing; an early `TypeError` fixes the silent bug now, and the full async path can be designed on its own. |
| O16 | JAX tracers | Recommended: `Map(executor="serial")` accepts tracers; every pool, `Cache`, `Timeout`, `RateLimit` and `Prefetch` raise `TypeError` on a tracer via `is_traced`. | Alternative: refuse tracers everywhere in the execution group. | A serial loop under `jit` is just unrolled Python and is useful for small $n$; sending a tracer to a thread or process is never correct. |

## As built

These notes come from the pull requests that implemented P1 to P5 (#209, #210, #211, #215, #216, #217).
Decisions without a note were built as written.

- **O2 (callables).**
  As decided, a bare `lambda` is refused with a `TypeError` that names `Lambda`.
  Only the per-item key of `GroupBy`, `CountBy`, `Unique` and `SortBy` accepts an operator; the dict operators take plain keys, because `Get(op)` would just be `op`.
  `Rename`'s old keys must be `str`, because they are JSON keys in the config.
  The lambda check exists twice, as `dicts.require_child` and as a private helper in `flow.py`; merging them is a small follow-up.
- **O4 (one engine).**
  The async executor runs its event loop on a helper thread, so async goes through the same window loop as every other executor.
  `Map.run` drives that loop from `asyncio.to_thread`, with the tasks on the caller's loop; cancelling `run()` does not stop the driver thread.
- **O6 (the four presets).**
  The presets keep their constructors, configs and hashes (gate G2-6), with three differences from the design.
  `ProcessMap` now accepts `on_error="collect"`, inherited from `Map`.
  `BatchedMap` does not get the `executor` argument or the `PartitionAll` internals yet, to keep its constructor unchanged: it runs `Map(op)` serially over `itertools.batched` lists and flattens.
  `AsyncMap` keeps its own running-loop error text, which points to `await async_map.run`.
- **O8 (backpressure).**
  `ordered` is an argument of `Map.iter`, not of the constructor, so it does not enter the config or the hash.
- **O9 (process start method).**
  `Map(executor="process")` uses spawn, as decided.
  The `ProcessMap` preset keeps the platform's start method, as the P2 fallback in [roadmap.md](roadmap.md) allows, and raises a `FutureWarning` at construction when that default is not spawn; so on Linux (fork) every `ProcessMap` warns.
- **O10 (pickling check).**
  `Cache` gained `__getstate__` and `__setstate__`: a pickled copy has a fresh lock, no entries and zero counters.
  A pickled or loaded `RateLimit` starts with fresh state.
  Building P4 also fixed a P1 bug: the `_MISSING` sentinel of `Get` was a bare `object()`, so a pickled `Get("id")` returned a copy of the sentinel instead of raising `KeyError`, and its hash held a memory address, which would have broken cache sharing across processes.
  The sentinel now survives pickling.
- **O11 (progress).**
  `Log`'s default message is `"call {n}: {type} {signature}"`, not the design's `"{op}"`, because `{op}` is not one of the three fields the design defines.
  A message naming any other field raises `ValueError` at construction.
  `Log` counts calls even when its logger is disabled for the level, but writes nothing; when a carrier arrives without its adapter, `{signature}` prints `?`.
- **O12 (disk cache).**
  The cache key is now `sha256(input key ‖ 0x00 ‖ pipeline_hash(inner))`, plus `‖ 0x00 ‖ pipeline_hash(key)` when `key` is set, so two classes with equal configs never share an entry, and neither do `key=Get("id")` and `key=Get("name")`.
  Only in-process keys changed; `pipeline_hash(Cache(...))` did not (gate G4-1), and the pinned test of the old framing became `test_hashing.py::test_cache_key_digest_framing`.
  `DirectoryStore` writes a format marker, `pipekit-store.json` with version 1, refuses a directory of another version, and refuses keys that are not 64-character hex, which blocks path traversal.
  `Cache.clear()` clears only a private store, never a shared named one, and `store` must be a `str`.
  To carry bound stores to process workers (execution.md §2.7), P4 also edited `execution.py`: owned pools get them through the initializer, borrowed non-thread executors with each task.
- **O15 (async composition).**
  Built in part.
  `Retry`, `Timeout`, `RateLimit` and `Until` refuse a child whose `_apply` is `async def` with a `TypeError` at construction (gate G3-7; `Until` checks both `op` and `pred`).
  `Try`, `Cache` and the iterable operators do not refuse yet; for `Cache` this was left out of #216 as a follow-up.

Four departures from [api/operators.md](api/operators.md) fall under no decision:

- **`Timeout` starts a daemon thread per call, not a shared pool.**
  A pool's workers are joined at interpreter exit, so one call that hangs for good would stop the process from exiting, and a bounded pool full of hung workers would make later calls time out without ever running.
  Abandoned calls are tracked process-wide; when more than `max_abandoned` are still running, a timeout logs a warning on `pipekit_ops.flow`.
- **`RateLimit` windows are half-open, $[t, t + T)$**, where $t \in \mathbb{R}$ is the oldest recorded start in seconds and $T \in \mathbb{R}_{>0}$ is `period`.
  A call that starts at $t_\text{now} \in \mathbb{R}$ is admitted when $t_\text{now} - t \ge T$; with a strict $>$, a fake clock spins forever on ties.
- **`Until` returns rather than raises when `pred` holds after the last allowed call**, and `max_iter=0` only checks the input.
- **`AssertSignature` raises at build time, rather than only reporting.**
  Its `compute_output_signature` makes `Sequential.summary()` raise the mismatch without running any data, which is how `compute_output_signature` errors already surface (as in pipekit-array's `MeanScalar`).
  An unknown or foreign signature passes through.

## Things this design does not decide

- **A cross-process rate limiter.**
  It needs shared state; the docs show how to divide the limit instead.
- **Async-aware wrappers.**
  Phase P6 designs `_apply_async`, and has not started; until then O15 makes the failure loud, for the four operators that refuse today.
- **A `Graph` that runs independent nodes in parallel.**
  `Graph` stays a serial topological walk; a parallel `Graph` evaluator would build on `Map` and the executor scope, and is a separate design.
