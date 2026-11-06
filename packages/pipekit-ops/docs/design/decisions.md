---
status: draft
version: 0.1.0
---

# Decisions

IDs O1 to O16.
Each row states the recommendation, the alternative it was weighed against, and why the recommendation wins.
Status is `proposed` for all of them until the first phase lands.

| ID | Topic | Recommended | Alternative | Why |
|---|---|---|---|---|
| O1 | Scope of core | Recommended: pipekit-ops adds container plumbing, flow limits, execution, a store protocol and serialisable checks, and never inspects a carrier or adds a dependency. | Alternative: a separate `pipekit-flow` package for the new operators. | The operators are small, stdlib-only and used by every sister package; a new package adds an install and a version to pin for about 1,400 lines. |
| O2 | Callables | Recommended: new operators take predicates, keys and folds as `Operator` instances or plain keys; a bare callable raises `TypeError` pointing to `Lambda`. | Alternative: accept callables, as toolz does, and set `forbid_in_yaml` on every new class. | With operators, a pipeline built from new ops round-trips and hashes; with callables, half the new catalogue would be flagged and `check_pickleable` would grow noisier. |
| O3 | Output type | Recommended: iterable ops return lists (dicts for the group-by family); `Map.iter` and `Prefetch` are the only lazy outputs. | Alternative: lazy iterators throughout, like toolz. | A `Graph` node may feed several consumers, and a `Cache` must hash its input; both break on a one-shot iterator. |
| O4 | One engine | Recommended: a single `Map(op, executor=…)` implements ordering, errors, backpressure, chunking and cancellation once. | Alternative: keep four classes and add the missing features to each. | Four copies of the same loop drift; today the classes already disagree on fail fast (0.20 s vs 1.01 s in the same test). |
| O5 | Where the executor lives | Recommended: `executor` is a name in the config; a live executor is bound with `executor_scope` and is not hashed. | Alternative: executor objects as config-carrying operators (`ThreadExecutor(8)`, `DaskExecutor(address)`). | The same pipeline must have the same `pipeline_hash` on a laptop and on a cluster; any `concurrent.futures.Executor` (dask, Ray, loky) then works with no code in pipekit. |
| O6 | The four presets | Recommended: keep `ThreadMap`, `ProcessMap`, `AsyncMap`, `BatchedMap` as subclasses of `Map` with unchanged constructors, configs and hashes; document them as presets. | Alternative: deprecate them in favour of `Map` and remove them in 0.3. | They appear in docs, tutorials and stored configs; their hashes are registry keys, and a deprecation would buy only a shorter list of names. |
| O7 | Default error policy | Recommended: `on_error="raise"` fails fast and cancels pending work; `"collect"` raises a `MapError` (an `ExceptionGroup`) with partial results; `"log_and_continue"` keeps today's `None` slots. | Alternative: make `"collect"` the default, as batch tools such as Prefect do. | Fail fast matches `ThreadMap` and `AsyncMap` today and Python's own `Executor.map`; collect is one argument away. |
| O8 | Backpressure | Recommended: `Map` pulls input lazily and keeps at most `max_in_flight` tasks, default $2 n_w$ for owned pools and 32 for borrowed ones. | Alternative: `Executor.map(…, buffersize=…)`, added in Python 3.14. | pipekit supports 3.12 and 3.13; and a hand-written loop over `submit` and `wait` is also what gives per-item error records and cancellation. |
| O9 | Process start method | Recommended: `start_method="spawn"` by default for owned process pools, with `"fork"` and `"forkserver"` as opt-outs. | Alternative: the platform default (fork on Linux up to 3.13). | fork deadlocks after JAX, BLAS or a thread pool has started threads, and makes pickling bugs platform-dependent. |
| O10 | Pickling check | Recommended: before the first remote submit, run `check_pickleable(op)` and then `pickle.dumps(op)` once; `Cache` and `RateLimit` define `__getstate__`. | Alternative: trust the `forbid_in_yaml` flag, as today. | The flag misses unflagged state: `ProcessMap(Cache(op))` fails today with a lock pickling error that `check_pickleable` does not report. |
| O11 | Progress | Recommended: `Map(…, progress=every)` logs counts and throughput through `logging`; `Log` is the pass-through observer. | Alternative: a `Progress` operator, or a tqdm extra. | A pass-through step cannot know $n$ or what is in flight; `Map` can. Core takes no dependency, and a tqdm logging handler is a user choice. |
| O12 | Disk cache | Recommended: a three-method `Store` protocol, `MemoryStore` and `DirectoryStore` in core, `Cache(inner, store=name, key=op)`, stores bound by `store_scope`; no `Checkpoint` class. | Alternative: `Cache(inner, path="/scratch")` with the path in the config, plus a separate `Checkpoint(store)` operator. | A path in the config makes the hash depend on the machine; `Checkpoint` would duplicate `Cache` with a store and clash with `pipekit_train.Checkpoint`. |
| O13 | Names from toolz | Recommended: use toolz names where semantics match (`PartitionAll`, `Concat`, `UpdateIn`, `Assoc`, `Dissoc`, `Get`, `GetIn`, `CountBy`, `Count`). | Alternative: the planned names (`Chunk`, `Flatten`, `OnKey`, `GetItem`). | The background page calls pipekit "toolz with each function replaced by a typed class"; matching names make that literal and avoid reusing `Count` for two meanings. |
| O14 | Two `BatchedMap`s | Recommended: the core class keeps `BatchedMap`; the planned pipekit-array class becomes `BatchAlong(inner, batch_size, dim)`, as the sibling proposal says. | Alternative: ADR A5 of pipekit-array: keep both names and let the import path disambiguate. | Hydra YAML and `from pipekit_ops import *` next to pipekit-array imports both see the bare class name; the core name has shipped since v0.1, the array one has not. |
| O15 | Async composition | Recommended: sync wrappers (`Retry`, `Try`, `Cache`, `Timeout`, `RateLimit`, `Until`, the iterable ops) refuse a child with an `async def _apply` at construction; async-aware versions come later through an optional `_apply_async`. | Alternative: make every wrapper detect and await coroutines. | Today `Retry(async_op)` returns an un-awaited coroutine and retries nothing; an early `TypeError` fixes the silent bug now, and the full async path can be designed on its own. |
| O16 | JAX tracers | Recommended: `Map(executor="serial")` accepts tracers; every pool, `Cache`, `Timeout`, `RateLimit` and `Prefetch` raise `TypeError` on a tracer via `is_traced`. | Alternative: refuse tracers everywhere in the execution group. | A serial loop under `jit` is just unrolled Python and is useful for small $n$; sending a tracer to a thread or process is never correct. |

## Things this design does not decide

- **A cross-process rate limiter.**
  It needs shared state; the docs show how to divide the limit instead.
- **Async-aware wrappers.**
  Phase P6 designs `_apply_async`; until then O15 makes the failure loud.
- **A `Graph` that runs independent nodes in parallel.**
  `Graph` stays a serial topological walk; a parallel `Graph` evaluator would build on `Map` and the executor scope, and is a separate design.
