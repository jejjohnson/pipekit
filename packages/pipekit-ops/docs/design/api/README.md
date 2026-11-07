---
status: draft
version: 0.1.0
---

# Operator catalogue

One table per group.
The columns are:

- **Operator**: the constructor, with its configuration arguments.
- **Signature**: what `__call__` takes and returns, then what `compute_output_signature` reports (`None` means "not tracked", as for `Fanout` and the parallel maps today).
- **Semantics**: one line; the detail is in [operators.md](operators.md) and [execution.md](execution.md).
- **Status**: `today` (ships at `903b0c3`, file named), `proposed (Pn)` (the roadmap phase in [../roadmap.md](../roadmap.md) that adds it), or `changed (Pn)` (ships, and this design changes it).

## Conventions shared by every new operator

- **Keys are plain values.**
  Where an argument is a key (`key`, `keys`), it is a `str`, an `int` or a tuple of them, and it is used for item lookup (`x[key]`), as in toolz's `getter` (`toolz/itertoolz.py`, `def getter`).
  Any argument that is a key may instead be an `Operator`, which is called on the item.
  A bare Python callable is refused with a `TypeError` that says to wrap it in `Lambda`.
- **Predicates and functions are operators.**
  `Filter(pred)`, `Until(op, pred, …)`, `AssertThat(pred)` and the rest take an `Operator`.
  Their config is then the child's config, so they round-trip whenever their children do, and no new class sets `forbid_in_yaml`.
- **Lists in, lists out.**
  Iterable operators accept any iterable and return a `list` (or a `dict` for the group-by family), as the parallel maps do today (`packages/pipekit-ops/src/pipekit_ops/parallel.py`).
  The two exceptions return a lazy iterator on purpose: `Prefetch` and `Map.iter`.
- **Several inputs, positionally.**
  `Merge` and `Zip` take their inputs as positional arguments, so they work as `Graph` nodes (`Merge()(a, b)`), and also accept one sequence of inputs, so they work inside a `Sequential` after `juxt` or `Fanout`.

## Dict plumbing

`Graph` and `Fanout` return dicts (`packages/pipekit-ops/src/pipekit_ops/_base/graph.py`, `packages/pipekit-ops/src/pipekit_ops/combine.py`), and STAC items are nested dicts.
Today only `Lambda` reshapes them.

| Operator | Signature | Semantics | Status |
|---|---|---|---|
| `Get(key, default=MISSING)` | `Mapping \| Sequence → Any`; `None` | `x[key]`; a tuple of keys returns a tuple; `default` replaces a missing key | today (`dicts.py`) |
| `GetIn(keys, default=MISSING)` | `Mapping → Any`; `None` | `x[k0][k1]…`; `default` replaces a missing path | today (`dicts.py`) |
| `Pick(keys)` | `Mapping → dict`; `None` | keep only `keys`, in the order given; a missing key raises `KeyError` | today (`dicts.py`) |
| `Dissoc(keys)` | `Mapping → dict`; `None` | drop `keys`; missing keys are ignored | today (`dicts.py`) |
| `Rename(mapping)` | `Mapping → dict`; `None` | rename keys by `mapping`; unlisted keys pass through; a rename onto an existing key raises `ValueError` | today (`dicts.py`) |
| `Merge()` | `*Mapping → dict`; `None` | union, later inputs win (toolz `merge`) | today (`dicts.py`) |
| `Assoc(key, op)` | `Mapping → dict`; `None` | add `key: op(x)` computed from the whole dict | today (`dicts.py`) |
| `UpdateIn(keys, op, default=MISSING)` | `Mapping → dict`; `None` | replace the value at a key path by `op(value)`; replaces the planned `OnKey` | today (`dicts.py`) |
| `ValMap(op)` | `Mapping → dict`; `None` | `op` on every value, keys kept (toolz `valmap`) | today (`dicts.py`) |

## Iterables

| Operator | Signature | Semantics | Status |
|---|---|---|---|
| `Map(op, executor="serial", …)` | `Iterable → list`; `None` | `op` on every item; the execution engine, see [execution.md](execution.md) | today (`execution.py`) |
| `Filter(pred)` | `Iterable → list`; `None` | keep items where `pred(item)` is truthy | today (`seq.py`) |
| `Not(pred)` | `Any → bool`; `None` | `not pred(x)`; the operator form of `complement` | today (`seq.py`) |
| `Remove(pred)` | `Iterable → list`; `None` | drop items where `pred(item)` is truthy (toolz `remove`) | today (`seq.py`) |
| `Reduce(op, initial=MISSING)` | `Iterable → Any`; `None` | left fold, `op(acc, item)`; empty input with no `initial` raises `TypeError` like `functools.reduce` | today (`seq.py`) |
| `Fold(op, initial, chunksize=128, combine=None, executor="scope")` | `Iterable → Any`; `None` | associative tree fold over chunks, parallel through `Map` (toolz `sandbox.fold`) | today (`execution.py`) |
| `PartitionAll(n)` | `Iterable → list[tuple]`; `None` | consecutive tuples of length `n`, the last may be shorter; replaces the planned `Chunk` | today (`seq.py`) |
| `Concat()` | `Iterable[Iterable] → list`; `None` | flatten one level (toolz `concat`); replaces the planned `Flatten` | today (`seq.py`) |
| `Zip(strict=True)` | `*Iterable → list[tuple]`; `None` | builtin `zip`, strict by default | today (`seq.py`) |
| `GroupBy(key)` | `Iterable → dict[Any, list]`; `None` | group items by `key`, first-seen key order (toolz `groupby`) | today (`seq.py`) |
| `CountBy(key)` | `Iterable → dict[Any, int]`; `None` | count items per `key` (toolz `countby`; `CountBy(Identity())` is toolz `frequencies`) | today (`seq.py`) |
| `Count()` | `Iterable → int`; `None` | number of items, lazy iterables included (toolz `count`) | today (`seq.py`) |
| `Unique(key=None)` | `Iterable → list`; `None` | first occurrence of each item, or of each `key(item)` (toolz `unique`) | today (`seq.py`) |
| `Take(n)` | `Iterable → list`; `None` | the first `n` items (toolz `take`), for dry runs | today (`seq.py`) |
| `SortBy(key, reverse=False)` | `Iterable → list`; `None` | stable sort by `key` | today (`seq.py`) |

## Flow

| Operator | Signature | Semantics | Status |
|---|---|---|---|
| `Retry(op, attempts, base_delay, on)` | `Any → Any`; `op`'s | exponential back-off, `base_delay * 2 ** (i - 1)` before attempt $i+1$ | today (`control.py`) |
| `Try(primary, fallback, on)` | `Any → Any`; common of both | toolz `excepts` | today (`control.py`) |
| `Timeout(op, seconds)` | `Any → Any`; `op`'s | raise `TimeoutError` if `op` has not returned after `seconds`; a thread is abandoned, not killed | proposed (P3) |
| `RateLimit(op, calls, period)` | `Any → Any`; `op`'s | at most `calls` starts of `op` in any window of `period` seconds, per process | proposed (P3) |
| `Until(op, pred, max_iter, on_max="raise")` | `Any → Any`; `None` | apply `op` until `pred(x)` is truthy; stateless sibling of `pipekit_cycle.Recurrence` | proposed (P3) |

## Execution

| Operator | Signature | Semantics | Status |
|---|---|---|---|
| `Map(op, executor, n_workers, max_in_flight, ordered, on_error, chunksize, start_method, progress)` | `Iterable → list`; `None` | one engine over serial, thread, process, async and any `concurrent.futures.Executor`; `progress` is P5 | today (`execution.py`), without `progress` |
| `Map.iter(items)`, `await Map.run(items)` | lazy iterator; awaitable list | streaming and running-event-loop entry points | today (`execution.py`) |
| `Prefetch(op, depth=2, executor="thread")` | `Iterable → Iterator`; `None` | read `depth` items ahead of the consumer, in input order | today (`execution.py`) |
| `executor_scope(executor)` | context manager | sets the executor that `Map(executor="scope")` uses (serial when none is set); not stored in any config | today (`execution.py`) |
| `ThreadMap(op, n_workers=8)` | `Iterable → list`; `None` | preset: `Map(op, executor="thread", …)`; config and hash unchanged | changed (P2) |
| `ProcessMap(op, n_workers=8, on_error="raise")` | `Iterable → list`; `None` | preset: `Map(op, executor="process", …)`; fails fast on `"raise"` | changed (P2) |
| `AsyncMap(op, semaphore=8)` | `Iterable → list`; `None` | preset: `Map(op, executor="async", max_in_flight=semaphore)` | changed (P2) |
| `BatchedMap(op, batch_size=8, flatten=True)` | `Iterable → list`; `None` | preset: `PartitionAll` then `Map` then optional `Concat`; gains `executor` | changed (P2) |
| `check_pickleable(op)` | `Operator → list[Operator]` | flagged operators in the tree | today (`parallel.py`); `Map` calls it, see [execution.md](execution.md) |

## Cache and store

| Operator | Signature | Semantics | Status |
|---|---|---|---|
| `Cache(inner, store=None, key=None)` | `Any → Any`; `inner`'s | memoise by input and the inner pipeline's hash; `store` names a `Store` in scope, `None` keeps today's private in-memory dict; `key` is an operator that computes the input key (toolz `memoize(key=…)`) | changed (P4) |
| `Memoize` | alias | `Memoize is Cache` | today (`cache.py`) |
| `Store` | protocol | `__contains__(key)`, `load(key)`, `save(key, value)` | proposed (P4) |
| `MemoryStore()` | `Store` | dict plus a lock; one process | proposed (P4) |
| `DirectoryStore(root, codec="pickle")` | `Store` | one file per key, atomic rename, safe across processes on one file system | proposed (P4) |
| `store_scope(**stores)` | context manager | binds store names to `Store` instances; not stored in any config | proposed (P4) |
| `Checkpoint(store)` | none | not added: it is `Cache(op, store=…)`, and `pipekit_train.Checkpoint` already owns the name | dropped |

## Observe and QC

| Operator | Signature | Semantics | Status |
|---|---|---|---|
| `Tap`, `Snapshot`, `ShapeTrace`, `Profile`, `Histogram` | pass-through | closure-holding observers | today (`observe.py`) |
| `Log(message="{op}", level="INFO", logger="pipekit_ops")` | `Any → Any`; passthrough | pass-through; logs one line per call with the call count and the input's `Signature` if an adapter is registered | proposed (P5) |
| `Map(…, progress=every)` | argument | log done / failed / in-flight counts and items per second every `every` items | proposed (P5) |
| `AssertThat(pred, message="predicate returned False")` | `Any → Any`; passthrough | `QCError` if `pred(x)` is falsy; the serialisable `AssertCallable` | proposed (P5) |
| `AssertSignature(pattern, dtype=None, missing="raise")` | `Any → Any`; passthrough | `QCError` unless `signature_of(x).matches(pattern)` (and the dtype agrees) | proposed (P5) |
| `Quarantine`, `AssertShape`, `AssertDType`, `AssertHasAttribute`, `AssertCallable`, `QCError` | pass-through | existing guards | today (`qc.py`) |
| `Progress`, observer `Count` | none | not added: `Map(progress=…)` and `Log`'s call count cover them, and `Count` keeps toolz's meaning | dropped |

## Name clash: two `BatchedMap`s

`pipekit_ops.BatchedMap` takes an iterable of items and calls the inner operator on lists of them (`packages/pipekit-ops/src/pipekit_ops/parallel.py`).
The pipekit-array design plans a `BatchedMap(inner, batch_size, axis)` that splits one array along an axis (`packages/pipekit-array/docs/design/api/operators.md`, ADR A5 in `packages/pipekit-array/docs/design/decisions.md`).
ADR A5 says the import path disambiguates.
This design disagrees (decision O14): code that imports both packages, or a Hydra YAML that names both, meets one name with two meanings.
The core name has shipped since v0.1, so the array operator should take the sibling proposal's name, `BatchAlong(inner, batch_size, dim)`.
