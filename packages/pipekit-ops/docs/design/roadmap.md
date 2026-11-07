---
status: draft
version: 0.1.0
---

# Roadmap and testing gates

Six phases, each a pull request or two.
P1 to P5 have shipped, in pipekit-ops 0.2.1; P6 is still deferred.

| Phase | Status | Pull requests |
|---|---|---|
| P1 | shipped | #209 |
| P2 | shipped | #210 (`Map`, `execution.py`), #211 (presets rebuilt on `Map`) |
| P3 | shipped | #215 |
| P4 | shipped | #216 |
| P5 | shipped | #217 |
| P6 | deferred | none yet |

Each phase below keeps its gate table as it was written, then says briefly what was measured when it landed.
The full numbers are in each pull request's description.

A phase is done when every gate in its table passes in CI, together with the four checks of the repository's pre-commit checklist (tests, `ruff check .`, `ruff format --check .`, `ty check`).

Gates that apply to every phase:

| ID | Gate | Pass condition |
|---|---|---|
| G0-1 | Round trip | every public operator added in the phase passes `pipekit_ops.testing.assert_round_trip(op, x)` on a fixed input: `loads(dumps(op))` has the same `pipeline_hash` and returns an equal output |
| G0-2 | Sweep | `public_operators(pipekit_ops)` lists no operator without a round-trip test case |
| G0-3 | No dependencies | `packages/pipekit-ops/pyproject.toml` still has `dependencies = []`, and importing `pipekit_ops` imports no module outside the stdlib (checked from `sys.modules` in a fresh interpreter) |
| G0-4 | No flags | no class added by this design sets `forbid_in_yaml = True` |

## P1: dict plumbing and iterables

Adds `Get`, `GetIn`, `Pick`, `Dissoc`, `Rename`, `Merge`, `Assoc`, `UpdateIn`, `ValMap`, `Not`, `Filter`, `Remove`, `Reduce`, `PartitionAll`, `Concat`, `Zip`, `GroupBy`, `CountBy`, `Count`, `Unique`, `Take`, `SortBy`.

| ID | Gate | Pass condition |
|---|---|---|
| G1-1 | toolz agreement | for 200 random inputs per operator (Hypothesis, fixed seed), each operator's output equals its toolz counterpart's, converted to a list where toolz is lazy: `Get`/`get`, `GetIn`/`get_in`, `Dissoc`/`dissoc`, `Merge`/`merge`, `ValMap`/`valmap`, `UpdateIn`/`update_in`, `Filter`/`filter`, `Remove`/`remove`, `PartitionAll`/`partition_all`, `Concat`/`concat`, `GroupBy`/`groupby`, `CountBy`/`countby`, `Count`/`count`, `Unique`/`unique`, `Take`/`take`; toolz is a test-only dependency |
| G1-2 | No mutation | for every dict operator, the input's `stable_json` before and after the call are equal |
| G1-3 | Strict zip | `Zip()([1, 2], [1])` raises `ValueError`; `Zip(strict=False)` returns `[(1, 1)]` |
| G1-4 | Callables refused | passing a bare `lambda` as a predicate or key to any P1 operator raises `TypeError` whose message contains `Lambda` |
| G1-5 | Running example | the example's search-and-screen `Graph` over a fixture of 300 STAC item dicts returns `n_found == 300` and the `n_kept` and `per_tile` values computed by a plain Python loop |

**Risk:** the new names crowd the top-level namespace (about 25 names).
**Fallback:** export them from `pipekit_ops.dicts` and `pipekit_ops.seq` only, and add top-level aliases later if users ask.

**Shipped** in #209, with all 22 operators also exported at top level (the fallback was not needed).
Measured:

- G1-1: Hypothesis with `max_examples=200` and a fixed seed; the 15 operators with a toolz counterpart agree with it, and the other 7 are checked against plain Python.
- G1-3: `Zip(strict=False)([1, 2], [1]) == [(1, 1)]`.
- G1-4: a bare `lambda` is refused by all 12 operators that take a predicate, key or function.
- G1-5: over 300 STAC-like item dicts the graph gives `n_found == 300`, with `n_kept` and `per_tile` equal to a plain loop, and the graph round-trips.
- G0-1 to G0-4: all 22 operators are in the round-trip sweep; a fresh interpreter that imports `pipekit_ops` loads only stdlib modules (`tests/test_no_dependencies.py`).

## P2: Map, executors, presets, Prefetch, Fold

Adds `Map`, `executor_scope`, `Prefetch`, `Fold`, `MapError`, rebuilds the four presets on `Map`, and gives `Cache` a `__getstate__` (decision O10) so it can go to a process pool.

| ID | Gate | Pass condition |
|---|---|---|
| G2-1 | Same outputs | for 1,000 integer items and an operator that sleeps a random 0 to 2 ms, the `serial`, `thread`, `process` and `async` executors and a borrowed `ThreadPoolExecutor(4)` return identical lists, in input order |
| G2-2 | Lazy input | `Map(op, executor="thread", n_workers=4, max_in_flight=8)` over a generator of 10,000 items never has more than 8 items pulled and not yet collected (a counting generator records the maximum) |
| G2-3 | Fail fast | with 2 workers, 100 items of 0.2 s and item 0 raising, `on_error="raise"` raises within 0.5 s on threads and within 1.5 s on processes (spawn start-up included); today `ProcessMap` takes about 10 s for the same input |
| G2-4 | Collect | with items 3 and 7 of 10 raising, `on_error="collect"` raises a `MapError` with exactly 2 sub-exceptions, notes naming items 3 and 7, and `results` holding the 8 other outputs in their slots |
| G2-5 | Log and continue | with item 3 of 10 raising, `on_error="log_and_continue"` returns 10 results with `None` at index 3 and emits exactly 1 warning on `pipekit_ops.parallel` |
| G2-6 | Hashes kept | `pipeline_hash` of `ThreadMap(Identity(), n_workers=4)`, `ProcessMap(Identity())`, `AsyncMap(Identity())`, `BatchedMap(Identity())` equals, respectively, `be9cbf3c…20bd`, `df4f9dd1…765b`, `2132e0bf…b081`, `70dec7fb…bf90` (full digests recorded at `903b0c3` in the test fixture) |
| G2-7 | Pickling caught early | `Map(Lambda(abs), executor="process")([1])` raises `TypeError` naming `Lambda` before any worker starts; `Map(Cache(Identity()), executor="process")([1, 2])` returns `[1, 2]` |
| G2-8 | Tracers | under `jax.jit`, `Map(op, executor="thread")` raises `TypeError`, and `Map(op, executor="serial")` returns the same values as the eager call (skipped when JAX is not installed) |
| G2-9 | Running loop | inside a running event loop, `Map(op, executor="async")(items)` raises `RuntimeError` whose message contains `run`, and `await Map(op, executor="async").run(items)` returns the eager result |
| G2-10 | Borrowed executor stays open | after `Map(..., executor="scope")` runs under `executor_scope(pool)`, `pool.submit(abs, -1).result() == 1` |
| G2-11 | Prefetch depth | `Prefetch(op, depth=4)` over 100 items never has more than 4 items pulled and not yet consumed; with item 0 blocked on an event, at most 3 later results finish and at most 4 items are pulled before the event is set |
| G2-12 | Fold equals reduce | for 10,000 random integers, `Fold(AddPair(), 0, chunksize=128, executor="thread")` equals `sum(items)`, where `AddPair` is a test operator returning `acc + item` |
| G2-13 | Nested maps | a `Map(executor="scope")` inside a `Map` task under a 2-worker pool completes 10 outer × 10 inner items within 5 s (no deadlock) |
| G2-14 | Ordered iterator window | `Map(op, executor="thread", n_workers=4, max_in_flight=4).iter(items, ordered=True)` over 100 items, with item 0 blocked on an event for 1 s, holds at most 4 items (running, queued or buffered) at every point, and yields `op(0) … op(99)` in order once the event is set |

**Risk:** spawn changes `ProcessMap` for Linux users whose scripts lack a `__main__` guard.
**Fallback:** keep `start_method=None` (platform default) on the `ProcessMap` preset for one minor release, with a `FutureWarning` when the default would differ, and use spawn only on `Map(executor="process")`.

**Risk:** the backpressure loop is the most intricate code in core.
**Fallback:** ship `Map` with the presets unchanged first, and move the presets onto `Map` in a second pull request once G2-1 to G2-5 hold for `Map`.

**Shipped** in two pull requests, as the second fallback suggests: #210 adds `Map`, `executor_scope`, `MapError`, `Failed`, `Prefetch` and `Fold`, and #211 rebuilds the four presets on `Map`.
The first fallback was taken too: `ProcessMap` keeps the platform's start method and raises a `FutureWarning` when that is not spawn, while `Map(executor="process")` uses spawn.
Measured:

- G2-2: at most 7 items were pulled and not yet collected, with a window of 8.
- G2-3: threads raised in 0.40 s (bound 0.5 s); `Map` on processes under spawn in 0.77 to 0.87 s, and `ProcessMap` in 0.429 s (bound 1.5 s).
  The wall-clock tests are marked `slow`, so CI skips them and runs timing-free versions instead (for `ProcessMap`: with 2 workers over 100 items and item 0 failing, at most 10 items start).
- G2-6: the four preset hashes are pinned in a test and unchanged.
- G2-8: run with JAX 0.10.1.
- G2-11: with the head item blocked, at most 4 items were held over the whole run.
- G2-13: 10 × 10 nested items finish in about 0.002 s.
- G2-14: at most 4 items held; the run took 1.005 to 1.009 s against the 1 s block, with results in order.
- G2-1, G2-4, G2-5, G2-7, G2-9, G2-10, G2-12: pass.

## P3: Timeout, RateLimit, Until

| ID | Gate | Pass condition |
|---|---|---|
| G3-1 | Timeout fires | `Timeout(op, seconds=0.1)` around an operator that sleeps 1 s raises `TimeoutError` within 0.15 s |
| G3-2 | Timeout passes | the same wrapper around an operator that sleeps 0.01 s returns its output |
| G3-3 | Retry on timeout | `Retry(Timeout(op, 0.1), attempts=3, base_delay=0, on=(TimeoutError,))` around an operator that hangs twice then returns, returns within 0.35 s |
| G3-4 | Rate | `RateLimit(op, calls=10, period=1.0)` called 50 times from 8 threads takes between 4.0 s and 4.5 s, and no window of 1 s contains more than 10 recorded start times |
| G3-5 | Until bound | `Until(op, pred, max_iter=5)` with a predicate that is never true raises `RuntimeError` after exactly 5 calls of `op`; with `on_max="return"` it returns $x_5$ |
| G3-6 | Until early | with a predicate already true on the input, `op` is called 0 times |
| G3-7 | Async children refused | `Retry`, `Timeout` and `RateLimit` wrapping an operator with `async def _apply` raise `TypeError` at construction |

**Risk:** abandoned threads from `Timeout` pile up when a service hangs for good.
**Fallback:** the `max_abandoned` warning, and `Timeout(…, isolate="process")` in a later phase.

**Shipped** in #215, in the module `pipekit_ops.flow`.
The `max_abandoned` warning shipped; `isolate="process"` has not.
Measured:

- G3-1: `Timeout` fired in 0.1006 s (limit 0.15 s).
- G3-2: a fast `op` returned in 0.0106 s.
- G3-3: 3 calls returned in 0.2029 s (limit 0.35 s).
- G3-4: 50 calls from 8 threads took 4.0021 s, with at most 10 starts in any 1 s window.
- G3-5 and G3-6: `Until` raises after exactly 5 calls (with `on_max="return"` it returns $x_5$), and calls `op` 0 times when `pred` already holds.
- G3-7: `Until` refuses async children too, for both `op` and `pred`, as decision O15 asks, so four operators pass the gate rather than three.

The wall-clock tests for G3-1, G3-3 and G3-4 are marked `slow`; each has a deterministic twin, on events or a fake clock, that runs in CI.

## P4: Store and the disk cache

| ID | Gate | Pass condition |
|---|---|---|
| G4-1 | Default unchanged | `Cache(Identity())` has `pipeline_hash` `24b9f307…0149`, as at `903b0c3`, and the same `hits` / `misses` sequence as today on the core-tour example (`(1, 2)`) |
| G4-2 | Survives a restart | a `Cache(op, store="s")` over a `DirectoryStore` computes 100 items once in one process; a second process with a fresh `Cache` returns the same 100 outputs with `misses == 0` |
| G4-3 | Atomic writes | 8 processes writing the same 100 keys concurrently leave 100 files, each loadable and equal to the computed value |
| G4-4 | Pickles | `pickle.loads(pickle.dumps(Cache(op)))` works, and the copy has no entries |
| G4-5 | Key by field | `Cache(op, key=Get("id"))` called on two dicts with the same `id` and different other fields runs `op` once |
| G4-6 | Unbound name | calling `Cache(op, store="missing")` outside any `store_scope` raises `LookupError` naming `missing` |
| G4-7 | Class in key | `Cache(A())` and `Cache(B())`, where `A` and `B` are different classes with equal configs, never share an entry in one store |

**Risk:** moving the config part of the key to `pipeline_hash(inner)` invalidates nothing on disk today (there is no disk backend), but a later change of `pipeline_hash` framing would.
**Fallback:** a `DirectoryStore` writes a `pipekit-store.json` with a format version, and refuses a directory with another version.

**Shipped** in #216, with the fallback built in from the start: `pipekit-store.json` records version 1.
Measured:

- G4-1: `pipeline_hash(Cache(Identity()))` is still `24b9f307…0149` (pinned), the config is still `{"inner": …}`, and the core-tour example still gives `(1, 2)`.
- G4-2: a first spawned process gives hits and misses `(0, 100)`; a second one, with a fresh `Cache`, gives `(100, 0)` and identical outputs.
- G4-3: 8 spawned writers saving the same 100 keys in shuffled order (20 KB values) leave exactly 100 files and no temporary files, all loadable and correct.
- G4-6: the `LookupError` names `'missing'` both outside any scope and inside another scope.
- G4-7: `A()` and `B()` with equal configs give 2 entries in one shared store.
- G4-4 and G4-5: pass.

## P5: Log, progress, AssertThat, AssertSignature

| ID | Gate | Pass condition |
|---|---|---|
| G5-1 | Log pass-through | `Log()` returns its input object unchanged (`is`) and emits exactly one record per call |
| G5-2 | Progress | `Map(op, progress=10)` over 100 items emits exactly 10 progress records, the last with `done == 100` |
| G5-3 | AssertThat | `AssertThat(Const(False))(x)` raises `QCError`; `AssertThat(Const(True))(x) is x` |
| G5-4 | AssertSignature | with a registered adapter, a carrier with dims `band y x` passes `"band y x"` and `"... y x"`, and fails `"time band y x"` with `QCError`; an unregistered carrier raises with `missing="raise"` and passes with `missing="skip"` |
| G5-5 | Build-time check | `Sequential.summary()` of a pipeline whose upstream signature is `time y x` followed by `AssertSignature("band y x")` reports the mismatch without running data |

**Shipped** in #217.
G5-1 to G5-5 pass (`tests/test_observe_qc_p5.py`, 32 tests).
`progress` is left out of `Map`'s config when it is `None`, and a test pins `pipeline_hash(Map(Identity()))` at `b5098ac5…`, its value before this phase, so every existing `Map` and preset keeps its hash.

## P6: async-aware wrappers (deferred)

`Retry`, `Timeout` and `RateLimit` gain an `_apply_async` used by `Map(executor="async")`, so an async read can be retried and rate-limited on one event loop.
Its gates are written with its design; the P3 refusal (G3-7) is lifted operator by operator.
Nothing of P6 has been built yet.

## Order and dependencies

```
P1  dict plumbing, iterables
 │
 ▼
P2  Map, executors, presets ──► P3  Timeout, RateLimit, Until
 │                               │
 ▼                               ▼
P4  Store, disk cache ────────► P5  Log, progress, asserts
                                 │
                                 ▼
                                P6  async-aware wrappers
```

P1 has no dependencies and unblocks `Fold` and `BatchedMap`'s rebuild.
P3 needs P2 only for `Timeout`'s shared pool and the async refusal; it can start in parallel once `Map`'s worker helpers exist.
As built, `Timeout` starts a daemon thread per call instead of using a shared pool (see [decisions.md](decisions.md)), and the five phases landed as a stack of pull requests in the order P1, P2, P3, P4, P5.
P4 needs `Get` from P1 for `key=` and the pickling rules from P2.
P5 needs `Map` for `progress`.
