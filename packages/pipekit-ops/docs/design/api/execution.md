---
status: draft
version: 0.1.0
---

# Execution

One operator, `Map`, runs another operator over many items.
Where the work happens is an argument: in the calling thread, on threads, on processes, on an event loop, or on any object that implements the stdlib `concurrent.futures.Executor` protocol.
The four wrappers that ship today become presets of it.

## 1. Today

`packages/pipekit-ops/src/pipekit_ops/parallel.py` ships four classes.
Each takes an iterable, calls `list(iterable)` first, and returns a list in input order.

| Class | Engine | Error behaviour, measured on `903b0c3` |
|---|---|---|
| `ThreadMap(op, n_workers=8)` | `ThreadPoolExecutor.map` | the first error in input order propagates; `Executor.map` cancels the futures that have not started. With 10 items of 0.2 s on 2 workers and item 0 failing, it raised after 0.20 s. |
| `ProcessMap(op, n_workers=8, on_error="raise")` | `ProcessPoolExecutor.submit` per item | `"raise"` re-raises when the loop reaches the failed future, then the `with` block waits for every queued item. The same test raised after 1.01 s: all ten items ran. `"log_and_continue"` logs and puts `None` in the slot. |
| `AsyncMap(op, semaphore=8)` | `asyncio.gather` under a semaphore; sync ops go through `asyncio.to_thread` | `gather` raises the first error; `asyncio.run` cancels the waiting tasks. Inside a running loop, `__call__` raises `RuntimeError` and points to `await op.run(items)`. |
| `BatchedMap(op, batch_size=8, flatten=True)` | a serial loop over slices | the error propagates. |

Five gaps follow from that code.

1. **No backpressure.**
   Every class materialises the whole input, and `ProcessMap` submits every item at once.
   For $N$ items that means $N$ pending futures and, if the results are large, all $N$ results in memory before the caller sees one.
2. **No collect mode.**
   An error either aborts the run or turns into `None`, which a later step cannot tell apart from a real `None`.
3. **No pluggable executor.**
   A dask or Ray cluster cannot be used without leaving pipekit, although both offer a `concurrent.futures.Executor`.
4. **Pickling is checked by a flag, not by pickling.**
   `check_pickleable` walks the tree for `forbid_in_yaml` classes.
   It misses objects that are not flagged and still do not pickle: `Cache` holds a `threading.Lock` (`packages/pipekit-ops/src/pipekit_ops/cache.py`), so `ProcessMap(Cache(op))` fails with `TypeError: cannot pickle '_thread.lock' object` while `check_pickleable(Cache(op))` returns `[]`.
5. **Async does not compose.**
   A wrapper whose child has an `async def _apply` calls the child and gets a coroutine back.
   `Retry(AsyncFetch(), attempts=3, …)(x)` returns an un-awaited coroutine, so nothing runs and nothing is retried; `AsyncMap` around that `Retry` returns a list of coroutines (checked on `903b0c3`).

## 2. The recommended design

### 2.1 One engine

```python
# proposed (P2)
class Map(Operator):
    def __init__(
        self,
        op: Operator,
        executor: str = "serial",
        n_workers: int | None = None,
        max_in_flight: int | None = None,
        on_error: str = "raise",
        chunksize: int = 1,
        start_method: str | None = None,
        progress: int | None = None,
    ) -> None: ...

    def _apply(self, items: Iterable[Any]) -> list[Any]: ...
    def iter(self, items: Iterable[Any], ordered: bool = True) -> Iterator[Any]: ...
    async def run(self, items: Iterable[Any]) -> list[Any]: ...
```

`executor` is a name, so the config stays JSON:

| `executor` | Runs on | Owned by `Map` | `n_workers` default |
|---|---|---|---|
| `"serial"` | the calling thread, a plain loop | nothing to own | 1 |
| `"thread"` | a `ThreadPoolExecutor` created per call | yes | 8 |
| `"process"` | a `ProcessPoolExecutor` created per call, with an explicit start method | yes | `os.process_cpu_count()` (3.13+), else `os.cpu_count()` |
| `"async"` | an event loop; `async def _apply` is awaited, a sync op runs in a thread | yes | 8 |
| `"scope"` | the executor set by `executor_scope(...)`, or serial if none is set | no: borrowed, never shut down | taken from the executor |

A live executor object is never part of the config.
It is bound around the call:

```python
from concurrent.futures import ThreadPoolExecutor

import pipekit_ops as pk

# Shapes: K kept items

pool = ThreadPoolExecutor(max_workers=8)
# proposed (P2)
reader = pk.Map(checked, executor="scope", max_in_flight=16)
# proposed (P2)
on_threads = pk.executor_scope(pool)
with on_threads:
    # (K items) → (K scenes)
    scenes = reader(kept)
```

`executor_scope` is a context manager over a `contextvars.ContextVar`.
`pipeline_hash(reader)` is the same on a laptop and on a cluster, because where the work runs is not part of what the pipeline computes.

A dask cluster plugs in through `distributed.Client.get_executor()` and Ray through `ray.util.multiprocessing` or a thin executor shim; pipekit-ops imports neither.
The protocol `Map` relies on is the stdlib one: `submit(fn, *args) -> Future`, plus `Future.result()`, `Future.cancel()`, `Future.done()` and `concurrent.futures.wait(..., return_when=FIRST_COMPLETED)`.

### 2.2 Ordering

- `Map.__call__` and `Map.run` always return a list in input order.
  Results are written into slot $i$ of a list of length $n$ as they complete, where $n$ is the number of items consumed and $i$ the item's position.
- `Map.iter(items, ordered=True)` yields results lazily in input order; with `ordered=False` it yields `(i, result)` pairs in completion order.
  The ordered iterator holds at most `max_in_flight` finished results while it waits for a slow item at the head.

### 2.3 Backpressure: a window of in-flight tasks

Let $W$ be `max_in_flight`.
`Map` pulls items from the input lazily and keeps at most $W$ tasks submitted and not yet collected.
When a task completes, it is collected and the next item is pulled and submitted.
The default is $W = 2 n_w$ for an owned pool, where $n_w$ is the number of workers, so each worker has one task running and one queued.
For a borrowed executor the default is $W = 32$, because `Map` cannot see how many workers it has.

The stdlib `Executor.map` submits every item before it yields the first result (on 3.12 and 3.13, which pipekit supports), so `Map` does not use it.
It runs its own loop over `submit` and `wait(FIRST_COMPLETED)`.

```python
# Shapes: n items; W = max_in_flight

# proposed (P2): the core loop of Map._apply, sketched
pending: dict[Future, int] = {}
for i, item in enumerate(items):
    # block until fewer than W tasks are in flight
    while len(pending) >= W:
        done = collect_one(pending, results)
    task = pool.submit(run_one, op, item)
    pending[task] = i
# drain the window
while pending:
    done = collect_one(pending, results)
```

Peak memory is then bounded by $W$ inputs and $W$ outputs in flight, plus the results already collected; `Map.iter` drops even those.

### 2.4 Error policy

`on_error` takes one of three values.

| `on_error` | Behaviour | Result |
|---|---|---|
| `"raise"` (default) | fail fast: on the first failure, stop pulling items, cancel every pending future, and re-raise | the exception, with a note `"Map: item i failed"` added through `BaseException.add_note` |
| `"collect"` | run every item, then raise if any failed | `MapError`, a subclass of `ExceptionGroup`, with one exception per failed item (each noted with its index) and the partial results on `MapError.results`, where a failed slot holds a `Failed(index, exception)` record |
| `"log_and_continue"` | log each failure as a warning on the `pipekit_ops.parallel` logger, with the traceback | a list with `None` in each failed slot, exactly as `ProcessMap` does today |

Fail fast changes what `ProcessMap(on_error="raise")` does: today it waits for every queued item; under `Map` it cancels what has not started.
Results stay the same; only the wall time drops.
`"raise"` re-raises the first failure *observed*, which on a pool may not be the first in input order; the note names the index.

`KeyboardInterrupt` and `SystemExit` are never collected or logged: they cancel the window and propagate, whatever `on_error` says.

### 2.5 Cancellation

- **Futures not yet started** are cancelled with `Future.cancel()`.
- **Owned pools** are shut down with `shutdown(wait=True, cancel_futures=True)`, so no work outlives the call.
- **Borrowed executors** are never shut down: `Map` cancels its own futures and leaves the pool to its owner.
- **Running thread tasks cannot be stopped.**
  Python has no safe way to kill a thread; a running task finishes and its result is dropped.
  `Timeout` (see [operators.md](operators.md)) bounds how long the caller waits, not how long the thread runs.
- **Async tasks** are cancelled with `Task.cancel()`, so an `async def _apply` sees `CancelledError` at its next `await`.

### 2.6 Chunking

With `chunksize` $c > 1$, `Map` submits lists of $c$ consecutive items to a worker-side helper that applies `op` to each item and returns one record per item.
That divides the number of submits, and the pickling of the task, by $c$, which pays off when one item costs less than a process round trip.
Errors stay per item: the helper catches each item's exception and returns it as a record, so `on_error` acts on items, not on chunks.
A sensible value for cheap items is $c = \lceil n / (4 n_w) \rceil$; the default is $c = 1$.

### 2.7 Pickling rules for processes and remote executors

Before it submits the first task to a process pool or a borrowed executor that is not a `ThreadPoolExecutor`, `Map` does two checks once per call:

1. `check_pickleable(op)` (`packages/pipekit-ops/src/pipekit_ops/parallel.py`); a non-empty result raises `TypeError` naming the flagged operators.
2. `pickle.dumps(op)`; a failure raises `TypeError` naming the class that failed.
   This catches what the flag misses, such as the lock inside `Cache`.

Operators that hold locks or counters implement `__getstate__` and `__setstate__`: `Cache` pickles its config without its lock or entries (a worker starts with an empty in-memory cache), and `RateLimit` pickles without its timestamps.

For an owned process pool, `op` is pickled once per worker, through the pool's `initializer`, not once per task.

**Start method.**
`Map(executor="process")` uses `start_method="spawn"` unless told otherwise.
`fork` copies the parent's threads' locks in whatever state they hold, which deadlocks after JAX, a BLAS library or a `ThreadPoolExecutor` has started threads.
The interpreter default differs by platform (fork on Linux up to 3.13, spawn on macOS and Windows), and Python 3.14 moves Linux to forkserver.
Pinning spawn gives the same behaviour everywhere and makes a pickling bug fail on a Linux laptop the way it would on a Mac.
It has two costs: the script needs an `if __name__ == "__main__":` guard, and each worker re-imports the modules, a few hundred milliseconds per worker.
`start_method="fork"` or `"forkserver"` opt out.

**Scopes cross into workers.**
Thread tasks run inside `contextvars.copy_context().run`, so `executor_scope` and `store_scope` (see [operators.md](operators.md)) are visible in worker threads.
Process and remote tasks carry the resolved stores with them, pickled; a `DirectoryStore` pickles as its path.

**No nested pools by default.**
Inside a `Map` task the executor scope is reset to serial.
An inner `Map(executor="scope")` then runs serially in its worker instead of submitting to the pool that runs it, which can deadlock a bounded pool.
An inner `Map` that names its own executor still gets it.

### 2.8 Async

- `Map(executor="async")` drives an event loop with `asyncio.run` when none is running, and keeps today's `RuntimeError` inside a running loop, which points to `await reader.run(items)`.
  Starting a second loop on another thread would block the caller's loop, which is worse than an error.
- A sync `op` runs through `asyncio.to_thread`, which uses the loop's default executor.
  That pool holds `min(32, os.cpu_count() + 4)` threads, so a `max_in_flight` above that is capped silently today; `Map` installs a default executor of `max_in_flight` threads for the loop it owns.
- `AsyncMap` detects coroutine `_apply` with `asyncio.iscoroutinefunction`, which Python 3.14 deprecates; `Map` uses `inspect.iscoroutinefunction`.
- **Wrappers refuse async children.**
  `Retry`, `Try`, `Cache`, `Timeout`, `RateLimit`, `Until` and the iterable operators are sync.
  Wrapping an operator whose `_apply` is a coroutine function raises `TypeError` at construction, instead of returning an un-awaited coroutine at call time (gap 5).
  Async-aware versions of `Retry`, `Timeout` and `RateLimit`, through an optional `_apply_async`, are phase P6.

### 2.9 JAX tracers

| Executor | Tracer input | Why |
|---|---|---|
| `"serial"` | allowed | a Python loop under `jax.jit` unrolls, the same as a `Sequential`; slow to compile for large $n$, but correct |
| any pool, async or borrowed | `TypeError` before anything is submitted | JAX's trace state is thread-local, and a tracer has no value to send to a process |

The check is `is_traced(items)` from `packages/pipekit-ops/src/pipekit_ops/_base/tracing.py`, the same one `Cache` uses, so pipekit-ops still never imports JAX.
The message points to `jax.vmap` or `jax.lax.map`.
`Cache`, `Timeout`, `RateLimit` and `Prefetch` refuse tracers for the same reasons; `Log` warns once, like `Tap`.

### 2.10 Prefetch

`Prefetch(op, depth=2, executor="thread")` returns a lazy iterator that runs `op` on up to `depth` items ahead of the consumer, in input order.
It is `Map.iter(items, ordered=True)` with $W$ set to `depth`, given its own name because the intent differs: overlap reading with computing, not parallelise the work.

```python
# Shapes: K kept items; d = 4 read-ahead

# proposed (P2)
# read scene i + 1 … i + 4 while scene i is screened
ahead = pk.Prefetch(cached, depth=4)
# proposed (P2)
screen_each = pk.Map(screen)
# today
read_and_screen = ahead | screen_each
# (K items) → (K summaries)
summaries = read_and_screen(kept)
```

`Map` pulls its input lazily (section 2.3), so at most 4 scenes are read and not yet screened.

### 2.11 The presets

The four classes stay, as thin subclasses of `Map` with their constructors and `get_config` unchanged:

| Preset | Is | Changes for users |
|---|---|---|
| `ThreadMap(op, n_workers)` | `Map(op, executor="thread", n_workers=n_workers)` | input is pulled lazily with $W = 2 n_w$ |
| `ProcessMap(op, n_workers, on_error)` | `Map(op, executor="process", n_workers=n_workers, on_error=on_error, start_method="spawn")` | fail fast on `"raise"`; spawn instead of the platform default; pickling checked before submit |
| `AsyncMap(op, semaphore)` | `Map(op, executor="async", max_in_flight=semaphore)` | `run` is the same; wrappers around async children now refuse at construction |
| `BatchedMap(op, batch_size, flatten)` | `PartitionAll(batch_size)`, then `Map(op)`, then a flatten step | gains an `executor` argument whose default `"serial"` keeps today's config |

Their `pipeline_hash` does not change: the hashes recorded on `903b0c3` become test fixtures (see [../roadmap.md](../roadmap.md)).
They are documented as presets and not deprecated (decision O6).

## 3. Running example

The read step of the Sentinel-2 job, end to end.
$K$ items survive the cloud screen; each read returns a scene carrier with dims `band y x`.

```python
from concurrent.futures import ThreadPoolExecutor

import pipekit_ops as pk

# Shapes: K ≈ 180 kept items; B = 4 bands; W = 16 in flight

# user
read = ReadBands(bands=("B02", "B03", "B04", "B08"))
# proposed (P3)
bounded = pk.Timeout(read, seconds=60.0)
# proposed (P3)
limited = pk.RateLimit(bounded, calls=10, period=1.0)
# today
retried = pk.Retry(limited, attempts=4, base_delay=1.0, on=(OSError, TimeoutError))
# proposed (P1)
scene_id = pk.Get("id")
# proposed (P4)
cached = pk.Cache(retried, store="scenes", key=scene_id)
# proposed (P5)
checked = cached | pk.AssertSignature("band y x")
# proposed (P2)
reader = pk.Map(checked, executor="scope", max_in_flight=16, on_error="collect")

pool = ThreadPoolExecutor(max_workers=8)
# proposed (P4)
scratch = pk.DirectoryStore("/scratch/s2")
# proposed (P2)
on_threads = pk.executor_scope(pool)
# proposed (P4)
in_scratch = pk.store_scope(scenes=scratch)
with on_threads, in_scratch:
    # (K items) → (K scenes)
    scenes = reader(kept)
```

`progress=20` would add a log line every 20 items (phase P5).

What each failure does:

- A hung read raises `TimeoutError` after 60 s; `Retry` tries again, up to 4 attempts.
- Every attempt, retries included, waits for a slot in the rate limiter, so the API sees at most 10 request starts per second from this process.
- A scene that fails 4 times is collected; the other scenes finish, and `MapError.results` holds them.
  A rerun reads only the failed scenes, because the rest are in `/scratch/s2`.

## 4. The alternative

**Executor objects as operators.**
`Map(op, executor=ThreadExecutor(8))`, where `ThreadExecutor`, `ProcessExecutor`, `AsyncExecutor` and `DaskExecutor` are config-carrying classes and `executor_scope` does not exist.

| | Recommended: named executor + scope | Alternative: executor objects |
|---|---|---|
| Hash on laptop vs cluster | the same | differs, so a registry sees two pipelines |
| dask, Ray | any `concurrent.futures.Executor`, no code in pipekit | a wrapper class each, in pipekit or in an extra |
| Reproducibility | the deployment must set the scope; forgetting it runs serially | the pipeline says where it runs |
| Testing | one `executor_scope(ThreadPoolExecutor(2))` in a fixture | build each pipeline twice |

The alternative is better when the execution venue is part of the scientific record.
For pipekit it is not: the outputs are the same by gate G2-1, so the venue belongs to the deployment.
