# pipekit-ops quickstart

The [README](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-ops#readme) screens a month of Sentinel-2 scene records with a filter, a retried and cached read on four threads, and a count per tile, and shows that the whole chain round-trips through JSON with the same content hash. This page picks up from there: the same composition on NumPy arrays with a `Graph` and an observer, then how the pieces work (the operator model, the wrappers, the execution engine, the dict and list operators, saving and hashing), the zero-dependency guarantee, what pipekit-ops deliberately leaves out, and how to move code written for older versions.

The notation is the README's. A pipeline $p = o_K \circ \ldots \circ o_1$ is a chain of $K$ operators, where $o_k : \mathcal{X}_{k-1} \to \mathcal{X}_k$ maps one carrier to the next, and $h(p)$ is its content hash.

## Operators on arrays: `|`, `Graph`, observers and JSON

We standardise a row of $N$ reflectance values $\mathbf{x} \in \mathbb{R}^N$ by a fixed mean $\mu$ and spread $\sigma$, so $\mathbf{z} = (\mathbf{x} - \mu) / \sigma \in \mathbb{R}^N$, clip $\mathbf{z}$ to $[-3, 3]$ to give $\tilde{\mathbf{z}} \in \mathbb{R}^N$, and then take the difference $\mathbf{d} = \mathbf{a} - \mathbf{b} \in \mathbb{R}^N$ of two dates $\mathbf{a}, \mathbf{b} \in \mathbb{R}^N$, each cleaned the same way. The carrier is a NumPy array this time, so the example also needs NumPy and jaxtyping (`uv add numpy jaxtyping`); pipekit-ops itself never imports them.

```python
import numpy as np
from jaxtyping import Float
from pipekit_ops import Graph, Input, Operator, Sequential, Tap, dumps, loads


# Shapes: N = 3 samples
Vector = Float[np.ndarray, " N"]


# user
class Standardize(Operator):
    """Centre and scale by a fixed mean and spread."""

    def __init__(self, *, mean: float, std: float) -> None:
        # μ
        self.mean = mean
        # σ
        self.std = std

    def _apply(self, x: Vector) -> Vector:
        # z = (x − μ) / σ
        return (x - self.mean) / self.std


# user
class Clip(Operator):
    """Clip to the interval [lo, hi]."""

    def __init__(self, *, lo: float, hi: float) -> None:
        self.lo = lo
        self.hi = hi

    def _apply(self, z: Vector) -> Vector:
        # z̃ = min(max(z, lo), hi)
        return np.clip(z, self.lo, self.hi)


# user
class Difference(Operator):
    """Subtract the second input from the first."""

    def _apply(self, a: Vector, b: Vector) -> Vector:
        # d = a − b
        return a - b


# user
def show(v: Vector) -> None:
    print("clean:", v)


# 1 · compose with |: a Sequential, itself an Operator

# user
# z = (x − 10) / 2
standardize = Standardize(mean=10.0, std=2.0)
# user
# z̃ = min(max(z, −3), 3)
clip = Clip(lo=-3.0, hi=3.0)
# today (pipekit-ops)
# o_2 ∘ o_1
clean: Sequential = standardize | clip

# (N,)
x: Vector = np.array([4.0, 10.0, 30.0])
# (N,) → (N,)
z: Vector = clean(x)

# 2 · wire a Graph: named inputs, shared operators, named outputs

# user
difference = Difference()
# today (pipekit-ops)
a = Input("a")
# today (pipekit-ops)
b = Input("b")
a_clean = clean(a)
b_clean = clean(b)
anomaly = difference(a_clean, b_clean)
# today (pipekit-ops)
graph = Graph(inputs={"a": a, "b": b}, outputs={"anomaly": anomaly})

# d = clean(a) − clean(b)
# (N,), (N,) → {"anomaly": (N,)}
out: dict[str, Vector] = graph(a=x, b=x + 2.0)

# 3 · observe without changing the flow

# today (pipekit-ops)
traced: Sequential = clean | Tap(show)
# (N,) → (N,), and show prints the clean values
same_values: Vector = traced(x)

# 4 · configuration is data: an operator round-trips through JSON

# today (pipekit-ops)
text: str = dumps(standardize)
# today (pipekit-ops)
same: Operator = loads(text)

print(z, out["anomaly"], same)
```

This prints `clean: [-3.  0.  3.]` from the `Tap`, then `[-3.  0.  3.] [-1. -1.  0.] Standardize(mean=10.0, std=2.0)`. The first is $\tilde{\mathbf{z}}$: $(4 - 10) / 2 = -3$, $0$, and $(30 - 10) / 2 = 10$ clipped to $3$. The second is $\mathbf{d}$ for $\mathbf{b} = \mathbf{x} + 2$, whose cleaned values are $[-2, 1, 3]$. The third is the standardiser rebuilt from its JSON, `{"class": "Standardize", "config": {"mean": 10.0, "std": 2.0}, "module": "__main__"}`. Inside the `Graph`, `clean(a)` called on an `Input` does not compute anything: it records a node, and the graph runs the nodes in topological order when it is called on data.

## The operator model

:::{image} ../assets/operator-light.svg
:alt: A pipekit Operator takes a carrier and returns one through a single _apply method; it composes into a Sequential (a | b | c), a Graph with named inputs and outputs, wrappers such as Retry, Cache and Tap, and stateful steps such as Cycle and TrainingLoop, each itself an Operator
:class: dark:hidden
:width: 100%
:::

:::{image} ../assets/operator-dark.svg
:alt: A pipekit Operator takes a carrier and returns one through a single _apply method; it composes into a Sequential (a | b | c), a Graph with named inputs and outputs, wrappers such as Retry, Cache and Tap, and stateful steps such as Cycle and TrainingLoop, each itself an Operator
:class: hidden dark:block
:width: 100%
:::

1. **An `Operator` has one method**, `_apply(self, carrier) -> carrier`. pipekit never inspects the carrier; domain packages narrow it.
2. **Composition is `|`.** `a | b | c` is a `Sequential`, flattened, so `(a | b) | c` and `a | (b | c)` are the same three steps. Flows that branch and merge are a `Graph`, built by calling operators on `Input`s; it rejects cycles and evaluates in topological order.
3. **Wrappers are operators too.** `Retry`, `Timeout`, `Try`, `Cache`, `Map` and the QC asserts take an operator and return one, so they nest anywhere.
4. **State is explicit.** A `StatefulOperator` maps $(x, s) \mapsto (x', s')$, a carrier and a state to the next carrier and state, and a `Sequential` or `Graph` threads the state through its stateful steps; pipekit-cycle's `Cycle` and `DACycle` and pipekit-train's `TrainingLoop` are built on it.
5. **Configuration is data.** An operator's constructor arguments, stored under their own names, are its config; `dumps` writes it as JSON and `loads` rebuilds the operator.
6. **Libraries plug in by protocol.** `Predictor`, `FittableTransformer` and the named-array adapters are runtime-checkable seams, so a library satisfies them without importing pipekit ([Protocols](../protocols.md)).

The long form is [Concepts](../concepts.md). A single operator also carries `backends` (the array backends it accepts) and `compute_output_signature` (how it changes a `Signature`), so `Sequential.summary` and `Graph.summary` can print shapes before anything runs.

## Wrappers: control, flow, cache and store, observe, QC

- **Control** puts decisions and failures inside the pipeline: `Branch`, `Switch`, `Try`, `Coalesce` and `Retry`. `Try` and `Retry` need their exception list, so neither is ever a silent catch-all.
- **Flow limits** bound a step: `Timeout` gives up after a number of seconds (the thread is abandoned, not killed), `RateLimit` caps calls per period across the threads of one process, and `Until` repeats a step until a predicate holds.
- **Cache and store.** `Cache(op)` keys on a SHA-256 of the input and the inner operator's `pipeline_hash`. `Cache(op, store="scenes", key=Get("id"))` keeps its entries in a named store bound with `store_scope`, so the store is not in the config and the pipeline hashes the same on a laptop and on a cluster. A `DirectoryStore` writes one pickle per key, atomically; pickle loads run code, so point it only at a directory you control.
- **Observers** pass the carrier through unchanged: `Tap`, `Log` and `ShapeTrace`, and the controllers `Snapshot` and `Histogram`, whose `.at(key)` gives the operator you place, and `Profile`, whose `.wrap(op)` does.
- **QC** turns "this won't break" into a permanent guard: `AssertShape`, `AssertDType`, `AssertThat` and `AssertSignature` raise `QCError`, while `Quarantine` returns a sentinel and keeps going.

Every wrapper is in the [core tour](../core-tour.md), with a run output for each.

A bigger graph puts these together: an anomaly detector that reads a stack of multi-channel images in parallel with retries and a cache, checks the stack, picks a composite by the number of exposures, scores against a background input and summarises, with four named outputs.

:::{image} ../assets/graph-light.svg
:alt: An anomaly detector as one pipekit Graph: sources are read in parallel (ThreadMap around Cache around Retry around ReadTile), stacked and checked, composited by a Branch (median for few exposures, mean for many), turned into a z-score against a background input and a channel ratio, scored, thresholded with a Tap, and summarised by a Fanout; composite, score, mask and summary are its outputs
:class: dark:hidden
:width: 640px
:::

:::{image} ../assets/graph-dark.svg
:alt: An anomaly detector as one pipekit Graph: sources are read in parallel (ThreadMap around Cache around Retry around ReadTile), stacked and checked, composited by a Branch (median for few exposures, mean for many), turned into a z-score against a background input and a channel ratio, scored, thresholded with a Tap, and summarised by a Fanout; composite, score, mask and summary are its outputs
:class: hidden dark:block
:width: 640px
:::

The wiring and the six domain operators, step by step and runnable, are in the tutorial [Build a bigger graph](../tutorials/anomaly-graph.md).

## Execution: `Map`, executors, `Prefetch`, `Fold` and the presets

`Map(op, executor=...)` is the one engine for running an operator over many items. `executor` is `"serial"`, `"thread"`, `"process"` (spawn by default), `"async"`, or `"scope"`, which borrows the `concurrent.futures.Executor` that an `executor_scope` binds (a Dask or Ray one included) and never shuts it down. The executor is not in the config, so the pipeline hashes the same wherever it runs. Input is pulled lazily, at most `max_in_flight` items are held at once, and `on_error` is `"raise"` (fail fast), `"collect"` (raise a `MapError` holding the partial results, with a `Failed` record per failure) or `"log_and_continue"`.

`Prefetch(op, depth)` reads up to `depth` items ahead of the consumer, and `Fold(op, initial, chunksize)` folds chunks in parallel through `Map` and then combines the partial results. `ThreadMap`, `ProcessMap`, `AsyncMap` and `BatchedMap` are presets of `Map` with their own arguments (`n_workers`, `semaphore`, `batch_size`). Process pools need every operator to pickle, so `check_pickleable(op)` walks a pipeline and returns the operators that hold a closure (they set `forbid_in_yaml = True`) before you fan out. The details are in the [execution design](../design-ops/api/execution.md).

## Dict and list plumbing

A `Graph` returns a dict, a `Fanout` returns a dict, and a STAC search returns a list of dicts. The dict operators (`Get`, `GetIn`, `Pick`, `Rename`, `Merge`, `Assoc`, `UpdateIn`, `ValMap`, `Dissoc`) reshape those without a `Lambda`, and the list operators (`Filter`, `Remove`, `GroupBy`, `CountBy`, `Reduce`, `Take`, `SortBy`, …) work on a list of items inside the pipeline. Their predicates and keys are operators too, so a chain built from them still round-trips through `dumps` and `loads`.

:::{note}
`pipekit_ops.Reduce` is a left fold over a list of items; `pipekit_array.Reduce` reduces an array over named dims. Import the packages under their own names (`pk.Reduce`, `pa.Reduce`) when you use both.
:::

## Saving and hashing

`dumps(op)` writes `op.state` as JSON with sorted keys, nested operators included: a child operator appears in its parent's config as its own record, a `Graph` records every node and its wiring, and an exception class is written as `{"exception": "builtins.OSError"}`. `loads` rebuilds the whole tree, as long as the modules that define the classes are imported. `pipeline_hash` hashes that record,

$$
h(p) = \mathrm{SHA256}(\mathrm{json}(s_p))
$$

where $s_p$ is the state of $p$ and $\mathrm{json}$ writes it with sorted keys; optional weight bytes are appended after a zero byte. So two pipelines share a hash exactly when they are configured the same, and the pipekit-experiment registries store models under it. Each public class is recorded under the module `pipekit_ops`, whichever area its file sits in, so moving a class between files inside the package does not change a hash.

For pipelines you did not write (uploaded by users, read from a queue), `loads_sandboxed` builds only the classes you have `register`ed, at any depth, and checks every class before constructing anything. The operators that hold a function or a debug payload (`Lambda`, `Sink`, `Tap`, `ShapeTrace`, `Branch`, `Switch`, `Coalesce`, `Quarantine` and `AssertCallable`) set `forbid_in_yaml`: `dumps` still writes them, but only as a debug record that names the function, `loads` refuses to rebuild them with a `RuntimeError`, and `pipeline_hash` hashes that record, so two different lambdas share a hash. When the pipeline must round-trip, use `Log` in place of `Tap` and an operator in place of a plain function wherever the operator takes one (the predicate of `Filter`, the key of `CountBy`); the flagged control-flow operators are rebuilt in code. `pipekit_ops.testing.assert_round_trip(op, *inputs)` checks the contract from your own test suite, and `public_operators(module)` lists the operators a module exports, so a sweep can fail on one that has no test case yet.

## The zero-dependency guarantee

pipekit-ops declares no dependencies, and we keep it that way: `import pipekit_ops` loads only the standard library, and `tests/test_no_dependencies.py` checks both (the import in a fresh interpreter). The one exception is `pipekit_ops.testing`, which imports NumPy lazily when `assert_round_trip` compares two arrays. Array-aware features find what they need through the carrier instead. `AssertShape` reads `shape`, `Cache` keys an array by its bytes, dtype and shape, and the named-array adapters (a `NamedArrayAdapter` registered by pipekit-array, gridax, geotoolz-array or another library) give a carrier's dims to `AssertSignature` and `Signature`. `on_import` runs a callback when a module such as `jax` is imported, without importing it, which is how pipekit-array's JAX pytree registrations stay optional.

So you get real value from this package without the rest of pipekit, and a library can depend on it without pulling in NumPy, JAX or anything else. If your science has its own carrier (AnnData, astropy tables, a `GeoTensor`), you write operators for it and get composition, control flow, caching, parallel maps, QC and a JSON round-trip for free. pipekit-ops and pipekit-array are the two foundations of the workspace, each usable on its own, and every other pipekit package builds on pipekit-ops; the [workspace README](https://github.com/jejjohnson/pipekit#the-layering) has the layering.

## What is not built

Two things are left out by design.

- There is no automatic parallelism: `Sequential` runs in order and `Graph` evaluates in one process.
- There is no graph scheduler: `Map(executor="scope")` can send the items of one step to a Dask or Ray executor, but scheduling whole pipelines across machines is left to Dask, Ray, Beam and orchestrators.

The open design questions are in the [roadmap](../design-ops/roadmap.md).

## Moving from older versions

**From `pipekit` before 0.2.** Until 0.2.0 this package was called `pipekit`. Change the import; every name and argument is unchanged.

```python
# Today: since 0.2.0 this line raises an ImportError that names the new import
# from pipekit import Operator, Retry, Sequential
# After:
# today (pipekit-ops)
from pipekit_ops import Operator, Retry, Sequential
```

Libraries should depend on `pipekit-ops`, not on `pipekit`: since 0.2.0 `pipekit` is the umbrella that installs the whole workspace, JAX included. States saved before 0.2.0 (`dumps` output, registry entries) still load: a module saved as `pipekit.<name>` resolves to `pipekit_ops.<name>`. Their `pipeline_hash` changes, because the hash covers each operator's module.

**From the flat layout before 0.4.** Until 0.4.0 the package was a flat list of modules (`pipekit_ops.control`, `pipekit_ops.serial`, `pipekit_ops.hashing`, …); since then it is the seven areas. Those modules are gone, so `import pipekit_ops.control` raises `ModuleNotFoundError`: import the name from `pipekit_ops` or from its area (`pipekit_ops.saving` for `sha256_hex`) instead. A state saved under one of the flat modules still loads, because `record_moves` reads those module names as `pipekit_ops`; its `pipeline_hash` changes once, since every class is now recorded under `pipekit_ops` (decision O17 in [Decisions](../design-ops/decisions.md)).

## Further reading

- [Patterns](../patterns.md) for batch jobs, services, orchestrators and audits, and [Guards](../guards.md) for `guard` and `jit_report`.
- [Why operators are classes, not functions](../background/toolz.md).
- The [design docs](../design-ops/README.md) (vision, decisions, roadmap, per-module API specs) and the master plan, [Report 2](https://github.com/jejjohnson/research_journal_v2/blob/main/notes/geotoolz/master_plan/toolz_2_pipekit.md).
- The [API reference](https://jejjohnson.github.io/pipekit/reference/pipekit-ops/).
- Tutorials: [a bigger graph](../tutorials/anomaly-graph.md), [ETL over many regions](../tutorials/etl-regions.md), [inference](../tutorials/inference.md).
- The sibling foundation, [pipekit-array](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-array), and the rest of the [family](https://github.com/jejjohnson/pipekit#the-packages).
