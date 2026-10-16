# A tour of the core

*Everything in `pipekit`, group by group: what it is, when to reach for it,
and what it does today.*

The core is pure Python with no dependencies. Its public surface is about
fifty names in thirteen modules, one module per job, so you can learn it a
group at a time. For why it is shaped this way (operators as classes, the
carrier left open), read the [background](background/toolz.md); for the
shapes real pipelines take, read [Patterns](patterns.md).

Every output on this page is from a real run of pipekit 0.0.2.

## The map

| Group | Module | Names | Reach for it when |
|---|---|---|---|
| Foundations | `_base/` | `Operator`, `ConfigMixin`, `Sequential`, `Graph`, `Input`, `Node` | always: writing a step, chaining steps, wiring a graph |
| Composition | `compose` | `pipe`, `compose`, `compose_left`, `juxt`, `complement` | you think in `toolz` |
| Building blocks | `blocks` | `Identity`, `Const`, `Lambda`, `Sink` | a no-op, a constant, a quick function, a write |
| Control flow | `control` | `Branch`, `Switch`, `Try`, `Coalesce`, `Retry` | a decision or a failure inside the pipeline |
| Observers | `observe` | `Tap`, `Snapshot`, `ShapeTrace`, `Profile`, `Histogram` | looking without changing the flow |
| Combination | `combine` | `Fanout` | one input, several named outputs |
| Caching | `cache` | `Cache`, `Memoize` | the same expensive call twice |
| Quality control | `qc` | `Quarantine`, `AssertShape`, `AssertDType`, `AssertHasAttribute`, `AssertCallable`, `QCError` | a guard that fails loudly, or sets bad inputs aside |
| Shape inference | `signature` | `Signature`, `compute_output_signature` | catching a shape mismatch before running |
| Parallelism | `parallel` | `ThreadMap`, `ProcessMap`, `AsyncMap`, `BatchedMap`, `check_pickleable` | many items, one operator |
| Serialisation | `serial` | `dumps`, `loads`, `loads_sandboxed`, `from_state_sandboxed`, `register`, `unregister`, `allowed` | configuration as data; untrusted pipelines |
| Hashing | `hashing` | `pipeline_hash` | one key for "the same pipeline again": a registry, a cache, an orchestrator task |
| State | `state` | `StatefulOperator`, `CarryState` | state carried across calls: time steps, training, streams |
| Protocols | `protocols` | `Predictor`, `FittableTransformer` | accepting a model without importing its library |

The sister packages build on this core: time stepping and data assimilation
in `pipekit-cycle`, training in `pipekit-train`, the model registry in
`pipekit-experiment`, benchmarks in `pipekit-evaluate`, Array API dispatch in
`pipekit-array`, JAX models in `pipekit-jax`.

The examples share one small operator:

```python
from pipekit import Operator


class Scale(Operator):
    """x ↦ a · x."""

    def __init__(self, factor: float) -> None:
        self.factor = factor  # stored under its argument's name → its config

    def _apply(self, x):
        return x * self.factor
```

## Foundations

### `Operator`

The class every step subclasses. You write two things: a constructor that
stores its arguments under the same names, and `_apply`. The base class
supplies the rest:

- **configuration**: `get_config()` is derived from the constructor's
  signature (`ConfigMixin`), and `state` is a JSON-safe record of the class
  and its config;
- **rebuilding**: `Operator.from_state(state)` finds the class among every
  imported subclass and calls it with the config;
- **the pipe**: `a | b` is a `Sequential`, flattened;
- **two modes in one `__call__`**: on data it runs `_apply`; on a graph
  placeholder (`Input` or `Node`) it records a node instead;
- **class flags**: `forbid_in_yaml` (holds a closure, so its config cannot be
  rebuilt) and `_terminal` (may only end a `Sequential`).

```python
import pipekit as pk

Scale(2.0).state
# → {'module': '__main__', 'class': 'Scale', 'config': {'factor': 2.0}}
pk.loads(pk.dumps(Scale(2.0)))   # → Scale(factor=2.0)
```

### `Sequential`

Apply operators left to right. It checks at construction that every item is
an operator and that a `_terminal` one comes last; `|` flattens, so
`(a | b) | c` and `a | (b | c)` are the same three steps. It has `len`,
indexing and slicing, a Keras-style `summary(signature)` and a `describe()`
tree, and the empty `Sequential` is the identity.

```python
from pipekit import Sequential, Signature

pipe = Scale(0.5) | Scale(2.0) | Scale(3.0)
pipe(10.0)         # → 30.0
len(pipe)          # → 3
pipe[1:]           # → Sequential([Scale(factor=2.0), Scale(factor=3.0)])
Sequential([])(7)  # → 7

print(pipe.summary(Signature(dims=(("y", 4), ("x", 4)), dtype="float32")))
# Operator   Input sig                      Output sig
# -----------------------------------------------------------------------
# [0] Scale  Signature((y=4, x=4) float32)  Signature((y=4, x=4) float32)
# [1] Scale  Signature((y=4, x=4) float32)  Signature((y=4, x=4) float32)
# [2] Scale  Signature((y=4, x=4) float32)  Signature((y=4, x=4) float32)
```

### `Graph`, `Input` and `Node`

For flows that branch and merge, with several inputs and named outputs.
Calling operators on `Input`s (and on what those calls return) records the
graph; `Graph(inputs=…, outputs=…)` wraps it, rejects cycles, and evaluates
in topological order. Inputs bind by keyword or by position; outputs come
back as a dict.

```python
from pipekit import Graph, Input, Lambda

a, b = Input("a"), Input("b")
s = Scale(2.0)(a)                                     # s = 2a
total = Lambda(lambda u, v: u + v, name="add")(s, b)  # s + b

g = Graph(inputs={"a": a, "b": b}, outputs={"s": s, "sum": total})
g(a=1.0, b=10.0)   # → {'s': 2.0, 'sum': 12.0}
g(1.0, 10.0)       # → the same; positional arguments bind in declaration order
```

A single-input graph composes into a `Sequential`
(`(Scale(2.0) | g1)(1.0)` → `{'y': 6.0}` for a graph `g1` with output
`y = 3x`). The [bigger-graph tutorial](tutorials/anomaly-graph.md) builds a
full one.

## Composition

Free functions that carry the `toolz` surface. They are not operators
themselves (except what they return). Below, `inc = Lambda(lambda v: v + 1)`.

| Function | `toolz` ancestor | What it does | Example |
|---|---|---|---|
| `pipe(value, *ops)` | `toolz.pipe` | `Sequential(ops)(value)` | |
| `compose(*ops)` | `toolz.compose` | right to left: `compose(f, g, h)` is `Sequential([h, g, f])` | `compose(Scale(2.0), inc)(3.0)` → `8.0` |
| `compose_left(*ops)` | `toolz.compose_left` | left to right, like `Sequential` | `compose_left(Scale(2.0), inc)(3.0)` → `7.0` |
| `juxt(*ops)` | `toolz.juxt` | each operator on the same input; a tuple | `juxt(Identity(), Scale(2.0))(5.0)` → `(5.0, 10.0)` |
| `complement(pred)` | `toolz.complement` | negates a predicate | `complement(str.isdigit)("a")` → `True` |

`juxt` returns a plain function, so it cannot sit inside a `Sequential`;
`Fanout` is the operator-shaped version.

## Building blocks

| Operator | Purpose | Holds a closure |
|---|---|---|
| `Identity()` | an explicit no-op; the default branch of `Branch` and `Switch` | no |
| `Const(value)` | return a fixed value, whatever the input | no |
| `Lambda(fn, name=…)` | wrap any callable; the escape hatch | yes |
| `Sink(write_fn, name=…)` | write as a side effect, return the input unchanged | yes |

`Sink` returns its input, so a pipeline can continue past it. An operator that
really ends a pipeline (writes and returns nothing) sets `_terminal = True`.

## Control flow

Decisions and failures as steps inside the pipeline, rather than `if` and
`try` around it.

| Operator | Purpose |
|---|---|
| `Branch(predicate, if_true, if_false=Identity())` | a binary choice |
| `Switch(key, cases, default=Identity())` | dispatch on `key(carrier)` |
| `Try(primary, fallback, on)` | run `fallback` when `primary` raises one of `on` |
| `Coalesce(sources, is_ok, on=())` | the first source whose output passes `is_ok` |
| `Retry(op, attempts, base_delay, on)` | retry on `on`; attempt *i* waits `base_delay · 2^(i−1)` |

```python
from pipekit import Coalesce, Const, Lambda, Retry, Try


def down(_):
    raise IOError("down")


Try(Lambda(down), Const(0.0), on=(IOError,))(1.0)    # → 0.0
Coalesce(
    [Lambda(down), Const(-1), Const(5)],
    is_ok=lambda r: r > 0,                            # -1 fails the check
    on=(IOError,),                                    # down() falls through
)(0)                                                  # → 5
Retry(flaky, attempts=5, base_delay=0.0, on=(ConnectionError,))(42)
# → 42, where flaky is an operator that raises ConnectionError twice, then succeeds
```

`Try` and `Retry` require their exception list, so neither is ever a silent
catch-all; when every source fails, `Coalesce` raises `RuntimeError`. The
[ETL tutorial](tutorials/etl-regions.md) nests all three around a reader.

## Observers

The carrier flows through unchanged; something useful happens on the side.

| Name | Purpose |
|---|---|
| `Tap(fn, name=…)` | call `fn` on the carrier; return the carrier |
| `Snapshot()` + `.at(key)` | capture intermediates by name in `snapshot.captures` |
| `ShapeTrace(printer=print, mode="every" \| "diff_only", label=None)` | log `shape` and `dtype` at each step |
| `Profile()` + `.wrap(op)` | time each call; `profile.report()` gives calls, total and mean per operator |
| `Histogram(bins=10, to_array=list)` + `.at(key)` | capture distributions; `to_array` turns the carrier into numbers |

`Snapshot`, `Profile` and `Histogram` are *controllers*, not operators: the
operators you place in the pipeline come from `.at()` and `.wrap()`, and the
data they collect lives on the controller.

```python
from pipekit import Histogram, Lambda, Profile, Snapshot

snap, prof, hist = Snapshot(), Profile(), Histogram(bins=2)
pipe = Lambda(lambda v: [v, 2 * v], name="pair") | snap.at("mid") | hist.at("out")
pipe(1.0)                   # → [1.0, 2.0]
snap.captures               # → {'mid': [1.0, 2.0]}
prof.wrap(Scale(3.0))(2.0)  # → 6.0;  prof.report() → {'Scale': {'calls': 1.0, ...}}
```

## Combination

`Fanout` runs several operators on the same input and returns a dict of
their outputs:

```python
from pipekit import Fanout, Lambda

Fanout({"n": Lambda(len), "u": Lambda(str.upper)})("ab")   # → {'n': 2, 'u': 'AB'}
```

## Caching

`Cache(inner)` (alias `Memoize`) memoises an operator in memory, keyed by a
SHA-256 of the input and the inner operator's configuration. Arrays are keyed
by their bytes, dtype and shape, so two large arrays never collide, and the
bookkeeping is thread-safe, so one `Cache` can be shared by `ThreadMap`
workers.

```python
from pipekit import Cache, Memoize

c = Cache(Scale(2.0))
c(1.0); c(1.0); c(2.0)
c.hits, c.misses    # → (1, 2)
Memoize is Cache    # → True
```

## Quality control

Assertions as pass-through operators: a research-time "this won't break"
becomes a permanent guard.

| Operator | Purpose | Holds a closure |
|---|---|---|
| `Quarantine(check, sentinel, on_quarantine=None)` | on failure call `on_quarantine` and return `sentinel`; never raises | yes |
| `AssertShape(expected_shape)` | raise `QCError` on a shape mismatch; `None` matches any size | no |
| `AssertDType(expected_dtype)` | raise `QCError` on a dtype mismatch | no |
| `AssertHasAttribute(name, value=…)` | raise `QCError` if an attribute is missing or wrong | no |
| `AssertCallable(predicate, message=…)` | raise `QCError` if a predicate fails | yes |

```python
AssertShape((2, 4))(x)   # x.shape == (2, 3)
# QCError: AssertShape: axis 1 expected 4, got 3 (expected=(2, 4), actual=(2, 3)).
```

The checks read `shape`, `dtype` and attributes and never look inside the
data, so they work on any carrier. `QCError` subclasses `AssertionError`.
`Quarantine` is the batch-friendly one: the
[inference](tutorials/inference.md) and [ETL](tutorials/etl-regions.md)
tutorials use it to set bad records aside and keep going.

## Shape inference

`Signature` is an immutable record of named dimensions and a dtype; a size of
`None` marks a dimension dynamic. An operator declares how it changes a
signature by overriding `compute_output_signature`; returning `None` means
"this operator does not track shape", and `Sequential.summary` shows `?` for
it.

```python
from pipekit import Signature

sig = Signature(dims=(("time", 12), ("y", 256), ("x", 256)), dtype="float32")
sig.format()                  # → '(time=12, y=256, x=256) float32'
sig.drop_dims(("time",))      # → Signature((y=256, x=256) float32)
sig.replace_dims({"x": 128})  # → Signature((time=12, y=256, x=128) float32)
```

In the core only `Sequential` propagates signatures; the payoff grows as the
operators you write (and domain libraries) declare theirs.

## Parallelism

Four shapes, one wrapper each. Every wrapper takes an iterable and returns a
list in input order.

| Shape | When | Constraint | Operator |
|---|---|---|---|
| Threads | I/O-bound: object-store reads, downloads | the I/O must release the GIL | `ThreadMap(op, n_workers=8)` |
| Processes | CPU-bound work | every step must pickle | `ProcessMap(op, n_workers=8, on_error="raise" \| "log_and_continue")` |
| Async | I/O on one thread: web services | `async def _apply`, or a sync operator run in a thread | `AsyncMap(op, semaphore=8)` |
| Batches | GPU inference, vectorised code | the inner operator takes a *list* of items | `BatchedMap(op, batch_size=8, flatten=True)` |

```python
from pipekit import AsyncMap, BatchedMap, Lambda, ThreadMap

ThreadMap(Scale(2.0), n_workers=2)([1, 2, 3])             # → [2.0, 4.0, 6.0]
add_up = Lambda(lambda b: [sum(b)], name="sum")
BatchedMap(add_up, batch_size=2)([1, 2, 3, 4, 5])         # → [3, 7, 5]
AsyncMap(Scale(2.0), semaphore=2)([1, 2])                 # → [2.0, 4.0]
```

**Pickling.** Process pools need every operator to pickle, and closures
(`Lambda`, `Tap`, `Sink`, a `Branch` predicate), open handles and captured
random generators do not. Operators that hold a closure carry
`forbid_in_yaml = True`, and `check_pickleable(op)` walks a whole pipeline,
however deeply nested, and returns the flagged operators before you fan out:

```python
from pipekit import Branch, Cache, Identity, Tap, ThreadMap, check_pickleable

pipe = ThreadMap(Cache(Scale(2.0) | Tap(print) | Branch(bool, Identity())))
[type(op).__name__ for op in check_pickleable(pipe)]   # → ['Tap', 'Branch']
```

**Async.** `AsyncMap(...)(items)` drives its own event loop, so it works in
scripts but not inside a running loop (a FastAPI or LitServe handler). There,
await its `run` method:

```python
async def handler(items):
    return await AsyncMap(Scale(2.0)).run(items)   # → [2.0, 4.0] for [1, 2]
```

**Deliberately left out.** No automatic parallelism (`Sequential` is
sequential; `Graph` evaluates in one process), no backend abstraction (the
wrappers are `concurrent.futures` and `asyncio`), no fault tolerance beyond
`on_error="log_and_continue"`, and no distributed execution: Dask, Ray, Beam
and orchestrators schedule pipekit operators; pipekit keeps them pickleable so
they can.

## Serialisation

| Name | What it does |
|---|---|
| `dumps(op)` | JSON-encode `op.state`, keys sorted, nested operators included |
| `loads(s)` | rebuild the whole tree through `Operator.from_state`; the defining modules must be imported |
| `register(cls)` | add a class to the sandbox allowlist (also a class decorator) |
| `unregister(cls)`, `allowed()` | remove from, and list, the allowlist |
| `loads_sandboxed(s)` | `loads`, but `PermissionError` for any class in the tree that is not registered |
| `from_state_sandboxed(state)` | the same check on a record already in memory |

The sandboxed loader is for pipelines you did not write (uploaded by users,
read from a queue): only classes you registered can be built, at any depth,
and every class is checked before anything is constructed. Nothing is
allowed by default, pipekit's own containers included.

```python
import pipekit as pk

pk.dumps(pk.Const(3))
# → '{"class": "Const", "config": {"value": 3}, "module": "pipekit.blocks"}'
pk.loads(pk.dumps(Scale(2.0)))             # → Scale(factor=2.0)
pk.loads_sandboxed(pk.dumps(Scale(2.0)))   # PermissionError: not registered
pk.register(Scale)
pk.loads_sandboxed(pk.dumps(Scale(2.0)))   # → Scale(factor=2.0)
pk.loads(pk.dumps(pk.Lambda(abs)))         # RuntimeError: marked forbid_in_yaml
```

Nested pipelines round-trip too. A child operator appears in its parent's
config as its own `state` record, a `Graph` records every node and its
wiring, and an exception class (`Try.on`, `Retry.on`) is written as
`{"exception": "builtins.KeyError"}`. `pipeline_hash` hashes that whole
record, so two pipelines share a hash exactly when they are configured the
same:

```python
from pipekit import Cache, Retry, Sequential, ThreadMap, pipeline_hash

pipe = ThreadMap(Cache(Retry(Scale(2.0), attempts=3, base_delay=0.5, on=(OSError,))))
back = pk.loads(pk.dumps(pipe))
back.state == pipe.state                      # → True
pipeline_hash(back) == pipeline_hash(pipe)    # → True

pk.register(Sequential)
pk.loads_sandboxed(pk.dumps(Sequential([Scale(2.0), pk.Identity()])))
# PermissionError: '__main__.Scale' is not in the allowlist (checked first)
```

`pipekit-experiment`'s `HydraConfigLoader` does the same with `_target_`
YAML: `to_yaml(op)` writes any pipeline, `from_yaml` loads it through
Hydra, and `from_yaml_sandboxed` loads untrusted YAML through the allowlist
without ever calling Hydra's `instantiate`.

## State

Most operators are stateless: `_apply(carrier) → carrier`. Some work carries
state from call to call: a data-assimilation cycle (forecast, analysis,
forecast…), a training loop (optimiser state, step, metrics), a streaming
statistic, a recurrent model. A `StatefulOperator` implements
`_apply(carrier, state) → (carrier, state)`, and a `CarryState` is the base
for that state, with `to_dict` / `from_dict` for checkpointing. A
`Sequential` that contains a stateful step threads the state through and
returns `(carrier, state)`; stateless steps pass it along untouched.

A streaming mean: after values x₁, …, xₙ the state is the running sum
Sₙ = ∑ᵢ xᵢ = Sₙ₋₁ + xₙ and the count n, and the step emits x̄ₙ = Sₙ / n.

```python
from pipekit import CarryState, StatefulOperator


class MeanState(CarryState):
    def __init__(self, total: float, n: int) -> None:
        self.total = total  # Sₙ = ∑ᵢ xᵢ
        self.n = n          # n


class StreamMean(StatefulOperator):
    def _apply(self, x, state):
        st = MeanState(state.total + x, state.n + 1)  # Sₙ = Sₙ₋₁ + xₙ;  n ← n + 1
        return st.total / st.n, st                    # x̄ₙ = Sₙ / n


pipe = Scale(1.0) | StreamMean()                    # a stateless step, then a stateful one
x_bar, st = pipe(4.0, state=MeanState(0.0, 0))      # x̄₁ = 4.0
x_bar, st = pipe(8.0, state=st)                     # x̄₂ = 6.0
st                                                  # → MeanState(total=12.0, n=2)
MeanState.from_dict(st.to_dict()) == st             # → True
pipe(4.0)   # TypeError: ... called without a `state` argument
```

A `Graph` threads state the same way. Call it with `state=` and it returns
`(outputs, state)`; the state passes through its stateful nodes in
topological order (the order `describe()` lists):

```python
from pipekit import Graph, Input

x = Input("x")
g = Graph(inputs={"x": x}, outputs={"mean": StreamMean()(Scale(1.0)(x))})
out, st = g(x=4.0, state=MeanState(0.0, 0))   # out == {"mean": 4.0}
out, st = g(x=8.0, state=st)                  # out == {"mean": 6.0}
```

`pipekit-cycle`'s `Cycle`, `DACycle` and `EnsembleDACycle` and
`pipekit-train`'s `TrainingLoop` are built on these two classes; see the
[data-assimilation](tutorials/ensemble-da.md) and
[training](tutorials/train-register-forecast.md) tutorials.

## Protocols

Model wrappers in the sister libraries rely on two duck-typed seams, written
down once as `runtime_checkable` Protocols: `Predictor` (anything with
`predict(x)`) and `FittableTransformer` (anything with `fit(x)` and
`transform(x)`). `isinstance(model, Predictor)` checks that the methods exist,
not their signatures, so a scikit-learn estimator, a JAX model wrapper or your
own class all qualify without importing pipekit. The cycle and experiment
packages define more such seams (`ForwardModel`, `AnalysisStep`,
`ModelRegistry`, …); see [Protocols](protocols.md).

## Limits today

What the core does not do yet, so you can plan around it:

- **`CarryState` hashes its fields.** A state holding a list or an array is
  unhashable; don't put states in sets or use them as dict keys.
- **Shape inference stops at `Sequential`.** `Graph`, `Fanout` and the
  parallel maps report their input signature as their output; `Graph` has
  `describe()` but no `summary()`.
- **`ProcessMap(on_error="log_and_continue")` prints to `stderr`.** It leaves
  `None` in the failed item's slot; it does not use `logging`.
- **`ShapeTrace` logs `shape` and `dtype` only.** A carrier with extra
  metadata worth printing (a coordinate system, units) needs a subclass that
  overrides `_apply`.

The design questions behind these, and the plan for them, are tracked in the
[GeoStack design reports](https://github.com/jejjohnson/geostack/tree/main/docs/reports)
and pipekit's issues.
