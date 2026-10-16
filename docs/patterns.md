# Patterns

*The shapes real pipelines take once they leave the notebook, which pipekit
pieces each one uses, and what is still missing.*

??? note "About this page"
    | | |
    |---|---|
    | **Covers** | Fourteen deployment patterns and four habits that cut across them |
    | **Read first** | [A tour of the core](core-tour.md) |
    | **Worked examples** | The [tutorials](tutorials/anomaly-graph.md), linked from each pattern |
    | **History** | Distilled from Report 9 of the [GeoStack design reports](https://github.com/jejjohnson/geostack/blob/main/docs/reports/09_use_cases.md), which walks twenty cases through one domain (a methane plume from a satellite scene) and now points here |

A pipeline is written once and then used in many places: a notebook, a batch
job, a web service, a scheduled family of runs, an audit. pipekit's bet is
that these are the *same object* in different harnesses, so the patterns
below are mostly about which harness to put around an `Operator`, and which
of its properties (a config, a hash, pickleability) the harness relies on.

Each pattern names its pipekit pieces, says what works today and what does
not, and links the tutorial that runs it end to end. The status column uses
three marks: **today** (ships and is tested), **partly** (works with a
workaround), **missing** (needs code that does not exist yet).

| # | Pattern | Pieces | Status | Tutorial |
|---|---|---|---|---|
| 1 | [Explore, then batch](#1-explore-then-batch) | `Tap`, `ShapeTrace`, `Profile`, `Snapshot` | today | [bigger graph](tutorials/anomaly-graph.md) |
| 2 | [ETL over many inputs](#2-etl-over-many-inputs) | `ProcessMap`, `Quarantine`, `Sink` | today | [ETL over many regions](tutorials/etl-regions.md) |
| 3 | [Validation as operators](#3-validation-as-operators) | `Assert*`, `Quarantine`, `QCError` | today | [inference](tutorials/inference.md) |
| 4 | [Fusing several inputs](#4-fusing-several-inputs) | `Graph`, `Input` | today | [bigger graph](tutorials/anomaly-graph.md) |
| 5 | [One preprocessing object for training and serving](#5-one-preprocessing-object-for-training-and-serving) | `Operator` config, registry | today | [train, register, forecast](tutorials/train-register-forecast.md) |
| 6 | [Cycles: time stepping and assimilation](#6-cycles-time-stepping-and-assimilation) | `StatefulOperator`, `pipekit-cycle` | today | [DA cycle](tutorials/da-cycle.md), [ensemble DA](tutorials/ensemble-da.md) |
| 7 | [An inference service](#7-an-inference-service) | `AsyncMap.run`, `BatchedMap`, the config hash | today | [inference](tutorials/inference.md) |
| 8 | [Promotion by tag](#8-promotion-by-tag) | `LocalModelRegistry.tag` | today | — |
| 9 | [Pipelines from users](#9-pipelines-from-users) | `register`, `loads_sandboxed`, `from_yaml_sandboxed` | today | — |
| 10 | [A pinned artifact](#10-a-pinned-artifact) | `InferenceArtifact`, `TrainingArtifact`, `to_yaml` | today | — |
| 11 | [Orchestrator tasks with deterministic keys](#11-orchestrator-tasks-with-deterministic-keys) | `pipeline_hash` | today | [ETL over many regions](tutorials/etl-regions.md) |
| 12 | [A/B comparison](#12-ab-comparison) | `Fanout` | partly | — |
| 13 | [Scheduled families, backfills and what-if reruns](#13-scheduled-families-backfills-and-what-if-reruns) | the config hash | today | — |
| 14 | [The feedback loop](#14-the-feedback-loop) | `pipekit-train`, registry | today | [train, register, forecast](tutorials/train-register-forecast.md) |

The snippets share one small operator, a detector with a threshold:

```python
from pipekit import Operator


class Threshold(Operator):
    """x ↦ 1[x > k]."""

    def __init__(self, *, k: float) -> None:
        self.k = k

    def _apply(self, x):
        return x > self.k
```

## Exploration and batch

### 1. Explore, then batch

**The problem.** A pipeline grows in a notebook, one step at a time, with
prints and plots between the steps. Then it has to run on a thousand inputs,
and the prints and plots have to go.

**The pattern.** Observers pass their input through unchanged, so you can
leave them in while you explore and drop them, or point them at a logger,
when you batch. `ShapeTrace` logs each step's shape and dtype, `Profile`
times a step, `Snapshot` keeps a copy of an intermediate, and `Tap` calls any
function on it. The object you explored with is the object you batch.

```python
from pipekit import Profile, ShapeTrace, Snapshot, Tap

snap, prof = Snapshot(), Profile()
explore = (
    clean | ShapeTrace() | snap.at("clean")   # log shape and dtype, keep a copy
    | features | Tap(plot)                    # look at the features
    | prof.wrap(model)                        # time the model
)
batch = clean | features | model             # the same steps, no observers
snap.captures["clean"], prof.report()        # what the exploration saw
```

**Today.** All of it. `ShapeTrace` logs shape and dtype only; a carrier with
more to say (units, a coordinate system) needs a subclass.

### 2. ETL over many inputs

**The problem.** One pipeline, many inputs (regions, files, stations, days),
some of them broken in ways you only find at scale.

**The pattern.** Extract with a list of keys, transform with
`ProcessMap(pipeline, on_error="log_and_continue")` so one bad input leaves a
`None` instead of killing the run, set malformed records aside with
`Quarantine`, and load with a `Sink` that writes one keyed output per input.
Run `check_pickleable(pipeline)` before fanning out: closures do not cross
process boundaries.

**Today.** All of it. `log_and_continue` prints to `stderr` rather than to
`logging`, so a run that must report its failures collects them itself (the
tutorial keeps a reject list per region).

### 3. Validation as operators

**The problem.** Checks scattered through glue code are skipped, duplicated,
or run in the wrong place.

**The pattern.** A check is a step. `AssertShape`, `AssertDType`,
`AssertHasAttribute` and `AssertCallable` raise `QCError` when an input is
wrong; `Quarantine(check, sentinel, on_quarantine)` replaces a bad input with a
sentinel and hands it to a callback, for streams where one bad record should
not stop the rest. Put the hard checks at the boundaries (after loading,
before the model, before writing) and the soft ones on the records.

**Today.** All of it. Array-aware checks (finite values, value ranges, across
array libraries) are planned for `pipekit-array`; until then
`AssertCallable` takes any predicate.

### 4. Fusing several inputs

**The problem.** A product needs two or more sources at once: two sensors, a
measurement and a reference, a feed and its timestamp.

**The pattern.** A `Graph`. Each source is an `Input`, each step is called on
the nodes it needs, and the outputs are named. Steps that take two carriers
(`_apply(self, a, b)`) are ordinary operators.

```python
from pipekit import Graph, Input

a, b = Input("a"), Input("b")
fused = Align()(a, b)
g = Graph(inputs={"a": a, "b": b}, outputs={"product": Model()(fused)})
g(a=scene_a, b=scene_b)["product"]
```

**Today.** All of it, but a `Graph` does not thread state, so a stateful step
belongs in a `Sequential` inside it, and shape inference stops at the graph's
edge.

## Machine learning

### 5. One preprocessing object for training and serving

**The problem.** Training normalises with one piece of code and serving with
another, and they drift: the classic train/serve skew.

**The pattern.** The fitted preprocessing is an operator whose fitted values
(a mean, a standard deviation, a vocabulary) are its *configuration*. Store
it in the registry next to the model, with its hash in the model's tags, and
load both by name at serving time. There is then exactly one
`Standardize(mean=…, std=…)` in the world, and the hash proves which one ran.

```python
scaler_hash = registry.store(scaler, name="scaler")
registry.store(model, name="model", weights=blob, tags={"scaler": scaler_hash})

serve = registry.load("scaler") | model_template.with_weights(
    registry.load_weights("model")
)
```

**Today.** All of it (`LocalModelRegistry`, `S3ModelRegistry`,
`JaxModelOp.with_weights`). The tutorial fits the scaler on the training split
only, so validation never sees its statistics.

### 6. Cycles: time stepping and assimilation

**The problem.** Forecast, observe, correct, repeat: the carrier is a state,
and every step needs the previous one.

**The pattern.** `pipekit-cycle`. A `Cycle` steps a forward model; a `DACycle`
or `EnsembleDACycle` alternates forecast and analysis, with observation
operators and noise models as Protocols, so the filter can come from any
library. Pre-processing (quality control, thinning, unit conversion of the
observations) and post-processing (verification, spread–skill) are ordinary
operators around the cycle.

**Today.** All of it.

## Serving

### 7. An inference service

**The problem.** A web handler receives a request, runs a pipeline and must
say exactly what produced its answer.

**The pattern.** Build the pipeline once at start-up. Inside the handler,
which already runs in an event loop, use `await AsyncMap(op).run(items)` (not
`AsyncMap(op)(items)`, which starts its own loop). Batch for a GPU with
`BatchedMap`, or let the model batch. Return the pipeline's content hash with
every answer, so a response can be traced to a configuration.

**Today.** All of it. pipekit provides no server: LitServe, FastAPI or
anything else hosts the operator.

### 8. Promotion by tag

**The problem.** Several candidate versions exist; one of them is "the" model
that serving uses, and promoting a new one (or rolling back) must not need a
redeploy.

**The pattern.** Store every candidate by content hash; point a human-readable
tag at the one in service. Serving loads by tag; promotion and rollback move
the tag. The same configuration always gets the same hash, so storing it
twice is harmless.

```python
import tempfile
from pipekit_experiment import LocalModelRegistry

registry = LocalModelRegistry(tempfile.mkdtemp())
v1 = registry.store(Threshold(k=3.0), name="detector-candidate")
v2 = registry.store(Threshold(k=2.5), name="detector-candidate")

registry.tag(v1, "detector", force=True)   # v1 in service
registry.tag(v2, "detector", force=True)   # promote v2
registry.load("detector")                  # → Threshold(k=2.5)
registry.tag(v1, "detector", force=True)   # roll back
registry.store(Threshold(k=3.0)) == v1     # → True: same config, same hash
```

**Today.** All of it.

### 9. Pipelines from users

**The problem.** Users (or another team, or a queue) send pipelines as data.
Building arbitrary classes from a string is code execution.

**The pattern.** Register the classes you allow, containers included, and
build untrusted input with `loads_sandboxed` (JSON) or
`HydraConfigLoader.from_yaml_sandboxed` (`_target_` YAML). Every class in the
tree is checked before anything is built, and anything unregistered raises
`PermissionError`. Operators holding closures are `forbid_in_yaml` and never
deserialise, registered or not.

```python
import pipekit as pk
from pipekit_experiment.adapters.hydra import HydraConfigLoader

pk.register(pk.Sequential)
pk.register(Threshold)

text = """
_target_: pipekit.Sequential
operators:
  - _target_: __main__.Threshold
    k: 2.0
"""
# a Sequential holding Threshold(k=2.0)
detector = HydraConfigLoader.from_yaml_sandboxed(text)
```

The YAML loader never calls Hydra's `instantiate`, which would call any
dotted path (`_target_: os.system`): it translates the YAML into a state
record and builds that through the allowlist. Interpolations (`${...}`) are
not resolved, so a sandboxed config cannot read the environment.

**Today.** All of it.

## Reproducibility and operations

### 10. A pinned artifact

**The problem.** A result has to be reproducible months later, for a review,
an audit or a paper: the exact pipeline, the exact weights, the exact
dependencies.

**The pattern.** `pipekit-experiment`'s `InferenceArtifact(pipeline_yaml,
pinned_model_hashes, backend_info, deps_lock, metadata)` records what ran;
`TrainingArtifact` records how the model was made (the training pipeline, the
dataset's hash, the model's hash, the tracker run). Keep the artifact with
the output.

For the `pipeline_yaml` field, `HydraConfigLoader.to_yaml(pipeline)` writes
the pipeline that actually ran, however deeply nested, and `from_yaml` loads
it back with the same `state`.

**Today.** All of it. `TrainingLoop.run()` returns a `TrainingArtifact` when
`pipekit-experiment` is installed.

### 11. Orchestrator tasks with deterministic keys

**The problem.** An orchestrator (Prefect, Dagster, Airflow, a cron job)
retries, resumes and backfills. Every task must know whether it has already
run.

**The pattern.** One task is one operator call on one input. Its key is a
hash of the operator's configuration and the input's identity, so a rerun
with the same pipeline and input finds its output and skips; a change to
either gives a new key.

```python
from pipekit import pipeline_hash


def task_key(op: Operator, input_id: str) -> str:
    """One key per (pipeline configuration, input)."""
    return f"{pipeline_hash(op)[:16]}-{input_id}"


# the same pipeline on the same day: the same key, so the rerun skips
task_key(Threshold(k=3.0), "2026-10-16") == task_key(Threshold(k=3.0), "2026-10-16")
# one argument changed: a new key
task_key(Threshold(k=3.0), "2026-10-16") != task_key(Threshold(k=2.5), "2026-10-16")
```

**Today.** All of it. `pipeline_hash` hashes the whole configuration: every
nested operator and, in a `Graph`, every node and its wiring. It is the hash
the registry stores under, so a task key and a registry entry agree.

### 12. A/B comparison

**The problem.** A new pipeline version must be compared with the current one
on the same inputs before it replaces it.

**The pattern.** `Fanout({"a": current, "b": candidate})` runs both on each
input and returns both outputs; a post-processing step compares them. Compare
the *distribution* of the differences, not only their mean: two versions can
agree on average and disagree badly on the inputs that matter.

```python
from pipekit import Fanout

ab = Fanout({"a": Threshold(k=3.0), "b": Threshold(k=2.5)})
ab(2.8)   # → {'a': False, 'b': True}
```

**Partly.** The side-by-side run ships; a diff of two pipelines'
configurations, and a report of where their outputs disagree, do not.
`pipekit-evaluate`'s scorers and `EvaluationReport` are the natural home.

### 13. Scheduled families, backfills and what-if reruns

**The problem.** The same pipeline runs every day (monitoring), over a past
period (backfill), and again with one setting changed (what-if).

**The pattern.** Key every output by the pipeline's hash and the input's date
or identity (pattern 11). A daily run and a backfill are the same task over
different dates. A what-if is a new operator with one constructor argument
changed, so it has a new hash and writes beside the original, never over it.

**Today.** All of it; the schedule belongs to the orchestrator.

### 14. The feedback loop

**The problem.** Reviewers accept or reject the pipeline's outputs, and those
verdicts should improve the next model.

**The pattern.** Verdicts become labels; a frozen label set gets a content
hash; `pipekit-train`'s `TrainingLoop` trains on it and returns a
`TrainingArtifact`; the new model is stored, compared with the current one
(pattern 12) and promoted by tag (pattern 8). Every model can be traced back
to the labels it saw.

**Today.** All of it, given a label store that hashes its frozen sets.

## Habits that cut across the patterns

### Composition in the core, infrastructure next door

The core composes and nothing else; deployment infrastructure lives in the
sister packages, each in one place.

| Concern | Lives in |
|---|---|
| Operators, composition, parallel maps, nested JSON, the sandbox allowlist | `pipekit` |
| `pipeline_hash` and the hashing primitives | `pipekit.hashing` |
| Content-addressed storage, tags, artifacts, YAML loading | `pipekit-experiment` |
| Training loops and backends | `pipekit-train` |
| Time stepping and assimilation | `pipekit-cycle` |
| Benchmarks, scorers and reports | `pipekit-evaluate` |
| JAX models with weight round-trips | `pipekit-jax` |
| Domain operators, readers and writers | your library, subclassing `Operator` |

### Parallelism is chosen per deployment

Nothing is parallel by default; each deployment picks its primitive.

| Deployment | Primitive |
|---|---|
| CPU-bound batch over many inputs | `ProcessMap` |
| I/O-bound reads | `ThreadMap` |
| I/O inside a running event loop (a server) | `await AsyncMap(op).run(items)` |
| GPU inference | `BatchedMap`, or the model batches |
| Across machines | the orchestrator, calling pickleable operators |

### YAML is the shared language, both ways

Every operator, in any library, subclasses `Operator`, so a Hydra YAML with
`_target_` paths can name steps from several libraries side by side.
`HydraConfigLoader.from_yaml` builds the nested pipeline, `to_yaml` writes one
built in Python, and `from_yaml_sandboxed` loads YAML you did not write. A
pipeline can start in Python or in YAML; either way the other form is one
call away.

### JAX speeds the model, not the pipeline

`pipekit-jax` wraps an Equinox model as an operator (`JaxModelOp`), with
byte-identical weight round-trips through the registry, and adds diffrax
forward models for `pipekit-cycle`. Inside the model, `jit` and `vmap` do
their work; the pipeline around it stays eager Python. That is a deliberate
split: pre- and post-processing are I/O and bookkeeping, where tracing buys
little and costs flexibility.

## One domain, all the patterns

The GeoStack design reports walk every pattern above through one example,
a methane plume from a satellite scene: read the scene, retrieve the
enhancement, detect the plume, estimate its emission rate, review it, and
retrain. [Report 9](https://github.com/jejjohnson/geostack/blob/main/docs/reports/09_use_cases.md)
has the twenty cases, the libraries each one needs, and the proposals that
close the gaps marked **partly** and **missing** here.
