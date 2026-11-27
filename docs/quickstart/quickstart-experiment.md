# pipekit-experiment quickstart

The [README](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-experiment) stores a plume detector's pre-processing, promotes it by tag, pins it in an inference record and watches the registry refuse it once its weights were edited. This page goes through the rest of the package in the same notation: the address and the layout on disk, the calling convention, tags, operators that do not round-trip, the registry on an object store, trackers and run records, and the three bridges. Every block on it runs as written with the base install, except the two marked as needing the `hydra` or `metaflow` extra.

(qs-exp-address)=
## The address of a model

A stored model is the state of an operator and, optionally, its weight bytes, and its address is

$$
h = \mathrm{sha256}\left(\mathrm{json}(s) \Vert \mathtt{0x00} \Vert w\right)
$$

where $s$ is the operator's state record (the nested module, class and config of `Operator.state`), $\mathrm{json}$ writes it with sorted keys, $w \in \{0, \dots, 255\}^{N_w}$ are the $N_w$ weight bytes ($N_w = 0$ when there are none), $\Vert$ joins byte strings, $\mathtt{0x00}$ is one zero byte that separates the two parts, and $h$ is the 64-character hex digest. It is `pipekit_ops.pipeline_hash(op, weights)`, the hash pipekit-ops uses wherever something must recognise the same pipeline again, so `store` returns it and you can compute it before storing anything.

`store(op, weights=...)` writes three files under `<root>/<h>/`: `operator.json` (the state, as the stable JSON above), `weights.bin` (the bytes, if there are any) and `metadata.json` (the tags). Storing the same content twice writes the same files again and merges the new tags into the old ones, so a later `store` that omits a tag never wipes it. `b""` and `None` hash alike, and `store` treats `b""` as no weights, so the layout never depends on which one you passed. Tag names live in `<root>/_tags/<name>`, one file per tag holding one hash.

```python
import tempfile

import pipekit_experiment as pe
from pipekit_ops import Pick, pipeline_hash


# Shapes: h is a 64-character hex digest; N_w weight bytes

# today (pipekit-ops)
prep = Pick(["xch4", "u10", "v10"])
# today (pipekit-experiment)
registry = pe.LocalModelRegistry(tempfile.mkdtemp())

# today (pipekit-experiment)
# h = sha256(json(s) ‖ 0x00 ‖ w), with N_w = 0
model_hash = registry.store(prep, tags={"sensor": "emit"})
# today (pipekit-experiment)
# b"" and None are both N_w = 0, so they give the same h; the tags merge
again = registry.store(prep, tags={"campaign": "permian"}, weights=b"")

# today (pipekit-ops)
# the registry's address is pipekit-ops's pipeline hash
matches = model_hash == again == pipeline_hash(prep)
# today (pipekit-experiment)
by_tags = registry.list(tags={"sensor": "emit", "campaign": "permian"})
print(matches, by_tags == [model_hash], registry.load_weights(model_hash))
```

This prints `True True None`: both calls landed on the address `pipeline_hash` gives, the entry now carries both tags, and there are no weights to read back.

`LocalModelRegistry` writes every file through a temp file and `os.replace`, so a crash mid-write never leaves a truncated `operator.json` or tag file behind. `S3ModelRegistry` is only as atomic as the store under it: object stores usually are, and a plain `file://` path through fsspec is not.

(qs-exp-convention)=
## The calling convention

Both registries share one implementation of the public methods; only the storage calls under them differ.

| Method | Takes | Returns | Raises |
|---|---|---|---|
| `store(op, *, name, tags, weights)` | an `Operator`, an optional tag name, tag values and bytes | $h$ | `TypeError` for anything that is not an `Operator` |
| `load(ref)` | a hash or a tag | the rebuilt operator | `KeyError`, `HashMismatchError` |
| `load_weights(ref)` | a hash or a tag | the bytes, or `None` | `KeyError`, `HashMismatchError` |
| `list(*, tags)` | tag values to match | the matching hashes, sorted | none of its own |
| `tag(hash, name, *, force)` | a stored hash and a name | `None` | `KeyError`, `FileExistsError` |
| `resolve_tag(name)` | a tag | $h$ | `KeyError` |

**References.** A `ref` is read as a hash when `<root>/<ref>/operator.json` exists and as a tag otherwise. Hashes, tags and references must be single path segments (no `/`, no `\`, not `.` or `..`, and not the reserved `_tags`), and anything else raises `ValueError`, so a caller-supplied string never escapes the registry root.

**Tags.** `store(op, name=...)` binds the name with `force=True`, so it moves an existing tag; `tag` refuses to move one unless you pass `force=True`. The `ModelRegistry` protocol has `store`, `load`, `load_weights`, `list` and `tag`; `resolve_tag` is a method of the two registries, not of the protocol.

(qs-exp-verify)=
## Loading verifies

`load(ref)` resolves the reference, re-hashes the stored text and weights and raises `HashMismatchError` (a `ValueError`) when they no longer give $h$. We write $s'$ and $w'$ for the state and the weights as they are read back, so the first check is

$$
\mathrm{sha256}\left(\mathrm{json}(s') \Vert \mathtt{0x00} \Vert w'\right) = h
$$

which the README's quickstart shows failing after an edit to `weights.bin`. `load` then rebuilds the operator with `Operator.from_state` and checks that the rebuilt operator hashes back to $h$, which catches a state that does not round-trip. `load_weights` makes the first check only, because it never rebuilds the operator.

:::{note}
An entry stored before pipekit 0.2.0, when the core was imported as `pipekit`, names the core's modules `pipekit.*`. It still loads: the stored text is checked against $h$ as usual, and the rebuilt operator is accepted when its state matches the stored one once those module names are updated with `pipekit_ops.upgrade_state`. Its operator then hashes to a new value, because the hash covers each operator's module.
:::

(qs-exp-tags)=
## Promotion and rollback by tag

A tag is the only mutable thing in the registry. Promoting a model, rolling it back and asking what is in service are all moves of one tag file, and the models under it never change ([pattern 8](../patterns.md#8-promotion-by-tag)).

```python
import tempfile

import pipekit_experiment as pe
from pipekit_ops import Const


# Shapes: h is a 64-character hex digest

# today (pipekit-experiment)
registry = pe.LocalModelRegistry(tempfile.mkdtemp())
# today (pipekit-ops)
# a detection threshold k = 3.0, stored and tagged in one call
v1 = registry.store(Const(3.0), name="detector")
# today (pipekit-ops)
# a candidate with k = 2.5
v2 = registry.store(Const(2.5))

try:
    # today (pipekit-experiment)
    # tag refuses to move a tag that already names a hash
    registry.tag(v2, "detector")
except FileExistsError:
    refused = True
# today (pipekit-experiment)
# promote v2 and read the threshold in service
registry.tag(v2, "detector", force=True)
promoted = registry.load("detector")(None)
# today (pipekit-experiment)
# roll back to v1
registry.tag(v1, "detector", force=True)
print(refused, promoted, registry.resolve_tag("detector") == v1)
```

This prints `True 2.5 True`: the plain `tag` call was refused, the promoted threshold was 2.5, and after the rollback the tag names $h$ of the first model again.

(qs-exp-forbid)=
## Operators that do not round-trip

An operator that holds a closure (a `Lambda`, a `CallableForward`, a JAX model wrapper) is marked `forbid_in_yaml`: its config names the callable but is a debug payload, not a recipe. `store` still writes it and its weights, but `load` cannot rebuild it, and `Operator.from_state` raises.

```python
import tempfile

import pipekit_experiment as pe
from pipekit_ops import Lambda


# Shapes: h is a 64-character hex digest; N_w = 4 weight bytes

# today (pipekit-experiment)
registry = pe.LocalModelRegistry(tempfile.mkdtemp())
# today (pipekit-ops)
# a closure: its config names the function but cannot rebuild it
model = Lambda(lambda scene: scene["xch4"] - 1900.0)
model_hash = registry.store(model, weights=b"\x00\x01\x02\x03")

try:
    # today (pipekit-experiment)
    registry.load(model_hash)
except RuntimeError as error:
    message = str(error).split(":")[0]
# today (pipekit-experiment)
# the weights are still checked against h and readable
print(message, registry.load_weights(model_hash))
```

This prints `from_state cannot reconstruct Lambda b'\x00\x01\x02\x03'`. Two lambdas share a hash, because the debug config names the callable (`<lambda>` for every lambda) and not its code. The pattern for such models is to rebuild a skeleton in code and read the verified weights into it, which is what `pipekit_jax.JaxModelOp.from_registry` does ([Getting started](../getting-started.md) has the round trip).

(qs-exp-s3)=
## The registry on an object store

`S3ModelRegistry` takes a root URI and the `storage_options` that fsspec passes to the filesystem (credentials, an endpoint), and runs the same store, load and tag code as `LocalModelRegistry`. It works with any fsspec scheme (`s3://`, `gs://`, `az://`, `memory://`, `file://`), with the scheme's own driver installed; the `s3` extra brings fsspec and s3fs, and constructing one without fsspec raises an `ImportError` that names the extra.

```python
import pipekit_experiment as pe
from pipekit_ops import Pick


# Shapes: h is a 64-character hex digest

# today (pipekit-experiment)
# any fsspec URI; memory:// needs only fsspec, s3:// also needs s3fs
registry = pe.S3ModelRegistry("memory://models")
# today (pipekit-ops)
prep = Pick(["xch4", "u10", "v10"])
# today (pipekit-experiment)
model_hash = registry.store(prep, name="production", weights=b"\x00\x01")
# today (pipekit-experiment)
served = registry.load("production")
print(isinstance(registry, pe.ModelRegistry), served.state == prep.state)
```

This prints `True True`: the fsspec registry satisfies the same protocol and serves the same operator by tag.

(qs-exp-trackers)=
## Trackers and run records

`ExperimentTracker` is four calls: `start_run(name, config)` returns a `Run`, `log_metrics(run, metrics, step)` records scalars at a step, `log_artifact(run, path, name)` attaches a file, and `end_run(run, status)` closes the run. The protocol is runtime-checkable and structural, so a tracker for MLflow, W&B or a CSV file satisfies it without inheriting from this package. No concrete tracker ships yet; the one below keeps everything in memory, with `RunMetrics` for the per-step series ($T$ logged steps) and `RunArtifacts` for the attached files.

```python
import uuid
from typing import Any

import pipekit_experiment as pe
from pipekit_ops import Pick, pipeline_hash


# Shapes: T = 3 logged steps; h is a 64-character hex digest


# user
# a tracker that keeps everything in memory, written to the protocol
class MemoryTracker:
    def __init__(self) -> None:
        # today (pipekit-experiment)
        self.history = pe.RunMetrics()
        # today (pipekit-experiment)
        self.files = pe.RunArtifacts()

    def start_run(self, name: str, config: dict[str, Any]) -> pe.Run:
        run_id = f"{name}-{uuid.uuid4().hex[:6]}"
        # today (pipekit-experiment)
        return pe.Run(run_id, config["pipeline_hash"], config=dict(config))

    def log_metrics(self, run: pe.Run, metrics: dict[str, float], step: int) -> None:
        for metric, value in metrics.items():
            self.history.log(metric, value, step)
        run.metrics = self.history.latest()

    def log_artifact(self, run: pe.Run, path: str, name: str) -> None:
        self.files.attach(name, path)
        run.artifacts = dict(self.files.items)

    def end_run(self, run: pe.Run, status: str) -> None:
        run.status = status


# today (pipekit-ops)
prep = Pick(["xch4", "u10", "v10"])
tracker = MemoryTracker()
run = tracker.start_run("plume-detector", {"pipeline_hash": pipeline_hash(prep)})
for step, loss in enumerate([0.9, 0.5, 0.3]):
    tracker.log_metrics(run, {"loss": loss}, step)
tracker.log_artifact(run, "s3://bucket/models/plume.bin", "weights")
tracker.end_run(run, "completed")

# today (pipekit-experiment)
# the protocol is checked by shape: MemoryTracker inherits nothing
print(isinstance(tracker, pe.ExperimentTracker), run.status, run.metrics)
print(tracker.history.values["loss"], run.to_dict()["artifacts"])
```

This prints

```
True completed {'loss': 0.3}
[(0, 0.9), (1, 0.5), (2, 0.3)] {'weights': 's3://bucket/models/plume.bin'}
```

that is, the tracker passes the protocol check, the run closed with its latest loss, the full series is kept as `(step, value)` pairs, and `Run.to_dict` gives the attached file by name. A `Run` also records its `pipeline_hash`, its `config` and UTC start and end times. `pipekit_train.LogToExperiment` forwards a training loop's config and metrics to any object with `start_run`, `log_metrics` and `end_run`, so a tracker like this one plugs into [pipekit-train](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-train) as it is.

`TrainingArtifact` records what a training run used: the training pipeline, the dataset's hash, the trained model's hash, the tracker's run, the registry URI, the backend and the lockfile. `pipekit_train.TrainingLoop.run()` returns one when this package is installed (the `experiment` extra of pipekit-train) and a local dataclass with the same fields otherwise. `InferenceArtifact` pins every registry-resolved operator of an inference pipeline by hash, as the README's quickstart does, and both records save to and load from JSON.

```python
import pathlib
import tempfile

import pipekit_experiment as pe
from pipekit_ops import Pick, dumps


# Shapes: h is a 64-character hex digest

root = pathlib.Path(tempfile.mkdtemp())
# today (pipekit-experiment)
registry = pe.LocalModelRegistry(root / "models")
# today (pipekit-ops)
prep = Pick(["xch4", "u10", "v10"])
# today (pipekit-experiment)
model_hash = registry.store(prep, name="plume-prep-v1")
# user
# the dataset's hash, as DVCDatasetVersioning.hash returns it
dataset_hash = "3863d0e317dee0a55c4e59d2ec0eef33.dir"

# today (pipekit-experiment)
# what a training run used: the pipeline, the data, the model and the stack
record = pe.TrainingArtifact(
    training_pipeline_yaml=dumps(prep),
    dataset_hash=dataset_hash,
    trained_model_hash=model_hash,
    tracker_run_id=None,
    model_registry_uri=str(registry.root),
    backend_info={"backend": "none"},
)
record.save(root / "training.json")
# today (pipekit-experiment)
reloaded = pe.TrainingArtifact.load(root / "training.json")
# today (pipekit-experiment)
rebuilt = reloaded.reload_model(registry)
print(reloaded == record, rebuilt.state == prep.state)
```

This prints `True True`: the record survives the JSON round trip unchanged, and `reload_model` brings back the operator it pinned. [Train an emulator](../tutorials/train-emulator.md) builds the same record from a real training loop.

(qs-exp-bridges)=
## The bridges

Each bridge lives in its own module under `pipekit_experiment.bridges` and imports its tool on first use, so the module imports without the tool and a CLI can branch on what is installed ([X1](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-experiment/docs/design/decisions.md) says why they are called bridges).

### DVC

`DVCDatasetVersioning(repo, dvc_executable="dvc")` calls the `dvc` CLI (`add`, `pull`, `push` and `status`) in the repository you give it, and raises a `RuntimeError` on the first such call when the executable is not on `PATH`. It never imports DVC's Python package, which the `dvc` extra installs along with the CLI. `hash(path)` reads the `md5:` or `hash:` line of `path.dvc` without calling the CLI, and `record_hash_in(path, sink)` puts that hash into a dict you pass, ready for a `TrainingArtifact`.

```python
import pathlib
import tempfile

from pipekit_experiment.bridges.dvc import DVCDatasetVersioning


# Shapes: the dataset hash is a 32-character md5 digest

repo = pathlib.Path(tempfile.mkdtemp())
# user
# the pointer file that `dvc add data/train` writes next to the data
(repo / "data").mkdir()
pointer = "outs:\n- md5: 3863d0e317dee0a55c4e59d2ec0eef33.dir\n  path: train\n"
(repo / "data" / "train.dvc").write_text(pointer)

# today (pipekit-experiment)
dvc = DVCDatasetVersioning(repo)
# today (pipekit-experiment)
# read from the pointer file, without calling the dvc CLI
meta: dict[str, str] = {}
dataset_hash = dvc.record_hash_in("data/train", meta)
print(dataset_hash, meta)
```

This prints `3863d0e317dee0a55c4e59d2ec0eef33.dir {'data/train': '3863d0e317dee0a55c4e59d2ec0eef33.dir'}`: the hash DVC recorded for the directory, keyed by the path you asked for.

### Hydra

`HydraConfigLoader` turns a pipeline into a Hydra config and back, with every method static. `to_yaml(op)` writes `_target_` YAML from `op.state`, nested operators included, and refuses an operator that holds a closure; `to_hydra_cfg(op)` gives a hydra-zen `builds(...)` spec of the operator; and `from_hydra_cfg(cfg)` and `from_yaml(text)` instantiate a config with `hydra.utils.instantiate`. `instantiate` will import and call any dotted path, so for YAML you did not write, use `from_yaml_sandboxed`, which builds only the classes registered with `pipekit_ops.register`, checks every class before it builds anything, does not resolve `${...}` interpolations and never calls `instantiate`. `to_yaml` and `from_yaml_sandboxed` need only PyYAML, which the `hydra` extra brings with hydra-core.

```python
from pipekit_experiment.bridges.hydra import HydraConfigLoader
from pipekit_ops import Get, Rename, Sequential, register


# Shapes: one scene of 3 named scalars

# today (pipekit-ops)
wind = Rename({"u10": "u", "v10": "v"}) | Get("u")
# today (pipekit-experiment)
# _target_ YAML written from wind.state, nested operators included
text = HydraConfigLoader.to_yaml(wind)

# today (pipekit-ops)
# untrusted YAML builds only the classes registered here
for cls in (Sequential, Rename, Get):
    register(cls)
# today (pipekit-experiment)
rebuilt = HydraConfigLoader.from_yaml_sandboxed(text)
print(rebuilt({"xch4": 1912.0, "u10": 3.1, "v10": -0.4}), rebuilt.state == wind.state)
print(text)
```

This prints

```
3.1 True
_target_: pipekit_ops.Sequential
operators:
- _target_: pipekit_ops.Rename
  mapping:
    u10: u
    v10: v
- _target_: pipekit_ops.Get
  key: u
```

that is, the rebuilt pipeline picks the eastward wind and has the same state as the original, and the YAML is the config Hydra reads. With hydra-core installed, the same text loads through Hydra itself; this block needs the `hydra` extra.

```python
from pipekit_experiment.bridges.hydra import HydraConfigLoader
from pipekit_ops import Get, Rename


# Shapes: one scene of 3 named scalars

# today (pipekit-ops)
wind = Rename({"u10": "u", "v10": "v"}) | Get("u")
# today (pipekit-experiment)
text = HydraConfigLoader.to_yaml(wind)
# today (pipekit-experiment)
# hydra.utils.instantiate: only for YAML you wrote yourself
rebuilt = HydraConfigLoader.from_yaml(text)
print(rebuilt.state == wind.state)
```

### Metaflow

`MetaflowStepAdapter.as_step(op, name=..., inputs=[...])` builds a step function that reads `self.<input>` for each input, applies the operator and writes the result to `self.<name>`, wrapped in Metaflow's `@step`. `decorate=False` gives the bare function, and `MetaflowStepAdapter.run(op, namespace, name=..., inputs=[...])` runs the same logic on any object with attributes, so both can be tested without Metaflow.

```python
from types import SimpleNamespace

from pipekit_experiment.bridges.metaflow import MetaflowStepAdapter
from pipekit_ops import Get, Rename


# Shapes: one scene of 3 named scalars

# today (pipekit-ops)
wind = Rename({"u10": "u", "v10": "v"}) | Get("u")
# today (pipekit-experiment)
# step(self) reads self.scene, applies wind and writes self.u
step = MetaflowStepAdapter.as_step(wind, name="u", inputs=["scene"], decorate=False)

# user
# a stand-in for the flow's self, so the step runs without Metaflow
flow = SimpleNamespace(scene={"xch4": 1912.0, "u10": 3.1, "v10": -0.4})
step(flow)
print(flow.u, step.__name__)
```

This prints `3.1 u`: the step wrote the eastward wind to `flow.u`, and the function carries the step's name. Inside a flow the decorated step is a class attribute of the `FlowSpec`; this block needs the `metaflow` extra.

```python
from metaflow import FlowSpec
from pipekit_experiment.bridges.metaflow import MetaflowStepAdapter
from pipekit_ops import Get, Rename


# today (pipekit-ops)
wind = Rename({"u10": "u", "v10": "v"}) | Get("u")


# user
class WindFlow(FlowSpec):
    # today (pipekit-experiment)
    # reads self.scene, writes self.u
    u = MetaflowStepAdapter.as_step(wind, name="u", inputs=["scene"])
```

(qs-exp-not-built)=
## Not built yet

- **MLflow and W&B adapters.** The `mlflow` and `wandb` extras pin those libraries, but no `ExperimentTracker` for either ships, and no concrete tracker of any kind ships yet. `pipekit_train.LogToExperiment` forwards to any object that satisfies the protocol, such as the one [above](#qs-exp-trackers).
- **Per-cell benchmark runs.** pipekit-evaluate's benchmark ladder does not write a `Run` per cell yet.

## Where to go next

- [Train an emulator](../tutorials/train-emulator.md), [Train, register, forecast](../tutorials/train-register-forecast.md) and [Run inference](../tutorials/inference.md) use the registry inside whole pipelines.
- [Patterns](../patterns.md) has one pre-processing object for training and serving, promotion by tag and the feedback loop.
- [Protocols](../protocols.md) puts `ExperimentTracker` and `ModelRegistry` beside the workspace's other protocols.
- The [API reference](https://jejjohnson.github.io/pipekit/reference/pipekit-experiment/) has every public name, generated from the docstrings, and master plan [Report 12](https://github.com/jejjohnson/research_journal_v2/blob/main/notes/geotoolz/master_plan/toolz_10_pipekit_experiment.md) is the design the package came from.
