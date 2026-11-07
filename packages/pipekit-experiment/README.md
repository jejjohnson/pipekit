<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/pipekit/main/docs/assets/packages/pipekit-experiment/logo-dark.svg">
    <img alt="pipekit-experiment" src="https://raw.githubusercontent.com/jejjohnson/pipekit/main/docs/assets/packages/pipekit-experiment/logo-light.svg" width="518">
  </picture>
</p>

**A content-addressed model registry, tracker protocols and reproducibility records for pipekit pipelines, with adapters for DVC, Hydra and Metaflow.**

<p align="center">
  <a href="https://github.com/jejjohnson/pipekit/actions/workflows/ci.yml"><img alt="Tests" src="https://github.com/jejjohnson/pipekit/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Version 0.1.0" src="https://img.shields.io/badge/version-0.1.0-blue">
  <img alt="Python 3.12+" src="https://img.shields.io/badge/python-3.12%2B-blue">
  <a href="https://opensource.org/licenses/MIT"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-yellow.svg"></a>
</p>

<p align="center">
  <a href="https://jejjohnson.github.io/pipekit/"><b>Docs</b></a> ·
  <a href="https://jejjohnson.github.io/pipekit/api/pipekit-experiment/"><b>API</b></a> ·
  <a href="https://jejjohnson.github.io/pipekit/tutorials/train-register-forecast/"><b>Tutorial</b></a>
</p>

## Why it exists

Consider a methane plume detector that runs on every EMIT scene.
Six months after a detection, someone asks which model produced it, which pre-processing sat in front of it, and whether the files on the bucket are still the ones we validated.
A directory called `models/plume_v3_final2/` does not answer that.

pipekit-experiment identifies a stored model by the hash of what it is: the operator's config and its weight bytes.
Names such as `production` are tags that point at a hash, so promoting a model moves a tag and never touches the model.
Loading checks the bytes against the hash, so a file changed behind the registry's back is refused instead of served.
Around the registry sit two protocols for external tools (`ExperimentTracker`, `ModelRegistry`), two JSON records that pin what a training or inference run used, and thin adapters for DVC, Hydra and Metaflow.

## Where it sits

pipekit-experiment is built on pipekit-ops; it does not use pipekit-array.
Both are libraries in their own right, which you can build on directly.

```
   your training and inference pipelines
                     │
   pipekit-train ────┤  ([experiment] extra: TrainingArtifact, LogToExperiment)
   pipekit-jax ──────┤  (JaxModelOp.from_registry reads load_weights)
                     │
          ┌──────────┴──────────┐
          │ pipekit-experiment  │  ◀ this package
          │ registry, trackers, │
          │  artifacts, adapters│
          └──────────┬──────────┘
            ┌────────┴────────┐
            │                 │
       pipekit-ops       pipekit-array
       (dependency:      (not used)
        Operator state,
        pipeline_hash)
```

- **pipekit-ops** is the one required dependency.
  The registry stores `Operator.state`, rebuilds with `Operator.from_state`, and hashes with `pipekit_ops.hashing.pipeline_hash`, so anything that round-trips through `pipekit_ops.dumps` / `loads` can be stored.
- **pipekit-array** is not a dependency and is not used; array-backed operators store like any other operator.

## Install

```bash
uv add pipekit-experiment
uv add 'pipekit-experiment[dvc,hydra,metaflow,s3]'   # combine the extras you need
```

| Extra | What it adds | When you need it |
| --- | --- | --- |
| *(none)* | `pipekit-ops` | the registry, protocols, run records and artifacts |
| `s3` | `fsspec`, `s3fs` | `S3ModelRegistry` on S3, or any other fsspec URI with its driver |
| `dvc` | `dvc` | `DVCDatasetVersioning`, which calls the `dvc` CLI |
| `hydra` | `hydra-core`, `hydra-zen` | `HydraConfigLoader`: pipelines to and from Hydra configs |
| `metaflow` | `metaflow` | `MetaflowStepAdapter`: an operator as a Metaflow `@step` |
| `mlflow` | `mlflow` | nothing yet: the extra pins the library, but no MLflow adapter ships |
| `wandb` | `wandb` | nothing yet: the extra pins the library, but no W&B adapter ships |

Each adapter imports its tool lazily and raises an `ImportError` with the install hint on first use, so the package imports cleanly with no extras.

## Quickstart

Store the pre-processing in front of the plume detector, with opaque weight bytes beside it, then promote it and load it back by tag.

```python
import tempfile

import pipekit_experiment as pe
from pipekit_ops import Pick, Rename

root = tempfile.mkdtemp()

# today (pipekit-ops)
# the scene pre-processing a plume detector was trained behind
prep = Pick(["xch4", "u10", "v10"]) | Rename({"u10": "u", "v10": "v"})
print(prep({"xch4": 1912.0, "u10": 3.1, "v10": -0.4, "qa": 1}))
# {'xch4': 1912.0, 'u': 3.1, 'v': -0.4}
weights = b"\x00\x01\x02\x03"  # user: opaque bytes, such as a model's weights

# today (pipekit-experiment)
registry = pe.LocalModelRegistry(root)
h = registry.store(prep, name="plume-prep-v1", tags={"sensor": "emit"}, weights=weights)
print(h == registry.store(prep, weights=weights))     # True: same content, same hash

# today (pipekit-experiment)
registry.tag(h, "production")                          # promote; force=True moves a tag
again = registry.load("production")                    # by tag, or by hash
print(again.state == prep.state)                       # True
print(registry.load_weights(h) == weights)             # True
print(registry.list(tags={"sensor": "emit"}) == [h])   # True
print(isinstance(registry, pe.ModelRegistry))          # True
```

Pin the hash in an inference record, and see the registry refuse a tampered entry:

```python
import pathlib

# today (pipekit-experiment)
# a reproducibility record that pins the model by hash
artifact = pe.InferenceArtifact(
    pipeline_yaml="...",  # user: your inference pipeline, from pipekit_ops.dumps
    pinned_model_hashes={"prep": h},
)
artifact.save(pathlib.Path(root) / "inference.json")
pinned = pe.InferenceArtifact.load(pathlib.Path(root) / "inference.json")
print(pinned.reload_models(registry)["prep"].state == prep.state)  # True

# user
# edit the stored weights behind the registry's back
(pathlib.Path(root) / h / "weights.bin").write_bytes(b"tampered")
try:
    registry.load(h)
except pe.HashMismatchError:
    print("refused")                                   # refused
```

## What ships

Version 0.1.0.

| Module | Public names | Purpose |
| --- | --- | --- |
| `registry` | `LocalModelRegistry`, `S3ModelRegistry`, `HashMismatchError` | Content-addressed storage on a local disk or any fsspec URI |
| `protocols` | `ModelRegistry`, `ExperimentTracker` | Runtime-checkable seams for registries and trackers |
| `run` | `Run`, `RunMetrics`, `RunArtifacts` | Plain records of a run, its metrics and its attached files |
| `artifacts` | `TrainingArtifact`, `InferenceArtifact` | JSON records that pin what a training or inference run used |
| `adapters.dvc` | `DVCDatasetVersioning` | Dataset hashes through the DVC CLI (`[dvc]`) |
| `adapters.hydra` | `HydraConfigLoader` | Pipelines to and from Hydra configs, with a sandboxed loader (`[hydra]`) |
| `adapters.metaflow` | `MetaflowStepAdapter` | An operator as a Metaflow `@step` (`[metaflow]`) |

The top-level names are exported from `pipekit_experiment`; the adapters are imported from their modules.

### Not built yet

- **MLflow and W&B adapters.** The `[mlflow]` and `[wandb]` extras exist, but no `ExperimentTracker` for either ships, and no concrete tracker of any kind ships yet.
  `pipekit_train.LogToExperiment` forwards to any object that satisfies the protocol.
- **Per-cell benchmark runs.** pipekit-evaluate's benchmark ladder does not write a `Run` per cell yet.

## How it works

### Content addressing

`store(op, weights=...)` writes three files under `<root>/<hash>/`: `operator.json` (the operator's state), `weights.bin` (the bytes, if any) and `metadata.json` (the tags).
The hash is `pipeline_hash(op, weights)`, a sha256 over the stable JSON of the state and the weight bytes, so storing the same content twice is a no-op that merges the tags.
`b""` and `None` weights hash the same, so the layout never depends on which one you passed.
Tags live in `<root>/_tags/<name>`, one hash per file.
`tag(h, name)` refuses to move an existing tag unless you pass `force=True`, which is how promotion stays deliberate.
`LocalModelRegistry` writes through a temp file and `os.replace`, so a crash never leaves half a file; `S3ModelRegistry` is as atomic as the object store under it.

### Loading verifies

`load(ref)` takes a hash or a tag, re-hashes the stored text and weights, and raises `HashMismatchError` when they no longer match.
It then rebuilds the operator with `Operator.from_state` and checks that the rebuilt operator hashes back to the same value.
An entry stored before pipekit 0.2.0 (when the core was imported as `pipekit`) still loads: the module names are updated with `pipekit_ops.upgrade_state` and the rest is checked as before.
References and tag names must be single path segments, so a caller-supplied string never escapes the registry root.

### Operators that do not round-trip

An operator marked `forbid_in_yaml` (a `Lambda`, a `CallableForward`, a JAX model wrapper) has a config that is a debug payload, not a recipe, so `load` cannot rebuild it and raises.
Its weights are still verified and readable with `load_weights`.
The pattern for those models is to rebuild a skeleton in code and read the weights into it, which is what `pipekit_jax.JaxModelOp.from_registry` does.

### Trackers and artifacts

`ExperimentTracker` is four calls (`start_run`, `log_metrics`, `log_artifact`, `end_run`), and an adapter for any tool satisfies it structurally.
`TrainingArtifact` records the training pipeline, the dataset hash, the trained-model hash, the tracker run and the lockfile; `pipekit_train.TrainingLoop.run()` returns one when this package is installed.
`InferenceArtifact` pins every registry-resolved operator of an inference pipeline by hash, and `reload_models(registry)` brings them back.

### The adapters

`DVCDatasetVersioning` shells out to the `dvc` CLI (`add`, `pull`, `push`, `hash`, `status`) and `record_hash_in(path, sink)` puts a dataset's hash into a dict you pass, ready for a `TrainingArtifact`.
`HydraConfigLoader` turns a pipeline into a Hydra config and back.
`from_yaml` calls `hydra.utils.instantiate`, which will import any dotted path, so for YAML you did not write use `from_yaml_sandboxed`, which builds only the classes you registered with `pipekit_ops.register`:

```python
from pipekit_experiment.adapters.hydra import HydraConfigLoader
from pipekit_ops import Get, Rename, Sequential, register

# today (pipekit-ops)
wind = Rename({"u10": "u", "v10": "v"}) | Get("u")
print(wind({"xch4": 1912.0, "u10": 3.1, "v10": -0.4}))  # 3.1

# today (pipekit-experiment)
text = HydraConfigLoader.to_yaml(wind)                   # through hydra_zen.builds
print(HydraConfigLoader.from_yaml(text).state == wind.state)  # True

# today (pipekit-ops)
# untrusted YAML builds only the classes you register
for cls in (Sequential, Rename, Get):
    register(cls)
print(HydraConfigLoader.from_yaml_sandboxed(text).state == wind.state)  # True
```

`MetaflowStepAdapter.as_step(op, name=..., inputs=[...])` builds a step that reads `self.<input>` for each input, applies the operator and writes `self.<name>`; `decorate=False` gives the bare function, which you can test without Metaflow:

```python
from metaflow import FlowSpec
from pipekit_experiment.adapters.metaflow import MetaflowStepAdapter


# today (pipekit-experiment)
class WindFlow(FlowSpec):
    # reads self.scene, writes self.u
    u = MetaflowStepAdapter.as_step(wind, name="u", inputs=["scene"])


# user
class Namespace:
    scene = {"xch4": 1912.0, "u10": 3.1, "v10": -0.4}


# today (pipekit-experiment)
step = MetaflowStepAdapter.as_step(wind, name="u", inputs=["scene"], decorate=False)
ns = Namespace()
step(ns)
print(ns.u)  # 3.1
```

## Learn more

- API reference: [pipekit-experiment](https://jejjohnson.github.io/pipekit/api/pipekit-experiment/).
- Tutorials: [Train, register, forecast](https://jejjohnson.github.io/pipekit/tutorials/train-register-forecast/), [Train an emulator](https://jejjohnson.github.io/pipekit/tutorials/train-emulator/).
- Design: master plan [Report 12](https://github.com/jejjohnson/research_journal_v2/blob/main/notes/geotoolz/master_plan/toolz_10_pipekit_experiment.md).
- The foundations: [pipekit-ops](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-ops) and [pipekit-array](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-array).
- Packages that use this one: [pipekit-train](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-train) (training artifacts, `LogToExperiment`), [pipekit-jax](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-jax) (`JaxModelOp.from_registry`).

## License

MIT. See [LICENSE](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-experiment/LICENSE).
