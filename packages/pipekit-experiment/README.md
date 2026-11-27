<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/pipekit/main/docs/assets/packages/pipekit-experiment/logo-dark.svg">
  <img alt="pipekit-experiment" src="https://raw.githubusercontent.com/jejjohnson/pipekit/main/docs/assets/packages/pipekit-experiment/logo-light.svg" width="518">
</picture>

**A content-addressed model registry, run records and tracker protocols for pipekit pipelines, on pipekit-ops, with bridges to DVC, Hydra and Metaflow, storage on any fsspec filesystem (s3fs for S3), and extras that pin MLflow and W&B for a tracker of your own.**

[![Tests](https://github.com/jejjohnson/pipekit/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/pipekit/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/pipekit/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/pipekit/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/pipekit/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/pipekit/actions/workflows/typecheck.yml)
[![Docs](https://github.com/jejjohnson/pipekit/actions/workflows/pages.yml/badge.svg)](https://jejjohnson.github.io/pipekit/)
[![codecov](https://codecov.io/gh/jejjohnson/pipekit/branch/main/graph/badge.svg)](https://codecov.io/gh/jejjohnson/pipekit)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-experiment/LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

📖 [Documentation](https://jejjohnson.github.io/pipekit/) ·
🚀 [Quickstart](https://jejjohnson.github.io/pipekit/quickstart-experiment/) ·
📚 [API reference](https://jejjohnson.github.io/pipekit/reference/pipekit-experiment/) ·
🧭 [Design](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-experiment/docs/design)

</div>

---

Consider a methane plume detector that runs on every EMIT scene. Six months after a detection, someone asks which model produced it, which pre-processing sat in front of it, and whether the files on the bucket are still the ones we validated. A directory called `models/plume_v3_final2/` does not answer that. So pipekit-experiment identifies a stored model by what it is, the operator's state and its weight bytes, and gives it one address,

```math
h = \mathrm{sha256}\left(\mathrm{json}(s) \Vert \mathtt{0x00} \Vert w\right)
```

where $`s`$ is the operator's state record (the nested module, class and config of `Operator.state`), $`\mathrm{json}`$ writes it with sorted keys, $`w \in \{0, \dots, 255\}^{N_w}`$ are the $`N_w`$ weight bytes ($`N_w = 0`$ when there are none), $`\Vert`$ joins byte strings, $`\mathtt{0x00}`$ is one zero byte that separates the two parts, and $`h`$ is the 64-character hex digest. Names such as `production` are tags that point at an $`h`$, so promoting a model moves a tag and never touches the model, and loading recomputes $`h`$ from the stored files, so a file changed behind the registry's back is refused instead of served.

```python
# Shapes: h is a 64-character hex digest
# (registry, prep, weights and pipeline_yaml as in the quickstart below)

# today (pipekit-experiment)
# h = sha256(json(s) ‖ 0x00 ‖ w), the address of the model
model_hash = registry.store(prep, weights=weights)
# today (pipekit-experiment)
# a tag names h; moving it promotes a model and never touches one
registry.tag(model_hash, "production")
# today (pipekit-experiment)
# rebuilt only when the stored files still hash to h
served = registry.load("production")
# today (pipekit-experiment)
# a run's record pins h, so the run leads back to the model it used
record = pe.InferenceArtifact(pipeline_yaml, pinned_model_hashes={"prep": model_hash})
```

None of it is a new serialisation format. The registry stores `Operator.state` from [pipekit-ops](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-ops), rebuilds with `Operator.from_state` and addresses with `pipekit_ops.pipeline_hash`, so anything that round-trips through `pipekit_ops.dumps` and `loads` can be stored, and `pipeline_hash(op, weights)` is the address before you store it. `S3ModelRegistry` runs the same code on any [fsspec](https://github.com/fsspec/filesystem_spec) URI, with [s3fs](https://github.com/fsspec/s3fs) for `s3://`. The bridges hand pipelines to the tools that run them: `DVCDatasetVersioning` calls the [DVC](https://github.com/iterative/dvc) CLI for dataset hashes, `HydraConfigLoader` writes and reads [Hydra](https://github.com/facebookresearch/hydra) `_target_` configs (with [hydra-zen](https://github.com/mit-ll-responsible-ai/hydra-zen) for `builds` specs), and `MetaflowStepAdapter` turns an operator into a [Metaflow](https://github.com/Netflix/metaflow) step. For [MLflow](https://github.com/mlflow/mlflow) and [W&B](https://github.com/wandb/wandb) the package gives the `ExperimentTracker` protocol and an extra that pins each library, and no adapter yet.

## 🌟 Highlights

- 🔐 **Content addresses**: `store` returns `pipeline_hash(op, weights)`, so storing the same operator and weights twice is a no-op that merges their tags, and `b""` and `None` weights give the same address.
- 🛡️ **Loading verifies**: `load` and `load_weights` re-hash the stored `operator.json` and `weights.bin` and raise `HashMismatchError` when they no longer match, and `load` also checks that the rebuilt operator hashes back to $`h`$.
- 🏷️ **Promotion by tag**: `tag` refuses to move a tag that already names a hash unless you pass `force=True`, so promoting and rolling back stay deliberate, and `list(tags=...)` filters models by their metadata.
- ☁️ **Two backends, one code path**: `LocalModelRegistry` writes through a temp file and `os.replace`, so a crash never leaves half a file, and `S3ModelRegistry` shares its store, load and tag code on any fsspec URI.
- 🧾 **Run records**: `TrainingArtifact` and `InferenceArtifact` are JSON records that pin what a run used by hash and bring the models back with `reload_model` and `reload_models`, beside the plain `Run`, `RunMetrics` and `RunArtifacts`.
- 🔌 **Seams checked by shape**: `ModelRegistry` and `ExperimentTracker` are runtime-checkable protocols, so a registry or tracker for any tool satisfies them without inheriting from this package.
- 🌉 **Bridges with a sandbox**: `HydraConfigLoader.from_yaml_sandboxed` builds a pipeline from untrusted `_target_` YAML using only the classes registered with `pipekit_ops.register`, and never calls `hydra.utils.instantiate`.

## 📐 One address, five questions

Every module answers a question about a stored model, and each answer is read from $`h`$. We write $`s'`$ and $`w'`$ for the state and the weights as they are read back from storage.

| Question | Equation | Module |
|---|---|---|
| Which model is this? | $`h = \mathrm{sha256}(\mathrm{json}(s) \Vert \mathtt{0x00} \Vert w)`$ | `registry`: `store` |
| Is it still the model we validated? | $`\mathrm{sha256}(\mathrm{json}(s') \Vert \mathtt{0x00} \Vert w') = h`$ | `registry`: `load`, `load_weights`, `HashMismatchError` |
| Which model is in service? | $`\mathrm{tag} \mapsto h`$ | `registry`: `tag`, `resolve_tag`, `list` |
| What did a run use? | the record holds every $`h`$ | `artifacts`, `run` |
| Can another tool store or track? | any object with the protocol's methods | `protocols`, `bridges` |

The second row is two checks. The stored text and bytes must hash to $`h`$, which catches a file edited after `store`, and the operator rebuilt from them must hash to $`h`$ as well, which catches a state that does not round-trip through `Operator.from_state`. An operator that holds a closure (a `Lambda`, a JAX model wrapper) is marked `forbid_in_yaml`, so `load` cannot rebuild it and raises, while its weights are still checked and readable with `load_weights`; the [quickstart](https://jejjohnson.github.io/pipekit/quickstart-experiment/) shows that path and the layout on disk.

## 📥 Install

pipekit-experiment is not on PyPI, so install it from the pipekit repository on GitHub. Its one dependency is pipekit-ops, which is not on PyPI either and has no third-party dependencies, so the two are installed together; the base install brings in no third-party package and needs Python 3.12 or newer.

```bash
pip install "pipekit-ops @ git+https://github.com/jejjohnson/pipekit.git#subdirectory=packages/pipekit-ops" "pipekit-experiment @ git+https://github.com/jejjohnson/pipekit.git#subdirectory=packages/pipekit-experiment"
```

Six extras add the tools: `s3` (fsspec and s3fs) for `S3ModelRegistry`, `dvc` (the DVC package, though the bridge calls the `dvc` CLI it installs), `hydra` (hydra-core and hydra-zen) for `HydraConfigLoader`, `metaflow` for `MetaflowStepAdapter`, and `mlflow` and `wandb`, which only pin those libraries for a tracker you write. Each bridge imports its tool on first use and says which extra to install when it is missing, so the package imports cleanly with no extras.

```bash
pip install "pipekit-ops @ git+https://github.com/jejjohnson/pipekit.git#subdirectory=packages/pipekit-ops" "pipekit-experiment[s3] @ git+https://github.com/jejjohnson/pipekit.git#subdirectory=packages/pipekit-experiment"
```

A project that depends on it pins one tag of the pipekit repository for every pipekit package it uses. uv reads `[tool.uv.sources]` only from the project it installs, so pipekit-experiment's requirement on pipekit-ops needs its own entry, and two different tags would give uv two URLs for pipekit-ops, which it reports as a conflict.

```toml
[project]
dependencies = ["pipekit-experiment[s3]"]

[tool.uv.sources]
pipekit-ops = { git = "https://github.com/jejjohnson/pipekit", subdirectory = "packages/pipekit-ops", tag = "pipekit-array-v0.3.1" }
pipekit-experiment = { git = "https://github.com/jejjohnson/pipekit", subdirectory = "packages/pipekit-experiment", tag = "pipekit-array-v0.3.1" }
```

## 🛰️ Quickstart: pin a plume detector's pre-processing

We store the pre-processing that sits in front of the plume detector, with four opaque weight bytes $`w`$ beside it ($`N_w = 4`$), promote it to `production`, pin it in an inference record and then edit the stored weights by hand. The operator's state $`s`$ is the record of `Pick` and `Rename` inside one `Sequential`, and the scene is one dict of four named scalars.

```python
import pathlib
import tempfile

import pipekit_experiment as pe
from pipekit_ops import Pick, Rename, dumps


# Shapes: one scene of 4 named scalars; h is a 64-character hex digest

root = pathlib.Path(tempfile.mkdtemp())
scene = {"xch4": 1912.0, "u10": 3.1, "v10": -0.4, "qa": 1}

# today (pipekit-ops)
# the pre-processing a plume detector was trained behind
prep = Pick(["xch4", "u10", "v10"]) | Rename({"u10": "u", "v10": "v"})
# user
# w, opaque bytes such as a model's weights
weights = b"\x00\x01\x02\x03"

# today (pipekit-experiment)
registry = pe.LocalModelRegistry(root)
# today (pipekit-experiment)
# h = sha256(json(s) ‖ 0x00 ‖ w)
model_hash = registry.store(prep, tags={"sensor": "emit"}, weights=weights)
# today (pipekit-experiment)
# the same s and w give the same h, so storing again is a no-op
same_hash = registry.store(prep, weights=weights) == model_hash
# today (pipekit-experiment)
# promote: the tag "production" now names h
registry.tag(model_hash, "production")
# today (pipekit-experiment)
# load by tag, after checking the stored files against h
served = registry.load("production")

# today (pipekit-experiment)
# an inference record that pins the model by h
record = pe.InferenceArtifact(
    pipeline_yaml=dumps(served),
    pinned_model_hashes={"prep": model_hash},
)
record.save(root / "inference.json")
# today (pipekit-experiment)
pinned = pe.InferenceArtifact.load(root / "inference.json")
reloaded = pinned.reload_models(registry)["prep"]

# user
# change the stored weights behind the registry's back
(root / model_hash / "weights.bin").write_bytes(b"tampered")
try:
    # today (pipekit-experiment)
    registry.load(model_hash)
    refused = False
except pe.HashMismatchError:
    refused = True

print(model_hash[:12], same_hash, served(scene), reloaded.state == prep.state, refused)
```

This prints `d415f4253757 True {'xch4': 1912.0, 'u': 3.1, 'v': -0.4} True True`: the first twelve characters of $`h`$, the second `store` landing on the same address, the pre-processing served by tag (the quality flag dropped and the winds renamed), the model the inference record pinned coming back with the same state, and the registry refusing the entry once its weights were edited. Nothing here is specific to a dict pipeline, so storing a fitted scaler, a threshold or a JAX model's weights works the same way.

The [quickstart](https://jejjohnson.github.io/pipekit/quickstart-experiment/) takes this further: the layout on disk and the calling convention, promotion and rollback by tag, operators that do not round-trip, the registry on an object store, a tracker written to the protocol, training records, and the DVC, Hydra and Metaflow bridges.

## 🔬 What else is in the box

The ten top-level names are exported from `pipekit_experiment` and listed in its `__all__`; the bridges are imported from their modules under `pipekit_experiment.bridges`, one module per tool. The package defines no Operator of its own: geostack's [pipekit shelf page](https://github.com/jejjohnson/geostack/blob/main/docs/design/skill_tree_pipekit.md) places it beside the role ladder, with orchestration, because it stores, records and hands over the steps and recipes that pipekit-ops builds.

| Module | Symbols | Notes |
|---|---|---|
| `pipekit_experiment.registry` | `LocalModelRegistry`, `S3ModelRegistry`, `HashMismatchError` | Content-addressed storage on a local disk or any fsspec URI (`s3`), with `store`, `load`, `load_weights`, `list`, `tag` and `resolve_tag` |
| `pipekit_experiment.protocols` | `ModelRegistry`, `ExperimentTracker` | Runtime-checkable seams for registries and trackers |
| `pipekit_experiment.run` | `Run`, `RunMetrics`, `RunArtifacts` | Plain records of a run, its metric series and its attached files |
| `pipekit_experiment.artifacts` | `TrainingArtifact`, `InferenceArtifact` | JSON records that pin what a training or inference run used |
| `pipekit_experiment.bridges.dvc` | `DVCDatasetVersioning` | Dataset hashes through the DVC CLI (`dvc`) |
| `pipekit_experiment.bridges.hydra` | `HydraConfigLoader` | Pipelines to and from Hydra `_target_` configs, with a sandboxed loader (`hydra`) |
| `pipekit_experiment.bridges.metaflow` | `MetaflowStepAdapter` | An operator as a Metaflow `@step` (`metaflow`) |

## 📓 Tutorials

| # | Tutorial | What it shows |
|---|---|---|
| 1 | [Train an emulator](https://jejjohnson.github.io/pipekit/train-emulator/) | a pipekit-train loop whose model is stored in a `LocalModelRegistry` and whose run returns a `TrainingArtifact` |
| 2 | [Train, register, forecast](https://jejjohnson.github.io/pipekit/train-register-forecast/) | a trained model registered by hash and dropped into a pipekit-cycle forecast |
| 3 | [Run inference](https://jejjohnson.github.io/pipekit/inference/) | an inference pipeline that loads its model from the registry |

## 📖 Documentation

- **[Quickstart](https://jejjohnson.github.io/pipekit/quickstart-experiment/)**: the address and the layout on disk, the calling convention, tags, operators that do not round-trip, the object-store backend, trackers and run records, the three bridges, and what is not built yet.
- **[Protocols](https://jejjohnson.github.io/pipekit/protocols/)**: the workspace's protocols side by side, `ExperimentTracker` and `ModelRegistry` among them.
- **[Patterns](https://jejjohnson.github.io/pipekit/patterns/)**: one pre-processing object for training and serving, promotion by tag, and the feedback loop, each built on the registry.
- **[API reference](https://jejjohnson.github.io/pipekit/reference/pipekit-experiment/)**: every public symbol, generated from the docstrings.
- **[Design decisions](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-experiment/docs/design/decisions.md)**: why the tool integrations are bridges (X1).

## 🧩 Ecosystem

pipekit-experiment is one package of the pipekit [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/), in the Core tier of the [GeoStack](https://github.com/jejjohnson/geostack). It owns the address of a stored model and the records that point at it, and composes the rest in:

| Concern | Owner |
|---|---|
| Operator state, `from_state`, `pipeline_hash`, `register` and the sandboxed loader | [`pipekit-ops`](https://github.com/jejjohnson/pipekit/tree/main/packages/pipekit-ops) |
| Object storage for `S3ModelRegistry` (the `s3` extra) | [`fsspec`](https://github.com/fsspec/filesystem_spec) and [`s3fs`](https://github.com/fsspec/s3fs) |
| Dataset versions for `DVCDatasetVersioning` (the `dvc` extra) | [`DVC`](https://github.com/iterative/dvc) |
| Configs for `HydraConfigLoader` (the `hydra` extra) | [`Hydra`](https://github.com/facebookresearch/hydra) and [`hydra-zen`](https://github.com/mit-ll-responsible-ai/hydra-zen) |
| Flow steps for `MetaflowStepAdapter` (the `metaflow` extra) | [`Metaflow`](https://github.com/Netflix/metaflow) |
| Experiment tracking behind `ExperimentTracker` (the `mlflow` and `wandb` extras, no adapter yet) | [`MLflow`](https://github.com/mlflow/mlflow) and [`W&B`](https://github.com/wandb/wandb) |

Dependencies point one way: pipekit-experiment imports pipekit-ops and nothing else from the workspace (not pipekit-array, though array-backed operators store like any other), and no package it builds on imports it. Inside the workspace, pipekit-train's `experiment` extra makes `TrainingLoop.run()` return a `TrainingArtifact` (it imports `pipekit_experiment.artifacts` when the package is installed), and its `Checkpoint` and `LogToExperiment` callbacks take any `ModelRegistry` or `ExperimentTracker` by shape; pipekit-jax imports nothing from it, and `JaxModelOp.from_registry` reads weights from any `ModelRegistry`; the `pipekit` umbrella installs it. Outside the workspace, vardax lists it in its `persist` extra and geostack's report notebooks import `LocalModelRegistry`.

## 🌳 Repository layout

```
packages/pipekit-experiment/
├── src/pipekit_experiment/
│   ├── __init__.py      ← the ten public names
│   ├── registry.py      ← LocalModelRegistry, S3ModelRegistry, HashMismatchError
│   ├── protocols.py     ← ModelRegistry, ExperimentTracker
│   ├── run.py           ← Run, RunMetrics, RunArtifacts
│   ├── artifacts.py     ← TrainingArtifact, InferenceArtifact
│   └── bridges/         ← dvc.py, hydra.py, metaflow.py, one module per tool (X1)
├── tests/               ← one file per module and per bridge
└── docs/design/         ← decisions.md
```

## ⚙️ Development

The package is developed inside the workspace, so clone the repository and run the tasks from its root.

```bash
git clone https://github.com/jejjohnson/pipekit.git
cd pipekit
make install                                        # uv sync + the pre-commit hooks
uv run pytest packages/pipekit-experiment/tests     # this package's tests
make lint                                           # ruff check on the whole repo
make typecheck                                      # ty check on every package
make docs                                           # the MyST prose and the API reference, with a link check
```

`make help` lists every task. [`AGENTS.md`](https://github.com/jejjohnson/pipekit/blob/main/AGENTS.md) holds the rules for contributors and coding agents across the workspace: the checks to pass before every commit, the docs build and the commit conventions.

## 📄 License

MIT. See [LICENSE](https://github.com/jejjohnson/pipekit/blob/main/packages/pipekit-experiment/LICENSE).
