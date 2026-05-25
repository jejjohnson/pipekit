---
status: draft
version: 0.1.0
---

# Example — Direct supervised (cloud-mask classifier)

The simplest training shape. A catalog of labeled scenes + a
preprocessing pipeline + a UNet. Maps to master plan §2.1 + §4.1.

```python
import pipekit as pk
import pipekit_train as pt
import pipekit_experiment as pe
import geocatalog as gc
import geopatcher as gp
import geotoolz as gz

# 1. Preprocess pipeline — re-used at inference time.
preprocess = pk.Sequential([
    gz.radiometry.ToFloat32(),
    gz.radiometry.PercentileClip(p_min=2, p_max=98),
])

# 2. Catalog of labeled scenes.
catalog = gc.open_catalog("s3://imeo/labeled-scenes.parquet")

# 3. CatalogDataset — yields (preprocessed_patch, label_mask) pairs.
train_ds = pt.CatalogDataset(
    catalog="s3://imeo/labeled-scenes.parquet",
    preprocess=preprocess,
    target_op=gz.cloud.LoadLabel(label_key="cloud_mask"),
    sampler=gp.SpatialRegularStride((256, 256)),
    seed=42,
    split="train",
)
val_ds = train_ds.with_split("val")

# 4. Model (JAX/Equinox). Wrapped as a pipekit operator.
import pipekit_jax as pj
import equinox as eqx
class UNet(eqx.Module): ...
model_op = pj.JaxModelOp(UNet(in_channels=4, out_channels=2, key=...))

# 5. Pipekit-experiment integration (optional but recommended).
registry = pe.LocalModelRegistry(root="./models")
tracker  = pe.adapters.mlflow.MLflowTracker(tracking_uri="http://mlflow:5000")

# 6. The training loop.
loop = pt.TrainingLoop(
    model_op=model_op,
    dataset=train_ds,
    val_dataset=val_ds,
    loss=pt.NLL(),
    optimizer_config={"name": "adamw", "lr": 1e-3, "weight_decay": 1e-4},
    max_steps=25_000,
    batch_size=16,
    eval_every_n_steps=500,
    log_every_n_steps=50,
    backend="equinox",
    callbacks=(
        pt.EarlyStopping(metric="val_loss", patience=5, mode="min"),
        pt.Checkpoint(every_n_steps=2_000, keep_last=3,
                      save_dir="./ckpts/cloud_mask", registry=registry),
        pt.LogToExperiment(tracker=tracker, run_name="cloud_mask_unet"),
    ),
    metric_writer=pt.JSONLWriter(path="./logs/cloud_mask.jsonl"),
    seed=42,
)

# 7. Run it.
trained_model_op, artifact = loop.run()
# trained_model_op is a pipekit.Operator — drops into any inference pipeline.
# artifact is a pipekit_experiment.TrainingArtifact with:
#   training_pipeline_yaml, dataset_hash, trained_model_hash,
#   tracker_run_id, model_registry_uri, backend_info, deps_lock.

# 8. Inference pipeline — train and inference share the preprocess step.
inference = pk.Sequential([
    preprocess,
    trained_model_op,
    gz.cloud.PostProcessMask(threshold=0.5),
])

# 9. Reproduce later from the artifact alone.
artifact.save("./artifacts/cloud_mask_v3.json")
# … some weeks later …
loaded = pe.TrainingArtifact.load("./artifacts/cloud_mask_v3.json")
reloaded_model = loaded.reload_model(registry)
# Same weights, byte-identical (modulo nondeterminism).
```

**What this example demonstrates:**

- `CatalogDataset` opening a `geocatalog` URI lazily, applying a
  `pipekit.Sequential` preprocess per row, tiling via
  `geopatcher.SpatialRegularStride`, splitting by row-hash.
- The training loop runs against the v0.1 Equinox adapter with the
  user's loss synthesised into a `TrainTask` automatically.
- The `Checkpoint` callback bridges to `pipekit-experiment`: every
  2000 steps a checkpoint lands on disk; the *final* checkpoint also
  flows into the `LocalModelRegistry` keyed by content hash.
- `LogToExperiment` forwards per-step metrics, per-eval metrics,
  hyper-parameters, and the final artifact URI to MLflow.
- The trained model drops into a `Sequential` for inference. The
  preprocess step is the same operator on both sides — zero rewrite.

**What changes for different backends:**

Swap `backend="equinox"` for `backend="lightning"` (v0.2) and supply
a PyTorch UNet via `pipekit-array.ModelOp` instead of a JaxModelOp.
The rest of the pipeline is unchanged.
