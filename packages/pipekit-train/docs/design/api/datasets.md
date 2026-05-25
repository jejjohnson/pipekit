---
status: draft
version: 0.1.0
---

# Layer 0 — Datasets

Four operators. The base + three concrete subclasses. Each yields
`(input, target)` pairs and provides a stable content hash that the
training artifact pins.

## `TrainingDataset`

```python
class TrainingDataset(Operator):
    """Base class — yields (input, target) pairs.

    Subclasses implement __iter__ + content_hash. The base provides
    pipekit Operator semantics: get_config / from_state / dumps,
    composition with Sequential, and split derivation.

    Attributes:
        seed: PRNG seed used by sampling subclasses.
        split: One of "train", "val", "test". Subclasses use this
            to partition deterministically by seed.
    """

    seed: int = 0
    split: Literal["train", "val", "test"] = "train"

    def __iter__(self) -> Iterator[tuple[Any, Any]]:
        raise NotImplementedError

    def content_hash(self) -> str:
        """Stable identifier for this dataset's contents.

        Default: sha256 over (type, get_config()).
        Subclasses override to fold in dataset-specific state
        (catalog URI, forward-model state_signature, etc.).
        """
        h = hashlib.sha256()
        h.update(type(self).__qualname__.encode())
        h.update(b"\x00")
        h.update(json.dumps(self.get_config(), sort_keys=True).encode())
        return h.hexdigest()

    def with_split(self, split: Literal["train", "val", "test"]) -> TrainingDataset:
        """Return a clone with a different split, same seed."""
        return dataclasses.replace(self, split=split)
```

**The `__iter__` discipline.** A `TrainingDataset` is a stream of
`(x, y)` pairs. It does not have a meaningful `_apply` — calling the
operator like a function raises `TypeError` (the carrier model
doesn't fit). Datasets compose into `TrainingLoop` via the
`dataset=...` argument, not via `Sequential`.

## `CatalogDataset`

For direct-supervised training over labeled scenes. Bridges to
`geocatalog` and `geopatcher`.

```python
class CatalogDataset(TrainingDataset):
    """Yields (preprocessed_carrier, label) pairs from a catalog.

    Args:
        catalog: A geocatalog.GeoCatalog (or StateCatalog) reference.
            Stored as a URI string for YAML round-trip.
        preprocess: Operator chain applied per-row before yielding.
            Typically the inference-time preprocessing pipeline.
        target_op: Operator that extracts the label from a catalog row.
        sampler: Optional geopatcher.SpatialSampler that tiles each
            row into patches. None means one yield per row.
        seed: PRNG seed.
        split: "train" / "val" / "test". Maps deterministically onto
            the catalog via split_op (default 80/10/10 by row hash).
        split_op: Operator that takes (row, seed, split) and returns
            True iff the row belongs to `split`. None uses the
            default row-hash splitter.
    """
    catalog: str                              # URI; loaded lazily
    preprocess: Operator
    target_op: Operator
    sampler: Operator | None = None
    split_op: Operator | None = None

    def content_hash(self) -> str:
        # Includes catalog URI + a catalog-side checksum if available
        # (e.g. parquet schema hash + row count) so cache invalidation
        # is reliable.
        ...
```

The `catalog` argument is a URI string, not an open catalog handle,
so the dataset round-trips through YAML. The handle is opened
lazily on first `__iter__`.

## `SimulationDataset`

The bridge to `pipekit-cycle`. For emulator training and amortized
inference.

```python
class SimulationDataset(TrainingDataset):
    """Yields (parameters, simulator_output) pairs.

    Used for both emulator training (network learns forward map) and
    amortized inference (network learns inverse map — the loss
    differs, not the data).

    Args:
        forward_model: Anything satisfying pipekit_cycle.ForwardModel.
        prior: Operator that samples parameter realizations.
            Called per __iter__ tick.
        n_samples: Total samples per epoch. With infinite
            iteration (Grain `num_epochs=None`), this controls
            steps-per-epoch.
        cycle: Optional pipekit_cycle.Cycle wrapping forward_model
            to roll out trajectories rather than single steps.
        obs_op: Optional pipekit_cycle.ObservationOperator applied
            after the forward model. Used for SBI training where the
            target is observed-space outputs, not model-space states.
        seed: PRNG seed.
        split: "train" / "val" / "test". Splits realised by
            partitioning the seed space (e.g. train uses seeds in
            [0, 0.8 * 2^32); val uses [0.8 * 2^32, 0.9 * 2^32); …).
    """
    forward_model: Operator                   # also satisfies ForwardModel
    prior: Operator
    n_samples: int
    cycle: Operator | None = None
    obs_op: Operator | None = None

    def content_hash(self) -> str:
        # Includes forward_model.state_signature, prior config,
        # n_samples, seed, split — anything that affects the data.
        ...
```

The split-by-seed-partition discipline is what makes
`SimulationDataset.with_split("val")` deterministic without
materializing the train set first.

## `CachedDataset`

A disk-backed cache around any `TrainingDataset`. Identical interface
to its source.

```python
class CachedDataset(TrainingDataset):
    """Disk-backed cache around any TrainingDataset.

    First epoch hits the source and writes to `cache_dir`. Subsequent
    epochs read from disk. Cache is keyed on `source.content_hash()`.

    Args:
        source: The wrapped TrainingDataset.
        cache_dir: Directory or fsspec URI (s3://, gcs://, …).
        format: Storage format. "zarr" for array-shaped pairs;
            "parquet" for record-shaped; "tfrecord" for very large
            heterogeneous batches.
    """
    source: TrainingDataset
    cache_dir: str
    format: Literal["zarr", "parquet", "tfrecord"] = "zarr"

    def content_hash(self) -> str:
        # Inherits source.content_hash() — the cache is just a
        # materialisation of the same logical contents.
        return self.source.content_hash()
```

**Cache invariants.** The cache assumes the source is deterministic
given `(get_config(), seed)`. If the underlying catalog mutates,
the cache stays stale until explicitly invalidated. The cache
exposes an `invalidate()` method that deletes the cache directory
keyed on the current content hash; full re-cache happens on the
next `__iter__`.

## Backend handoff

Each backend adapter translates `TrainingDataset` into its native
loader idiom:

| Backend     | Translation                                          |
|-------------|------------------------------------------------------|
| Equinox     | Wrap as a `grain.MapDataset` source, then            |
|             | `grain.DataLoader(operations=[..., grain.Batch(N)])` |
| Lightning   | Wrap as a `torch.utils.data.IterableDataset`, then   |
|             | `LightningDataModule` with train/val/test loaders    |
| Keras       | `tf.data.Dataset.from_generator(dataset.__iter__)`   |

The user does not normally see this — they pass the
`TrainingDataset` to `TrainingLoop(dataset=...)` and the adapter
does the translation. Power users who want backend-native loaders
can construct them directly and bypass Layer 0 entirely.
