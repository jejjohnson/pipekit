"""Layer 0 — training datasets.

Carrier-agnostic dataset operators that yield ``(input, target)``
pairs and provide a stable content hash.

Classes in this module:

- `TrainingDataset` — abstract `Operator` base. Defines the
  ``__iter__`` + ``content_hash`` + ``with_split`` contract.
- `IterableDataset` — generic escape hatch wrapping any iterable
  with an explicit content hash. Streaming, no random access.
- `SimulationDataset` — wraps a ``pipekit_cycle.ForwardModel`` + a
  prior operator to generate ``(params, simulator_output)`` pairs.
  Supports both single-step emulation and trajectory rollouts.
- `CachedDataset` — disk/fsspec-backed cache (zarr in v0.1).

See ``docs/design/api/datasets.md``.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterable, Iterator
from typing import Any, ClassVar, Literal, cast

from pipekit import Operator


Split = Literal["train", "val", "test"]


def _stable_json(obj: Any) -> str:
    """sha256-stable JSON serialisation with ``default=repr`` fallback."""
    return json.dumps(obj, sort_keys=True, default=repr)


def _hash_config(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(_stable_json(p).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


class TrainingDataset(Operator):
    """Abstract base — yields ``(input, target)`` pairs.

    Subclasses implement ``__iter__`` and ``content_hash``. The base
    provides:

    - `pipekit.Operator` semantics — get_config / dumps / loads.
    - `with_split` for deterministic train/val/test split derivation.
    - A default ``content_hash`` over ``(type, get_config())`` that
      subclasses can extend or override.

    The training surface is the **iterator**, not ``_apply``. Calling
    a `TrainingDataset` like ``dataset()`` is undefined and raises
    ``TypeError`` — datasets compose into `TrainingLoop` via the
    ``dataset=`` constructor argument, not via `Sequential`.

    Attributes:
        seed: PRNG seed used by sampling subclasses. Default ``0``.
        split: ``"train"`` (default), ``"val"``, or ``"test"``.
    """

    # Datasets serialise their config but the data source typically
    # references external state (catalog URI, forward model with closures).
    # Mark the class so YAML loaders can decide policy per subclass.
    forbid_in_yaml: ClassVar[bool] = False

    def __init__(self, seed: int = 0, split: Split = "train") -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(
                f"split must be one of 'train', 'val', 'test'; got {split!r}."
            )
        self.seed = int(seed)
        self.split = split

    def __iter__(self) -> Iterator[tuple[Any, Any]]:
        raise NotImplementedError(
            f"{type(self).__name__} must implement __iter__ to yield "
            "(input, target) pairs."
        )

    def _apply(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError(
            f"{type(self).__name__} is iterated, not called. Use "
            "`for x, y in dataset:` or pass it to `TrainingLoop(dataset=...)`."
        )

    def content_hash(self) -> str:
        """Stable identifier for this dataset's contents.

        Default folds ``(type, get_config())`` into a sha256. Subclasses
        with external state (catalog URIs, forward-model signatures)
        override to fold those in as well.
        """
        return _hash_config(type(self).__qualname__, self.get_config())

    def with_split(self, split: Split) -> TrainingDataset:
        """Return a shallow clone of this dataset with a different split.

        Same ``seed``; only the split label changes. Subclasses that
        carry mutable state should override.
        """
        clone = self.__class__.__new__(self.__class__)
        clone.__dict__.update(self.__dict__)
        clone.split = split
        return clone


class IterableDataset(TrainingDataset):
    """Generic escape hatch — wrap any iterable as a `TrainingDataset`.

    Useful when the data source doesn't fit the catalog / simulation
    patterns: an existing PyTorch dataset, a generator over remote
    files, a user-built sequence of synthetic batches.

    Args:
        source: Either an iterable of ``(input, target)`` pairs, or a
            zero-arg callable returning such an iterable (callable form
            is required when the iterable is single-pass — the trainer
            needs to iterate multiple epochs).
        content_hash: User-supplied content hash. Required, because we
            can't introspect ``source`` deterministically.
        seed: PRNG seed (unused by the base wrapper; surfaced for
            consistency with sampling subclasses).
        split: ``"train"`` (default), ``"val"``, or ``"test"``. The
            base wrapper does NOT partition; it's the user's
            responsibility to pass a split-specific source.

    Because the source is an opaque user closure, this class sets
    ``forbid_in_yaml = True`` — the YAML round-trip is debug-only.

    Equinox-adapter note: IterableDatasets aren't indexable, so the
    adapter falls back from Grain's `MapDataset` (indexed) to plain
    Python iteration with manual batching.
    """

    forbid_in_yaml: ClassVar[bool] = True
    __config_mixin_auto__ = False

    def __init__(
        self,
        source: Iterable[tuple[Any, Any]] | Callable[[], Iterable[tuple[Any, Any]]],
        content_hash: str,
        seed: int = 0,
        split: Split = "train",
    ) -> None:
        super().__init__(seed=seed, split=split)
        if not (callable(source) or hasattr(source, "__iter__")):
            raise TypeError(
                "IterableDataset.source must be an iterable or a zero-arg "
                f"callable returning one; got {type(source).__name__}."
            )
        if not isinstance(content_hash, str) or not content_hash:
            raise ValueError(
                "IterableDataset.content_hash must be a non-empty string — "
                "the framework can't derive a hash for opaque iterables."
            )
        self.source = source
        self._content_hash = content_hash

    def __iter__(self) -> Iterator[tuple[Any, Any]]:
        # ty can't narrow the union; cast both branches via Any.
        raw_source = cast(Any, self.source)
        source = raw_source() if callable(raw_source) else raw_source
        yield from source

    def content_hash(self) -> str:
        return _hash_config(
            "IterableDataset",
            {"hash": self._content_hash, "seed": self.seed, "split": self.split},
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "content_hash": self._content_hash,
            "seed": self.seed,
            "split": self.split,
        }


class SimulationDataset(TrainingDataset):
    """Yields ``(parameters, simulator_output)`` pairs.

    Used for emulator training and amortized inference. The dataset
    samples a prior, evaluates the forward model (single step or
    trajectory), and optionally pushes the output through an
    observation operator (for SBI where the network learns to invert
    *observations*, not raw model states).

    Args:
        forward_model: Anything satisfying
            ``pipekit_cycle.ForwardModel`` — exposes ``step(state, dt)``
            and ``dt``. Workspace-internal package; resolves through the
            ``[cycle]`` extra at runtime.
        prior: Zero-arg `Operator` returning a parameter sample.
        n_samples: Total samples per epoch.
        cycle: Optional ``pipekit_cycle.Cycle`` for trajectory rollouts.
            When ``None``, the dataset emits single-step pairs.
        obs_op: Optional ``pipekit_cycle.ObservationOperator``. When
            set, the output is passed through it before yielding
            (used for SBI training where the target is observed-space
            outputs rather than latent state).
        seed: PRNG seed for the prior. Each sample uses
            ``seed + split_offset + i`` so splits don't overlap.
        split: ``"train"``, ``"val"``, or ``"test"``. Partitions the
            seed space (80/10/10 by default).

    The content hash includes the forward-model ``state_signature``
    (when available) and ``dt``, the prior config, ``n_samples``,
    ``seed``, and ``split`` — anything that affects the data.
    """

    forbid_in_yaml: ClassVar[bool] = True  # forward_model often carries closures
    __config_mixin_auto__ = False

    # Split → (numerator_start, numerator_end) over a 10ths denominator
    # so the integer split sizes are exact (80 / 10 / 10).
    _SPLIT_TENTHS: ClassVar[dict[Split, tuple[int, int]]] = {
        "train": (0, 8),
        "val": (8, 9),
        "test": (9, 10),
    }

    def __init__(
        self,
        forward_model: Any,
        prior: Operator,
        n_samples: int,
        cycle: Operator | None = None,
        obs_op: Any | None = None,
        seed: int = 0,
        split: Split = "train",
    ) -> None:
        super().__init__(seed=seed, split=split)
        for name in ("step", "dt"):
            if not hasattr(forward_model, name):
                raise TypeError(
                    f"SimulationDataset.forward_model must satisfy "
                    f"pipekit_cycle.ForwardModel (missing `{name}`). Got "
                    f"{type(forward_model).__name__}."
                )
        if not isinstance(prior, Operator):
            raise TypeError(
                "SimulationDataset.prior must be a pipekit.Operator; got "
                f"{type(prior).__name__}."
            )
        if n_samples <= 0:
            raise ValueError(
                f"SimulationDataset.n_samples must be positive; got {n_samples}."
            )
        self.forward_model = forward_model
        self.prior = prior
        self.n_samples = int(n_samples)
        self.cycle = cycle
        self.obs_op = obs_op

    # --- iteration / indexing -------------------------------------------

    def __len__(self) -> int:
        start_t, end_t = self._SPLIT_TENTHS[cast(Split, self.split)]
        end = self.n_samples * end_t // 10
        start = self.n_samples * start_t // 10
        return end - start

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        n = len(self)
        if idx < 0:
            idx += n
        if not 0 <= idx < n:
            raise IndexError(idx)
        # Per-sample deterministic seed so splits don't overlap.
        sample_seed = self.seed + self._split_offset() + idx
        # The prior's call protocol: takes an optional seed kwarg if it
        # accepts one; otherwise just call.
        try:
            params = self.prior(seed=sample_seed)
        except TypeError:
            params = self.prior()
        if self.cycle is not None:
            output = self.cycle(params)
        else:
            output = self.forward_model.step(params, self.forward_model.dt)
        if self.obs_op is not None:
            output = self.obs_op(output)
        return params, output

    def __iter__(self) -> Iterator[tuple[Any, Any]]:
        for i in range(len(self)):
            yield self[i]

    def _split_offset(self) -> int:
        """Stable per-split offset into the seed space."""
        start_t, _ = self._SPLIT_TENTHS[cast(Split, self.split)]
        return self.n_samples * start_t // 10

    # --- hashing / config ----------------------------------------------

    def content_hash(self) -> str:
        fm_sig = getattr(self.forward_model, "state_signature", None)
        fm_dt = getattr(self.forward_model, "dt", None)
        return _hash_config(
            "SimulationDataset",
            {
                "forward_model": {
                    "class": type(self.forward_model).__qualname__,
                    "state_signature": repr(fm_sig),
                    "dt": fm_dt,
                },
                "prior": {
                    "class": type(self.prior).__qualname__,
                    "config": self.prior.get_config(),
                },
                "n_samples": self.n_samples,
                "cycle": (
                    type(self.cycle).__qualname__ if self.cycle is not None else None
                ),
                "obs_op": (
                    type(self.obs_op).__qualname__ if self.obs_op is not None else None
                ),
                "seed": self.seed,
                "split": self.split,
            },
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "forward_model": type(self.forward_model).__qualname__,
            "prior": type(self.prior).__qualname__,
            "n_samples": self.n_samples,
            "cycle": (
                type(self.cycle).__qualname__ if self.cycle is not None else None
            ),
            "obs_op": (
                type(self.obs_op).__qualname__ if self.obs_op is not None else None
            ),
            "seed": self.seed,
            "split": self.split,
        }


class CachedDataset(TrainingDataset):
    """Disk-backed cache around any `TrainingDataset`.

    First epoch hits the source and writes to ``cache_dir``. Subsequent
    epochs read from disk. Cache is keyed on ``source.content_hash()``
    so cache files are reused across runs as long as the source's
    config and seed haven't changed.

    Args:
        source: The wrapped `TrainingDataset`. Must be indexable
            (provide ``__len__`` + ``__getitem__``) — the cache writes
            into a fixed-shape zarr store, which requires up-front
            knowledge of the dataset length. `IterableDataset` is not
            cacheable.
        cache_dir: Local filesystem directory. Created on first write.
            v0.1 supports local paths only; fsspec URIs (``s3://``,
            ``gcs://``) are scheduled for v0.2 alongside an
            ``fsspec``-backed zarr store.
        format: Storage format. v0.1 ships ``"zarr"`` only;
            ``"parquet"`` and ``"tfrecord"`` raise NotImplementedError
            and are scheduled for v0.2.

    Raises:
        TypeError: if ``source`` doesn't provide ``__len__`` + ``__getitem__``.
        NotImplementedError: if ``format`` is not ``"zarr"`` in v0.1.
        ImportError: lazily, if ``zarr`` is not installed when the
            cache is first read or written.

    Notes:
        - The cache writes raw NumPy-compatible arrays. Non-array
          carriers (objects, ragged structures) must be converted
          upfront — `CachedDataset` is for array-shaped pairs.
        - The cache is content-addressed, not split-addressed. Calling
          ``cached.with_split("val")`` returns a `CachedDataset`
          wrapping ``source.with_split("val")``; the cache for that
          split lives at a different path because its content hash
          differs.
        - To force re-cache, delete the cache directory or change a
          source-config field.
    """

    forbid_in_yaml: ClassVar[bool] = True
    __config_mixin_auto__ = False

    def __init__(
        self,
        source: TrainingDataset,
        cache_dir: str | os.PathLike[str],
        format: Literal["zarr", "parquet", "tfrecord"] = "zarr",
    ) -> None:
        if not isinstance(source, TrainingDataset):
            raise TypeError(
                "CachedDataset.source must be a TrainingDataset; got "
                f"{type(source).__name__}."
            )
        if not (hasattr(source, "__len__") and hasattr(source, "__getitem__")):
            raise TypeError(
                "CachedDataset.source must be indexable (provide __len__ + "
                f"__getitem__); got {type(source).__name__}. Wrap streaming "
                "sources differently or materialise them first."
            )
        if format != "zarr":
            raise NotImplementedError(
                f"CachedDataset format={format!r} is scheduled for v0.2. "
                "Use format='zarr' for v0.1; see docs/design/api/datasets.md."
            )
        super().__init__(seed=source.seed, split=cast(Split, source.split))
        self.source = source
        self.cache_dir = str(cache_dir)
        self.format: Literal["zarr", "parquet", "tfrecord"] = format

    def with_split(self, split: Split) -> CachedDataset:
        return CachedDataset(
            source=self.source.with_split(split),
            cache_dir=self.cache_dir,
            format=self.format,
        )

    def content_hash(self) -> str:
        # Inherits source.content_hash() — the cache is a materialisation
        # of the same logical contents.
        return self.source.content_hash()

    # --- cache layout --------------------------------------------------

    def _cache_path(self) -> str:
        return os.path.join(self.cache_dir, self.content_hash())

    def _materialised(self) -> bool:
        """Check whether the cache has already been written."""
        try:
            import zarr
        except ImportError:
            return False
        path = self._cache_path()
        # zarr's open auto-detects existing stores.
        try:
            import zarr

            root = zarr.open(path, mode="r")
        except (FileNotFoundError, KeyError, ValueError):
            return False
        return "x" in root and "y" in root

    def _materialise(self) -> None:
        """Write the source into the zarr cache (first epoch)."""
        import numpy as np
        import zarr

        # `self.source` is typed as TrainingDataset; the indexability
        # guard in __init__ means it also implements __len__/__getitem__.
        # ty can't combine the runtime guard with the static type, so
        # we cast at the boundary.
        source = cast(Any, self.source)
        n = len(source)
        if n == 0:
            raise ValueError(
                "CachedDataset cannot materialise an empty source — "
                "we need at least one sample to infer the array "
                "shapes / dtypes for the zarr store. Check the source "
                "dataset's split sizes or content_hash."
            )
        first_x, first_y = source[0]
        first_x = np.asarray(first_x)
        first_y = np.asarray(first_y)

        path = self._cache_path()
        os.makedirs(path, exist_ok=True)
        root = cast(Any, zarr.open(path, mode="w"))
        x_arr = root.create_array("x", shape=(n, *first_x.shape), dtype=first_x.dtype)
        y_arr = root.create_array("y", shape=(n, *first_y.shape), dtype=first_y.dtype)
        x_arr[0] = first_x
        y_arr[0] = first_y
        for i in range(1, n):
            xi, yi = source[i]
            x_arr[i] = np.asarray(xi)
            y_arr[i] = np.asarray(yi)

    def _open_cache(self) -> tuple[Any, Any]:
        import zarr

        root = cast(Any, zarr.open(self._cache_path(), mode="r"))
        return root["x"], root["y"]

    # --- iteration / indexing -----------------------------------------

    def __len__(self) -> int:
        return len(cast(Any, self.source))

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        if not self._materialised():
            self._materialise()
        x_arr, y_arr = self._open_cache()
        return x_arr[idx], y_arr[idx]

    def __iter__(self) -> Iterator[tuple[Any, Any]]:
        if not self._materialised():
            self._materialise()
        x_arr, y_arr = self._open_cache()
        n = x_arr.shape[0]
        for i in range(n):
            yield x_arr[i], y_arr[i]

    # --- explicit invalidation ----------------------------------------

    def invalidate(self) -> None:
        """Delete the cache directory (for the current content hash).

        After calling this, the next iteration re-materialises from
        the source. Useful when the cache becomes corrupt or when
        you want to force a re-bake without changing source config.
        """
        import shutil

        path = self._cache_path()
        if os.path.exists(path):
            shutil.rmtree(path)

    def get_config(self) -> dict[str, Any]:
        return {
            "source": type(self.source).__qualname__,
            "source_hash": self.source.content_hash(),
            "cache_dir": self.cache_dir,
            "format": self.format,
        }
