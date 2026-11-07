---
status: draft
version: 0.1.0
---

# Inference: `ModelOp`

> **Status: not built** (phase D of `boundaries.md` §4).
> `pipekit_array.inference` does not exist in pipekit-array 0.1.1, so every call to `ModelOp` below is `proposed`.
> Two shipped pieces cover part of the ground: `BatchAlong` (`pipekit_array.batching`) splits a carrier for chunked inference, and `Estimator` (`pipekit_array.estimator`, `[sklearn]`) fits and applies a scikit-learn-style estimator on a named carrier.

`ModelOp` is the framework-agnostic trained-model wrapper.
It holds a callable (a NumPy function, an sklearn `predict`, a raw torch `nn.Module`, a Keras model, …) and calls it on the input.

```python
# proposed (phase D)
from pipekit_array.inference import ModelOp

# Wrap a Keras model; method defaults to "__call__".
op = ModelOp(model=keras_model)
y = op(x)  # x: a NumPy, torch or JAX array, as the model expects

# Wrap an sklearn estimator's predict method.
op = ModelOp(model=sklearn_clf, method="predict")
y_hat = op(x_test)  # (N, D) → (N,)
```

## Why this exists

There are three reasons:

1. **It closes the train-to-serve loop for pipekit-train.**
   The Lightning and Keras adapters (v0.2 / v0.3 of pipekit-train; see `docs/design/api/adapters.md` in that package) need to return a `pipekit_ops.Operator` from `loop.run()` that wraps the trained `nn.Module` or `keras.Model`.
   That operator is `ModelOp`.
2. **It is the natural home for sklearn and NumPy-ML pipelines.**
   A lot of working code wraps a fitted `RandomForestClassifier` or a trained sklearn pipeline, and `ModelOp` is the one-line shim that makes them compose into a pipekit `Sequential`.
3. **It generalises `pipekit_array.BatchAlong`'s `inner` argument.**
   `BatchAlong(ModelOp(big_model), size=32, dim="sample")` is the canonical "stream a large array through a model in chunks" idiom.

The split with `pipekit_jax.JaxModelOp` is ADR A8: the Equinox weight-blob round trip lives in pipekit-jax (which carries the Orbax dependency), and pipekit-array's `ModelOp` stays generic and Orbax-free.

## Contract

### Constructor

```python
ModelOp(
    model: Any,                         # the trained model object
    method: str = "__call__",           # method to invoke on `model`
    batch_size: int | None = None,      # if set, batch automatically
    method_kwargs: dict | None = None,  # forwarded to model.<method>(x, **kwargs)
)
```

**`model`**: any callable or object with the named `method`.
There is no isinstance check, and the duck typing is intentional.
If `getattr(model, method)` is not callable at construction time, it raises `TypeError` immediately.

**`method`**: the name of the method to call.
It defaults to `"__call__"`, which is the right shape for raw Keras, PyTorch and Equinox modules; sklearn estimators use `"predict"`, `"predict_proba"` or `"transform"`.

**`batch_size`**: if set, the model call is wrapped in `BatchAlong(self._inner_op, size=batch_size, dim=...)` internally.
It is a convenience shorthand, equivalent to `BatchAlong(ModelOp(model, method), size=..., dim=...)`.
`BatchAlong` cuts a named dim of a named carrier, so the design still has to say which dim, or how a raw array maps to one.

**`method_kwargs`**: passed through on every call.
They are useful for models with `training=False` flags, dropout suppression and the like.
Keys must be strings and values must be JSON-serialisable, because they are re-emitted in `get_config()`.

### `_apply(x)`

```python
def _apply(self, x):
    fn = getattr(self.model, self.method)
    return fn(x, **(self.method_kwargs or {}))
```

If `batch_size` is set, it is the equivalent batched call.
There is no Array API dispatch, because the model owns the backend; `ModelOp`'s job is plumbing, not numerics.

### `get_config()`

```python
{
    # The model object is the artifact. We don't serialise it; the
    # config records the call shape only.
    "model": {
        "class": type(self.model).__name__,
        "module": type(self.model).__module__,
        # Optionally a content hash if the model exposes one (sklearn
        # doesn't; a PyTorch state_dict can be hashed; this is opt-in
        # per backend).
    },
    "method": self.method,
    "batch_size": self.batch_size,
    "method_kwargs": dict(self.method_kwargs or {}),
}
```

Note: the *trained weights* are not in the config.
`ModelOp` round-trips the *call shape*, and the weights are the user's artifact to keep out of band.
For Equinox, use `pipekit_jax.JaxModelOp`, which *does* round-trip weights through `pipekit-experiment.ModelRegistry`.

### `compute_output_signature(sig)`

It returns `None`, because the operator can't know the model's output shape without running it, so `Sequential.summary` prints `?` for the output.
Override it per backend in user code if shape inference matters (a Keras model can report its own via `model.compute_output_shape`).

### `forbid_in_yaml`

```python
forbid_in_yaml: ClassVar[bool] = True
```

The model itself isn't YAML-round-trippable (the trained weights are the artifact, not part of the config).
The `check_pickleable` lint catches `ModelOp` instances in `ProcessMap`-deployable pipelines and reminds the user to make sure the model pickles.

## Use cases

### Sklearn classifier in a pipekit pipeline

```python
# proposed (phase D)
from pipekit_array.inference import ModelOp

# proposed (phase C)
from pipekit_array.qc import AssertNoNaN

pipe = AssertNoNaN() | ModelOp(rf_classifier, method="predict_proba")
probs = pipe(X_test)  # (n_samples, n_features) → (n_samples, n_classes)
```

On a named carrier, `Estimator(rf_classifier, sample_dims=..., method="predict_proba")` already does this today, fitting by dim name (`pipekit_array.estimator`).

### Keras model wrapped after pipekit-train (planned for v0.3 of pipekit-train)

```python
# Inside pipekit_train.adapters.keras.run():
def run(loop):
    keras_model, history = _train(loop)
    # proposed (phase D)
    return ModelOp(model=keras_model), TrainingArtifact(...)
```

### Large-input batched inference

```python
# proposed (phase D)
op = ModelOp(big_torch_model, method="forward", batch_size=64)
preds = op(huge_input_array)  # batches internally
```

This is equivalent to:

```python
# today (pipekit-array): BatchAlong; proposed (phase D): ModelOp
op = BatchAlong(ModelOp(big_torch_model, method="forward"), size=64, dim="sample")
```

The `batch_size=` constructor shorthand is purely for ergonomics; under the hood it is the explicit `BatchAlong` composition.

## Per-backend notes

- **PyTorch:** `model.eval()` and `torch.no_grad()` are *not* automatic.
  The user is responsible for putting the model in eval mode and disabling autograd if they want inference semantics.
  The docstring should say so, and cite a one-line `eval` Tap as the pattern.
- **Keras 3:** `model(x, training=False)` is the inference-time signature, so pass `method_kwargs={"training": False}`.
- **sklearn:** `method="predict"` for the dense prediction, `"predict_proba"` for the probabilistic one and `"transform"` for transformers.
  The model must be fitted before `ModelOp` is constructed; we don't check, and sklearn raises `NotFittedError` on the first call.
- **JAX (`jax.numpy` arrays):** raw JAX functions work (`ModelOp(model=lambda x: ...)`); for an `eqx.Module`, use `pipekit_jax.JaxModelOp` instead, so the weights round-trip.

## Failure modes

| Symptom | Likely cause |
| --- | --- |
| `TypeError: model.<method> is not callable` | wrong `method` name; check `dir(model)` |
| `NotFittedError` (sklearn) | model not fitted before wrapping |
| `RuntimeError: leaf Variable that requires grad ...` | torch model not in `eval()` + `no_grad()` |
| Pipeline can't pickle for `ProcessMap` | model not pickleable; switch to `ThreadMap` or pin |

All four should be documented in the operator's docstring, each with its one-line fix.
