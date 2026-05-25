---
status: draft
version: 0.1.0
---

# Inference — `ModelOp`

`ModelOp` is the framework-agnostic trained-model wrapper. It holds
a callable (numpy function, sklearn `predict`, raw torch
`nn.Module`, Keras model, …) and calls it on the input.

```python
from pipekit_array.inference import ModelOp


# Wrap a Keras model:
op = ModelOp(model=keras_model)  # method defaults to "__call__"
y = op(x)  # x: numpy / torch / jax array per the model's expectations

# Wrap an sklearn estimator's predict method:
op = ModelOp(model=sklearn_clf, method="predict")
y_hat = op(x_test)
```

## Why this exists

Three reasons:

1. **It closes the train→serve loop for pipekit-train.** The Lightning
   and Keras adapters (v0.2 / v0.3 of pipekit-train, see
   `docs/design/api/adapters.md` in that package) need to return a
   `pipekit.Operator` from `loop.run()` that wraps the trained
   `nn.Module` / `keras.Model`. That operator is `ModelOp`.

2. **It's the natural home for sklearn / numpy-ML pipelines.** A lot
   of working code wraps a fitted `RandomForestClassifier` or a
   trained sklearn pipeline. `ModelOp` is the one-line shim that
   makes them compose into a pipekit `Sequential`.

3. **It generalises `pipekit-array.parallel.BatchedMap`'s `inner`
   argument.** `BatchedMap(ModelOp(big_model), batch_size=32)` is
   the canonical "stream a large array through a model in chunks"
   idiom.

The split with `pipekit_jax.JaxModelOp` is in ADR A8 — Equinox
weight-blob round-trip lives in pipekit-jax (which carries the Orbax
dep); pipekit-array's `ModelOp` stays generic and Orbax-free.

## Contract

### Constructor

```python
ModelOp(
    model: Any,                       # the trained model object
    method: str = "__call__",         # method to invoke on `model`
    batch_size: int | None = None,    # if set, batch automatically
    method_kwargs: dict | None = None,  # forwarded to model.<method>(x, **kwargs)
)
```

**`model`**: any callable / object with the named `method`. No
isinstance check — the duck-typing is intentional. If `getattr(model,
method)` is not callable at construction time, raise `TypeError`
immediately.

**`method`**: name of the method to call. Defaults to `"__call__"`,
which is the right shape for raw Keras / PyTorch / equinox modules.
sklearn estimators use `"predict"` / `"predict_proba"` / `"transform"`.

**`batch_size`**: if set, wraps the actual model call in
`BatchedMap(self._inner_op, batch_size=batch_size, axis=0)`
internally. Convenience shorthand; equivalent to
`BatchedMap(ModelOp(model, method), batch_size=...)`.

**`method_kwargs`**: passed through on every call. Useful for
models with `training=False` flags, dropout suppression, etc.
Keys must be strings; values must be JSON-serialisable (we
re-emit them in `get_config()`).

### `_apply(x)`

```python
def _apply(self, x):
    fn = getattr(self.model, self.method)
    return fn(x, **(self.method_kwargs or {}))
```

If `batch_size` is set, the equivalent batched call. No Array API
dispatch — the model itself owns the backend. `ModelOp`'s job is
plumbing, not numerics.

### `get_config()`

```python
{
    # The model object is the artifact. We don't serialise it; the
    # config records the call shape only.
    "model": {
        "class": type(self.model).__name__,
        "module": type(self.model).__module__,
        # Optionally a content hash if model exposes one (sklearn
        # doesn't; PyTorch state_dict can be hashed; this is opt-in
        # per backend).
    },
    "method": self.method,
    "batch_size": self.batch_size,
    "method_kwargs": dict(self.method_kwargs or {}),
}
```

Note: the *trained weights* are not in the config. `ModelOp` round-
trips the *call shape*; the weights are the user's artifact to
preserve out-of-band. (For Equinox, use `pipekit_jax.JaxModelOp`,
which *does* round-trip weights through `pipekit-experiment.ModelRegistry`.)

### `compute_output_signature(sig)`

Returns `None` — the operator can't know the model's output shape
without running it. `Sequential.summary` will print `?` for the
output. Override per-backend in user code if shape inference matters
(e.g. a Keras model can self-report via `model.compute_output_shape`).

### `forbid_in_yaml`

```python
forbid_in_yaml: ClassVar[bool] = True
```

Because the model itself isn't YAML-round-trippable (the trained
weights are the artifact, not part of the config). The
`check_pickleable` lint catches `ModelOp` instances in
`ProcessMap`-deployable pipelines and reminds the user to ensure the
model is pickleable.

## Use cases

### Sklearn classifier in a pipekit pipeline

```python
from pipekit_array.inference import ModelOp
from pipekit_array.qc import AssertNoNaN


pipe = AssertNoNaN() | ModelOp(rf_classifier, method="predict_proba")
probs = pipe(X_test)  # shape (n_samples, n_classes)
```

### Keras model wrapped after pipekit-train (planned v0.3 of pipekit-train)

```python
# Inside pipekit_train.adapters.keras.run():
def run(loop):
    keras_model, history = _train(loop)
    return ModelOp(model=keras_model), TrainingArtifact(...)
```

### Large-input batched inference

```python
op = ModelOp(big_torch_model, method="forward", batch_size=64)
preds = op(huge_input_array)  # auto-batches internally
```

Equivalent to:

```python
op = BatchedMap(ModelOp(big_torch_model, method="forward"), batch_size=64)
```

The `batch_size=` constructor shorthand is purely for ergonomics;
under the hood it's the explicit `BatchedMap` composition.

## Per-backend notes

- **PyTorch:** `model.eval()` and `torch.no_grad()` are *not*
  automatic. The user is responsible for putting the model in
  eval mode and disabling autograd if they want inference semantics.
  Document this in the docstring; cite a one-liner `eval` Tap as the
  pattern.
- **Keras 3:** `model(x, training=False)` is the inference-time
  signature; pass `method_kwargs={"training": False}`.
- **sklearn:** `method="predict"` for the dense prediction;
  `"predict_proba"` for probabilistic; `"transform"` for transformers.
  The model must be `.fit`-ed before `ModelOp` is constructed (we
  don't check; sklearn raises `NotFittedError` on the first call).
- **JAX (`jax.numpy` arrays):** raw JAX functions work
  (`ModelOp(model=lambda x: ...)`); for `eqx.Module`, use
  `pipekit_jax.JaxModelOp` instead so weights round-trip.

## Failure modes

| Symptom                                               | Likely cause                                       |
| ----------------------------------------------------- | -------------------------------------------------- |
| `TypeError: model.<method> is not callable`           | wrong `method` name; check `dir(model)`            |
| `NotFittedError` (sklearn)                            | model not `.fit`-ed before wrapping                |
| `RuntimeError: leaf Variable that requires grad ...` | torch model not in `eval()` + `no_grad()`           |
| Pipeline can't pickle for `ProcessMap`                | model not pickleable; switch to `ThreadMap` or pin |

All four are documented in the operator's docstring with the
single-line fix.
