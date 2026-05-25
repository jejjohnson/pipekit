# pipekit-jax

JAX / Equinox carrier integration for `pipekit`. Ships one operator:

- **`JaxModelOp`** — wraps an `eqx.Module` as a `pipekit.Operator`,
  with `serialize_weights` / `with_weights` methods so trained
  weights round-trip byte-identically through any
  `pipekit_experiment.ModelRegistry` via the registry's `weights`
  blob.

`JaxModelOp` is the public successor to the in-package
`pipekit_train.adapters.equinox.EquinoxModelOp` stand-in that the
v0.1 `pipekit-train` release shipped with — same constructor
signature, drop-in swap. Once `pipekit-jax` is installed,
`Checkpoint(registry=…)` will store the full weight blob alongside
the operator state, closing the v0.1 reproducibility caveat
documented in `pipekit-train/docs/design/boundaries.md §13.1`.

## Quick start

```python
import equinox as eqx
import jax
from pipekit_jax import JaxModelOp

mlp = eqx.nn.MLP(in_size=2, out_size=2, width_size=32, depth=2, key=jax.random.key(0))
op = JaxModelOp(mlp)

# Serialise weights for storage:
weights = op.serialize_weights()       # bytes — eqx.tree_serialise_leaves

# Reload weights into a fresh skeleton:
fresh = JaxModelOp(eqx.nn.MLP(in_size=2, out_size=2, width_size=32, depth=2, key=jax.random.key(1)))
restored = fresh.with_weights(weights)
# `restored.module` carries the original weights, byte-identical.
```

## Registry round-trip

```python
from pipekit_experiment import LocalModelRegistry
from pipekit_jax import JaxModelOp

registry = LocalModelRegistry(root="./models")

# Store a trained model:
trained_op = JaxModelOp(trained_eqx_module)
hash_ = registry.store(trained_op, weights=trained_op.serialize_weights())

# Reload later — needs a template (same module structure, fresh
# weights) because eqx.Module structure isn't recoverable from JSON
# alone:
template = JaxModelOp(fresh_eqx_module_of_same_shape)
reloaded = template.from_registry(registry, hash_)
# reloaded.module has the byte-identical trained weights.
```

See [`docs/design/`](docs/design/) for the design + boundary notes.
