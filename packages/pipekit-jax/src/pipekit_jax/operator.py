"""`JaxOperator` — a pipekit `Operator` that is also an `eqx.Module`.

Domain libraries (finitevolX, spectraldiffx, somax, …) need operators
that are both:

- **pipekit operators** — dual-mode ``__call__`` (eager / graph node),
  ``|`` composition, and a JSON ``state`` / ``from_state`` round trip;
- **Equinox modules** — array fields are pytree leaves, so the operator
  works under ``jax.jit`` / ``jax.grad`` / ``jax.vmap``, and config-only
  fields are ``eqx.field(static=True)``.

The two compose because Equinox generates a dataclass ``__init__`` from
the fields, and that is exactly the signature pipekit's ``ConfigMixin``
reads to build ``get_config()``. `JaxOperator` makes the combination
explicit, coerces array fields to JSON in ``get_config()``, and adds the
`WeightsMixin` byte round trip.

Writing one::

    class Diffuse(JaxOperator):
        '''u ↦ u + κ ∇²u on a periodic 1-D grid (κ learnable).'''

        dx: float = eqx.field(static=True)        # config, not a leaf
        kappa: jax.Array = eqx.field(converter=jnp.asarray)   # leaf

        def _apply(self, u):
            lap = (jnp.roll(u, -1) - 2 * u + jnp.roll(u, 1)) / self.dx**2
            return u + self.kappa * lap

Learnable fields: give them ``converter=jnp.asarray``. A JSON reload
passes a Python float back into the constructor, and a Python float is
not an inexact array — without the converter the reloaded parameter is
silently treated as a constant by ``eqx.filter_grad``.
`pipekit_jax.testing.check_jax_operator` catches this.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import numpy as np
from pipekit import Operator

from pipekit_jax.weights import WeightsMixin


def jsonable(value: Any) -> Any:
    """Coerce a config value to JSON builtins.

    Arrays (JAX or NumPy) become a Python scalar (0-d) or nested lists;
    tuples / lists / dicts are converted recursively; everything else is
    returned unchanged.
    """
    if isinstance(value, np.ndarray | np.generic) or eqx.is_array(value):
        arr = np.asarray(value)
        return arr.item() if arr.ndim == 0 else arr.tolist()
    if isinstance(value, tuple | list):
        return type(value)(jsonable(v) for v in value)
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    return value


class JaxOperator(Operator, WeightsMixin, eqx.Module):
    """Base class for pipekit operators whose fields are Equinox fields.

    Subclasses declare fields as on any ``eqx.Module`` and implement
    ``_apply``. ``Operator`` comes first in the MRO, so pipekit's
    dual-mode ``__call__`` is the one that runs (Equinox defines no
    ``__call__``).

    - **Config**: ``get_config()`` is auto-derived from the generated
      ``__init__`` and coerced to JSON builtins (`jsonable`), so
      ``op.state`` survives ``json.dumps`` and ``Operator.from_state``.
      Operators holding non-array Python objects (callables, modules
      without a JSON form) should set ``forbid_in_yaml = True`` and
      persist through `serialize_weights` / `with_weights`.
    - **Leaves**: inexact-array fields are differentiable; mark
      config-only fields ``eqx.field(static=True)``; give learnable
      fields ``converter=jnp.asarray`` (see the module docstring).
    - **Weights**: `serialize_weights` / `with_weights` /
      `from_registry` persist the operator's own array leaves.
    """

    def get_config(self) -> dict[str, Any]:
        """JSON-builtin constructor config (arrays → scalars / lists)."""
        return {k: jsonable(v) for k, v in super().get_config().items()}
