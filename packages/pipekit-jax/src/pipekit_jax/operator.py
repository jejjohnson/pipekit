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
from pipekit._base.operator import _is_json_primitive

from pipekit_jax.weights import WeightsMixin


def jsonable(value: Any) -> Any:
    """Coerce a config value to JSON builtins.

    Arrays (JAX or NumPy) become a Python scalar (0-d) or nested lists;
    tuples (including named tuples) become plain tuples and lists plain
    lists, converted recursively — JSON drops the subclass anyway; dicts
    are converted recursively; everything else is returned unchanged.
    """
    if isinstance(value, np.ndarray | np.generic) or eqx.is_array(value):
        arr = np.asarray(value)
        return arr.item() if arr.ndim == 0 else arr.tolist()
    if isinstance(value, tuple):
        return tuple(jsonable(v) for v in value)
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    return value


def _is_nested_operator(value: Any) -> bool:
    """A ``{"class", "config"}`` payload (or a list of them) for a nested operator."""
    if isinstance(value, list | tuple):
        return bool(value) and all(_is_nested_operator(v) for v in value)
    return isinstance(value, dict) and set(value) == {"class", "config"}


def _json_safe(value: Any) -> bool:
    """Whether ``json.dumps`` can encode ``value`` (str-keyed dicts allowed)."""
    if value is None or isinstance(value, bool | int | float | str):
        return True
    if isinstance(value, list | tuple):
        return all(_json_safe(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _json_safe(v) for k, v in value.items())
    return False


def _round_trips(value: Any) -> bool:
    """Whether a config value is acceptable in a JSON ``state`` record.

    JSON primitives (and lists / tuples of them) round-trip through
    ``Operator.from_state``. Nested-operator payloads are pipekit's
    documented debug form for containers (``Sequential``, ``Cycle``, …):
    they are rebuilt by a YAML / Hydra loader, not ``from_state``, so
    they are allowed — provided everything inside them is JSON-encodable
    (a nested complex array or callable is not).
    """
    return _is_json_primitive(value) or (
        _is_nested_operator(value) and _json_safe(value)
    )


class JaxOperator(Operator, WeightsMixin, eqx.Module):
    """Base class for pipekit operators whose fields are Equinox fields.

    Subclasses declare fields as on any ``eqx.Module`` and implement
    ``_apply``. ``Operator`` comes first in the MRO, so pipekit's
    dual-mode ``__call__`` is the one that runs (Equinox defines no
    ``__call__``).

    - **Config**: ``get_config()`` is auto-derived from the generated
      ``__init__`` and coerced to JSON builtins (`jsonable`), so
      ``op.state`` survives ``json.dumps`` and ``Operator.from_state``.
      Real numbers, strings, booleans and (nested) lists / tuples of
      them round-trip; ``op.state`` raises ``TypeError`` for complex
      values and plain dicts, which ``from_state`` cannot rebuild.
      Nested-operator payloads (``{"class", "config"}``) are allowed —
      pipekit's debug form for operators that hold operators. Operators
      holding those, or non-array Python objects (callables, modules
      without a JSON form), should set ``forbid_in_yaml = True`` and
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

    @property
    def state(self) -> dict[str, Any]:
        """JSON-serialisable state record, validated for a round trip.

        Raises:
            TypeError: if a config value cannot survive ``json.dumps`` →
                ``Operator.from_state`` (complex numbers, dicts) and the
                class is not marked ``forbid_in_yaml``.
        """
        record = super().state
        if not type(self).forbid_in_yaml:
            bad = [k for k, v in record["config"].items() if not _round_trips(v)]
            if bad:
                raise TypeError(
                    f"{type(self).__name__}: config field(s) {bad} cannot round-trip "
                    "through JSON and Operator.from_state (complex values and "
                    "dicts are not supported). Store them as real lists / tuples "
                    "of pairs, or set `forbid_in_yaml = True` and persist arrays "
                    "with serialize_weights()."
                )
        return record
