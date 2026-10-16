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

Compiling: use ``eqx.filter_jit(op)`` (or ``jax.jit`` on a function that
takes ``op`` as an argument). ``jax.jit(op)`` hashes the callable, and an
``eqx.Module`` with array fields is unhashable.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import equinox as eqx
import numpy as np
from pipekit import Operator

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


def _json_safe(value: Any) -> bool:
    """Whether ``json.dumps`` can encode ``value`` (str-keyed dicts allowed)."""
    if value is None or isinstance(value, bool | int | float | str):
        return True
    if isinstance(value, list | tuple):
        return all(_json_safe(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _json_safe(v) for k, v in value.items())
    return False


def _debug_repr(value: Any) -> Any:
    """A JSON-safe stand-in for a config value with no JSON form."""
    if isinstance(value, list | tuple):
        return [v if _json_safe(v) else _debug_repr(v) for v in value]
    if isinstance(value, dict):
        return {
            str(k): v if _json_safe(v) else _debug_repr(v) for k, v in value.items()
        }
    if isinstance(value, complex):
        return f"<complex {value!r}>"
    if hasattr(value, "__qualname__"):
        return f"<{getattr(value, '__module__', '?')}.{value.__qualname__}>"
    return f"<{type(value).__module__}.{type(value).__qualname__}>"


def _has_legacy_payload(value: Any) -> bool:
    """Whether ``value`` holds a dict shaped ``{"class", "config"}``.

    ``Operator.from_state`` refuses that shape: it is the nested-operator
    payload written before nested round-trips existed (nested operators
    are now full ``{"module", "class", "config"}`` records), so an
    ordinary option dict that happens to use those two keys cannot
    round-trip either.
    """
    if isinstance(value, list | tuple):
        return any(_has_legacy_payload(v) for v in value)
    if not isinstance(value, dict):
        return False
    if set(value) == {"class", "config"}:
        return True
    return any(_has_legacy_payload(v) for v in value.values())


def _round_trips(value: Any) -> bool:
    """Whether a config value survives ``json.dumps`` → ``Operator.from_state``.

    Anything JSON-encodable does: primitives, lists, str-keyed dicts and
    nested operators' ``state`` records, which ``from_state`` rebuilds
    recursively. Complex values and callables do not, and neither does a
    dict shaped like a pre-round-trip nested payload (`_has_legacy_payload`).
    """
    return _json_safe(value) and not _has_legacy_payload(value)


class JaxOperator(Operator, WeightsMixin, eqx.Module):
    """Base class for pipekit operators whose fields are Equinox fields.

    Subclasses declare fields as on any ``eqx.Module`` and implement
    ``_apply``. ``Operator`` comes first in the MRO, so pipekit's
    dual-mode ``__call__`` is the one that runs (Equinox defines no
    ``__call__``).

    - **Config**: ``get_config()`` is auto-derived from the generated
      ``__init__`` and coerced to JSON builtins (`jsonable`), so
      ``op.state`` survives ``json.dumps`` and ``Operator.from_state``.
      Real numbers, strings, booleans, (nested) lists / tuples and
      str-keyed dicts of them round-trip, and so do nested operators
      (their ``state`` records, which ``from_state`` rebuilds);
      ``op.state`` raises ``TypeError`` for complex values, and for
      array fields whose converter would not restore their dtype (JSON
      drops dtypes: a ``float16`` field needs a casting converter).
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

    @property
    def state(self) -> dict[str, Any]:
        """JSON-serialisable state record, validated for a round trip.

        Raises:
            TypeError: if a config value cannot survive ``json.dumps`` →
                ``Operator.from_state`` (complex numbers, dicts) and the
                class is not marked ``forbid_in_yaml``.
        """
        record = super().state
        if type(self).forbid_in_yaml:
            # Not round-trippable by design, but still JSON-encodable so a
            # registry can store the record next to the weight blob.
            record["config"] = {
                k: v if _json_safe(v) else _debug_repr(v)
                for k, v in record["config"].items()
            }
            return record
        if not type(self).forbid_in_yaml:
            bad = [k for k, v in record["config"].items() if not _round_trips(v)]
            if bad:
                raise TypeError(
                    f"{type(self).__name__}: config field(s) {bad} cannot round-trip "
                    "through JSON and Operator.from_state (complex values, and "
                    "dicts keyed exactly {'class', 'config'}, are not supported). "
                    "Store them as real numbers or lists of pairs, or set "
                    "`forbid_in_yaml = True` and persist arrays with "
                    "serialize_weights()."
                )
            lossy = self._dtype_lossy_fields()
            if lossy:
                raise TypeError(
                    f"{type(self).__name__}: array field(s) {lossy} would reload "
                    "with a different dtype: JSON drops dtypes and the field's "
                    "converter does not restore it. Give the field a casting "
                    "converter, e.g. eqx.field(converter=lambda x: "
                    "jnp.asarray(x, jnp.float16)), or set `forbid_in_yaml = True` "
                    "and persist arrays with serialize_weights()."
                )
        return record

    def _dtype_lossy_fields(self) -> list[str]:
        """Array config fields whose JSON round trip changes their dtype.

        Rebuilds each array field the way ``from_state`` does (its JSON
        value passed through the field's converter) and compares dtypes.
        A field without a converter reloads as a Python scalar / list;
        `check_jax_operator` reports that case as a lost learnable leaf.
        """
        lossy = []
        for f in dataclasses.fields(self):
            converter = f.metadata.get("converter")
            if not f.init or converter is None:
                continue
            value = getattr(self, f.name)
            if not eqx.is_array(value):
                continue
            rebuilt = converter(jsonable(value))
            if not eqx.is_array(rebuilt) or rebuilt.dtype != value.dtype:
                lossy.append(f.name)
        return lossy
