"""Dict plumbing — reshape the dicts that `Graph`, `Fanout` and STAC items carry.

Every reshape that used to be a `Lambda` (and so stopped a pipeline from
round-tripping) is a small operator here, named after its toolz function:

| Operator                     | Semantics                                        |
| ---------------------------- | ------------------------------------------------ |
| `Get(key, default)`          | ``x[key]``; a tuple of keys returns a tuple      |
| `GetIn(keys, default)`       | ``x[k₀][k₁]…``                                   |
| `Pick(keys)`                 | keep only ``keys``, in the order given           |
| `Dissoc(keys)`               | drop ``keys``; missing keys are ignored          |
| `Rename(mapping)`            | rename keys; unlisted keys pass through          |
| `Merge()`                    | union of several dicts, later inputs win         |
| `Assoc(key, op)`             | add ``key: op(x)``                               |
| `UpdateIn(keys, op, default)`| replace the value at a key path by ``op(value)`` |
| `ValMap(op)`                 | ``op`` on every value                            |

Keys are plain values (a `str`, an `int` or a tuple of them) and functions
are operators, so every operator here has a faithful config. A bare Python
callable is refused with a `TypeError` that points to `Lambda`. No operator
mutates its input: each dict it returns is a new dict, whose values are
shared with the input, not copied.

See the pipekit-ops design, ``api/operators.md`` section 1.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pipekit_ops._base.operator import (
    _MISSING,
    ConfigMixin,
    Operator,
    nested_config,
    require_operator,
)


__all__ = [
    "Assoc",
    "Dissoc",
    "Get",
    "GetIn",
    "Merge",
    "Pick",
    "Rename",
    "UpdateIn",
    "ValMap",
]

#: A plain key: one index into a container.
Key = str | int

#: What a missing key raises when a ``default`` stands in for it (toolz ``get``).
_LOOKUP_ERRORS = (KeyError, IndexError, TypeError)


class _ContainerOperator(Operator):
    """Shared plumbing for the dict and iterable operators.

    The config is the constructor's arguments, with child operators as
    their nested state and an omitted ``default`` left out. The output is
    a container, not one carrier, so its signature is not tracked.
    """

    def get_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {}
        for name, value in ConfigMixin.get_config(self).items():
            if value is _MISSING:
                continue
            if isinstance(value, Operator):
                value = nested_config(value)
            config[name] = value
        return config

    def compute_output_signature(self, input_signature: Any) -> Any:
        """``None``: the output is a container, not one carrier."""
        return None

    def __repr__(self) -> str:
        fields = ConfigMixin.get_config(self).items()
        params = ", ".join(f"{k}={v!r}" for k, v in fields if v is not _MISSING)
        return f"{type(self).__name__}({params})"


def require_child(value: Any, owner: str, field: str) -> None:
    """Raise ``TypeError`` unless ``value`` is an `Operator`; name `Lambda` if callable.

    Args:
        value: The candidate predicate, key or function.
        owner: The validating class's name (for the message).
        field: The parameter name.
    """
    if callable(value) and not isinstance(value, Operator):
        raise TypeError(
            f"{owner}.{field} must be an Operator, got {type(value).__name__}. "
            "Wrap a plain function in `Lambda(fn)` (it will not round-trip "
            "through dumps / loads)."
        )
    require_operator(value, owner, field)


def plain_key(value: Any, owner: str, field: str) -> Key | tuple[Key, ...]:
    """Check a plain key (a `str`, an `int` or a tuple of them); lists become tuples.

    A key reloads from JSON as a list, so lists are accepted and turned back
    into tuples: the rebuilt operator then picks several keys again.

    Args:
        value: The candidate key.
        owner: The validating class's name (for the message).
        field: The parameter name.

    Returns:
        The key, with a list turned into a tuple.

    Raises:
        TypeError: if ``value`` is anything else; a function is told to
            use `Lambda`.
    """
    if isinstance(value, str | int):
        return value
    if isinstance(value, tuple | list) and all(isinstance(k, str | int) for k in value):
        return tuple(value)
    hint = (
        " To call a function on the input, use `Lambda(fn)`." if callable(value) else ""
    )
    raise TypeError(
        f"{owner}.{field} must be a str, an int or a tuple of them, "
        f"got {type(value).__name__}.{hint}"
    )


def key_path(value: Any, owner: str, field: str) -> tuple[Key, ...]:
    """Check a key path (a tuple of `str` / `int`); a list becomes a tuple.

    Raises:
        TypeError: if ``value`` is not a tuple or list of them (a bare
            `str` is refused: it would walk its characters).
    """
    if isinstance(value, tuple | list) and all(isinstance(k, str | int) for k in value):
        return tuple(value)
    raise TypeError(
        f"{owner}.{field} must be a tuple of str / int keys, "
        f"got {type(value).__name__}."
    )


def lookup(key: Key | tuple[Key, ...] | Operator, item: Any) -> Any:
    """``key(item)`` for an operator, ``item[key]``, or a tuple for a tuple of keys."""
    if isinstance(key, Operator):
        return key(item)
    if isinstance(key, tuple):
        return tuple(item[k] for k in key)
    return item[key]


class Get(_ContainerOperator):
    """Index into a mapping or sequence (toolz ``get``).

    x ↦ x[k], or x ↦ (x[k₀], x[k₁], …) for a tuple of keys.

    Args:
        key: A `str` or `int`, or a tuple of them to pick several items.
        default: Returned in place of a missing item (a `KeyError`,
            `IndexError` or `TypeError` on lookup, as in toolz). Omit it to
            let the error propagate.

    Raises:
        TypeError: at construction, if ``key`` is not a plain key.

    Examples:
        >>> from pipekit_ops import Get
        >>> Get("id")({"id": "S2A_31TCJ", "cloud": 12.0})
        'S2A_31TCJ'
        >>> Get(("a", "z"), default=None)({"a": 1})
        (1, None)
        >>> Get(-1)([10, 20, 30])
        30
    """

    def __init__(self, key: Key | tuple[Key, ...], default: Any = _MISSING) -> None:
        self.key = plain_key(key, "Get", "key")
        self.default = default

    def _apply(self, x: Any) -> Any:
        if self.default is _MISSING:
            return lookup(self.key, x)
        if isinstance(self.key, tuple):
            return tuple(_get_or(x, k, self.default) for k in self.key)
        return _get_or(x, self.key, self.default)


def _get_or(x: Any, key: Key, default: Any) -> Any:
    """``x[key]``, or ``default`` if the lookup fails."""
    try:
        return x[key]
    except _LOOKUP_ERRORS:
        return default


class GetIn(_ContainerOperator):
    """Walk a key path through nested containers (toolz ``get_in``).

    ``x ↦ x[k₀][k₁]…[kₘ₋₁]``; the empty path returns x.

    Args:
        keys: The path, a tuple of `str` / `int` keys.
        default: Returned when the path is missing (a `KeyError`,
            `IndexError` or `TypeError` on the way, as in toolz). Omit it
            to let the error propagate.

    Raises:
        TypeError: at construction, if ``keys`` is not a tuple of keys.

    Examples:
        >>> from pipekit_ops import GetIn
        >>> item = {"properties": {"eo:cloud_cover": 12.5}}
        >>> GetIn(("properties", "eo:cloud_cover"))(item)
        12.5
        >>> GetIn(("properties", "s2:mgrs_tile"), default="?")(item)
        '?'
    """

    def __init__(self, keys: Sequence[Key], default: Any = _MISSING) -> None:
        self.keys = key_path(keys, "GetIn", "keys")
        self.default = default

    def _apply(self, x: Any) -> Any:
        try:
            for key in self.keys:
                x = x[key]
        except _LOOKUP_ERRORS:
            if self.default is _MISSING:
                raise
            return self.default
        return x


class Pick(_ContainerOperator):
    """Keep only some keys, in the order given.

    x ↦ {k: x[k] for k in keys}.

    Args:
        keys: The keys to keep, a tuple of `str` / `int`.

    Raises:
        TypeError: at construction, if ``keys`` is not a tuple of keys.
        KeyError: at call time, if a key is missing from the input.

    Examples:
        >>> from pipekit_ops import Pick
        >>> Pick(("b", "a"))({"a": 1, "b": 2, "c": 3})
        {'b': 2, 'a': 1}
        >>> Pick(("z",))({"a": 1})  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        KeyError: 'z'
    """

    def __init__(self, keys: Sequence[Key]) -> None:
        self.keys = key_path(keys, "Pick", "keys")

    def _apply(self, x: Mapping[Any, Any]) -> dict[Any, Any]:
        return {key: x[key] for key in self.keys}


class Dissoc(_ContainerOperator):
    """Drop some keys; missing keys are ignored (toolz ``dissoc``).

    x ↦ {k: v for k, v in x if k ∉ keys}.

    Args:
        keys: The keys to drop, a tuple of `str` / `int`.

    Raises:
        TypeError: at construction, if ``keys`` is not a tuple of keys.

    Examples:
        >>> from pipekit_ops import Dissoc
        >>> Dissoc(("scenes",))({"n_kept": 3, "scenes": [0, 1, 2]})
        {'n_kept': 3}
        >>> Dissoc(("missing",))({"a": 1})
        {'a': 1}
    """

    def __init__(self, keys: Sequence[Key]) -> None:
        self.keys = key_path(keys, "Dissoc", "keys")

    def _apply(self, x: Mapping[Any, Any]) -> dict[Any, Any]:
        dropped = set(self.keys)
        return {key: value for key, value in x.items() if key not in dropped}


class Rename(_ContainerOperator):
    """Rename keys; unlisted keys pass through and the order is kept.

    x ↦ {m(k): v for k, v in x}, with m(k) = mapping[k] if k is listed, else k.

    Args:
        mapping: Old key ↦ new key. Old keys are `str` (they are JSON object
            keys in the config); new keys are `str` or `int`, all distinct.
            An old key missing from the input is ignored.

    Raises:
        TypeError: at construction, if a key of ``mapping`` is not a `str`
            or a new key is not a `str` / `int`.
        ValueError: at construction, if two old keys share a new key; at
            call time, if a new key collides with a key the input keeps.

    Examples:
        >>> from pipekit_ops import Rename
        >>> Rename({"n_found": "found"})({"n_found": 300, "n_kept": 212})
        {'found': 300, 'n_kept': 212}
        >>> Rename({"a": "b"})({"a": 1, "b": 2})  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValueError: Rename: 'a' -> 'b' collides with the kept key 'b'.
    """

    def __init__(self, mapping: Mapping[str, Key]) -> None:
        for old, new in mapping.items():
            if not isinstance(old, str):
                raise TypeError(
                    f"Rename.mapping keys must be str (JSON object keys), "
                    f"got {type(old).__name__}."
                )
            if not isinstance(new, str | int):
                raise TypeError(
                    f"Rename.mapping[{old!r}] must be a str or an int, "
                    f"got {type(new).__name__}."
                )
        if len(set(mapping.values())) != len(mapping):
            raise ValueError(f"Rename.mapping sends two keys to one: {mapping!r}.")
        self.mapping = dict(mapping)

    def _apply(self, x: Mapping[Any, Any]) -> dict[Any, Any]:
        for old, new in self.mapping.items():
            # a target that is itself renamed away frees its slot (a swap)
            if old in x and new in x and new not in self.mapping:
                raise ValueError(
                    f"Rename: {old!r} -> {new!r} collides with the kept key {new!r}."
                )
        return {self.mapping.get(key, key): value for key, value in x.items()}


class Merge(_ContainerOperator):
    """Union of several mappings; later inputs win (toolz ``merge``).

    (x₁, …, xₘ) ↦ {k: xⱼ[k]}, over every key k of any input, with j the last
    input that holds k.

    Takes the mappings as positional arguments, so it works as a `Graph`
    node, or as one sequence of mappings, so it works after `juxt`.

    Examples:
        >>> from pipekit_ops import Merge
        >>> Merge()({"a": 1, "b": 1}, {"b": 2})
        {'a': 1, 'b': 2}
        >>> Merge()([{"a": 1}, {"c": 3}])
        {'a': 1, 'c': 3}
    """

    def _apply(self, *dicts: Any) -> dict[Any, Any]:
        if len(dicts) == 1 and not isinstance(dicts[0], Mapping):
            dicts = tuple(dicts[0])
        merged: dict[Any, Any] = {}
        for mapping in dicts:
            merged.update(mapping)
        return merged


class Assoc(_ContainerOperator):
    """Add a key whose value is computed from the whole dict (toolz ``assoc``).

    x ↦ {**x, key: op(x)}.

    Args:
        key: The key to set, a `str` or `int`; an existing value is replaced.
        op: The `Operator` called on the whole input.

    Raises:
        TypeError: at construction, if ``key`` is not a `str` / `int` or
            ``op`` is not an `Operator`.

    Examples:
        >>> from pipekit_ops import Assoc, Get
        >>> Assoc("tile", Get("t"))({"t": "31TCJ"})
        {'t': '31TCJ', 'tile': '31TCJ'}
    """

    def __init__(self, key: Key, op: Operator) -> None:
        if not isinstance(key, str | int):
            raise TypeError(
                f"Assoc.key must be a str or an int, got {type(key).__name__}."
            )
        require_child(op, "Assoc", "op")
        self.key = key
        self.op = op

    def _apply(self, x: Mapping[Any, Any]) -> dict[Any, Any]:
        return {**x, self.key: self.op(x)}


class UpdateIn(_ContainerOperator):
    """Replace the value at a key path by ``op(value)`` (toolz ``update_in``).

    x ↦ a copy of x with x[k₀]…[kₘ₋₁] replaced by op(x[k₀]…[kₘ₋₁]).

    Every dict along the path is copied; nothing else is.

    Args:
        keys: The path, a non-empty tuple of `str` / `int` keys.
        op: The `Operator` called on the value at the path.
        default: Stands in for a missing value: ``op(default)`` is stored,
            and missing dicts along the path are created. Omit it to raise
            `KeyError` on a missing key instead (toolz always fills).

    Raises:
        TypeError: at construction, if ``keys`` is not a tuple of keys or
            ``op`` is not an `Operator`.
        ValueError: at construction, if ``keys`` is empty.

    Examples:
        >>> from pipekit_ops import Const, UpdateIn
        >>> zero_cloud = UpdateIn(("properties", "cloud"), Const(0.0))
        >>> zero_cloud({"properties": {"cloud": 9}})
        {'properties': {'cloud': 0.0}}
        >>> UpdateIn(("a", "b"), Const(1), default=None)({})
        {'a': {'b': 1}}
    """

    def __init__(
        self, keys: Sequence[Key], op: Operator, default: Any = _MISSING
    ) -> None:
        self.keys = key_path(keys, "UpdateIn", "keys")
        if not self.keys:
            raise ValueError("UpdateIn.keys must hold at least one key.")
        require_child(op, "UpdateIn", "op")
        self.op = op
        self.default = default

    def _apply(self, x: Mapping[Any, Any]) -> dict[Any, Any]:
        return self._update(x, self.keys)

    def _update(self, x: Mapping[Any, Any], keys: tuple[Key, ...]) -> dict[Any, Any]:
        key, rest = keys[0], keys[1:]
        if key in x:
            value = x[key]
        elif self.default is _MISSING:
            raise KeyError(key)
        else:
            value = {} if rest else self.default
        out = dict(x)
        out[key] = self._update(value, rest) if rest else self.op(value)
        return out


class ValMap(_ContainerOperator):
    """Apply an operator to every value, keeping the keys (toolz ``valmap``).

    x ↦ {k: op(v) for k, v in x}.

    Args:
        op: The `Operator` called on each value.

    Raises:
        TypeError: at construction, if ``op`` is not an `Operator`.

    Examples:
        >>> from pipekit_ops import Count, ValMap
        >>> ValMap(Count())({"31TCJ": [0, 1], "31TDJ": [2]})
        {'31TCJ': 2, '31TDJ': 1}
    """

    def __init__(self, op: Operator) -> None:
        require_child(op, "ValMap", "op")
        self.op = op

    def _apply(self, x: Mapping[Any, Any]) -> dict[Any, Any]:
        return {key: self.op(value) for key, value in x.items()}
