"""Iterables — filter, fold, group and count lists of items inside a pipeline.

| Operator                    | Output                                           |
| --------------------------- | ------------------------------------------------ |
| `Not(pred)`                 | ``not pred(x)``                                  |
| `Filter(pred)`              | items with ``pred(item)`` truthy                 |
| `Remove(pred)`              | items with ``pred(item)`` falsy                  |
| `Reduce(op, initial)`       | left fold, ``op((acc, item))``                   |
| `PartitionAll(n)`           | tuples of ``n`` consecutive items                |
| `Concat()`                  | one level flattened                              |
| `Zip(strict=True)`          | tuples of aligned items, strict by default       |
| `GroupBy(key)`              | ``dict[group, list]``                            |
| `CountBy(key)`              | ``dict[group, int]``                             |
| `Count()`                   | the number of items                              |
| `Unique(key=None)`          | the first occurrence of each item or key         |
| `Take(n)`                   | the first ``n`` items                            |
| `SortBy(key, reverse)`      | a stable sort                                    |

Every operator accepts any iterable and returns a `list` (a `dict` for the
group-by family; `Not` returns a `bool`), so an output can be inspected,
cached and fed to two `Graph` nodes. Predicates and folds are operators;
a ``key`` is an operator or a plain key (a `str`, an `int` or a tuple of
them, looked up as ``item[key]`` the way toolz's ``getter`` does). A bare
Python callable is refused with a `TypeError` that points to `Lambda`.

See the pipekit-ops design, ``api/operators.md`` section 2.
"""

from __future__ import annotations

import functools
import itertools
from collections.abc import Iterable, Sized
from typing import Any

from pipekit_ops._base.operator import _MISSING, Operator
from pipekit_ops.dicts import (
    Key,
    _ContainerOperator,
    lookup,
    plain_key,
    require_child,
)


__all__ = [
    "Concat",
    "Count",
    "CountBy",
    "Filter",
    "GroupBy",
    "Not",
    "PartitionAll",
    "Reduce",
    "Remove",
    "SortBy",
    "Take",
    "Unique",
    "Zip",
]

#: A ``key`` argument: an operator called on the item, or a plain key.
KeyLike = Key | tuple[Key, ...] | Operator


def _key(value: Any, owner: str, field: str) -> KeyLike:
    """Check a ``key`` argument: an `Operator`, or a plain key (lists become tuples)."""
    if isinstance(value, Operator):
        return value
    return plain_key(value, owner, field)


def _count(n: Any, owner: str, low: int) -> int:
    """Check an integer count ``n ≥ low``."""
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError(f"{owner}.n must be an int, got {type(n).__name__}.")
    if n < low:
        raise ValueError(f"{owner}.n must be >= {low}, got {n}.")
    return n


class Not(_ContainerOperator):
    """Negate a predicate; the operator form of `complement`.

    x ↦ ¬ pred(x).

    Args:
        pred: The predicate `Operator`.

    Raises:
        TypeError: at construction, if ``pred`` is not an `Operator`.

    Examples:
        >>> from pipekit_ops import Const, Not
        >>> Not(Const(0))("anything")
        True
    """

    def __init__(self, pred: Operator) -> None:
        require_child(pred, "Not", "pred")
        self.pred = pred

    def _apply(self, x: Any) -> bool:
        return not self.pred(x)


class Filter(_ContainerOperator):
    """Keep the items a predicate accepts (builtin ``filter``).

    (x₀, …, xₙ₋₁) ↦ [xᵢ | pred(xᵢ) truthy], in input order.

    Args:
        pred: The predicate `Operator`, often a `Sequential` that reads a
            field and compares it.

    Raises:
        TypeError: at construction, if ``pred`` is not an `Operator`.

    Examples:
        >>> from pipekit_ops import Filter, Get
        >>> Filter(Get("clear"))([{"clear": True}, {"clear": False}])
        [{'clear': True}]
        >>> Filter(lambda x: x)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        TypeError: Filter.pred must be an Operator, got function. Wrap a plain
        function in `Lambda(fn)` ...
    """

    def __init__(self, pred: Operator) -> None:
        require_child(pred, "Filter", "pred")
        self.pred = pred

    def _apply(self, items: Iterable[Any]) -> list[Any]:
        return [item for item in items if self.pred(item)]


class Remove(_ContainerOperator):
    """Drop the items a predicate accepts (toolz ``remove``).

    (x₀, …, xₙ₋₁) ↦ [xᵢ | pred(xᵢ) falsy], in input order.

    Args:
        pred: The predicate `Operator`.

    Raises:
        TypeError: at construction, if ``pred`` is not an `Operator`.

    Examples:
        >>> from pipekit_ops import Get, Remove
        >>> Remove(Get("cloudy"))([{"cloudy": True}, {"cloudy": False}])
        [{'cloudy': False}]
    """

    def __init__(self, pred: Operator) -> None:
        require_child(pred, "Remove", "pred")
        self.pred = pred

    def _apply(self, items: Iterable[Any]) -> list[Any]:
        return [item for item in items if not self.pred(item)]


class Reduce(_ContainerOperator):
    """Left fold (``functools.reduce``), with the operator called on one pair.

    a₀ = initial, aᵢ₊₁ = op((aᵢ, xᵢ)); returns aₙ. Without ``initial``,
    a₁ = x₀.

    ``op`` takes one ``(acc, item)`` tuple because an operator takes one
    carrier: an operator written for `Reduce` reads ``acc, item = pair``.

    Args:
        op: The fold `Operator`, called on ``(acc, item)``.
        initial: The starting accumulator. Omit it to start from the first
            item.

    Raises:
        TypeError: at construction, if ``op`` is not an `Operator`; at call
            time, on an empty input with no ``initial``.

    Examples:
        >>> from pipekit_ops import Get, Reduce
        >>> Reduce(Get(1))([3, 4, 5])
        5
        >>> Reduce(Get(0), initial="start")([3, 4, 5])
        'start'
    """

    def __init__(self, op: Operator, initial: Any = _MISSING) -> None:
        require_child(op, "Reduce", "op")
        self.op = op
        self.initial = initial

    def _apply(self, items: Iterable[Any]) -> Any:
        # aᵢ₊₁ = op((aᵢ, xᵢ))
        step = lambda acc, item: self.op((acc, item))
        if self.initial is _MISSING:
            return functools.reduce(step, items)
        return functools.reduce(step, items, self.initial)


class PartitionAll(_ContainerOperator):
    """Split into tuples of ``n`` consecutive items (toolz ``partition_all``).

    (x₀, …, xₘ₋₁) ↦ [(x₀ … xₙ₋₁), (xₙ … x₂ₙ₋₁), …]; the last may be shorter.

    Args:
        n: The tuple length, ``n ≥ 1``.

    Raises:
        TypeError: at construction, if ``n`` is not an `int`.
        ValueError: at construction, if ``n < 1``.

    Examples:
        >>> from pipekit_ops import PartitionAll
        >>> PartitionAll(2)([1, 2, 3, 4, 5])
        [(1, 2), (3, 4), (5,)]
    """

    def __init__(self, n: int) -> None:
        self.n = _count(n, "PartitionAll", 1)

    def _apply(self, items: Iterable[Any]) -> list[tuple[Any, ...]]:
        return list(itertools.batched(items, self.n))


class Concat(_ContainerOperator):
    """Flatten exactly one level (toolz ``concat``).

    (s₀, s₁, …) ↦ s₀ ⧺ s₁ ⧺ …

    Examples:
        >>> from pipekit_ops import Concat
        >>> Concat()([[1, 2], [3], [[4]]])
        [1, 2, 3, [4]]
    """

    def _apply(self, iterables: Iterable[Iterable[Any]]) -> list[Any]:
        return list(itertools.chain.from_iterable(iterables))


class Zip(_ContainerOperator):
    """Align several iterables into tuples (builtin ``zip``), strict by default.

    (s₁, …, sₘ) ↦ [(s₁[i], …, sₘ[i]) for i < n].

    Takes the iterables as positional arguments, so it works as a `Graph`
    node, or as one sequence of iterables, so it works after `juxt`.

    Args:
        strict: Raise `ValueError` when the inputs differ in length. With
            ``False``, stop at the shortest, as the builtin does.

    Examples:
        >>> from pipekit_ops import Zip
        >>> Zip()(["a", "b"], [1, 2])
        [('a', 1), ('b', 2)]
        >>> Zip(strict=False)([1, 2], [1])
        [(1, 1)]
        >>> Zip()([1, 2], [1])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValueError: zip() argument 2 is shorter than argument 1
    """

    def __init__(self, strict: bool = True) -> None:
        self.strict = strict

    def _apply(self, *iterables: Iterable[Any]) -> list[tuple[Any, ...]]:
        if len(iterables) == 1:
            iterables = tuple(iterables[0])
        return list(zip(*iterables, strict=self.strict))


class GroupBy(_ContainerOperator):
    """Group items by a key, in first-seen order (toolz ``groupby``).

    (x₀, …, xₙ₋₁) ↦ {g: [xᵢ | key(xᵢ) = g]}, each list in input order.

    Args:
        key: An `Operator` called on each item, or a plain key looked up
            as ``item[key]`` (a tuple of keys groups by a tuple).

    Raises:
        TypeError: at construction, if ``key`` is neither.

    Examples:
        >>> from pipekit_ops import GroupBy
        >>> GroupBy(0)(["ab", "b", "ac"])
        {'a': ['ab', 'ac'], 'b': ['b']}
        >>> items = [{"tile": "A", "i": 0}, {"tile": "B", "i": 1}]
        >>> GroupBy(("tile", "i"))(items)
        {('A', 0): [{'tile': 'A', 'i': 0}], ('B', 1): [{'tile': 'B', 'i': 1}]}
    """

    def __init__(self, key: KeyLike) -> None:
        self.key = _key(key, "GroupBy", "key")

    def _apply(self, items: Iterable[Any]) -> dict[Any, list[Any]]:
        groups: dict[Any, list[Any]] = {}
        for item in items:
            groups.setdefault(lookup(self.key, item), []).append(item)
        return groups


class CountBy(_ContainerOperator):
    """Count items per key, in first-seen order (toolz ``countby``).

    (x₀, …, xₙ₋₁) ↦ {g: #{i | key(xᵢ) = g}}. ``CountBy(Identity())`` is
    toolz ``frequencies``.

    Args:
        key: An `Operator` called on each item, or a plain key looked up
            as ``item[key]``.

    Raises:
        TypeError: at construction, if ``key`` is neither.

    Examples:
        >>> from pipekit_ops import CountBy, Identity
        >>> CountBy("tile")([{"tile": "A"}, {"tile": "B"}, {"tile": "A"}])
        {'A': 2, 'B': 1}
        >>> CountBy(Identity())("abca")
        {'a': 2, 'b': 1, 'c': 1}
    """

    def __init__(self, key: KeyLike) -> None:
        self.key = _key(key, "CountBy", "key")

    def _apply(self, items: Iterable[Any]) -> dict[Any, int]:
        counts: dict[Any, int] = {}
        for item in items:
            group = lookup(self.key, item)
            counts[group] = counts.get(group, 0) + 1
        return counts


class Count(_ContainerOperator):
    """The number of items (toolz ``count``).

    (x₀, …, xₙ₋₁) ↦ n; ``len(x)`` when defined, else the iterable is consumed.

    Examples:
        >>> from pipekit_ops import Count
        >>> Count()([1, 2, 3])
        3
        >>> Count()(i for i in range(5) if i % 2)
        2
    """

    def _apply(self, items: Iterable[Any]) -> int:
        if isinstance(items, Sized):
            return len(items)
        return sum(1 for _ in items)


class Unique(_ContainerOperator):
    """The first occurrence of each item, or of each key (toolz ``unique``).

    (x₀, …, xₙ₋₁) ↦ [xᵢ | key(xᵢ) ∉ {key(xⱼ) | j < i}]; items or keys must be
    hashable.

    Args:
        key: ``None`` to compare items, an `Operator` called on each item,
            or a plain key looked up as ``item[key]``.

    Raises:
        TypeError: at construction, if ``key`` is none of these.

    Examples:
        >>> from pipekit_ops import Unique
        >>> Unique()([3, 1, 3, 2, 1])
        [3, 1, 2]
        >>> Unique("tile")([{"tile": "A", "i": 0}, {"tile": "A", "i": 1}])
        [{'tile': 'A', 'i': 0}]
    """

    def __init__(self, key: KeyLike | None = None) -> None:
        self.key = None if key is None else _key(key, "Unique", "key")

    def _apply(self, items: Iterable[Any]) -> list[Any]:
        seen: set[Any] = set()
        kept: list[Any] = []
        for item in items:
            marker = item if self.key is None else lookup(self.key, item)
            if marker not in seen:
                seen.add(marker)
                kept.append(item)
        return kept


class Take(_ContainerOperator):
    """The first ``n`` items (toolz ``take``), for dry runs.

    (x₀, …, xₘ₋₁) ↦ [x₀, …, x_min(n, m)₋₁]; a lazy input is read no further.

    Args:
        n: How many items, ``n ≥ 0``.

    Raises:
        TypeError: at construction, if ``n`` is not an `int`.
        ValueError: at construction, if ``n < 0``.

    Examples:
        >>> from pipekit_ops import Take
        >>> Take(2)([5, 6, 7])
        [5, 6]
        >>> Take(3)(iter(range(10**9)))
        [0, 1, 2]
    """

    def __init__(self, n: int) -> None:
        self.n = _count(n, "Take", 0)

    def _apply(self, items: Iterable[Any]) -> list[Any]:
        return list(itertools.islice(items, self.n))


class SortBy(_ContainerOperator):
    """A stable sort by key (builtin ``sorted``).

    (x₀, …, xₙ₋₁) ↦ the items ordered by key(xᵢ); equal keys keep input order.

    Args:
        key: An `Operator` called on each item, or a plain key looked up
            as ``item[key]``.
        reverse: Sort in descending order (still stable).

    Raises:
        TypeError: at construction, if ``key`` is neither.

    Examples:
        >>> from pipekit_ops import SortBy, Take
        >>> SortBy("cloud")([{"cloud": 30}, {"cloud": 5}])
        [{'cloud': 5}, {'cloud': 30}]
        >>> (SortBy(0, reverse=True) | Take(1))([(1, "a"), (3, "b"), (2, "c")])
        [(3, 'b')]
    """

    def __init__(self, key: KeyLike, reverse: bool = False) -> None:
        self.key = _key(key, "SortBy", "key")
        self.reverse = reverse

    def _apply(self, items: Iterable[Any]) -> list[Any]:
        return sorted(
            items, key=functools.partial(lookup, self.key), reverse=self.reverse
        )
