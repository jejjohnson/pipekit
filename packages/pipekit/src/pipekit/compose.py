"""Group B — composition convenience.

Small free functions (not `Operator` subclasses) that mirror the
`toolz` composition surface. They cost almost nothing to implement
and signal the lineage clearly.

- `pipe(value, *operators)` — toolz.pipe; sugar for ``Sequential(ops)(value)``.
- `compose(*operators)` — toolz.compose; right-to-left composition.
- `compose_left(*operators)` — toolz.compose_left; explicit `Sequential` alias.
- `complement(predicate)` — toolz.complement; negate a callable predicate.
- `juxt(*operators)` — toolz.juxt; tuple-returning multi-output.

`juxt` is the tuple-keyed sibling of `Fanout` (dict-keyed, Group F).

See master plan Report 2, Group B.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pipekit._base.operator import Operator
from pipekit._base.sequential import Sequential


def pipe(value: Any, *operators: Operator) -> Any:
    """Apply ``operators`` left-to-right to ``value``.

    Equivalent to ``Sequential(operators)(value)`` but with less
    ceremony when you have a value in hand and want to thread it
    through a sequence of operators inline.

    Example::

        result = pipe(gt, Scale(2.0), AddConst(1.0))
        # same as: Sequential([Scale(2.0), AddConst(1.0)])(gt)
    """
    return Sequential(list(operators))(value)


def compose(*operators: Operator) -> Sequential:
    """Right-to-left composition (mathematical order).

    ``compose(f, g, h)`` is equivalent to ``Sequential([h, g, f])``.
    The rightmost operator runs first, matching the way function
    composition is written in math: ``f ∘ g ∘ h``.

    Example::

        pipe = compose(Scale(2.0), AddConst(1.0))   # adds 1, then scales
        assert pipe(3.0) == 8.0
    """
    return Sequential(list(reversed(operators)))


def compose_left(*operators: Operator) -> Sequential:
    """Left-to-right composition (pipeline order).

    Explicit alias of ``Sequential([...])`` for users who prefer the
    free-function style. ``compose_left(f, g, h)`` is exactly
    ``Sequential([f, g, h])``.
    """
    return Sequential(list(operators))


def complement(predicate: Callable[..., bool]) -> Callable[..., bool]:
    """Return a predicate that negates ``predicate``.

    Useful for inverting a predicate without lambda noise::

        is_clean = lambda gt: not has_nans(gt)
        # equivalent:
        is_clean = complement(has_nans)

        branch = Branch(predicate=is_clean, if_true=Identity(), if_false=Clean())
    """

    def negated(*args: Any, **kwargs: Any) -> bool:
        return not predicate(*args, **kwargs)

    negated.__name__ = f"complement({getattr(predicate, '__name__', 'predicate')})"
    return negated


def juxt(*operators: Operator) -> Callable[..., tuple[Any, ...]]:
    """Return a callable that applies each operator to its input and tuples the results.

    ``juxt(f, g, h)(x)`` returns ``(f(x), g(x), h(x))``. The dict-keyed
    analogue is `Fanout` (Group F); use `juxt` when you want a positional
    tuple without naming the outputs.

    Example::

        triple = juxt(Identity(), Scale(2.0), Scale(3.0))
        assert triple(5.0) == (5.0, 10.0, 15.0)
    """
    ops = list(operators)

    def juxtaposed(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        return tuple(op(*args, **kwargs) for op in ops)

    juxtaposed.__name__ = "juxt"
    return juxtaposed
