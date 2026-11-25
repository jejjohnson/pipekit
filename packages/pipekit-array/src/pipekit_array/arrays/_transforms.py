"""JAX's transformations on JAX data, and a plain equivalent elsewhere (ADR A27).

`vmap` and `scan` are loops off JAX. `stop_gradient` detaches on torch, and
`optimization_barrier` is an identity off JAX. JAX is imported on JAX data only.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from typing import Any

import array_api_compat

from pipekit_array.arrays._backend import namespace_of
from pipekit_array.arrays._host import is_host
from pipekit_array.arrays._numerics import constant


__all__ = ["optimization_barrier", "scan", "stop_gradient", "vmap"]


def vmap(fn: Callable[..., Any], in_axes: int | None | tuple = 0) -> Callable[..., Any]:
    """A function mapped over an axis of its arguments.

    On JAX data it is ``jax.vmap``. Elsewhere ``fn`` is called once per entry of
    the axis, and the results are stacked, which is slower than a function that
    broadcasts.

    Args:
        fn: A function of arrays, or of nested tuples, lists and dicts of them.
        in_axes: The axis to map over, of every argument or, in a tuple, of each
            one; ``None`` passes an argument whole to every call.

    Returns:
        The mapped function. The mapped axis comes first in its results.

    Raises:
        ValueError: (the mapped function) if no argument is mapped, or if the
            mapped axes differ in size or, off JAX, are empty.
        BackendError: (the mapped function) if a mapped argument is not an array
            of a known library.

    Example:
        >>> import numpy as np
        >>> vmap(np.linalg.norm)(np.asarray([[3.0, 4.0], [6.0, 8.0]]))
        array([ 5., 10.])
        >>> vmap(np.dot, in_axes=(0, None))(np.eye(2), np.asarray([1.0, 2.0]))
        array([1., 2.])
    """

    def mapped(*args: Any) -> Any:
        if _on_jax(args):
            import jax

            return jax.vmap(fn, in_axes=in_axes)(*args)
        axes = in_axes if isinstance(in_axes, tuple) else (in_axes,) * len(args)
        fronted = [
            (arg, None) if axis is None else (_moved(arg, axis), 0)
            for arg, axis in zip(args, axes, strict=True)
        ]
        size = _leading_size("vmap", [arg for arg, axis in fronted if axis == 0])
        calls = [
            fn(*(arg if axis is None else _entry(arg, k) for arg, axis in fronted))
            for k in range(size)
        ]
        return _stacked("vmap", calls)

    return mapped


def scan(
    f: Callable[[Any, Any], tuple[Any, Any]],
    init: Any,
    xs: Any = None,
    length: int | None = None,
    reverse: bool = False,
) -> tuple[Any, Any]:
    """A recurrence: (c_k, y_k) = f(c_{k-1}, x_k) for k = 0, …, K - 1.

    On JAX data, in ``init`` or ``xs``, it is ``jax.lax.scan``. Elsewhere ``f``
    is called once per step, which is slower than compiled code: a recurrence
    should have few steps over large arrays there.

    Args:
        f: The step, from the carry and a slice of ``xs`` to the next carry and
            an output.
        init: c_{-1}, the first carry. Off JAX it may be any object.
        xs: The inputs, sliced along their first axis: an array, or nested
            tuples, lists and dicts of arrays. ``None`` gives each step ``None``.
        length: K, for ``xs=None``; otherwise the size of the first axis.
        reverse: Run from the last input to the first. Each output keeps the
            place of its input.

    Returns:
        The last carry, and the outputs stacked along a new first axis; an
        output of ``None`` stays ``None``.

    Raises:
        ValueError: If the inputs differ in length, or their number is unknown.
            Off JAX also if there is no step, because the shape of an output is
            then unknown.
        BackendError: If an input is not an array of a known library.

    Example:
        >>> import numpy as np
        >>> total, partial = scan(lambda c, x: (c + x, c + x), 0.0, np.arange(4.0))
        >>> float(total), partial.tolist()
        (6.0, [0.0, 1.0, 3.0, 6.0])
    """
    if _on_jax((init, xs)):
        import jax

        return jax.lax.scan(f, init, xs, length=length, reverse=reverse)
    inputs = [] if xs is None else [xs]
    if not inputs and length is None:
        msg = "scan needs xs or a length"
        raise ValueError(msg)
    size = _leading_size("scan", inputs, length=length)
    steps = range(size)
    carry, outputs = init, []
    for k in reversed(steps) if reverse else steps:
        carry, output = f(carry, _entry(xs, k))
        outputs.append(output)
    return carry, _stacked("scan", outputs[::-1] if reverse else outputs)


def stop_gradient(x: Any) -> Any:
    """The same values, as a constant of the derivative.

    Args:
        x: An array, or nested tuples, lists and dicts of them.

    Returns:
        ``jax.lax.stop_gradient(x)`` on JAX, the detached tensors on torch, and
        ``x`` itself on a library without automatic differentiation.

    Raises:
        BackendError: If a leaf is not an array of a known library.

    Example:
        >>> stop_gradient([1.0])
        [1.0]
    """
    if _on_jax(x):
        import jax

        return jax.lax.stop_gradient(x)
    xp = namespace_of(*_leaves(x))
    if array_api_compat.is_torch_namespace(xp):
        return _tree_map(lambda leaf: leaf if is_host(leaf) else constant(leaf, xp), x)
    return x


def optimization_barrier(x: Any) -> Any:
    """The same values, which XLA may not fuse with what comes before.

    Args:
        x: An array, or nested tuples, lists and dicts of them.

    Returns:
        ``jax.lax.optimization_barrier(x)`` on JAX, and ``x`` itself elsewhere.

    Raises:
        BackendError: If a leaf is not an array of a known library.

    Example:
        >>> optimization_barrier([1.0])
        [1.0]
    """
    if _on_jax(x):
        import jax

        return jax.lax.optimization_barrier(x)
    namespace_of(*_leaves(x))
    return x


def _on_jax(tree: Any) -> bool:
    """Whether a leaf is a JAX array; nothing is, until JAX is imported."""
    jax = sys.modules.get("jax")
    if jax is None:
        return False
    leaves = jax.tree_util.tree_leaves(tree)
    return any(isinstance(leaf, jax.Array) for leaf in leaves)


def _tree_map(fn: Callable[..., Any], tree: Any, *rest: Any) -> Any:
    """``fn`` of each leaf of a tree, and of the same leaf of the others."""
    if tree is None:
        return None
    if isinstance(tree, dict):
        return {
            key: _tree_map(fn, value, *(other[key] for other in rest))
            for key, value in tree.items()
        }
    if isinstance(tree, (tuple, list)):
        values = [_tree_map(fn, *pair) for pair in zip(tree, *rest, strict=True)]
        # a NamedTuple takes its fields one by one
        return type(tree)(*values) if hasattr(tree, "_fields") else type(tree)(values)
    return fn(tree, *rest)


def _leaves(tree: Any) -> Iterator[Any]:
    """The leaves of a tree."""
    if isinstance(tree, dict):
        tree = tuple(tree.values())
    if isinstance(tree, (tuple, list)):
        for value in tree:
            yield from _leaves(value)
    elif tree is not None:
        yield tree


def _moved(tree: Any, axis: int) -> Any:
    """Every leaf with an axis moved to the front."""

    def move(leaf: Any) -> Any:
        xp = namespace_of(leaf)
        return xp.moveaxis(leaf, axis, 0)

    return _tree_map(move, tree)


def _entry(tree: Any, k: int) -> Any:
    """Entry k of the first axis of every leaf."""
    return _tree_map(lambda leaf: leaf[k, ...], tree)


def _leading_size(name: str, tree: Any, *, length: int | None = None) -> int:
    """The common size of the first axis of every leaf."""
    sizes = set()
    for leaf in _leaves(tree):
        namespace_of(leaf)
        if not getattr(leaf, "ndim", 0):
            msg = f"{name} needs an axis to run over, got an array without one"
            raise ValueError(msg)
        sizes.add(leaf.shape[0])
    if length is not None:
        sizes.add(length)
    if len(sizes) != 1:
        msg = f"{name} needs one length to run over, got {sorted(sizes) or 'none'}"
        raise ValueError(msg)
    return sizes.pop()


def _stacked(name: str, results: list[Any]) -> Any:
    """The results of the steps, each leaf stacked along a new first axis."""
    if not results:
        msg = f"{name} over no steps: the shape of a result is unknown before one ran"
        raise ValueError(msg)

    def stack(*leaves: Any) -> Any:
        xp = namespace_of(*leaves)
        # asarray would detach a torch tensor from its graph
        return xp.stack([xp.asarray(x) if is_host(x) else x for x in leaves])

    return _tree_map(stack, *results)
