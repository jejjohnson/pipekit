"""What the array API standard leaves out: indexed updates, loops, padding.

A function written for JAX uses ``x.at[index].set``, ``jax.lax.scan`` or
``jax.vmap``, and cannot run on NumPy. Each hook here is JAX's own call on JAX
data, so compiled code lowers as before, and a plain equivalent on every other
array library. The hooks take the library of their data, and join NumPy data to
it, by the host rule of `pipekit_array.functions`.

- `set_at`, `add_at`, `multiply_at`: a copy of an array with some entries changed.
- `vmap`, `scan`: a map over an axis and a recurrence.
- `pad`: zeros around an array. `pipekit_array.pad` is the one for named
  dimensions.
- `stop_gradient`, `optimization_barrier`: identities that steer JAX.

This module needs ``pipekit-array[numpy,compat]``, so importing
``pipekit_array`` does not import it. It imports JAX on JAX data only.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from types import ModuleType
from typing import Any

from pipekit_ops import BackendError

from pipekit_array.functions import (
    _constant,
    _is_host,
    like,
    namespace_of,
    table_like,
)


# isort: split
# after the module above, whose error names the extras to install
import array_api_compat
import numpy as np


__all__ = [
    "add_at",
    "multiply_at",
    "optimization_barrier",
    "pad",
    "scan",
    "set_at",
    "stop_gradient",
    "vmap",
]


# --- indexed updates ---


def set_at(x: Any, index: Any, values: Any) -> Any:
    """A copy of an array with some entries replaced: y = x, then y[index] = v.

    On JAX it is ``x.at[index].set(values)``.

    Args:
        x: The array, which is not changed.
        index: What ``x[index]`` accepts: integers, slices, an ellipsis, a
            boolean mask or an integer array. An entry that an integer array
            names twice takes one of its two values.
        values: The new entries, which broadcast to ``x[index]`` and take the
            dtype of ``x``.

    Returns:
        y, in ``x``'s library, shape and dtype.

    Raises:
        BackendError: If the array library does not take that kind of index, or
            if two libraries other than NumPy meet in one call.
        IndexError: If the index is out of bounds. JAX does not report it, and
            leaves the array as it is.

    Example:
        >>> import numpy as np
        >>> set_at(np.zeros(3), slice(1, None), [1.0, 2.0])
        array([0., 1., 2.])
    """
    xp, x, index = _update_inputs(x, index, values)
    if array_api_compat.is_jax_namespace(xp):
        return x.at[index].set(values)
    return _assign("set_at", xp, x, index, values)


def add_at(x: Any, index: Any, values: Any) -> Any:
    """A copy of an array with some entries added to: y = x, then y[index] += v.

    An entry that the index names twice is added to twice, which ``y[index] += v``
    does not do on NumPy. On JAX it is ``x.at[index].add(values)``.

    Args:
        x: The array, which is not changed.
        index: As in `set_at`; an integer array needs NumPy or JAX.
        values: What to add, which broadcasts to ``x[index]``.

    Returns:
        y, in ``x``'s library, shape and dtype.

    Raises:
        BackendError: As `set_at`.
        IndexError: As `set_at`.

    Example:
        >>> import numpy as np
        >>> add_at(np.zeros(4), np.asarray([0, 2, 2]), [1.0, 10.0, 100.0])
        array([  1.,   0., 110.,   0.])
    """
    xp, x, index = _update_inputs(x, index, values)
    if array_api_compat.is_jax_namespace(xp):
        return x.at[index].add(values)
    if _has_integer_array(xp, index):
        return _accumulate("add_at", np.add, xp, x, index, values)
    # the index names an entry once
    return _assign("add_at", xp, x, index, x[index] + like(values, x, xp))


def multiply_at(x: Any, index: Any, values: Any) -> Any:
    """A copy of an array with some entries multiplied: y = x, then y[index] *= v.

    An entry that the index names twice is multiplied twice. On JAX it is
    ``x.at[index].multiply(values)``.

    Args:
        x: The array, which is not changed.
        index: As in `set_at`; an integer array needs NumPy or JAX.
        values: The factors, which broadcast to ``x[index]``.

    Returns:
        y, in ``x``'s library, shape and dtype.

    Raises:
        BackendError: As `set_at`.
        IndexError: As `set_at`.

    Example:
        >>> import numpy as np
        >>> multiply_at(np.ones(3), np.asarray([1, 1]), [2.0, 3.0])
        array([1., 6., 1.])
    """
    xp, x, index = _update_inputs(x, index, values)
    if array_api_compat.is_jax_namespace(xp):
        return x.at[index].multiply(values)
    if _has_integer_array(xp, index):
        return _accumulate("multiply_at", np.multiply, xp, x, index, values)
    return _assign("multiply_at", xp, x, index, x[index] * like(values, x, xp))


# --- loops ---


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


# --- padding, and identities that steer JAX ---


def pad(x: Any, widths: tuple[tuple[int, int], ...]) -> Any:
    """An array with zeros around it.

    Args:
        x: The array.
        widths: How many zeros before and after, for each axis.

    Returns:
        The larger array, in ``x``'s library and dtype, on its device.

    Raises:
        BackendError: If ``x`` is not an array of a known library.
        ValueError: If ``widths`` is not one pair per axis.

    Example:
        >>> import numpy as np
        >>> pad(np.ones(2), ((1, 2),))
        array([0., 1., 1., 0., 0.])
    """
    xp = namespace_of(x)
    x = xp.asarray(x) if _is_host(x) else x
    if len(widths) != x.ndim:
        msg = f"pad needs one pair of widths per axis, got {len(widths)} for {x.ndim}"
        raise ValueError(msg)
    native = getattr(xp, "pad", None)
    if native is not None:
        return native(x, widths)

    def zeros(width: int, axis: int) -> Any:
        shape = (*x.shape[:axis], width, *x.shape[axis + 1 :])
        return xp.zeros(shape, dtype=x.dtype, device=array_api_compat.device(x))

    for axis, (before, after) in enumerate(widths):
        x = xp.concat([zeros(before, axis), x, zeros(after, axis)], axis=axis)
    return x


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
        return _tree_map(
            lambda leaf: leaf if _is_host(leaf) else _constant(leaf, xp), x
        )
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


# --- the update of a copy ---


def _update_inputs(x: Any, index: Any, values: Any) -> tuple[ModuleType, Any, Any]:
    """The namespace, the array in it, and the index with its arrays in it."""
    parts = index if isinstance(index, tuple) else (index,)
    arrays = [part for part in parts if hasattr(part, "dtype")]
    xp = namespace_of(x, values, *arrays)
    x = xp.asarray(x) if _is_host(x) else x
    if array_api_compat.is_jax_namespace(xp):
        return xp, x, index
    parts = tuple(
        table_like(part, x, xp) if isinstance(part, np.ndarray) else part
        for part in parts
    )
    return xp, x, parts if isinstance(index, tuple) else parts[0]


def _has_integer_array(xp: ModuleType, index: Any) -> bool:
    """Whether an index may name an entry twice."""
    parts = index if isinstance(index, tuple) else (index,)
    return any(
        isinstance(part, list)
        or (getattr(part, "ndim", 0) > 0 and xp.isdtype(part.dtype, "integral"))
        for part in parts
    )


def _assign(name: str, xp: ModuleType, x: Any, index: Any, values: Any) -> Any:
    """A copy of ``x`` with ``values`` at the index."""
    # astype, because asarray would detach a torch tensor from its graph
    y = xp.astype(x, x.dtype, copy=True)
    try:
        y[index] = like(values, x, xp)
    except (IndexError, NotImplementedError) as err:
        strict = array_api_compat.is_array_api_strict_namespace(xp)
        if isinstance(err, NotImplementedError) or (
            strict and _has_integer_array(xp, index)
        ):
            raise _refusal(name, xp) from err
        raise
    return y


def _accumulate(
    name: str, ufunc: np.ufunc, xp: ModuleType, x: Any, index: Any, values: Any
) -> Any:
    """A copy of ``x``, updated once for each time the index names an entry."""
    if not array_api_compat.is_numpy_namespace(xp):
        raise _refusal(name, xp)
    y = np.array(x)
    ufunc.at(y, index, like(values, x, xp))
    return y


def _refusal(name: str, xp: ModuleType) -> BackendError:
    library = xp.__name__.removeprefix("array_api_compat.").partition(".")[0]
    return BackendError(f"{name} does not take this index on {library}")


# --- nested arrays ---


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
        return xp.stack([xp.asarray(x) if _is_host(x) else x for x in leaves])

    return _tree_map(stack, *results)
