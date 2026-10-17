"""Contract checks for pipekit operators.

Call these from a library's test suite on every operator it ships:

- `assert_round_trip(op, *inputs)` — the config round-trip contract
  (roadmap gate A1): ``loads(dumps(op))`` has the same `pipeline_hash`
  and returns the same output on a fixed input.
- `public_operators(module)` — every `Operator` subclass a module
  exports in its ``__all__``, so a sweep can fail on an operator that
  has no test case yet.

Example::

    from pipekit.testing import assert_round_trip

    def test_scale_round_trips():
        assert_round_trip(Scale(factor=2.0), 3.0)
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import ModuleType
from typing import Any

from pipekit._base.operator import Operator
from pipekit.hashing import pipeline_hash
from pipekit.parallel import check_pickleable
from pipekit.serial import dumps, loads


__all__ = ["assert_round_trip", "public_operators"]


def assert_round_trip(op: Operator, *inputs: Any, **kwargs: Any) -> Operator:
    """Assert that ``op`` survives ``loads(dumps(op))`` unchanged.

    Checks, in order:

    1. ``dumps`` then ``loads`` rebuild an operator of the same class.
    2. The rebuilt operator has the same `pipeline_hash`, so its whole
       configuration (nested operators and `Graph` wiring included)
       came back.
    3. Called on the same ``inputs`` and ``kwargs``, both operators
       return the same output. Comparison is exact: arrays (anything
       exposing ``__array__``) must match in type, shape, dtype and
       bytes, so ``NaN`` equals ``NaN`` but ``-0.0`` differs from
       ``0.0``; floats compare exactly (``NaN`` equal to ``NaN``);
       dicts (whatever their key order, as with ``==``), lists and
       tuples are compared item by item; any other value with ``==``.
       Array outputs need NumPy installed.

    Both operators receive the same input objects: an operator that
    mutates its input breaks the carrier contract and may fail here.

    Args:
        op: The operator (or `Sequential` / `Graph`) under test.
        *inputs: Positional inputs for the call.
        **kwargs: Keyword inputs for the call (a `Graph`'s named inputs).

    Returns:
        The rebuilt operator, for further checks.

    Raises:
        RuntimeError: if ``op``, or any operator nested in it, is marked
            ``forbid_in_yaml`` (its config is a debug payload, so the
            round trip is not expected to hold).
        AssertionError: on the first violated property, naming the
            operator.

    Example::

        >>> from pipekit import Const
        >>> from pipekit.testing import assert_round_trip
        >>> back = assert_round_trip(Const(3), "ignored")
        >>> back.value
        3
    """
    name = type(op).__name__
    flagged = check_pickleable(op)
    if flagged:
        names = ", ".join(type(f).__name__ for f in flagged)
        raise RuntimeError(
            f"assert_round_trip({name}): refuses operators marked forbid_in_yaml "
            f"({names}); their config is a debug payload, not a round trip."
        )

    try:
        back = loads(dumps(op))
    except Exception as exc:
        raise AssertionError(
            f"{name}: loads(dumps(op)) failed with {type(exc).__name__}: {exc}"
        ) from exc
    _require(
        type(back) is type(op),
        f"{name}: loads(dumps(op)) rebuilt a {type(back).__name__}",
    )

    expected_hash = pipeline_hash(op)
    actual_hash = pipeline_hash(back)
    _require(
        actual_hash == expected_hash,
        (
            f"{name}: pipeline_hash changed across loads(dumps(op)) "
            f"({expected_hash[:12]} -> {actual_hash[:12]}).\n"
            f"  before: {op.state!r}\n  after:  {back.state!r}"
        ),
    )

    expected = op(*inputs, **kwargs)
    try:
        actual = back(*inputs, **kwargs)
    except Exception as exc:
        raise AssertionError(
            f"{name}: the rebuilt operator raised {type(exc).__name__} on inputs "
            f"the original accepts: {exc}"
        ) from exc
    _assert_same(actual, expected, f"{name} output")
    return back


def public_operators(module: ModuleType) -> dict[str, type[Operator]]:
    """Every `Operator` subclass named in ``module.__all__``.

    Args:
        module: An imported package or module with an ``__all__``.

    Returns:
        ``{public name: class}``, in ``__all__`` order. Abstract bases
        (`Operator` itself, `StatefulOperator`, …) are included: the
        caller decides what to skip.

    Raises:
        AttributeError: if ``module`` has no ``__all__``.
    """
    found: dict[str, type[Operator]] = {}
    for name in module.__all__:
        obj = getattr(module, name)
        if isinstance(obj, type) and issubclass(obj, Operator):
            found[name] = obj
    return found


def _require(ok: bool, message: str) -> None:
    """Raise `AssertionError` unless ``ok`` (kept under ``python -O``)."""
    if not ok:
        raise AssertionError(message)


def _assert_same(actual: Any, expected: Any, where: str) -> None:
    """Exact, recursive equality; raise `AssertionError` naming ``where``."""
    _require(
        type(actual) is type(expected),
        f"{where}: type {type(actual).__name__} != {type(expected).__name__}",
    )
    if isinstance(expected, Mapping):
        # key order is not compared, as with ``==``: `dumps` sorts keys
        _require(
            set(actual) == set(expected),
            (
                f"{where}: keys {sorted(map(repr, actual))} != "
                f"{sorted(map(repr, expected))}"
            ),
        )
        for key in expected:
            _assert_same(actual[key], expected[key], f"{where}[{key!r}]")
        return
    if isinstance(expected, (list, tuple)):
        _require(
            len(actual) == len(expected),
            f"{where}: length {len(actual)} != {len(expected)}",
        )
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            _assert_same(a, e, f"{where}[{i}]")
        return
    if hasattr(expected, "__array__"):
        _assert_same_array(actual, expected, where)
        return
    if isinstance(expected, float) and math.isnan(expected):
        _require(math.isnan(actual), f"{where}: {actual!r} != nan")
        return
    _require(actual == expected, f"{where}: {actual!r} != {expected!r}")


def _assert_same_array(actual: Any, expected: Any, where: str) -> None:
    """Bitwise equality of two arrays: shape, dtype and bytes."""
    import numpy as np  # lazy: pipekit itself has no third-party dependency

    a = np.asarray(actual)
    e = np.asarray(expected)
    _require(a.shape == e.shape, f"{where}: shape {a.shape} != {e.shape}")
    _require(a.dtype == e.dtype, f"{where}: dtype {a.dtype} != {e.dtype}")
    if e.dtype.hasobject:
        _assert_same(a.tolist(), e.tolist(), where)
        return
    _require(
        np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(e).tobytes(),
        f"{where}: array values differ\n  actual:   {a!r}\n  expected: {e!r}",
    )
