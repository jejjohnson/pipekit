"""Gate A1, config round-trip: every shipped pipekit operator survives it.

For every concrete operator in ``pipekit.__all__`` not marked
``forbid_in_yaml``, ``loads(dumps(op))`` has the same `pipeline_hash`
and returns the same output on a fixed input (`assert_round_trip`).
The operators are collected from ``__all__``, so a new operator without
a case in `CASES` fails `test_every_operator_has_a_case`.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from typing import Any

import pipekit
import pytest
from pipekit import (
    AssertCallable,
    AssertDType,
    AssertHasAttribute,
    AssertShape,
    AsyncMap,
    BatchedMap,
    Cache,
    Const,
    Fanout,
    Graph,
    Identity,
    Input,
    Lambda,
    Memoize,
    Operator,
    ProcessMap,
    Retry,
    Sequential,
    ThreadMap,
    Try,
)
from pipekit.testing import _assert_same, assert_round_trip, public_operators


class _Scale(Operator):
    """Multiply by a constant: a round-trippable leaf for the containers."""

    def __init__(self, factor: float) -> None:
        self.factor = factor

    def _apply(self, x: float) -> float:
        return x * self.factor


class _Add(Operator):
    """Sum two carriers: the join of the two-input `Graph`."""

    def _apply(self, a: float, b: float) -> float:
        return a + b


@dataclasses.dataclass(frozen=True)
class _Field:
    """A carrier with the attributes the QC asserts read."""

    shape: tuple[int, ...] = (2, 3)
    dtype: str = "float32"
    units: str = "K"


def _two_input_graph() -> Graph:
    """Two inputs, a shared operator used on both, a passthrough output."""
    a = Input("a")
    b = Input("b")
    clean = _Scale(2.0) | Const(5.0) | _Scale(3.0)
    total = _Add()(clean(a), _Scale(10.0)(clean(b)))
    return Graph(inputs={"a": a, "b": b}, outputs={"total": total, "raw": a})


#: Abstract bases: they have no behaviour of their own to round-trip.
SKIP: dict[str, str] = {
    "Operator": "abstract base: `_apply` raises NotImplementedError",
    "StatefulOperator": "abstract base: `_apply` raises NotImplementedError",
}

#: name -> (build the operator, positional inputs, keyword inputs)
CASES: dict[str, tuple[Callable[[], Operator], tuple[Any, ...], dict[str, Any]]] = {
    "AssertDType": (lambda: AssertDType("float32"), (_Field(),), {}),
    "AssertHasAttribute": (lambda: AssertHasAttribute("units", "K"), (_Field(),), {}),
    "AssertShape": (lambda: AssertShape((2, None)), (_Field(),), {}),
    "AsyncMap": (lambda: AsyncMap(_Scale(2.0), semaphore=2), ([1.0, 2.0],), {}),
    "BatchedMap": (
        lambda: BatchedMap(Identity(), batch_size=2, flatten=True),
        ([1, 2, 3],),
        {},
    ),
    "Cache": (lambda: Cache(_Scale(2.0)), (3.0,), {}),
    "Const": (lambda: Const({"a": [1, 2]}), (0.0,), {}),
    "Fanout": (lambda: Fanout({"a": _Scale(2.0), "b": Identity()}), (3.0,), {}),
    "Graph": (_two_input_graph, (), {"a": 1.0, "b": 2.0}),
    "Identity": (Identity, (3.0,), {}),
    "Memoize": (lambda: Memoize(_Scale(2.0)), (3.0,), {}),
    # `Const` is importable in a worker process whatever the start method.
    "ProcessMap": (lambda: ProcessMap(Const(7), n_workers=2), ([1, 2],), {}),
    "Retry": (
        lambda: Retry(_Scale(2.0), attempts=2, base_delay=0.0, on=(KeyError,)),
        (3.0,),
        {},
    ),
    "Sequential": (lambda: Sequential([_Scale(2.0), _Scale(0.5)]), (3.0,), {}),
    "ThreadMap": (lambda: ThreadMap(_Scale(2.0), n_workers=2), ([1.0, 2.0],), {}),
    "Try": (
        lambda: Try(primary=_Scale(2.0), fallback=Identity(), on=(KeyError,)),
        (3.0,),
        {},
    ),
}

OPERATORS = public_operators(pipekit)
ROUND_TRIPPABLE = sorted(n for n, cls in OPERATORS.items() if not cls.forbid_in_yaml)


def test_every_operator_has_a_case():
    missing = [n for n in ROUND_TRIPPABLE if n not in CASES and n not in SKIP]
    assert not missing, (
        f"Gate A1: {missing} have no round-trip case. Add a constructor and an "
        "input to CASES in this file (or, for an abstract base only, to SKIP)."
    )


def test_no_stale_cases():
    stale = sorted(set(CASES) - set(ROUND_TRIPPABLE))
    assert not stale, f"CASES names operators that are not shipped: {stale}"


@pytest.mark.parametrize("name", ROUND_TRIPPABLE)
def test_operator_round_trips(name):
    if name in SKIP:
        pytest.skip(SKIP[name])
    make, args, kwargs = CASES[name]
    assert_round_trip(make(), *args, **kwargs)


def test_two_input_graph_with_a_shared_operator_round_trips():
    graph = _two_input_graph()
    back = assert_round_trip(graph, a=1.0, b=2.0)
    assert back(a=1.0, b=2.0) == {"total": 165.0, "raw": 1.0}
    # the shared `clean` pipeline is still one object, used by both inputs
    assert back._order[0].operator is back._order[1].operator


def test_graph_nested_in_a_pipeline_round_trips():
    x = Input("x")
    inner = Graph(inputs={"x": x}, outputs={"y": _Scale(3.0)(x)})
    assert_round_trip(Cache(_Scale(2.0) | inner), 1.0)


# --- the helper itself -----------------------------------------------------


class _Lossy(Operator):
    """`from_config` drops ``k``, so the round trip changes the hash."""

    def __init__(self, k: int = 1) -> None:
        self.k = k

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> _Lossy:
        return cls()

    def _apply(self, x: int) -> int:
        return x + self.k


class _Hidden(Operator):
    """``offset`` is left out of the config, so the output changes."""

    __config_exclude__ = ("offset",)

    def __init__(self, offset: int = 0) -> None:
        self.offset = offset

    def _apply(self, x: int) -> int:
        return x + self.offset


class _Fragile(Operator):
    """Hashes its tuple of axes, which reloads from JSON as an unhashable list."""

    def __init__(self, axes: tuple[int, ...]) -> None:
        self.axes = axes

    def _apply(self, x: int) -> int:
        return x + hash(self.axes) % 2


class _Arrays(Operator):
    """Return a fixed array, configured by a list of floats."""

    def __init__(self, values: list[float]) -> None:
        self.values = values

    def _apply(self, x: Any) -> Any:
        import numpy as np

        return {"out": (np.asarray(self.values, dtype=np.float32), x)}


def test_helper_returns_the_rebuilt_operator():
    back = assert_round_trip(_Scale(2.0), 3.0)
    assert isinstance(back, _Scale)
    assert back.factor == 2.0


def test_helper_catches_a_changed_hash():
    with pytest.raises(AssertionError, match=r"_Lossy: pipeline_hash changed"):
        assert_round_trip(_Lossy(k=2), 1)


def test_helper_catches_a_changed_output():
    with pytest.raises(AssertionError, match=r"_Hidden output: 1 != 6"):
        assert_round_trip(_Hidden(offset=5), 1)


def test_helper_catches_a_rebuilt_operator_that_raises():
    with pytest.raises(AssertionError, match=r"_Fragile: the rebuilt operator raised"):
        assert_round_trip(_Fragile(axes=(0, 1)), 1)


def test_helper_catches_a_failed_load():
    with pytest.raises(AssertionError, match=r"Const: loads\(dumps\(op\)\) failed"):
        # a non-JSON config value cannot be dumped
        assert_round_trip(Const(_Hidden()), 1)


@pytest.mark.parametrize(
    "op",
    [Lambda(abs), AssertCallable(callable), Cache(_Scale(2.0) | Lambda(abs))],
    ids=["Lambda", "AssertCallable", "nested Lambda"],
)
def test_helper_refuses_forbid_in_yaml(op):
    with pytest.raises(RuntimeError, match="forbid_in_yaml"):
        assert_round_trip(op, 1.0)


def test_helper_compares_arrays_bitwise_inside_containers():
    np = pytest.importorskip("numpy")
    assert_round_trip(_Arrays([1.0, float("nan"), -0.0]), (1, [2.0]))
    # bitwise: -0.0 and 0.0 differ, NaN equals NaN
    _assert_same(np.array([np.nan]), np.array([np.nan]), "nan")
    with pytest.raises(AssertionError, match="array values differ"):
        _assert_same(np.array([-0.0]), np.array([0.0]), "zero")
    with pytest.raises(AssertionError, match="dtype"):
        _assert_same(np.zeros(2, np.float32), np.zeros(2, np.float64), "dtype")
    with pytest.raises(AssertionError, match=r"x\['b'\]\[1\]: 3 != 4"):
        _assert_same({"b": (1, 3)}, {"b": (1, 4)}, "x")


def test_dumps_output_is_plain_json():
    """The round trip goes through text, not an in-memory state."""
    text = pipekit.dumps(_two_input_graph())
    assert isinstance(json.loads(text), dict)
