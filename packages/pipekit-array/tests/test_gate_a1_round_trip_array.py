"""Gate A1, config round-trip: every shipped pipekit-array operator survives it.

For every operator in ``pipekit_array.__all__`` not marked
``forbid_in_yaml``, ``loads(dumps(op))`` has the same `pipeline_hash`
and returns the same output on a fixed input (`assert_round_trip`).
A new operator without a case in `CASES` fails
`test_every_operator_has_a_case`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pipekit_array
import pytest
from pipekit_array import (
    ApplyAlong,
    ArgReduce,
    BatchAlong,
    BroadcastLike,
    Coarsen,
    ConcatenateAlong,
    CumReduce,
    Diff,
    DimArray,
    Dot,
    ExpandDims,
    ISel,
    MeanScalar,
    MergeDims,
    Pad,
    Quantile,
    Reduce,
    Rename,
    Roll,
    Sel,
    Shift,
    SplitDims,
    Squeeze,
    StackAlong,
    TakeAlong,
    Transpose,
    Where,
    Window,
)
from pipekit_ops import Operator
from pipekit_ops.testing import assert_round_trip, public_operators


np = pytest.importorskip("numpy")

X = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
PAIR = [X[0], X[1]]
#: a named cube (time, band, x) with labelled bands, and an index into its time
CUBE = DimArray(X, ("time", "band", "x"), {"band": ("B4", "B8", "B11")})
BEST = DimArray(np.array([[0, 1, 0, 1]] * 3), ("band", "x"))
#: CUBE's first scene, keeping a length-1 time dim; a (member,) carrier
SCENE = DimArray(X[:1], ("time", "band", "x"), {"band": ("B4", "B8", "B11")})
MEMBERS = DimArray(np.zeros(5, dtype=np.float32), ("member",))
#: a (time, x) mask for CUBE
VALID = DimArray(np.array([[True, False, True, True]] * 2), ("time", "x"))

#: Abstract bases: none ship in pipekit-array yet.
SKIP: dict[str, str] = {}

#: name -> (build the operator, positional inputs)
CASES: dict[str, tuple[Callable[[], Operator], tuple[Any, ...]]] = {
    "ArgReduce": (lambda: ArgReduce("argmax", dim="time"), (CUBE,)),
    "ConcatenateAlong": (lambda: ConcatenateAlong(axis=1), (PAIR,)),
    # a slice index is stored as start/stop/step
    "ISel": (lambda: ISel(dim="x", index=slice(1, None, 2)), (CUBE,)),
    # a tuple of axes comes back from JSON as a list
    "MeanScalar": (lambda: MeanScalar(axis=(0, 2), keepdims=True), (X,)),
    "Reduce": (lambda: Reduce("mean", dim=("time", "x")), (CUBE,)),
    "Sel": (lambda: Sel(dim="band", label=["B8", "B4"]), (CUBE,)),
    "StackAlong": (lambda: StackAlong(axis=-1), (PAIR,)),
    "TakeAlong": (lambda: TakeAlong(dim="time"), (CUBE, BEST)),
    "Transpose": (lambda: Transpose(order=("x", "time", "band")), (CUBE,)),
    # Phase 1: dims
    "BroadcastLike": (lambda: BroadcastLike(exclude=("x",)), (CUBE, MEMBERS)),
    "ExpandDims": (lambda: ExpandDims(("member", "run"), axis=-1), (CUBE,)),
    # several merges keep their order through JSON, which sorts object keys
    "MergeDims": (lambda: MergeDims(tx=("time", "x"), b=("band",)), (CUBE,)),
    "Rename": (lambda: Rename(band="channel", x="col"), (CUBE,)),
    "SplitDims": (lambda: SplitDims(x={"x": 2, "half": 2}), (CUBE,)),
    "Squeeze": (lambda: Squeeze("time"), (SCENE,)),
    # Phase 2: index-only transforms
    "CumReduce": (lambda: CumReduce("cumulative_sum", "time"), (CUBE,)),
    "Diff": (lambda: Diff("band", n=2, label="lower"), (CUBE,)),
    # a (before, after) pair comes back from JSON as a list
    "Pad": (lambda: Pad(mode="edge", x=(1, 2), time=1), (CUBE,)),
    "Roll": (lambda: Roll(time=1, x=-1), (CUBE,)),
    "Shift": (lambda: Shift(fill_value=0.0, x=2), (CUBE,)),
    "Where": (lambda: Where(other=-1.0), (CUBE, VALID)),
    # Phase 3: windows
    "Window": (lambda: Window(window_dim="w", stride=2, x=2), (CUBE,)),
    "Coarsen": (lambda: Coarsen("max", boundary="trim", band=2, x=3), (CUBE,)),
    # a nested operator round-trips through its own state record
    "ApplyAlong": (lambda: ApplyAlong(Reduce("sum", dim="x"), dim="band"), (CUBE,)),
    "BatchAlong": (
        lambda: BatchAlong(Reduce("mean", dim="x"), size=2, dim="band", pad="last"),
        (CUBE,),
    ),
    # Phase 4: statistics and algebra
    # q comes back from JSON as an ordered list
    "Quantile": (lambda: Quantile([0.9, 0.1], dim=("time", "x")), (CUBE,)),
    "Dot": (lambda: Dot(dim="band"), (CUBE, CUBE)),
}

OPERATORS = public_operators(pipekit_array)
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
    make, args = CASES[name]
    assert_round_trip(make(), *args)
