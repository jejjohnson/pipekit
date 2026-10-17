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
from pipekit import Operator
from pipekit.testing import assert_round_trip, public_operators
from pipekit_array import ConcatenateAlong, MeanScalar, StackAlong


np = pytest.importorskip("numpy")

X = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
PAIR = [X[0], X[1]]

#: Abstract bases: none ship in pipekit-array yet.
SKIP: dict[str, str] = {}

#: name -> (build the operator, positional inputs)
CASES: dict[str, tuple[Callable[[], Operator], tuple[Any, ...]]] = {
    "ConcatenateAlong": (lambda: ConcatenateAlong(axis=1), (PAIR,)),
    # a tuple of axes comes back from JSON as a list
    "MeanScalar": (lambda: MeanScalar(axis=(0, 2), keepdims=True), (X,)),
    "StackAlong": (lambda: StackAlong(axis=-1), (PAIR,)),
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
