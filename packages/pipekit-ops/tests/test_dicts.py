"""Dict plumbing (P1): toolz agreement (G1-1), no mutation (G1-2), G1-4."""

from __future__ import annotations

from typing import Any

import pytest
import toolz
from hypothesis import given, settings, strategies as st
from pipekit_ops import (
    Assoc,
    Const,
    Count,
    Dissoc,
    Get,
    GetIn,
    Identity,
    Lambda,
    Merge,
    Operator,
    Pick,
    Rename,
    Sequential,
    UpdateIn,
    ValMap,
    dumps,
    loads,
)
from pipekit_ops.hashing import stable_json


#: G1-1: 200 random inputs per operator, from a fixed seed.
GATE = settings(max_examples=200, derandomize=True, deadline=None, database=None)

KEYS = st.sampled_from(["a", "b", "c", "d", "e"])
LEAVES = st.integers(-5, 5)
FLAT = st.dictionaries(KEYS, LEAVES, max_size=5)
NESTED = st.recursive(
    LEAVES, lambda inner: st.dictionaries(KEYS, inner, max_size=4), max_leaves=12
).filter(lambda x: isinstance(x, dict))
PATHS = st.lists(KEYS, max_size=4)


class _Inc(Operator):
    """x ↦ x + 1, a round-trippable leaf for the operator arguments."""

    def _apply(self, x: int) -> int:
        return x + 1


def _outcome(fn: Any, *args: Any) -> tuple[str, Any]:
    """``("ok", value)`` or ``("raise", exception type)``, to compare failures too."""
    try:
        return "ok", fn(*args)
    except Exception as exc:
        return "raise", type(exc)


# --- G1-1: toolz agreement -------------------------------------------------


@GATE
@given(FLAT, KEYS)
def test_get_matches_toolz(d, key):
    assert Get(key, default=None)(d) == toolz.get(key, d, None)
    assert _outcome(Get(key), d)[0] == _outcome(toolz.get, key, d)[0]


@GATE
@given(FLAT, st.lists(KEYS, min_size=1, max_size=4))
def test_get_tuple_matches_toolz_list(d, keys):
    # toolz picks several keys with a list; pipekit with a tuple
    assert Get(tuple(keys), default=None)(d) == toolz.get(keys, d, None)


@GATE
@given(st.lists(LEAVES, max_size=6), st.integers(-8, 8))
def test_get_index_matches_toolz(seq, i):
    assert Get(i, default=None)(seq) == toolz.get(i, seq, None)


@GATE
@given(NESTED, PATHS)
def test_get_in_matches_toolz(d, path):
    assert GetIn(tuple(path), default=None)(d) == toolz.get_in(path, d, None)
    assert _outcome(GetIn(tuple(path)), d) == _outcome(
        toolz.get_in, path, d, None, True
    )


@GATE
@given(FLAT, st.lists(KEYS, max_size=5))
def test_dissoc_matches_toolz(d, keys):
    out = Dissoc(tuple(keys))(d)
    assert out == toolz.dissoc(d, *keys)
    # toolz rebuilds from a set when most keys go; pipekit keeps input order
    assert list(out) == [k for k in d if k not in keys]


@GATE
@given(st.lists(FLAT, min_size=1, max_size=4))
def test_merge_matches_toolz(dicts):
    expected = toolz.merge(*dicts)
    assert list(Merge()(*dicts).items()) == list(expected.items())
    assert Merge()(dicts) == expected


@GATE
@given(FLAT)
def test_valmap_matches_toolz(d):
    op = _Inc()
    assert ValMap(op)(d) == toolz.valmap(op, d)


@GATE
@given(NESTED, st.lists(KEYS, min_size=1, max_size=4))
def test_update_in_matches_toolz(d, path):
    op = _Inc()
    ours = _outcome(UpdateIn(tuple(path), op, default=0), d)
    theirs = _outcome(toolz.update_in, d, path, op, 0)
    assert ours == theirs


@GATE
@given(NESTED, st.lists(KEYS, min_size=1, max_size=4))
def test_update_in_without_default_raises_on_a_missing_key(d, path):
    try:
        toolz.get_in(path, d, no_default=True)
    except (KeyError, TypeError):
        present = False
    else:
        present = True
    if present:
        ours = _outcome(UpdateIn(tuple(path), _Inc()), d)
        assert ours == _outcome(toolz.update_in, d, path, _Inc())
    else:
        with pytest.raises((KeyError, TypeError)):
            UpdateIn(tuple(path), _Inc())(d)


# --- the operators toolz has no counterpart for ------------------------------


@GATE
@given(FLAT, st.lists(KEYS, max_size=5, unique=True))
def test_pick_keeps_listed_keys_in_order(d, keys):
    if all(k in d for k in keys):
        assert list(Pick(tuple(keys))(d).items()) == [(k, d[k]) for k in keys]
    else:
        with pytest.raises(KeyError):
            Pick(tuple(keys))(d)


@GATE
@given(FLAT, st.dictionaries(KEYS, st.sampled_from(["v", "w", "x"]), max_size=3))
def test_rename_keeps_order_and_unlisted_keys(d, mapping):
    if len(set(mapping.values())) != len(mapping):
        return
    out = Rename(mapping)(d)
    assert list(out) == [mapping.get(k, k) for k in d]
    assert list(out.values()) == list(d.values())


def test_rename_refuses_a_collision_and_allows_a_swap():
    with pytest.raises(ValueError, match="collides with the kept key 'b'"):
        Rename({"a": "b"})({"a": 1, "b": 2})
    assert Rename({"a": "b", "b": "a"})({"a": 1, "b": 2}) == {"b": 1, "a": 2}
    assert Rename({"z": "b"})({"b": 2}) == {"b": 2}
    with pytest.raises(ValueError, match="two keys to one"):
        Rename({"a": "x", "b": "x"})
    with pytest.raises(TypeError, match="JSON object keys"):
        Rename({0: "x"})  # ty: ignore[invalid-argument-type]


@GATE
@given(FLAT)
def test_assoc_adds_a_computed_key(d):
    out = Assoc("n", Count())(d)
    assert out == {**d, "n": len(d)}


def test_get_misses_raise_without_default():
    with pytest.raises(KeyError):
        Get("x")({})
    with pytest.raises(IndexError):
        Get(3)([1])
    with pytest.raises(KeyError):
        GetIn(("a", "x"))({"a": {}})


def test_update_in_refuses_an_empty_path():
    with pytest.raises(ValueError, match="at least one key"):
        UpdateIn((), Identity())


def test_merge_as_a_graph_node():
    from pipekit_ops import Graph, Input

    a, b = Input("a"), Input("b")
    graph = Graph(inputs={"a": a, "b": b}, outputs={"m": Merge()(a, b)})
    assert graph(a={"x": 1}, b={"x": 2, "y": 3}) == {"m": {"x": 2, "y": 3}}


# --- round trip of list-valued keys ------------------------------------------


def test_tuple_keys_come_back_as_tuples():
    back = loads(dumps(Get(("a", "b"))))
    assert back(d := {"a": 1, "b": 2}) == (1, 2)
    assert loads(dumps(Pick(("b",))))(d) == {"b": 2}
    assert loads(dumps(GetIn(["a"])))(d) == 1


def test_omitted_default_is_left_out_of_the_config():
    assert Get("a").get_config() == {"key": "a"}
    assert Get("a", default=None).get_config() == {"key": "a", "default": None}
    assert repr(Get("a")) == "Get(key='a')"


def test_output_signature_is_not_tracked():
    assert Get("a").compute_output_signature(None) is None
    assert ValMap(Identity()).compute_output_signature(None) is None


# --- G1-2: no mutation -------------------------------------------------------

DICT_OPS: dict[str, Any] = {
    "Get": Get("a", default=None),
    "GetIn": GetIn(("a", "b"), default=None),
    "Pick": Pick(()),
    "Dissoc": Dissoc(("a", "b")),
    "Rename": Rename({"a": "z"}),
    "Assoc": Assoc("a", Const({"new": 1})),
    "UpdateIn": UpdateIn(("a", "b", "c"), Const(99), default=None),
    "ValMap": ValMap(Const(0)),
}


@pytest.mark.parametrize("name", sorted(DICT_OPS))
@GATE
@given(d=NESTED)
def test_dict_operators_never_mutate(name, d):
    before = stable_json(d)
    _outcome(DICT_OPS[name], d)
    assert stable_json(d) == before


@GATE
@given(st.lists(NESTED, min_size=1, max_size=3))
def test_merge_never_mutates(dicts):
    before = stable_json(dicts)
    Merge()(*dicts)
    Merge()(dicts)
    assert stable_json(dicts) == before


# --- G1-4: bare callables refused --------------------------------------------

FUNCTION_ARGS = {
    "Get": lambda f: Get(f),
    "Assoc.op": lambda f: Assoc("k", f),
    "UpdateIn.op": lambda f: UpdateIn(("k",), f),
    "ValMap": lambda f: ValMap(f),
}


@pytest.mark.parametrize("name", sorted(FUNCTION_ARGS))
def test_bare_callables_are_refused(name):
    with pytest.raises(TypeError, match="Lambda"):
        FUNCTION_ARGS[name](lambda x: x)


def test_non_keys_are_refused():
    with pytest.raises(TypeError, match="tuple of str / int"):
        GetIn("properties")  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="must be a str, an int"):
        Get(1.5)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match=r"Assoc\.key"):
        Assoc(None, Identity())  # ty: ignore[invalid-argument-type]


def test_wrapping_in_lambda_works_and_is_flagged():
    op = ValMap(Lambda(abs))
    assert op({"a": -1}) == {"a": 1}
    with pytest.raises(RuntimeError, match="forbid_in_yaml"):
        loads(dumps(Sequential([op])))
