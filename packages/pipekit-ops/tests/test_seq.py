"""Iterables (P1): toolz agreement (G1-1), strict zip (G1-3), no callables (G1-4),
and the running example (G1-5)."""

from __future__ import annotations

import itertools
import random
from typing import Any

import pytest
import toolz
from hypothesis import given, settings, strategies as st
from pipekit_ops import (
    Concat,
    Count,
    CountBy,
    Dissoc,
    Filter,
    Get,
    GetIn,
    Graph,
    GroupBy,
    Identity,
    Input,
    Not,
    Operator,
    PartitionAll,
    Reduce,
    Remove,
    Rename,
    SortBy,
    Take,
    Unique,
    Zip,
    juxt,
)
from pipekit_ops.testing import assert_round_trip


#: G1-1: 200 random inputs per operator, from a fixed seed.
GATE = settings(max_examples=200, derandomize=True, deadline=None, database=None)

INTS = st.lists(st.integers(-10, 10), max_size=20)
RECORDS = st.lists(
    st.fixed_dictionaries({"tile": st.sampled_from("ABC"), "i": st.integers(0, 3)}),
    max_size=15,
)


class _IsEven(Operator):
    """x ↦ x is even."""

    def _apply(self, x: int) -> bool:
        return x % 2 == 0


class _Mod3(Operator):
    """x ↦ x mod 3, a grouping key."""

    def _apply(self, x: int) -> int:
        return x % 3


class _AddPair(Operator):
    """(a, x) ↦ a + x, a fold for `Reduce`."""

    def _apply(self, pair: tuple[int, int]) -> int:
        acc, item = pair
        return acc + item


# --- G1-1: toolz agreement -------------------------------------------------


@GATE
@given(INTS)
def test_filter_matches_builtin(xs):
    assert Filter(_IsEven())(xs) == list(filter(_IsEven(), xs))


@GATE
@given(INTS)
def test_remove_matches_toolz(xs):
    assert Remove(_IsEven())(xs) == list(toolz.remove(_IsEven(), xs))


@GATE
@given(INTS, st.integers(1, 6))
def test_partition_all_matches_toolz(xs, n):
    assert PartitionAll(n)(xs) == list(toolz.partition_all(n, xs))


@GATE
@given(st.lists(INTS, max_size=5))
def test_concat_matches_toolz(xss):
    assert Concat()(xss) == list(toolz.concat(xss))


@GATE
@given(INTS)
def test_group_by_matches_toolz(xs):
    ours = GroupBy(_Mod3())(xs)
    theirs = toolz.groupby(_Mod3(), xs)
    assert list(ours.items()) == list(theirs.items())


@GATE
@given(RECORDS)
def test_group_by_plain_keys_match_toolz_getter(records):
    assert GroupBy("tile")(records) == toolz.groupby("tile", records)
    # toolz picks several keys with a list; pipekit with a tuple
    assert GroupBy(("tile", "i"))(records) == toolz.groupby(["tile", "i"], records)


@GATE
@given(RECORDS)
def test_count_by_matches_toolz(records):
    ours = CountBy("tile")(records)
    assert list(ours.items()) == list(toolz.countby("tile", records).items())
    assert CountBy(Identity())([r["i"] for r in records]) == toolz.frequencies(
        r["i"] for r in records
    )


@GATE
@given(INTS)
def test_count_matches_toolz(xs):
    assert Count()(xs) == toolz.count(xs)
    assert Count()(iter(xs)) == toolz.count(iter(xs))


@GATE
@given(RECORDS)
def test_unique_matches_toolz(records):
    ids = [r["i"] for r in records]
    assert Unique()(ids) == list(toolz.unique(ids))
    assert Unique("tile")(records) == list(toolz.unique(records, key=Get("tile")))


@GATE
@given(INTS, st.integers(0, 25))
def test_take_matches_toolz(xs, n):
    assert Take(n)(xs) == list(toolz.take(n, xs))


# --- the operators toolz has no counterpart for ------------------------------


@GATE
@given(INTS)
def test_reduce_matches_functools(xs):
    assert Reduce(_AddPair(), initial=0)(xs) == sum(xs)
    if xs:
        assert Reduce(_AddPair())(xs) == sum(xs)


def test_reduce_of_empty_input_without_initial_raises():
    with pytest.raises(TypeError, match="empty"):
        Reduce(_AddPair())([])


@GATE
@given(INTS)
def test_not_negates(xs):
    assert [Not(_IsEven())(x) for x in xs] == [x % 2 != 0 for x in xs]


@GATE
@given(RECORDS, st.booleans())
def test_sort_by_is_stable(records, reverse):
    expected = sorted(records, key=lambda r: r["tile"], reverse=reverse)
    assert SortBy("tile", reverse=reverse)(records) == expected


def test_take_reads_a_lazy_input_no_further():
    source = iter(range(100))
    assert Take(3)(source) == [0, 1, 2]
    assert next(source) == 3


def test_zip_takes_one_sequence_of_inputs_after_juxt():
    pair = juxt(Get("a"), Get("b"))
    assert Zip()(pair({"a": [1, 2], "b": "xy"})) == [(1, "x"), (2, "y")]


def test_counts_must_be_valid():
    with pytest.raises(ValueError, match=r"PartitionAll.n must be >= 1"):
        PartitionAll(0)
    with pytest.raises(ValueError, match=r"Take.n must be >= 0"):
        Take(-1)
    with pytest.raises(TypeError, match=r"Take.n must be an int"):
        Take(2.0)  # ty: ignore[invalid-argument-type]


# --- G1-3: strict zip --------------------------------------------------------


def test_zip_is_strict_by_default():
    with pytest.raises(ValueError):
        Zip()([1, 2], [1])
    assert Zip(strict=False)([1, 2], [1]) == [(1, 1)]


# --- G1-4: bare callables refused --------------------------------------------

FUNCTION_ARGS = {
    "Not": lambda f: Not(f),
    "Filter": lambda f: Filter(f),
    "Remove": lambda f: Remove(f),
    "Reduce": lambda f: Reduce(f),
    "GroupBy": lambda f: GroupBy(f),
    "CountBy": lambda f: CountBy(f),
    "Unique": lambda f: Unique(f),
    "SortBy": lambda f: SortBy(f),
}


@pytest.mark.parametrize("name", sorted(FUNCTION_ARGS))
def test_bare_callables_are_refused(name):
    with pytest.raises(TypeError, match="Lambda"):
        FUNCTION_ARGS[name](lambda x: x)


# --- G1-5: the running example -----------------------------------------------

N_TILES = 25
SCENES_PER_TILE = 12


class _StacSearch(Operator):
    """A stand-in STAC search: 12 item dicts on each of 25 tiles, from a seed."""

    def __init__(self, collection: str, month: str, seed: int = 0) -> None:
        self.collection = collection
        self.month = month
        self.seed = seed

    def _apply(self, query: Any) -> list[dict[str, Any]]:
        rng = random.Random(self.seed)
        tiles = [f"31T{chr(65 + i // 5)}{chr(65 + i % 5)}" for i in range(N_TILES)]
        items = []
        for tile, k in itertools.product(tiles, range(SCENES_PER_TILE)):
            items.append(
                {
                    "id": f"S2_{tile}_{self.month}_{k:02d}",
                    "collection": self.collection,
                    "properties": {
                        "eo:cloud_cover": round(rng.uniform(0.0, 100.0), 2),
                        "s2:mgrs_tile": tile,
                    },
                    "assets": {
                        band: {"href": f"s3://bucket/{tile}/{k}/{band}.tif"}
                        for band in ("B02", "B03", "B04", "B08")
                    },
                }
            )
        rng.shuffle(items)
        return items


class _MaxCloud(Operator):
    """cloud cover ↦ cloud cover ≤ limit."""

    def __init__(self, limit: float) -> None:
        self.limit = limit

    def _apply(self, cloud_cover: float) -> bool:
        return cloud_cover <= self.limit


def _search_and_screen() -> Graph:
    """The search-and-screen part of the vision's job (no reads: those are P2+)."""
    search = _StacSearch(collection="sentinel-2-l2a", month="2026-10")
    cloud_cover = GetIn(("properties", "eo:cloud_cover"))
    clear_enough = cloud_cover | _MaxCloud(40.0)
    keep = Filter(clear_enough)
    tile_id = GetIn(("properties", "s2:mgrs_tile"))
    per_tile = CountBy(tile_id)

    items = Input("items")
    found = search(items)
    kept = keep(found)
    return Graph(
        inputs={"items": items},
        outputs={
            "n_found": Count()(found),
            "n_kept": Count()(kept),
            "per_tile": per_tile(kept),
        },
    )


def test_running_example_matches_a_plain_loop():
    job = _search_and_screen()
    summary = job(items="query")

    found = _StacSearch(collection="sentinel-2-l2a", month="2026-10")("query")
    n_kept = 0
    per_tile: dict[str, int] = {}
    for item in found:
        if item["properties"]["eo:cloud_cover"] <= 40.0:
            n_kept += 1
            tile = item["properties"]["s2:mgrs_tile"]
            per_tile[tile] = per_tile.get(tile, 0) + 1

    assert summary["n_found"] == 300
    assert summary["n_kept"] == n_kept
    assert 0 < n_kept < 300
    assert summary["per_tile"] == per_tile
    assert list(summary["per_tile"]) == list(per_tile)


def test_running_example_round_trips_and_reports():
    back = assert_round_trip(_search_and_screen(), items="query")
    summary = back(items="query")
    report = (Dissoc(("per_tile",)) | Rename({"n_found": "found", "n_kept": "kept"}))(
        summary
    )
    assert report == {"found": 300, "kept": summary["n_kept"]}
