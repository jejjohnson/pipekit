"""Tests for `pipekit_ops.store` and the store-backed `Cache` — phase P4.

Gates G4-1 to G4-7 of ``packages/pipekit-ops/docs/design/roadmap.md``.
The cross-process gates (G4-2, G4-3) run their work in spawned processes
through the module-level tasks in ``_store_ops.py``.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import pickle
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import pytest
from pipekit_ops import (
    Cache,
    DirectoryStore,
    Get,
    Identity,
    Map,
    MemoryStore,
    Operator,
    Store,
    dumps,
    executor_scope,
    loads,
    pipeline_hash,
    store_scope,
)

from tests._store_ops import Payload, run_cached, write_all


#: ``pipeline_hash(Cache(Identity()))`` before P4, computed at c16b7eb (G4-1)
CACHE_IDENTITY_HASH = "24b9f307d9e6d72d942868cdd7d1f03edec714e12396e41abd6c7b4e78e30149"


class Counting(Operator):
    """x ↦ x, counting calls on the class."""

    calls = 0

    def _apply(self, x):
        type(self).calls += 1
        return x


class A(Operator):
    """One class of a pair with equal (empty) configs."""

    def _apply(self, x):
        return ("A", x)


class B(Operator):
    """The other class of the pair; its output differs from `A`'s."""

    def _apply(self, x):
        return ("B", x)


class _Scale(Operator):
    """The core-tour example's operator."""

    def __init__(self, factor: float) -> None:
        self.factor = factor

    def _apply(self, x: float) -> float:
        return x * self.factor


@pytest.fixture(autouse=True)
def _reset_counting():
    Counting.calls = 0
    yield


def _spawn_pool(n: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(n, mp_context=multiprocessing.get_context("spawn"))


# --- G4-1: the default is unchanged ---------------------------------------


def test_g4_1_default_hash_and_config_unchanged():
    op = Cache(Identity())
    assert pipeline_hash(op) == CACHE_IDENTITY_HASH
    assert op.get_config() == {
        "inner": {"module": "pipekit_ops.blocks", "class": "Identity", "config": {}}
    }


def test_g4_1_core_tour_hits_and_misses():
    c = Cache(_Scale(2.0))
    assert [c(1.0), c(1.0), c(2.0)] == [2.0, 2.0, 4.0]
    assert (c.hits, c.misses) == (1, 2)


def test_store_and_key_enter_the_config_and_hash_only_when_set():
    plain = Cache(Identity())
    named = Cache(Identity(), store="scenes")
    keyed = Cache(Identity(), key=Get("id"))
    assert named.get_config()["store"] == "scenes"
    assert keyed.get_config()["key"]["class"] == "Get"
    assert "key" not in named.get_config()
    assert "store" not in keyed.get_config()
    assert len({pipeline_hash(op) for op in (plain, named, keyed)}) == 3


# --- G4-2: a DirectoryStore survives a restart -----------------------------


def test_g4_2_survives_a_restart(tmp_path):
    items = list(range(100))
    with _spawn_pool(1) as first:
        outputs, hits, misses = first.submit(run_cached, str(tmp_path), items).result()
    assert (hits, misses) == (0, 100)
    # a new pool is a new process, with a fresh Cache
    with _spawn_pool(1) as second:
        again, hits, misses = second.submit(run_cached, str(tmp_path), items).result()
    assert (hits, misses) == (100, 0)
    assert again == outputs == [Payload()(i) for i in items]


# --- G4-3: atomic writes ----------------------------------------------------


def test_g4_3_concurrent_writers_leave_whole_files(tmp_path):
    items = list(range(100))
    root = str(tmp_path)
    start_at = time.time() + 2.0
    with _spawn_pool(8) as pool:
        futures = [
            pool.submit(write_all, root, items, seed, start_at) for seed in range(8)
        ]
        assert [f.result() for f in futures] == [100] * 8

    entries = [
        os.path.join(directory, name)
        for directory, _, names in os.walk(root)
        for name in names
        if name != "pipekit-store.json"
    ]
    # no temporary file is left behind, and no key is written twice
    assert len(entries) == 100
    assert all(name.endswith(".pkl") for name in entries)
    store = DirectoryStore(root)
    cached = Cache(Payload(), store="s")
    for i in items:
        assert store.load(cached._make_key(i)) == Payload()(i)


# --- G4-4: pickling ---------------------------------------------------------


def test_g4_4_cache_pickles_without_entries():
    c = Cache(Counting(), store=None, key=Get(0))
    c((1, "a"))
    copy = pickle.loads(pickle.dumps(c))
    assert len(copy._store) == 0
    assert (copy.hits, copy.misses) == (0, 0)
    assert pipeline_hash(copy) == pipeline_hash(c)
    assert copy((1, "b")) == (1, "b")


def test_named_cache_pickles_with_its_store_name():
    copy = pickle.loads(pickle.dumps(Cache(Identity(), store="scenes")))
    assert copy.store == "scenes"


# --- G4-5: key by field -----------------------------------------------------


def test_g4_5_key_by_field_runs_op_once():
    c = Cache(Counting(), key=Get("id"))
    first = c({"id": "S2A", "cloud": 3})
    second = c({"id": "S2A", "cloud": 9})
    assert Counting.calls == 1
    assert second == first
    assert (c.hits, c.misses) == (1, 1)


def test_key_refuses_a_plain_function():
    with pytest.raises(TypeError, match="Lambda"):
        Cache(Identity(), key=lambda item: item["id"])  # type: ignore[arg-type]


def test_store_must_be_a_name():
    with pytest.raises(TypeError, match="store name"):
        Cache(Identity(), store=MemoryStore())  # type: ignore[arg-type]


def test_two_key_operators_do_not_share_entries():
    store = MemoryStore()
    by_id = Cache(Identity(), store="s", key=Get("id"))
    by_name = Cache(Identity(), store="s", key=Get("name"))
    with store_scope(s=store):
        by_id({"id": "a", "name": "b"})
        assert by_name({"id": "b", "name": "a"}) == {"id": "b", "name": "a"}
    assert len(store) == 2


# --- G4-6: unbound name -----------------------------------------------------


def test_g4_6_unbound_name_raises_lookup_error():
    with pytest.raises(LookupError, match="missing"):
        Cache(Identity(), store="missing")(1)


def test_unbound_inside_another_scope():
    with store_scope(other=MemoryStore()), pytest.raises(LookupError, match="missing"):
        Cache(Identity(), store="missing")(1)


# --- G4-7: the class is in the key ------------------------------------------


def test_g4_7_equal_configs_of_different_classes_never_share():
    assert A().get_config() == B().get_config()
    store = MemoryStore()
    with store_scope(s=store):
        assert Cache(A(), store="s")(1) == ("A", 1)
        assert Cache(B(), store="s")(1) == ("B", 1)
    assert len(store) == 2


def test_g4_7_private_stores_key_by_class_too():
    assert Cache(A())._make_key(1) != Cache(B())._make_key(1)


# --- store_scope ------------------------------------------------------------


def test_store_scope_nests_and_resets():
    outer, inner = MemoryStore(), MemoryStore()
    cached = Cache(Identity(), store="s")
    with store_scope(s=outer):
        cached(1)
        with store_scope(s=inner, t=MemoryStore()) as bound:
            assert set(bound) == {"s", "t"}
            cached(2)
        cached(3)
    assert (len(outer), len(inner)) == (2, 1)
    with pytest.raises(LookupError):
        cached(1)


def test_store_scope_refuses_a_non_store():
    with pytest.raises(TypeError, match="Store"), store_scope(s={}):  # type: ignore[arg-type]
        pass


def test_store_scope_reaches_map_threads():
    store = MemoryStore()
    reader = Map(Cache(Counting(), store="s"), executor="thread", n_workers=4)
    with store_scope(s=store):
        assert reader([1, 2, 3, 1]) == [1, 2, 3, 1]
    assert len(store) == 3


def test_store_scope_reaches_a_borrowed_thread_pool():
    store = MemoryStore()
    reader = Map(Cache(Identity(), store="s"), executor="scope")
    with ThreadPoolExecutor(2) as pool, executor_scope(pool), store_scope(s=store):
        assert reader([1, 2]) == [1, 2]
    assert len(store) == 2


def test_store_scope_travels_to_process_workers(tmp_path):
    """Process workers get the bound stores pickled (execution.md §2.7)."""
    reader = Map(Cache(Payload(8), store="s"), executor="process", n_workers=2)
    with store_scope(s=DirectoryStore(tmp_path)):
        outputs = reader(list(range(6)))
    assert outputs == [Payload(8)(i) for i in range(6)]
    # the workers wrote to the parent's directory
    _, hits, misses = run_cached_small(str(tmp_path))
    assert (hits, misses) == (6, 0)


def run_cached_small(root: str) -> tuple[list, int, int]:
    cached = Cache(Payload(8), store="s")
    with store_scope(s=DirectoryStore(root)):
        outputs = [cached(i) for i in range(6)]
    return outputs, cached.hits, cached.misses


def test_unpicklable_store_is_refused_before_a_process_starts():
    class Local(MemoryStore):
        pass

    reader = Map(Cache(Identity(), store="s"), executor="process")
    with store_scope(s=Local()), pytest.raises(TypeError, match="'s'"):
        reader([1])


# --- MemoryStore ------------------------------------------------------------


def test_memory_store_is_a_store_and_loads_what_it_saves():
    store = MemoryStore()
    key = "f" * 64
    assert isinstance(store, Store)
    assert key not in store
    with pytest.raises(KeyError):
        store.load(key)
    store.save(key, 1)
    assert key in store and store.load(key) == 1
    copy = pickle.loads(pickle.dumps(store))
    assert copy.load(key) == 1 and copy._lock is not store._lock
    store.clear()
    assert len(store) == 0


def test_cache_clear_leaves_named_stores_alone():
    store = MemoryStore()
    cached = Cache(Identity(), store="s")
    with store_scope(s=store):
        cached(1)
        cached.clear()
        cached(1)
    assert (cached.hits, len(store)) == (1, 1)


def test_memory_store_is_safe_under_threads():
    store = MemoryStore()
    keys = [f"{i:064x}" for i in range(200)]

    def write():
        for key in keys:
            store.save(key, key)

    threads = [threading.Thread(target=write) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(store) == 200


# --- DirectoryStore ---------------------------------------------------------


def test_directory_store_layout_and_marker(tmp_path):
    store = DirectoryStore(tmp_path / "nested" / "root")
    key = "ab" + "0" * 62
    store.save(key, {"id": "S2A"})
    assert (tmp_path / "nested" / "root" / "ab" / f"{key}.pkl").is_file()
    marker = json.loads(
        (tmp_path / "nested" / "root" / "pipekit-store.json").read_text()
    )
    assert marker == {"format": "pipekit-store", "version": 1}
    assert key in DirectoryStore(tmp_path / "nested" / "root")


def test_directory_store_refuses_another_version(tmp_path):
    (tmp_path / "pipekit-store.json").write_text('{"version": 2}')
    with pytest.raises(ValueError, match="version 2"):
        DirectoryStore(tmp_path)


def test_directory_store_refuses_keys_that_are_not_digests(tmp_path):
    store = DirectoryStore(tmp_path)
    for bad in ("../etc/passwd", "AB" * 32, "a" * 63):
        with pytest.raises(ValueError, match="hex digests"):
            store.save(bad, 1)


def test_directory_store_missing_key_and_codec(tmp_path):
    store = DirectoryStore(tmp_path)
    assert "0" * 64 not in store
    with pytest.raises(KeyError):
        store.load("0" * 64)
    with pytest.raises(ValueError, match="codec"):
        DirectoryStore(tmp_path, codec="json")


def test_directory_store_pickles_as_its_path(tmp_path):
    store = DirectoryStore(tmp_path)
    store.save("1" * 64, [1])
    copy = pickle.loads(pickle.dumps(store))
    assert (copy.root, copy.codec) == (str(tmp_path), "pickle")
    assert copy.load("1" * 64) == [1]
    assert len(pickle.dumps(store)) < 300


def test_directory_store_overwrite_leaves_no_temporary_files(tmp_path):
    store = DirectoryStore(tmp_path)
    for value in range(5):
        store.save("2" * 64, value)
    assert store.load("2" * 64) == 4
    assert os.listdir(tmp_path / "22") == ["2" * 64 + ".pkl"]


# --- round trip -------------------------------------------------------------


def test_named_keyed_cache_round_trips_through_dumps():
    op = Cache(Counting(), store="scenes", key=Get("id"))
    back = loads(dumps(op))
    assert pipeline_hash(back) == pipeline_hash(op)
    assert back.store == "scenes" and isinstance(back.key, Get)
    store = MemoryStore()
    with store_scope(scenes=store):
        op({"id": 1, "v": 2})
        back({"id": 1, "v": 3})
    # the reloaded cache finds the original's entry in the shared store
    assert (back.hits, len(store)) == (1, 1)


def test_keyed_cache_shares_entries_with_process_workers(tmp_path):
    """A pickled `Get` keys the same in a worker as in the parent."""
    items = [{"id": i, "v": i} for i in range(4)]
    reader = Map(Cache(Identity(), store="s", key=Get("id")), executor="process")
    parent = Cache(Identity(), store="s", key=Get("id"))
    with store_scope(s=DirectoryStore(tmp_path)):
        assert reader(items) == items
        assert [parent({"id": i}) for i in range(4)] == items
    assert (parent.hits, parent.misses) == (4, 0)


def test_pickled_get_still_raises_on_a_missing_key():
    copy = pickle.loads(pickle.dumps(Get("id")))
    with pytest.raises(KeyError):
        copy({})


def test_store_scope_travels_to_a_borrowed_process_pool(tmp_path):
    reader = Map(Cache(Payload(8), store="s"), executor="scope")
    store = DirectoryStore(tmp_path)
    with _spawn_pool(2) as pool, executor_scope(pool), store_scope(s=store):
        assert reader(list(range(6))) == [Payload(8)(i) for i in range(6)]
    assert run_cached_small(str(tmp_path))[1:] == (6, 0)
