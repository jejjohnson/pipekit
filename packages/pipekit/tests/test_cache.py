"""Tests for `pipekit.cache` — Group G."""

from __future__ import annotations

import pytest
from pipekit import Cache, Memoize, Operator


class Counter(Operator):
    """Operator that increments a class-level counter on each _apply."""

    calls = 0

    def __init__(self, label: str = "c") -> None:
        self.label = label

    def _apply(self, x):
        type(self).calls += 1
        return x * 2


@pytest.fixture(autouse=True)
def _reset_counter():
    Counter.calls = 0
    yield


def test_cache_hits_and_misses():
    c = Cache(Counter())
    assert c(3) == 6
    assert c(3) == 6
    assert c(4) == 8
    assert Counter.calls == 2  # only two unique inputs
    assert c.hits == 1
    assert c.misses == 2


def test_cache_clear_drops_entries():
    c = Cache(Counter())
    c(1)
    c.clear()
    c(1)
    assert Counter.calls == 2
    assert c.misses == 2


def test_cache_key_depends_on_inner_config():
    a = Cache(Counter("a"))
    b = Cache(Counter("b"))
    a(3)
    b(3)
    # Different caches, different counters — but inner.config differs anyway.
    assert a.hits == 0
    assert b.hits == 0


def test_memoize_is_alias_of_cache():
    assert Memoize is Cache


def test_cache_rejects_non_operator():
    with pytest.raises(TypeError):
        Cache(lambda x: x)  # type: ignore[arg-type]


def test_cache_get_config_includes_inner():
    c = Cache(Counter("hello"))
    cfg = c.get_config()
    assert "inner" in cfg
    assert cfg["inner"]["class"] == "Counter"


def test_cache_handles_unhashable_inputs():
    """Lists aren't hashable; Cache falls back to repr-based keys."""
    c = Cache(Counter())
    c([1, 2, 3])
    c([1, 2, 3])
    # Both calls share a repr-stable key, so the second is a hit.
    assert c.hits == 1
    assert c.misses == 1


class _FakeArray:
    """Array-like with a truncated repr — the NumPy large-array hazard."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self.shape = (len(data),)
        self.dtype = "uint8"

    def tobytes(self) -> bytes:
        return self._data

    def __mul__(self, k):  # let Counter's `x * 2` succeed
        return self._data * k

    def __repr__(self) -> str:  # deliberately elides content, like numpy
        return f"_FakeArray(shape={self.shape}, [...])"


def test_cache_array_like_inputs_do_not_collide_on_truncated_repr():
    """Two arrays with identical reprs but different bytes must miss twice."""
    inner = Counter()
    op = Cache(inner)
    a = _FakeArray(b"\x01" * 8)
    b = _FakeArray(b"\x02" * 8)
    assert repr(a) == repr(b)  # the trap the byte-content key avoids
    op(a)
    op(b)
    assert op.misses == 2
    assert Counter.calls == 2


def test_cache_array_like_identical_content_hits():
    inner = Counter()
    op = Cache(inner)
    op(_FakeArray(b"\x03" * 4))
    op(_FakeArray(b"\x03" * 4))
    assert (op.hits, op.misses) == (1, 1)
    assert Counter.calls == 1


def test_cache_is_thread_safe_under_concurrent_access():
    """Counters must stay consistent when hammered from many threads."""
    import threading

    inner = Counter()
    op = Cache(inner)
    op(1)  # warm the key so every thread hits
    n_threads, per_thread = 8, 50

    def hammer():
        for _ in range(per_thread):
            assert op(1) == 2

    threads = [threading.Thread(target=hammer) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert op.hits == n_threads * per_thread
    assert op.misses == 1
