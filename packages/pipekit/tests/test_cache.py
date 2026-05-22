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
