"""Operators that can't work on a JAX tracer say so instead of misbehaving."""

from __future__ import annotations

import subprocess
import sys
import warnings

import pytest
from pipekit import Cache, Lambda, Sequential
from pipekit._base.tracing import is_traced
from pipekit.blocks import Sink
from pipekit.observe import Profile, ShapeTrace, Snapshot, Tap


jax = pytest.importorskip("jax")
jnp = jax.numpy


def _double() -> Lambda:
    return Lambda(lambda x: 2 * x)


def test_is_traced_finds_a_tracer_inside_a_pytree() -> None:
    seen: list[bool] = []

    def f(x):
        seen.extend([is_traced(x), is_traced({"a": [x]}), is_traced(1.0)])
        return x

    jax.jit(f)(jnp.ones(2))
    assert seen == [True, True, False]
    assert not is_traced(jnp.ones(2))


def test_cache_raises_under_jit_and_names_it() -> None:
    cache = Cache(_double())
    with pytest.raises(TypeError, match=r"jax\.jit"):
        jax.jit(cache)(jnp.arange(3.0))
    assert (cache.hits, cache.misses) == (0, 0)


def test_cache_still_memoises_concrete_jax_arrays() -> None:
    cache = Cache(_double())
    x = jnp.arange(3.0)
    assert bool(jnp.array_equal(cache(x), cache(x)))
    assert (cache.hits, cache.misses) == (1, 1)


def test_snapshot_raises_under_jit_and_stores_nothing() -> None:
    snapshot = Snapshot()
    with pytest.raises(TypeError, match="traced"):
        jax.jit(snapshot.at("z"))(jnp.ones(2))
    assert snapshot.captures == {}


def test_sink_raises_under_jit_and_writes_nothing() -> None:
    written: list[object] = []
    with pytest.raises(TypeError, match="traced"):
        jax.jit(Sink(written.append))(jnp.ones(2))
    assert written == []


def test_tap_warns_once_per_instance_and_still_fires() -> None:
    fired: list[object] = []
    tap = Tap(fired.append)
    with pytest.warns(UserWarning, match="trace time"):
        jax.jit(tap)(jnp.ones(2))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # A new shape forces a second trace of the same instance.
        jax.jit(tap)(jnp.ones(3))
    assert len(fired) == 2


def test_profile_warns_once_per_instance() -> None:
    profile = Profile()
    op = profile.wrap(_double())
    with pytest.warns(UserWarning, match="trace time"):
        jax.jit(op)(jnp.ones(2))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        jax.jit(op)(jnp.ones(3))


def test_shape_trace_prints_once_and_never_warns() -> None:
    lines: list[str] = []
    op = jax.jit(ShapeTrace(printer=lines.append))
    x = jnp.ones(2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for _ in range(3):
            op(x)
    assert len(lines) == 1


def test_observers_stay_silent_on_concrete_values() -> None:
    fired: list[object] = []
    written: list[object] = []
    snapshot = Snapshot()
    profile = Profile()
    pipe = Sequential(
        [
            Tap(fired.append),
            snapshot.at("z"),
            Sink(written.append),
            profile.wrap(_double()),
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = pipe(jnp.ones(2))
    assert bool(jnp.array_equal(out, 2 * jnp.ones(2)))
    assert len(fired) == len(written) == 1
    assert "z" in snapshot.captures


def test_operators_accept_a_dict_jax_cannot_flatten() -> None:
    # JAX sorts dict keys when flattening, and 0 and "a" can't be compared.
    carrier = {0: "zero", "a": "letter"}
    with pytest.raises(TypeError):
        jax.tree_util.tree_leaves(carrier)

    assert not is_traced(carrier)
    fired: list[object] = []
    written: list[object] = []
    snapshot = Snapshot()
    profile = Profile()
    cache = Cache(Lambda(lambda x: x))
    pipe = Sequential(
        [
            Tap(fired.append),
            snapshot.at("z"),
            Sink(written.append),
            profile.wrap(Lambda(lambda x: x)),
            cache,
        ]
    )
    assert pipe(carrier) == carrier
    assert fired == written == [carrier]
    assert snapshot.captures["z"] == carrier


def test_core_works_with_jax_absent() -> None:
    # Setting a sys.modules entry to None makes `import jax` raise ImportError.
    code = (
        "import sys; sys.modules['jax'] = None\n"
        "from pipekit import Cache, Lambda\n"
        "c = Cache(Lambda(lambda x: 2 * x)); c(1); c(1)\n"
        "assert (c.hits, c.misses) == (1, 1)\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
