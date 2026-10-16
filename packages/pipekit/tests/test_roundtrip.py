"""Nested pipelines round-trip through `dumps` / `loads` (and the sandbox)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pipekit import (
    AsyncMap,
    BatchedMap,
    Cache,
    Fanout,
    Graph,
    Identity,
    Input,
    Lambda,
    Operator,
    ProcessMap,
    Retry,
    Sequential,
    ThreadMap,
    Try,
    dumps,
    loads,
    loads_sandboxed,
    register,
    unregister,
)


class FlakyError(Exception):
    """An exception defined outside the standard library."""


class Built(Operator):
    """Counts its constructions, to show a refused pipeline builds nothing."""

    count = 0

    def __init__(self, k: int = 0) -> None:
        type(self).count += 1
        self.k = k

    def _apply(self, x: Any) -> Any:
        return x


def round_trip(op: Operator) -> Operator:
    return loads(dumps(op))


def test_sequential_round_trips(Scale_cls, AddConst_cls):
    pipe = Scale_cls(2.0) | AddConst_cls(1.0)
    back = round_trip(pipe)
    assert isinstance(back, Sequential)
    assert back.state == pipe.state
    assert back(3.0) == pipe(3.0) == 7.0


def test_wrappers_round_trip_nested(Scale_cls):
    """ThreadMap(Cache(Retry(...))): three wrappers deep, each rebuilt."""
    inner = Retry(Scale_cls(2.0), attempts=3, base_delay=0.0, on=(FlakyError,))
    pipe = ThreadMap(Cache(inner), n_workers=2)
    back = round_trip(pipe)
    assert back.state == pipe.state
    assert back([1.0, 2.0]) == [2.0, 4.0]
    assert back.op.inner.on == (FlakyError,)


@pytest.mark.parametrize(
    "make",
    [
        lambda S: Fanout({"a": S(2.0), "b": Identity()}),
        lambda S: Try(primary=S(2.0), fallback=Identity(), on=(KeyError,)),
        lambda S: ProcessMap(S(2.0), n_workers=2, on_error="log_and_continue"),
        lambda S: AsyncMap(S(2.0), semaphore=3),
        lambda S: BatchedMap(Identity(), batch_size=4, flatten=True),
    ],
    ids=["Fanout", "Try", "ProcessMap", "AsyncMap", "BatchedMap"],
)
def test_containers_round_trip(make, Scale_cls):
    op = make(Scale_cls)
    back = round_trip(op)
    assert type(back) is type(op)
    assert back.state == op.state


def test_graph_round_trips(Scale_cls, AddConst_cls, TwoInput_cls):
    """Shared operators, a nested Sequential and a passthrough output."""
    a = Input("a")
    b = Input("b")
    clean = Scale_cls(2.0) | AddConst_cls(1.0)
    total = TwoInput_cls()(clean(a), clean(b))
    graph = Graph(inputs={"a": a, "b": b}, outputs={"total": total, "raw": a})

    back = round_trip(graph)
    assert isinstance(back, Graph)
    assert back.state == graph.state
    assert back(a=1.0, b=2.0) == graph(a=1.0, b=2.0) == {"total": 8.0, "raw": 1.0}
    # the shared `clean` is still one object, used by both nodes
    first, second = back._order[0], back._order[1]
    assert first.operator is second.operator


def test_graph_nested_in_a_sequential_round_trips(Scale_cls):
    x = Input("x")
    inner = Graph(inputs={"x": x}, outputs={"y": Scale_cls(3.0)(x)})
    pipe = Scale_cls(2.0) | inner
    back = round_trip(pipe)
    assert back.state == pipe.state
    assert back(1.0) == {"y": 6.0}


def test_tuples_survive_an_in_memory_round_trip(Scale_cls):
    """`from_state` on an in-memory state keeps tuples; JSON makes them lists."""
    op = Retry(Scale_cls(2.0), attempts=2, base_delay=0.0, on=(KeyError,))
    assert Operator.from_state(op.state).on == (KeyError,)


def test_a_closure_anywhere_in_the_tree_is_refused(Scale_cls):
    pipe = Cache(Scale_cls(2.0) | Lambda(abs))
    with pytest.raises(RuntimeError, match="forbid_in_yaml"):
        round_trip(pipe)


def test_exception_records_resolve_only_loaded_exception_classes():
    state = Try(primary=Identity(), fallback=Identity(), on=(KeyError,)).state
    text = json.dumps(state)
    not_loaded = text.replace("builtins.KeyError", "no_such_module.Boom")
    with pytest.raises(TypeError, match="not a loaded exception class"):
        loads(not_loaded)
    not_an_exception = text.replace("builtins.KeyError", "builtins.int")
    with pytest.raises(TypeError, match="not a loaded exception class"):
        loads(not_an_exception)


def test_graph_config_rejects_a_forward_reference(Scale_cls):
    x = Input("x")
    graph = Graph(inputs={"x": x}, outputs={"y": Scale_cls(2.0)(x)})
    state = graph.state
    state["config"]["nodes"][0]["parents"] = [{"node": 0}]
    with pytest.raises(ValueError, match="not an earlier node"):
        Operator.from_state(state)


def test_sandbox_checks_every_nested_class():
    Built.count = 0
    register(Sequential)
    try:
        text = dumps(Sequential([Built(1), Built(2)]))
        built_before = Built.count
        with pytest.raises(PermissionError, match="Built"):
            loads_sandboxed(text)
        # refused before anything was constructed
        assert Built.count == built_before

        register(Built)
        back = loads_sandboxed(text)
        assert back.state == loads(text).state
    finally:
        unregister(Sequential)
        unregister(Built)


def test_sandbox_needs_the_container_registered_too():
    register(Built)
    try:
        with pytest.raises(PermissionError, match="Sequential"):
            loads_sandboxed(dumps(Sequential([Built(1)])))
    finally:
        unregister(Built)


def test_from_state_sandboxed_takes_a_record_in_memory():
    from pipekit import from_state_sandboxed

    state = Sequential([Built(1)]).state
    with pytest.raises(PermissionError, match="Sequential"):
        from_state_sandboxed(state)
    register(Sequential)
    register(Built)
    try:
        assert from_state_sandboxed(state).state == state
    finally:
        unregister(Sequential)
        unregister(Built)
