"""Tests for `pipekit._base.graph` — Graph + Input + Node."""

from __future__ import annotations

import pytest
from pipekit import (
    CarryState,
    Graph,
    Input,
    Node,
    Operator,
    Sequential,
    Signature,
    StatefulOperator,
)


def test_calling_operator_on_input_returns_node(Scale_cls):
    src = Input("x")
    node = Scale_cls(2.0)(src)
    assert isinstance(node, Node)
    assert not isinstance(node, Input)
    assert node.operator is not None
    assert isinstance(node.operator, Scale_cls)
    assert node.parents == (src,)


def test_input_is_subclass_of_node():
    """Report 2.A.3 — Input subclasses Node so dispatch treats them uniformly."""
    assert issubclass(Input, Node)
    src = Input("x")
    assert isinstance(src, Node)


def test_input_identity_equality():
    """Inputs compare by identity — required for the id-keyed eval cache."""
    a = Input("x")
    b = Input("x")  # same name, different instance
    assert a != b
    assert hash(a) != hash(b)


def test_simple_graph_evaluation(Scale_cls, AddConst_cls):
    x = Input("x")
    y = Scale_cls(2.0)(x)
    z = AddConst_cls(1.0)(y)

    g = Graph(inputs={"x": x}, outputs={"out": z})
    result = g(x=3.0)
    assert result == {"out": 7.0}


def test_multi_input_multi_output(TwoInput_cls, Scale_cls):
    a = Input("a")
    b = Input("b")
    sum_node = TwoInput_cls()(a, b)
    scaled = Scale_cls(2.0)(sum_node)

    g = Graph(
        inputs={"a": a, "b": b},
        outputs={"sum": sum_node, "scaled": scaled},
    )
    result = g(a=1.0, b=2.0)
    assert result == {"sum": 3.0, "scaled": 6.0}


def test_positional_args_bind_in_declaration_order(TwoInput_cls):
    a = Input("a")
    b = Input("b")
    out = TwoInput_cls()(a, b)
    g = Graph(inputs={"a": a, "b": b}, outputs={"out": out})

    # Positional: a=10, b=5
    assert g(10.0, 5.0) == {"out": 15.0}


def test_mixing_positional_and_keyword_raises(TwoInput_cls):
    a = Input("a")
    b = Input("b")
    out = TwoInput_cls()(a, b)
    g = Graph(inputs={"a": a, "b": b}, outputs={"out": out})

    with pytest.raises(TypeError, match=r"positional args .* or keyword inputs"):
        g(1.0, b=2.0)


def test_missing_input_raises(TwoInput_cls):
    a = Input("a")
    b = Input("b")
    out = TwoInput_cls()(a, b)
    g = Graph(inputs={"a": a, "b": b}, outputs={"out": out})

    with pytest.raises(ValueError, match=r"missing required input\(s\): \['b'\]"):
        g(a=1.0)


def test_wrong_positional_arity_raises(TwoInput_cls):
    a = Input("a")
    b = Input("b")
    out = TwoInput_cls()(a, b)
    g = Graph(inputs={"a": a, "b": b}, outputs={"out": out})

    with pytest.raises(TypeError, match="expected 2 positional"):
        g(1.0)


def test_diamond_dependency_evaluates_once(Scale_cls, TwoInput_cls):
    """Diamond: x -> scale -> add(scale_out, scale_out). Tests cache correctness."""
    x = Input("x")
    scaled = Scale_cls(2.0)(x)
    doubled = TwoInput_cls()(scaled, scaled)

    g = Graph(inputs={"x": x}, outputs={"out": doubled})
    assert g(x=3.0) == {"out": 12.0}  # 3*2 + 3*2


def test_cycle_detection(Scale_cls):
    """A node referenced as its own ancestor should raise."""
    x = Input("x")
    node1 = Scale_cls(2.0)(x)
    # Forcibly create a cycle by mutating parents
    node2 = Scale_cls(3.0)(node1)
    node1.parents = (node2,)  # cycle: node1 -> node2 -> node1

    with pytest.raises(ValueError, match="Cycle detected"):
        Graph(inputs={"x": x}, outputs={"out": node2})


def test_undeclared_input_raises(Scale_cls):
    declared = Input("declared")
    undeclared = Input("undeclared")
    out = Scale_cls(2.0)(undeclared)

    with pytest.raises(ValueError, match="referenced by an output but not declared"):
        Graph(inputs={"declared": declared}, outputs={"out": out})


def test_passthrough_output_returns_input_directly(Scale_cls):
    """A Graph can list an Input directly as an output (passthrough)."""
    x = Input("x")
    g = Graph(inputs={"x": x}, outputs={"echo": x})
    assert g(x=42.0) == {"echo": 42.0}


def test_graph_is_an_operator_and_composes_typewise(Scale_cls):
    """A Graph is an Operator and can be `|`-composed.

    Evaluating ``graph | normal_op`` requires the next op to consume
    Graph's ``dict`` output, which is a downstream concern. Here we
    confirm only the type relationship — composition itself works.
    """
    x = Input("x")
    y = Scale_cls(2.0)(x)
    g = Graph(inputs={"x": x}, outputs={"y": y})

    assert isinstance(g, Operator)
    composed = g | Scale_cls(3.0)
    assert isinstance(composed, Sequential)
    assert len(composed) == 2


def test_get_config_records_the_whole_topology(Scale_cls, AddConst_cls):
    x = Input("x")
    h = Scale_cls(2.0)(x)
    y = AddConst_cls(1.0)(h)
    g = Graph(inputs={"x": x}, outputs={"y": y, "raw": x})
    cfg = g.get_config()
    assert cfg["inputs"] == ["x"]
    assert [op["class"] for op in cfg["operators"]] == ["Scale", "AddConst"]
    assert cfg["operators"][0]["config"] == {"factor": 2.0}
    assert cfg["nodes"] == [
        {"operator": 0, "parents": [{"input": "x"}]},
        {"operator": 1, "parents": [{"node": 0}]},
    ]
    assert cfg["outputs"] == {"y": {"node": 1}, "raw": {"input": "x"}}


def test_state_differs_when_an_upstream_step_differs(Scale_cls, AddConst_cls):
    """Two graphs that differ only upstream of their outputs must not collide."""

    def build(factor: float) -> Graph:
        x = Input("x")
        y = AddConst_cls(1.0)(Scale_cls(factor)(x))
        return Graph(inputs={"x": x}, outputs={"y": y})

    assert build(2.0).state != build(3.0).state
    assert build(2.0).state == build(2.0).state


def test_get_config_records_a_shared_operator_once(Scale_cls, TwoInput_cls):
    """An operator used by two nodes appears once in `operators`."""
    a = Input("a")
    b = Input("b")
    scale = Scale_cls(2.0)
    out = TwoInput_cls()(scale(a), scale(b))
    cfg = Graph(inputs={"a": a, "b": b}, outputs={"out": out}).get_config()
    assert [op["class"] for op in cfg["operators"]] == ["Scale", "TwoInput"]
    assert [node["operator"] for node in cfg["nodes"]] == [0, 0, 1]
    assert cfg["nodes"][2]["parents"] == [{"node": 0}, {"node": 1}]


def test_get_config_distinguishes_wiring(TwoInput_cls):
    """Swapping which input feeds which argument changes the config."""

    def build(swap: bool) -> Graph:
        a = Input("a")
        b = Input("b")
        out = TwoInput_cls()(b, a) if swap else TwoInput_cls()(a, b)
        return Graph(inputs={"a": a, "b": b}, outputs={"out": out})

    assert build(True).state != build(False).state


def test_describe_includes_topology(Scale_cls, AddConst_cls):
    x = Input("x")
    y = Scale_cls(2.0)(x)
    z = AddConst_cls(1.0)(y)
    g = Graph(inputs={"x": x}, outputs={"z": z})
    text = g.describe()
    assert "inputs=['x']" in text
    assert "outputs=['z']" in text
    assert "Scale" in text and "AddConst" in text


def test_repr(Scale_cls):
    x = Input("x")
    y = Scale_cls(2.0)(x)
    g = Graph(inputs={"x": x}, outputs={"y": y})
    assert repr(g) == "Graph(inputs=[x], outputs=[y])"


def test_node_rejects_non_node_parents(Scale_cls):
    """Constructing a Node with non-Node parents raises immediately."""
    with pytest.raises(TypeError, match="expected Node"):
        Node(operator=Scale_cls(2.0), parents=(42,))


# ── state ────────────────────────────────────────────────────────────────────


class Tally(CarryState):
    """Which stateful steps ran, in order, and a running total."""

    def __init__(self, seen: tuple[str, ...] = (), total: float = 0.0) -> None:
        self.seen = seen
        self.total = total


class Accumulate(StatefulOperator):
    """x ↦ x, and the state records the step's name and adds x to the total."""

    def __init__(self, name: str) -> None:
        self.name = name

    def _apply(self, x, state):
        return x, Tally((*state.seen, self.name), state.total + x)


def test_graph_threads_state_through_a_stateful_node(Scale_cls):
    x = Input("x")
    y = Accumulate("acc")(Scale_cls(2.0)(x))
    g = Graph(inputs={"x": x}, outputs={"y": y})

    out, state = g(x=3.0, state=Tally())
    assert out == {"y": 6.0}
    assert state.seen == ("acc",)
    assert state.total == 6.0

    # the returned state feeds the next call
    _, state = g(x=1.0, state=state)
    assert state.total == 8.0


def test_graph_threads_state_in_topological_order(Scale_cls):
    x = Input("x")
    first = Accumulate("first")(x)
    second = Accumulate("second")(Scale_cls(10.0)(first))
    g = Graph(inputs={"x": x}, outputs={"y": second})
    _, state = g(x=1.0, state=Tally())
    assert state.seen == ("first", "second")
    assert state.total == 11.0


def test_stateful_graph_requires_a_state():
    x = Input("x")
    g = Graph(inputs={"x": x}, outputs={"y": Accumulate("acc")(x)})
    assert g._is_stateful
    with pytest.raises(TypeError, match="without a `state`"):
        g(x=1.0)


def test_stateful_graph_cannot_name_an_input_state():
    s = Input("state")
    with pytest.raises(ValueError, match="named 'state'"):
        Graph(inputs={"state": s}, outputs={"y": Accumulate("acc")(s)})


def test_stateless_graph_keeps_an_input_named_state(Scale_cls):
    s = Input("state")
    g = Graph(inputs={"state": s}, outputs={"y": Scale_cls(2.0)(s)})
    assert g(state=3.0) == {"y": 6.0}


def test_stateless_graph_passes_a_given_state_through(Scale_cls):
    """As for Sequential: supplying a state to a stateless graph returns it."""
    x = Input("x")
    g = Graph(inputs={"x": x}, outputs={"y": Scale_cls(2.0)(x)})
    out, state = g(x=1.0, state="unchanged")
    assert out == {"y": 2.0}
    assert state == "unchanged"


def test_stateful_graph_inside_a_stateful_sequential(Scale_cls):
    x = Input("x")
    inner = Graph(inputs={"x": x}, outputs={"y": Accumulate("inner")(x)})
    pipe = Sequential([Scale_cls(2.0), inner])
    assert pipe._is_stateful
    out, state = pipe(1.0, Tally())
    assert out == {"y": 2.0}
    assert state.seen == ("inner",)


def test_stateful_graph_nested_in_a_graph():
    x = Input("x")
    inner = Graph(inputs={"x": x}, outputs={"y": Accumulate("inner")(x)})
    outer_x = Input("x")
    outer = Graph(
        inputs={"x": outer_x},
        outputs={"z": Accumulate("outer")(outer_x), "w": inner(outer_x)},
    )
    out, state = outer(x=2.0, state=Tally())
    assert out == {"z": 2.0, "w": {"y": 2.0}}
    # outputs are visited in declaration order, so "z" comes before "w"
    assert state.seen == ("outer", "inner")
    assert state.total == 4.0


# ── signatures ───────────────────────────────────────────────────────────────
class MeanOver(Operator):
    """Mean over one named dimension: drops it from the signature."""

    def __init__(self, dim: str) -> None:
        self.dim = dim

    def _apply(self, x):
        return x

    def compute_output_signature(self, signature):
        return None if signature is None else signature.drop_dims((self.dim,))


def _cube() -> Signature:
    return Signature(dims=(("time", 12), ("y", 64), ("x", 64)), dtype="float32")


def test_output_signatures_propagate_through_nodes(TwoInput_cls):
    a = Input("a")
    b = Input("b")
    mean_a = MeanOver("time")(a)
    mean_b = MeanOver("time")(b)
    g = Graph(
        inputs={"a": a, "b": b},
        outputs={"anomaly": TwoInput_cls()(mean_a, mean_b), "mean_a": mean_a},
    )
    image = Signature(dims=(("y", 64), ("x", 64)), dtype="float32")
    assert g.output_signatures({"a": _cube(), "b": _cube()}) == {
        "anomaly": image,
        "mean_a": image,
    }
    # inputs that disagree make a multi-input node unknown, not wrong
    other = _cube().replace_dims({"x": 32})
    assert g.output_signatures({"a": _cube(), "b": other})["anomaly"] is None


def test_output_signatures_take_a_bare_signature_for_one_input():
    x = Input("x")
    g = Graph(inputs={"x": x}, outputs={"y": MeanOver("time")(x)})
    assert g.output_signatures(_cube())["y"].dims == (("y", 64), ("x", 64))


def test_output_signatures_reject_undeclared_inputs():
    x = Input("x")
    g = Graph(inputs={"x": x}, outputs={"y": MeanOver("time")(x)})
    with pytest.raises(ValueError, match="undeclared"):
        g.output_signatures({"z": _cube()})


def test_summary_lists_nodes_and_outputs():
    x = Input("x")
    g = Graph(inputs={"x": x}, outputs={"y": MeanOver("time")(x)})
    table = g.summary(_cube())
    assert "[0] MeanOver(x)" in table
    assert "(time=12, y=64, x=64) float32" in table
    assert "→ y = [0]" in table
    assert "(y=64, x=64) float32" in table


def test_a_graph_reports_no_single_signature_inside_a_pipeline():
    x = Input("x")
    g = Graph(inputs={"x": x}, outputs={"y": MeanOver("time")(x)})
    assert g.compute_output_signature(_cube()) is None
