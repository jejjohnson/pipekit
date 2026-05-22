"""Tests for `pipekit._base.graph` — Graph + Input + Node."""

from __future__ import annotations

import pytest
from pipekit import Graph, Input, Node, Operator, Sequential


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


def test_get_config_lists_outputs_by_class(Scale_cls):
    x = Input("x")
    y = Scale_cls(2.0)(x)
    g = Graph(inputs={"x": x}, outputs={"y": y})
    cfg = g.get_config()
    assert cfg["inputs"] == ["x"]
    assert cfg["outputs"]["y"]["class"] == "Scale"
    assert cfg["outputs"]["y"]["config"] == {"factor": 2.0}


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
