"""Register pipekit's composition containers as JAX pytrees.

Importing `pipekit_jax` registers `pipekit.Sequential`, `pipekit.Graph`
and `pipekit_jax.JaxModelOp` with `jax.tree_util`, so JAX transformations
see *through* a pipeline to the leaves of the operators it holds:

- ``Sequential`` — children are its operators, in order.
- ``Graph`` — children are the node operators in topological order;
  the topology (inputs, node wiring, outputs) is static aux data that
  holds no operators or parameters.
- `JaxModelOp` — the child is its wrapped ``eqx.Module``.

Without the registration a container is a single opaque leaf, and a
learnable coefficient on an operator inside it is a constant to
``jax.grad`` / ``eqx.filter_grad``. pipekit core stays JAX-free: the
registration is an import side effect of this package, like enabling a
``jax.config`` flag.

What the leaves are depends on the operators. An operator that is
itself a pytree (an ``eqx.Module`` subclass, or `JaxModelOp` via its
module) contributes its own leaves. A plain `pipekit.Operator` is one opaque
leaf — fine under ``eqx.filter_jit`` / ``eqx.filter_grad`` (non-array
leaves are static there), but not as a direct argument to
``jax.jit``.

Caveats:

- Registration is by exact type. Subclasses of ``Sequential`` /
  ``Graph`` are not registered automatically.
- Operator *sharing* is not preserved. An operator instance used by two
  graph nodes flattens to two children; after an unflatten with new
  leaves they are two independent objects. (Equinox modules behave
  the same way.)
- Other containers (``Fanout``, ``Branch``, ``Switch``, ``Cache``,
  ``Try``, the parallel maps) are not registered yet.

Example:
    >>> import equinox as eqx, jax, jax.numpy as jnp
    >>> import pipekit_jax  # registers the containers
    >>> from pipekit import Operator, Sequential
    >>> class Scale(Operator, eqx.Module):
    ...     factor: jax.Array
    ...     def _apply(self, x):
    ...         return self.factor * x
    >>> pipe = Sequential([Scale(jnp.asarray(2.0)), Scale(jnp.asarray(3.0))])
    >>> len(jax.tree.leaves(pipe))
    2
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
from pipekit import Graph, Sequential
from pipekit._base.graph import Input, Node

from pipekit_jax.model_op import JaxModelOp


# ---------------------------------------------------------------------
# Sequential
# ---------------------------------------------------------------------


def _sequential_flatten(seq: Sequential) -> tuple[list[Any], None]:
    return list(seq.operators), None


def _sequential_unflatten(_: None, operators: Any) -> Sequential:
    # Bypass ``Sequential.__init__``: JAX rebuilds trees with placeholder
    # leaves (``None`` from ``eqx.partition``, sentinels in ``vmap`` axis
    # specs), which the constructor's ``require_operator`` check rejects.
    new = object.__new__(Sequential)
    new.operators = list(operators)
    return new


# ---------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------

# A reference to a graph vertex: ("in", input_name) or ("node", index into
# the topological order).
_Ref = tuple[str, Any]


class _GraphTopology(NamedTuple):
    """Wiring of a `Graph` with no operators in it — the pytree aux data.

    Plain tuples of strings and ints: hashable, compared by value, and
    holding no reference to the source graph or its parameters. Two
    graphs with the same wiring therefore share a treedef (no retrace),
    and rebuilding a graph never keeps the original alive.
    """

    inputs: tuple[str, ...]
    parents: tuple[tuple[_Ref, ...], ...]  # per node, in topological order
    outputs: tuple[tuple[str, _Ref], ...]


def _graph_flatten(graph: Graph) -> tuple[list[Any], _GraphTopology]:
    input_name = {id(node): name for name, node in graph.inputs.items()}
    index = {id(node): i for i, node in enumerate(graph._order)}

    def ref(node: Node) -> _Ref:
        if id(node) in input_name:
            return ("in", input_name[id(node)])
        return ("node", index[id(node)])

    topology = _GraphTopology(
        inputs=tuple(graph.inputs),
        parents=tuple(tuple(ref(p) for p in node.parents) for node in graph._order),
        outputs=tuple((name, ref(node)) for name, node in graph.outputs.items()),
    )
    return [node.operator for node in graph._order], topology


def _graph_unflatten(topology: _GraphTopology, operators: Any) -> Graph:
    inputs = {name: Input(name) for name in topology.inputs}
    nodes: list[Node] = []

    def resolve(r: _Ref) -> Node:
        kind, key = r
        return inputs[key] if kind == "in" else nodes[key]

    for op, parents in zip(operators, topology.parents, strict=True):
        nodes.append(Node(operator=op, parents=tuple(resolve(p) for p in parents)))
    new = object.__new__(Graph)
    new.inputs = inputs
    new.outputs = {name: resolve(r) for name, r in topology.outputs}
    new._order = nodes
    return new


# ---------------------------------------------------------------------
# JaxModelOp — a plain Operator wrapping an eqx.Module
# ---------------------------------------------------------------------


def _model_op_flatten(op: JaxModelOp) -> tuple[list[Any], None]:
    return [op.module], None


def _model_op_unflatten(_: None, children: Any) -> JaxModelOp:
    # Bypass __init__: its isinstance(module, eqx.Module) check would
    # reject the placeholder leaves JAX passes during transformations.
    new = object.__new__(JaxModelOp)
    new.module = children[0]
    return new


jax.tree_util.register_pytree_node(
    Sequential, _sequential_flatten, _sequential_unflatten
)
jax.tree_util.register_pytree_node(Graph, _graph_flatten, _graph_unflatten)
jax.tree_util.register_pytree_node(JaxModelOp, _model_op_flatten, _model_op_unflatten)


__all__: list[str] = []
