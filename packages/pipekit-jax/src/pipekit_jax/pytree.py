"""Register pipekit's composition containers as JAX pytrees.

Importing `pipekit_jax` registers `pipekit.Sequential` and
`pipekit.Graph` with `jax.tree_util`, so JAX transformations see
*through* a pipeline to the leaves of the operators it holds:

- ``Sequential`` — children are its operators, in order.
- ``Graph`` — children are the node operators in topological order;
  the topology (inputs, node wiring, outputs) is static aux data.

Without the registration a container is a single opaque leaf, and a
learnable coefficient on an operator inside it is a constant to
``jax.grad`` / ``eqx.filter_grad``. pipekit core stays JAX-free: the
registration is an import side effect of this package, like enabling a
``jax.config`` flag.

What the leaves are depends on the operators. An operator that is
itself a pytree (an ``eqx.Module`` subclass such as `JaxModelOp`)
contributes its own leaves. A plain `pipekit.Operator` is one opaque
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

from typing import Any

import jax
from pipekit import Graph, Sequential
from pipekit._base.graph import Node


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


def _graph_flatten(graph: Graph) -> tuple[list[Any], Graph]:
    # The aux data must compare equal across flatten → unflatten round
    # trips, or `tree_map` / `eqx.filter_jit` see two different
    # treedefs. `Graph` compares by identity, so every rebuilt graph
    # points back at the one original template.
    template = getattr(graph, "_pytree_template", graph)
    return [node.operator for node in graph._order], template


def _graph_unflatten(template: Graph, operators: Any) -> Graph:
    remap: dict[int, Node] = {id(i): i for i in template.inputs.values()}
    for node, op in zip(template._order, operators, strict=True):
        parents = tuple(remap[id(p)] for p in node.parents)
        remap[id(node)] = Node(operator=op, parents=parents)
    new = object.__new__(Graph)
    new.inputs = template.inputs
    new.outputs = {name: remap[id(node)] for name, node in template.outputs.items()}
    new._order = [remap[id(node)] for node in template._order]
    new._pytree_template = template
    return new


jax.tree_util.register_pytree_node(
    Sequential, _sequential_flatten, _sequential_unflatten
)
jax.tree_util.register_pytree_node(Graph, _graph_flatten, _graph_unflatten)


__all__: list[str] = []
