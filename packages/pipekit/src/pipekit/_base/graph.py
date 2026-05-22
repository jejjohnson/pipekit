"""Graph composition — symbolic operator graphs.

`Sequential` covers the linear case; `Graph` covers the rest:
branching outputs, multi-input fusion, diamond dependencies.
Construction is symbolic — calling an `Operator` on `Input` / `Node`
instances builds up a graph; running it is `_apply(**inputs_by_name)`
(or positionally).

Pattern::

    img = Input("image")
    ref = Input("reference")
    ndvi = NDVI(red_idx=2, nir_idx=3)(img)
    rmse = RMSE(axis=(-2, -1))(ndvi, ref)

    g = Graph(
        inputs={"image": img, "reference": ref},
        outputs={"ndvi": ndvi, "rmse": rmse},
    )
    result = g(image=img_gt, reference=ref_gt)
    # {"ndvi": GeoTensor, "rmse": scalar}

`Input` is a subclass of `Node` (master plan Report 2.A.3): the
topological sort can treat all vertices uniformly, with the special
case being "this node has no operator — supply its value from the
caller's kwargs".

See master plan Report 2, Group A.3.
"""

from __future__ import annotations

from typing import Any

from pipekit._base.operator import Carrier, Operator


class Node:
    """A vertex in a `Graph`.

    Created automatically by `Operator.__call__` when any positional
    argument is a `Node` (or `Input`, which subclasses `Node`). Carries
    the operator and its parent vertices. Compared by identity so the
    ``id(...)``-keyed evaluation cache during topological sort works.

    Raises:
        TypeError: if any element of ``parents`` is not a `Node`.
            `Graph` evaluation walks ``parents`` recursively and indexes
            an ``id(...)``-keyed cache; non-Node parents would crash with
            an unhelpful ``KeyError`` deep in the traversal.
    """

    __slots__ = ("operator", "parents")

    def __init__(
        self,
        operator: Operator | None,
        parents: tuple[Any, ...],
    ) -> None:
        for i, p in enumerate(parents):
            if not isinstance(p, Node):
                raise TypeError(
                    f"Node.parents[{i}] is {type(p).__name__}, expected Node. "
                    "Wrap literal values in `Const(value)` to lift them into "
                    "the graph."
                )
        self.operator = operator
        self.parents = parents

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)


class Input(Node):
    """A named entry point into a `Graph`.

    `Operator.__call__` recognises `Node` (and any subclass, including
    `Input`) as graph-construction mode. `Graph._apply` supplies the
    value at run-time from the caller's kwargs / positional args.

    Subclassing `Node` (master plan Report 2.A.3, "xr_toolz shape")
    lets the topological sort treat all vertices uniformly with one
    special case: "this node has no operator — source it from the
    caller's inputs".
    """

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        super().__init__(operator=None, parents=())
        self.name = name


class Graph(Operator):
    """A symbolic operator graph with multiple inputs and outputs.

    Construction is implicit — calling operators on `Input` / `Node`
    instances builds the graph; ``Graph(inputs=..., outputs=...)``
    wraps the result. ``_apply(**inputs)`` evaluates in topological
    order.

    Inherits from `Operator` so a `Graph` satisfies the same interface
    as any other operator. ``Operator.__call__`` dispatches keyword
    args straight through to `Graph._apply`; positional args are bound
    to declared `Input`s in declaration order.

    Args:
        inputs: Map of ``input-name → Input`` placeholders. The keys
            are the keyword names accepted by ``__call__``.
        outputs: Map of ``output-name → Node`` (or ``Input``, if the
            output is a direct passthrough). The keys are the keys of
            the returned dict.

    Raises:
        ValueError: if the graph contains a cycle, if an `Input`
            referenced by a node isn't declared in ``inputs``, or if
            a required input is missing at call time.
    """

    __config_mixin_auto__ = False  # custom get_config — topology, not kwargs

    def __init__(
        self,
        inputs: dict[str, Input],
        outputs: dict[str, Node],
    ) -> None:
        self.inputs = inputs
        self.outputs = outputs
        self._order = self._topological_sort()

    def _topological_sort(self) -> list[Node]:
        """Return the topological ordering of non-input `Node`s.

        Inputs are sources (supplied by the caller) and are not
        included in the work list. Direct-passthrough outputs (an
        `Input` listed under ``outputs=``) are likewise skipped.
        """
        declared_inputs = set(map(id, self.inputs.values()))
        order: list[Node] = []
        visited: set[int] = set()
        on_stack: set[int] = set()

        def visit(node: Node) -> None:
            node_id = id(node)
            if node_id in visited:
                return
            if node_id in on_stack:
                raise ValueError(
                    "Cycle detected in graph — operator graphs must be DAGs."
                )
            on_stack.add(node_id)
            for parent in node.parents:
                visit(parent)
            on_stack.discard(node_id)
            visited.add(node_id)
            if isinstance(node, Input):
                if node_id not in declared_inputs:
                    raise ValueError(
                        f"Input {node.name!r} is referenced by an output but "
                        "not declared in `inputs=`."
                    )
                return
            order.append(node)

        for output in self.outputs.values():
            visit(output)
        return order

    def _apply(self, *args: Carrier, **inputs: Carrier) -> dict[str, Any]:
        """Evaluate the graph with the supplied inputs.

        Accepts inputs either positionally (bound to declared `Input`s
        in declaration order) or by keyword. The positional form lets
        single-input graphs compose into `Sequential` cleanly and lets
        graphs nest inside other graphs — both shapes route values
        through `Operator.__call__`, which only splat positionally.

        Args:
            *args: One value per declared `Input`, in declaration order.
                Mutually exclusive with ``**inputs``.
            **inputs: One value per declared `Input`, keyed by name.

        Returns:
            ``{output-name: result}`` for each declared output.
        """
        if args and inputs:
            raise TypeError(
                "Graph._apply accepts either positional args (bound to inputs "
                "in declaration order) or keyword inputs, not both."
            )
        if args:
            if len(args) != len(self.inputs):
                raise TypeError(
                    f"Graph expected {len(self.inputs)} positional argument(s) "
                    f"to bind to inputs {list(self.inputs)}, got {len(args)}."
                )
            inputs = dict(zip(self.inputs, args, strict=True))

        missing = set(self.inputs) - set(inputs)
        if missing:
            raise ValueError(f"Graph missing required input(s): {sorted(missing)}")

        cache: dict[int, Any] = {
            id(self.inputs[name]): inputs[name] for name in self.inputs
        }
        for node in self._order:
            assert node.operator is not None  # guaranteed by _topological_sort
            node_args = tuple(cache[id(p)] for p in node.parents)
            # Route through __call__ so nested operators (Graph, Sequential)
            # get their own dispatch, not just bare _apply.
            cache[id(node)] = node.operator(*node_args)

        return {name: cache[id(node)] for name, node in self.outputs.items()}

    def get_config(self) -> dict[str, Any]:
        """Best-effort config — node operators' configs, by output name.

        Graphs are inherently runtime-defined (the topology comes from
        Python object identity), so this is a debug repr rather than
        a faithful YAML round-trip. A YAML-format graph would store
        the topology as a list of (op, parent-keys) records.
        """
        return {
            "inputs": list(self.inputs),
            "outputs": {
                name: {
                    "class": type(node.operator).__name__
                    if node.operator is not None
                    else "Input",
                    "config": node.operator.get_config()
                    if node.operator is not None
                    else {},
                }
                for name, node in self.outputs.items()
            },
        }

    def __repr__(self) -> str:
        ins = ", ".join(self.inputs)
        outs = ", ".join(self.outputs)
        return f"Graph(inputs=[{ins}], outputs=[{outs}])"

    def describe(self) -> str:
        """Return a short topology description.

        Lists inputs, internal nodes (in topological order), and
        outputs. Useful for notebook exploration.
        """
        lines = [
            f"Graph(inputs={list(self.inputs)}, outputs={list(self.outputs)})",
            "  internal nodes (topological order):",
        ]
        for i, node in enumerate(self._order):
            op_name = type(node.operator).__name__ if node.operator else "Input"
            lines.append(f"    [{i}] {op_name}")
        return "\n".join(lines)
