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

from typing import Any, Self

from pipekit._base.operator import _MISSING, Carrier, Operator
from pipekit.adapters import signature_of


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

    State:
        A graph that contains a stateful node (a `StatefulOperator`, or
        a `Sequential` / `Graph` that contains one) threads a carry state
        the way `Sequential` does: call it as
        ``graph(a=..., b=..., state=s)`` and it returns
        ``(outputs, state)``. The state passes through the stateful nodes
        in topological order (the order `describe` lists); stateless
        nodes ignore it. Inside a stateful `Sequential` the graph receives
        ``(carrier, state)`` positionally, like any stateful step.

    Raises:
        ValueError: if the graph contains a cycle, if an `Input`
            referenced by a node isn't declared in ``inputs``, if a
            required input is missing at call time, or if a stateful
            graph declares an input named ``state`` (the name is the
            carry state's).
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
        if self._is_stateful and "state" in inputs:
            raise ValueError(
                "A stateful Graph cannot declare an input named 'state': "
                "`state=` is how its carry state is passed. Rename the input."
            )

    @property
    def _is_stateful(self) -> bool:  # type: ignore[override]
        """True if any node's operator is stateful (as for `Sequential`)."""
        return any(
            getattr(node.operator, "_is_stateful", False) for node in self._order
        )

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

    def _apply(self, *args: Carrier, **inputs: Carrier) -> Any:
        """Evaluate the graph with the supplied inputs (and carry state).

        Accepts inputs either positionally (bound to declared `Input`s
        in declaration order) or by keyword. The positional form lets
        single-input graphs compose into `Sequential` cleanly and lets
        graphs nest inside other graphs — both shapes route values
        through `Operator.__call__`, which only splat positionally.

        Args:
            *args: One value per declared `Input`, in declaration order,
                then, for a stateful call, the state. Mutually exclusive
                with ``**inputs``.
            **inputs: One value per declared `Input`, keyed by name, and
                ``state=`` for a stateful call (unless ``state`` is a
                declared input of a stateless graph).

        Returns:
            ``{output-name: result}`` for each declared output; for a
            stateful call, ``(outputs, state)``.

        Raises:
            TypeError: if positional and keyword inputs are mixed, if the
                number of positional arguments fits neither the inputs
                nor the inputs plus a state, or if a stateful graph is
                called without a state.
        """
        if args and inputs:
            raise TypeError(
                "Graph._apply accepts either positional args (bound to inputs "
                "in declaration order) or keyword inputs, not both."
            )
        state: Any = _MISSING
        if args and len(args) == len(self.inputs) + 1:
            # (carrier..., state): how a stateful Sequential or Graph calls it
            *args_list, state = args
            args = tuple(args_list)
        if "state" in inputs and "state" not in self.inputs:
            state = inputs.pop("state")
        if self._is_stateful and state is _MISSING:
            raise TypeError(
                "Graph contains stateful operator(s) but was called without a "
                "`state` argument. Supply the initial CarryState: "
                "graph(..., state=s)."
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
            if state is not _MISSING and getattr(node.operator, "_is_stateful", False):
                # stateful nodes take the state last and hand it on, in order
                cache[id(node)], state = node.operator(*node_args, state)
            else:
                cache[id(node)] = node.operator(*node_args)

        outputs = {name: cache[id(node)] for name, node in self.outputs.items()}
        if state is _MISSING:
            return outputs
        return outputs, state

    def get_config(self) -> dict[str, Any]:
        """The whole topology: inputs, operators, nodes and outputs.

        Every node is recorded, upstream ones included, so two graphs
        that differ anywhere have different configs (and so different
        `state` records and content hashes). The record::

            {
                "inputs": ["image", "reference"],
                "operators": [<state>, ...],
                "nodes": [{"operator": 0, "parents": [{"input": "image"}]}, ...],
                "outputs": {"ndvi": {"node": 0}, "raw": {"input": "image"}},
            }

        ``operators`` lists each distinct operator once, by identity, in
        first-use order, so an operator shared by several nodes is not
        duplicated. ``nodes`` is in topological order; a parent or an
        output refers either to a declared input by name or to an
        earlier node by index.
        """
        input_names = {id(node): name for name, node in self.inputs.items()}
        node_index = {id(node): i for i, node in enumerate(self._order)}
        operator_index: dict[int, int] = {}
        operators: list[dict[str, Any]] = []
        nodes: list[dict[str, Any]] = []

        def ref(node: Node) -> dict[str, Any]:
            if id(node) in input_names:
                return {"input": input_names[id(node)]}
            return {"node": node_index[id(node)]}

        for node in self._order:
            assert node.operator is not None  # guaranteed by _topological_sort
            op_id = id(node.operator)
            if op_id not in operator_index:
                operator_index[op_id] = len(operators)
                operators.append(node.operator.state)
            parents = [ref(parent) for parent in node.parents]
            nodes.append({"operator": operator_index[op_id], "parents": parents})

        return {
            "inputs": list(self.inputs),
            "operators": operators,
            "nodes": nodes,
            "outputs": {name: ref(node) for name, node in self.outputs.items()},
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Rebuild a graph from the topology `get_config` records.

        Re-creates each `Input`, then calls each node's operator on its
        parents in the recorded order, which wires the nodes exactly as
        they were; an operator shared by several nodes stays one object.

        Args:
            config: A `get_config` record whose ``operators`` have already
                been rebuilt into `Operator` instances.

        Raises:
            ValueError: if a reference names an undeclared input or a node
                that comes later in the order.
        """
        inputs = {name: Input(name) for name in config["inputs"]}
        operators: list[Operator] = config["operators"]
        built: list[Node] = []

        def lookup(ref: dict[str, Any]) -> Node:
            if "input" in ref:
                if ref["input"] not in inputs:
                    raise ValueError(f"Graph config: undeclared input {ref!r}.")
                return inputs[ref["input"]]
            index = ref["node"]
            if not 0 <= index < len(built):
                raise ValueError(f"Graph config: {ref!r} is not an earlier node.")
            return built[index]

        for spec in config["nodes"]:
            parents = [lookup(ref) for ref in spec["parents"]]
            built.append(operators[spec["operator"]](*parents))
        outputs = {name: lookup(ref) for name, ref in config["outputs"].items()}
        return cls(inputs=inputs, outputs=outputs)

    def __repr__(self) -> str:
        ins = ", ".join(self.inputs)
        outs = ", ".join(self.outputs)
        return f"Graph(inputs=[{ins}], outputs=[{outs}])"

    def compute_output_signature(self, input_signature: Any) -> Any:
        """``None``: a graph returns a dict of outputs, not one carrier.

        Use `output_signatures` for each output's signature, and
        `summary` for the whole table.
        """
        return None

    def output_signatures(self, input_signatures: Any) -> dict[str, Any]:
        """Each output's signature, propagated node by node without data.

        Args:
            input_signatures: ``{input name: Signature}``; a missing name
                means unknown. A single-input graph also takes the bare
                signature.

        Returns:
            ``{output name: Signature | None}``; ``None`` where an
            operator does not track shape, or where a multi-input node's
            inputs disagree (see `summary`).
        """
        signatures = self._node_signatures(input_signatures)
        return {name: signatures[id(node)] for name, node in self.outputs.items()}

    def summary(self, input_signatures: Any = None) -> str:
        """A Keras-style table of the nodes, their input and output signatures.

        Nodes are listed in topological order. Signatures propagate
        through each node's `compute_output_signature`; a node with
        several inputs gets its operator's signature for them only when
        they all agree (an elementwise join), and ``?`` otherwise, since
        `compute_output_signature` takes one input. The outputs follow.

        Args:
            input_signatures: As for `output_signatures`; ``None`` lists
                the structure with every signature unknown.
        """
        signatures = self._node_signatures(input_signatures)
        names = {id(node): name for name, node in self.inputs.items()}

        def label(node: Node) -> str:
            if id(node) in names:
                return names[id(node)]
            return f"[{self._order.index(node)}]"

        def show(signature: Any) -> str:
            if signature is None:
                return "?"
            # a Signature's compact "(y=64, x=64) float32" form
            fmt = getattr(signature, "format", None)
            return fmt() if callable(fmt) else repr(signature)

        rows: list[tuple[str, str, str]] = []
        for i, node in enumerate(self._order):
            parents = ", ".join(label(parent) for parent in node.parents)
            sig_in = ", ".join(show(signatures[id(p)]) for p in node.parents) or "-"
            name = f"[{i}] {type(node.operator).__name__}({parents})"
            rows.append((name, sig_in, show(signatures[id(node)])))
        for out_name, node in self.outputs.items():
            rows.append(
                (f"→ {out_name} = {label(node)}", "", show(signatures[id(node)]))
            )

        w_name = max([len("Node"), *(len(r[0]) for r in rows)])
        w_in = max([len("Input sig"), *(len(r[1]) for r in rows)])
        header = f"{'Node'.ljust(w_name)}  {'Input sig'.ljust(w_in)}  Output sig"
        body = [f"{n.ljust(w_name)}  {i.ljust(w_in)}  {o}" for n, i, o in rows]
        return "\n".join([header, "-" * len(header), *body])

    def _node_signatures(self, input_signatures: Any) -> dict[int, Any]:
        """``id(node) → Signature | None`` for every input and node."""
        if input_signatures is None:
            input_signatures = {}
        elif not isinstance(input_signatures, dict):
            if len(self.inputs) != 1:
                raise TypeError(
                    "This graph has several inputs: pass "
                    "{input name: Signature}, not a single signature."
                )
            input_signatures = {next(iter(self.inputs)): input_signatures}
        unknown = sorted(set(input_signatures) - set(self.inputs))
        if unknown:
            raise ValueError(f"Signatures given for undeclared input(s) {unknown}.")

        signatures: dict[int, Any] = {}
        for name, node in self.inputs.items():
            signatures[id(node)] = signature_of(input_signatures.get(name))
        for node in self._order:
            assert node.operator is not None  # guaranteed by _topological_sort
            parents = [signatures[id(parent)] for parent in node.parents]
            if not parents:
                joined = None
            elif all(parent == parents[0] for parent in parents):
                joined = parents[0]
            else:
                # inputs disagree: compute_output_signature takes one input
                signatures[id(node)] = None
                continue
            signatures[id(node)] = node.operator.compute_output_signature(joined)
        return signatures

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
