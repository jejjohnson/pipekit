"""Metaflow adapter — wrap a pipekit operator as a Metaflow ``@step``.

Pipekit handles the compute; Metaflow handles scheduling and
cross-step artifact tracking. The adapter is pure glue:

- `as_step(op, name, inputs)` — returns a function suitable for
  decoration with ``@step`` inside a `FlowSpec`.
- `MetaflowStepAdapter.run(op, inputs)` — invoke an operator with
  inputs read from ``self.<input_name>`` and write outputs to
  ``self.<name>``.

The adapter doesn't import ``metaflow`` at module load — the import
happens only when the user actually constructs a Metaflow flow with
the produced step.

Behind ``pipekit-experiment[metaflow]`` extra.

See master plan Report 12, section 3.5.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pipekit import Operator


def _require_metaflow() -> Any:
    try:
        import metaflow  # ty: ignore[unresolved-import]
    except ImportError as e:
        raise ImportError(
            "MetaflowStepAdapter requires metaflow. Install with "
            "`pip install pipekit-experiment[metaflow]`."
        ) from e
    return metaflow


class MetaflowStepAdapter:
    """Adapt a pipekit `Operator` for use as a Metaflow ``@step``.

    Use either:

    - `as_step(op, name, inputs)` to obtain a callable that reads its
      inputs from ``self.<input_name>`` and writes the operator's
      output to ``self.<name>``. Apply ``@step`` to it inside a
      ``FlowSpec``.
    - `run(op, namespace, name, inputs)` to invoke the same logic
      synchronously, passing a dict / SimpleNamespace as ``namespace``.
      Useful for testing flows outside Metaflow.

    The adapter never inspects Metaflow internals; it only relies on
    the ``self`` namespace conventions and the ``@step`` decorator.
    """

    @staticmethod
    def run(
        op: Operator,
        namespace: Any,
        *,
        name: str,
        inputs: list[str],
    ) -> Any:
        """Invoke ``op`` against attributes of ``namespace`` and persist its output.

        Reads each ``input_name`` from ``namespace`` and calls
        ``op(*args)``. Writes the result back as ``namespace.<name>``.
        Returns the result.

        Args:
            op: The pipekit operator to invoke.
            namespace: An object exposing ``inputs`` as attributes
                (typically ``self`` inside a Metaflow ``@step``).
            name: Attribute name under which to store the output.
            inputs: List of input attribute names to read from
                ``namespace`` in order.
        """
        if not isinstance(op, Operator):
            raise TypeError(
                f"MetaflowStepAdapter.run expects an Operator, got {type(op).__name__}."
            )
        args = [getattr(namespace, n) for n in inputs]
        out = op(*args)
        setattr(namespace, name, out)
        return out

    @staticmethod
    def as_step(
        op: Operator,
        *,
        name: str,
        inputs: list[str],
        decorate: bool = True,
    ) -> Callable[..., Any]:
        """Build a Metaflow step function around ``op``.

        Returns a function with signature ``step(self)`` that reads
        ``self.<input>`` for each ``inputs`` entry, applies ``op``,
        and writes the result to ``self.<name>``.

        Args:
            op: The pipekit operator to wrap.
            name: Attribute name for the output (also used as
                ``__name__`` of the returned function).
            inputs: Input attribute names, in argument order.
            decorate: If ``True`` (default), wrap the returned
                function with ``metaflow.step`` so it's drop-in for a
                ``FlowSpec``. Set ``False`` to obtain the bare callable
                for testing.

        Raises:
            ImportError: if ``decorate=True`` and ``metaflow`` isn't
                installed.
        """
        if not isinstance(op, Operator):
            raise TypeError(
                f"MetaflowStepAdapter.as_step expects an Operator, "
                f"got {type(op).__name__}."
            )

        def _step(self: Any) -> None:
            MetaflowStepAdapter.run(op, self, name=name, inputs=inputs)

        _step.__name__ = name
        _step.__doc__ = (
            f"Metaflow step wrapping {type(op).__name__}; "
            f"reads {inputs!r}, writes {name!r}."
        )
        if not decorate:
            return _step
        metaflow = _require_metaflow()
        return metaflow.step(_step)
