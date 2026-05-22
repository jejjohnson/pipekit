"""Group F — combination.

One-input → many-outputs operators. `Fanout` is dict-keyed; the
tuple-keyed sibling is `juxt` (in `compose.py`).

`Fanout` is intentionally minimal — it's sugar over a single-input
`Graph` with multiple named outputs. Carrier-specific merging
(`Augment`, `ApplyToEach`, anything that uses ``xr.merge``) stays in
sister libraries (xr_toolz) because it isn't carrier-agnostic.

See master plan Report 2, Group F.
"""

from __future__ import annotations

from typing import Any

from pipekit._base.operator import Operator


class Fanout(Operator):
    """Apply each named operator to the same input; return a dict of outputs.

    The dict-keyed companion to `juxt`. Each operator runs on the same
    carrier; the results are collected under their keys.

    Args:
        branches: Mapping from output name to `Operator`.

    Raises:
        TypeError: if any branch isn't an `Operator`.
        ValueError: if ``branches`` is empty.

    Example::

        rgb = Fanout({
            "red":   SelectBand(0),
            "green": SelectBand(1),
            "blue":  SelectBand(2),
        })
        out = rgb(image)
        # {"red": ..., "green": ..., "blue": ...}
    """

    __config_mixin_auto__ = False  # custom config — nested operators

    def __init__(self, branches: dict[str, Operator]) -> None:
        if not branches:
            raise ValueError("Fanout requires at least one branch.")
        for name, op in branches.items():
            if not isinstance(op, Operator):
                raise TypeError(
                    f"Fanout.branches[{name!r}] must be an Operator, "
                    f"got {type(op).__name__}."
                )
        self.branches = dict(branches)

    def _apply(self, x: Any) -> dict[str, Any]:
        return {name: op(x) for name, op in self.branches.items()}

    def get_config(self) -> dict[str, Any]:
        return {
            "branches": {
                name: {"class": type(op).__name__, "config": op.get_config()}
                for name, op in self.branches.items()
            }
        }

    def __repr__(self) -> str:
        names = ", ".join(self.branches)
        return f"Fanout([{names}])"
