"""The ``level x rung x complexity`` cube.

A benchmark *cell* is a ``(data level, inference rung, complexity step)``
triple — the single mental model of ``benchmark_ladder.md`` §2.4. The
earlier ``level x rung`` matrix is one face of the cube, read at a fixed
complexity step.
"""

from __future__ import annotations

from dataclasses import dataclass


#: The staged inference ladder (``benchmark_ladder.md`` §2.2).
RUNG_NAMES: dict[int, str] = {
    1: "simple_model",
    2: "model_based_inference",
    3: "emulator",
    4: "emulator_based_inference",
    5: "amortized_predictor",
    6: "improve",
}

#: The L0-L4 data hierarchy (``benchmark_ladder.md`` §2.1).
LEVEL_NAMES: dict[str, str] = {
    "L0": "unstructured_obs",
    "L1": "structured_obs",
    "L2": "gap_filled_obs",
    "L3": "analysis",
    "L4": "reanalysis_forecast",
}


@dataclass(frozen=True)
class Cell:
    """One cell of the benchmark cube.

    Attributes:
        level: The data-hierarchy level (``"L0"`` … ``"L4"``).
        rung: The staged-ladder rung (1-6).
        complexity: The per-rung complexity step (``0`` = simplest).
    """

    level: str
    rung: int
    complexity: int = 0

    def __post_init__(self) -> None:
        if self.level not in LEVEL_NAMES:
            raise ValueError(
                f"unknown level {self.level!r}; expected one of {sorted(LEVEL_NAMES)}"
            )
        if self.rung not in RUNG_NAMES:
            raise ValueError(
                f"unknown rung {self.rung!r}; expected one of {sorted(RUNG_NAMES)}"
            )
        if self.complexity < 0:
            raise ValueError(f"complexity must be >= 0, got {self.complexity}")

    @property
    def rung_name(self) -> str:
        """The human-readable name of :attr:`rung`."""
        return RUNG_NAMES[self.rung]

    @property
    def label(self) -> str:
        """A compact ``level/rung/complexity`` string key for reports."""
        return f"{self.level}/r{self.rung}/c{self.complexity}"
