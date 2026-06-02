"""The "simpler thing is ground truth" rule, encoded as data.

The single invariant that makes the ladder a benchmark and not five
scripts (``benchmark_ladder.md`` §4.3) runs along **both** axes of §2:

* the **staged** axis — the previous rung is the reference;
* the **complexity** axis — the adjacent (simpler) step is the reference.

Both are *data, not code branches*: the default tables below live on the
:class:`ReferenceRule`, which a domain may override (e.g. an oracle-less
domain falls back to known-θ coverage only).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum


class Axis(Enum):
    """Which benchmark axis a reference is resolved along."""

    STAGED = "staged"  # §2.2 inference ladder
    COMPLEXITY = "complexity"  # §2.3 per-rung depth


class ReferenceSource(Enum):
    """Where a rung/step's ground truth comes from."""

    NONE = "none"  # generative rung; not scored against a reference
    KNOWN_THETA = "known_theta"  # synthetic truth from the simulator
    SIMULATOR_OUTPUT = "simulator_output"  # rung-1 forward outputs
    ORACLE_POSTERIOR = "oracle_posterior"  # rung-2 oracle posterior
    PREVIOUS_RUNG = "previous_rung"  # the staged step just below
    SIMPLER_STEP = "simpler_step"  # the complexity step just below
    BEST_VALIDATED_BELOW = "best_validated_below"  # for rung 6 "improve"


#: Staged-axis defaults (``benchmark_ladder.md`` §4.3, first table).
DEFAULT_STAGED: Mapping[int, ReferenceSource] = {
    1: ReferenceSource.NONE,
    2: ReferenceSource.KNOWN_THETA,
    3: ReferenceSource.SIMULATOR_OUTPUT,
    4: ReferenceSource.ORACLE_POSTERIOR,
    5: ReferenceSource.ORACLE_POSTERIOR,
    6: ReferenceSource.BEST_VALIDATED_BELOW,
}

#: Complexity-axis defaults (``benchmark_ladder.md`` §4.3, second table):
#: a richer step reproduces the simpler one, except the amortized head,
#: whose calibrated upgrade is judged against the oracle posterior.
DEFAULT_COMPLEXITY: Mapping[int, ReferenceSource] = {
    1: ReferenceSource.SIMPLER_STEP,
    2: ReferenceSource.SIMPLER_STEP,
    3: ReferenceSource.SIMPLER_STEP,
    4: ReferenceSource.SIMPLER_STEP,
    5: ReferenceSource.ORACLE_POSTERIOR,
    6: ReferenceSource.BEST_VALIDATED_BELOW,
}


@dataclass(frozen=True)
class ReferenceRule:
    """Resolve the reference source for a rung along a given axis.

    Attributes:
        staged: Per-rung reference sources on the staged axis.
        complexity: Per-rung reference sources on the complexity axis.

    A complexity step of ``0`` is the simplest step on its rung and has no
    simpler neighbour, so it is resolved as :attr:`ReferenceSource.NONE`
    on the complexity axis regardless of the table.
    """

    staged: Mapping[int, ReferenceSource] = field(
        default_factory=lambda: dict(DEFAULT_STAGED)
    )
    complexity: Mapping[int, ReferenceSource] = field(
        default_factory=lambda: dict(DEFAULT_COMPLEXITY)
    )

    def source(
        self, rung: int, *, axis: Axis = Axis.STAGED, complexity: int = 0
    ) -> ReferenceSource:
        """Return the reference source for ``rung`` along ``axis``.

        Args:
            rung: The staged-ladder rung (1-6).
            axis: Which axis to resolve the reference along.
            complexity: The complexity step; only consulted for
                :attr:`Axis.COMPLEXITY`, where step ``0`` has no simpler
                neighbour.

        Returns:
            The :class:`ReferenceSource` for that rung/step.
        """
        if axis is Axis.COMPLEXITY:
            if complexity <= 0:
                return ReferenceSource.NONE
            return self.complexity.get(rung, ReferenceSource.SIMPLER_STEP)
        return self.staged.get(rung, ReferenceSource.NONE)
