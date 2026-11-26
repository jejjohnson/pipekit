"""Where the ground truth of a score comes from."""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum


class ReferenceSource(Enum):
    """Where a rung/step's ground truth comes from."""

    NONE = "none"  # generative rung; not scored against a reference
    KNOWN_THETA = "known_theta"  # synthetic truth from the simulator
    SIMULATOR_OUTPUT = "simulator_output"  # rung-1 forward outputs
    ORACLE_POSTERIOR = "oracle_posterior"  # rung-2 oracle posterior
    PREVIOUS_RUNG = "previous_rung"  # the staged step just below
    SIMPLER_STEP = "simpler_step"  # the complexity step just below
    BEST_VALIDATED_BELOW = "best_validated_below"  # for rung 6 "improve"


def as_sources(
    sources: ReferenceSource | str | Iterable[ReferenceSource | str],
) -> frozenset[ReferenceSource]:
    """Normalise one reference source or several to a set of members.

    Args:
        sources: A :class:`ReferenceSource`, its string value, or an
            iterable of either.

    Returns:
        The sources as a frozenset of members.

    Raises:
        ValueError: If a string is not a :class:`ReferenceSource` value.
    """
    # A bare string is one source, not an iterable of characters.
    if isinstance(sources, ReferenceSource | str):
        sources = (sources,)
    return frozenset(ReferenceSource(s) for s in sources)
