"""Carrier-agnostic protocols for the benchmark ladder.

These mirror the style of :mod:`pipekit_cycle.protocols`: small,
runtime-checkable structural interfaces that domain packages
(``plumax``, ``somax``, ``xtremax``, …) satisfy *without importing*
``pipekit-evaluate``. They are the orchestration layer described in
``packages/pipekit-evaluate/docs/design/benchmark_ladder.md`` §4 — the
"who plays which rung" contracts — and sit on top of (not inside) the
``Unit x Lens x Stage`` scorer taxonomy (:mod:`pipekit_evaluate.scoring`).

A benchmark ``Task`` reuses ``pipekit_cycle``'s ``ForwardModel`` /
``ObservationOperator`` as its simulator and observation operator, so a
benchmark is assembled from the same parts a ``DACycle`` is. Those types
are referenced under :data:`typing.TYPE_CHECKING` only, so importing this
module never pulls in ``pipekit-cycle``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pipekit_evaluate.scoring._protocols import Estimate


if TYPE_CHECKING:
    from collections.abc import Mapping

    from pipekit_cycle.protocols import ForwardModel, ObservationOperator


@runtime_checkable
class Prior(Protocol):
    """A minimal prior over task parameters ``θ``.

    Deliberately thin (see ``benchmark_ladder.md`` §8, decision 2): a
    ``sample`` surface with an optional ``log_prob``, so neither NumPyro
    nor ``distrax`` is a hard dependency of the framework core.

    Members:
        sample(n, key): Draw ``n`` samples of ``θ``.
        log_prob(theta): Optional log-density at ``theta``; implementations
            without a tractable density may raise ``NotImplementedError``.
    """

    def sample(self, n: int, *, key: Any = None) -> Any: ...

    def log_prob(self, theta: Any) -> Any: ...


@runtime_checkable
class Task(Protocol):
    """One benchmark scenario: prior + simulator + obs operator + data.

    The simulator and observation operator are *exactly* the
    ``pipekit_cycle`` protocols, so no parallel hierarchy is introduced.

    Members:
        name: Human-readable task identifier.
        domain: Domain tag (``"plume"`` | ``"ocean"`` | ``"land"`` | …).
        prior(): The prior over task parameters ``θ`` (rung 1 input).
        simulator(): The rung-1 generative ``ForwardModel`` (``θ → state``).
        observation_operator(): The ``state → observations`` map.
        datasets(): The ``L0..L4`` products by key, plus any known truth
            (e.g. ``"theta"``, ``"state"``, ``"obs"``) used as references.
    """

    name: str
    domain: str

    def prior(self) -> Prior: ...

    def simulator(self) -> ForwardModel: ...

    def observation_operator(self) -> ObservationOperator: ...

    def datasets(self) -> Mapping[str, Any]: ...


@runtime_checkable
class Estimator(Protocol):
    """Any rung that maps observations → an :class:`Estimate`.

    The same protocol covers the model-based oracle (rung 2), emulator-based
    inference (rung 4) and the amortized predictor (rung 5). ``rung``
    locates the estimator on the staged axis (§2.2) and ``complexity``
    locates it on the per-rung complexity axis (§2.3).

    Members:
        rung: Position on the staged ladder (1-6).
        complexity: Position on the per-rung complexity axis (``0`` =
            simplest). Two estimators sharing a ``rung`` but differing in
            ``complexity`` are the cheap/rich pair scored against each other.
        fit(task): Return a fitted estimator (a no-op ``return self`` for an
            already-trained or closed-form rung-2 oracle).
        __call__(obs): Produce an :class:`Estimate` from observations.
    """

    rung: int
    complexity: int

    def fit(self, task: Task) -> Estimator: ...

    def __call__(self, obs: Any) -> Estimate: ...


@runtime_checkable
class Oracle(Protocol):
    """The designated reference :class:`Estimator` for a task (usually rung 2).

    Not yet consumed by `Benchmark` — reference selection currently
    lives inline in ``Benchmark._oracle_prediction``; this seam exists
    for the planned scorer taxonomy, where reference policy becomes
    pluggable.

    Members:
        reference_for(task): Return the estimator whose output is the
            reference posterior for the fast rungs of ``task``.
    """

    def reference_for(self, task: Task) -> Estimator: ...
