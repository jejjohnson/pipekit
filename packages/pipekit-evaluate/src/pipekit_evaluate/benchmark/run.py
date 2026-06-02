"""The benchmark runner: fill cube cells, resolve references, score.

:class:`Benchmark` realises ``benchmark_ladder.md`` §4.2 (the ``Benchmark``
dataclass) and §4.3 (the reference rule). It is deliberately carrier- and
scorer-agnostic: it fits each estimator, resolves the reference for that
cell via :class:`~pipekit_evaluate.benchmark.reference.ReferenceRule`, and
delegates the actual critique to the supplied :class:`Scorer`s. Writing a
``pipekit_experiment.Run`` per cell is left to a later slice.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from pipekit_evaluate.benchmark.cube import Cell
from pipekit_evaluate.benchmark.protocols import Estimate, Estimator, Scorer, Task
from pipekit_evaluate.benchmark.reference import (
    Axis,
    ReferenceRule,
    ReferenceSource,
)


@dataclass(frozen=True)
class CellResult:
    """The scored outcome for one cube cell.

    Attributes:
        cell: The ``(level, rung, complexity)`` cell.
        reference: Which reference source this cell was scored against.
        scores: Named metric values (namespaced by scorer), empty when the
            cell is a generative rung or its reference was unavailable.
        note: Human-readable explanation when ``scores`` is empty.
    """

    cell: Cell
    reference: ReferenceSource
    scores: Mapping[str, float] = field(default_factory=dict)
    note: str = ""


@dataclass(frozen=True)
class BenchmarkReport:
    """The collected results of a benchmark run.

    Attributes:
        task: The task name.
        domain: The task domain.
        axis: The axis the references were resolved along.
        results: One :class:`CellResult` per estimator.
    """

    task: str
    domain: str
    axis: Axis
    results: Sequence[CellResult]

    def to_dict(self) -> dict[str, dict[str, float]]:
        """Flatten to ``{cell label: {metric: value}}`` for serialisation."""
        return {r.cell.label: dict(r.scores) for r in self.results}


@dataclass
class Benchmark:
    """Run the ``level x rung x complexity`` cube and score each cell.

    Attributes:
        task: The scenario under test.
        estimators: The rungs/steps to fill in (each carries ``rung`` and
            ``complexity``).
        scorers: The ``Unit x Lens`` critiques applied to every scored cell.
        level: The data-hierarchy level the run sits at (default ``"L4"``).
        axis: Which axis to resolve references along (default staged).
        reference_rule: The "simpler thing is ground truth" tables.
        reference: An explicit oracle estimator; when ``None`` the rung-2
            estimator among ``estimators`` (lowest complexity) is used.
        obs_key, truth_key, state_key: Keys into ``task.datasets()`` for the
            observations fed to estimators and the known-θ / simulator-state
            references.
    """

    task: Task
    estimators: Sequence[Estimator]
    scorers: Sequence[Scorer]
    level: str = "L4"
    axis: Axis = Axis.STAGED
    reference_rule: ReferenceRule = field(default_factory=ReferenceRule)
    reference: Estimator | None = None
    obs_key: str = "obs"
    truth_key: str = "theta"
    state_key: str = "state"

    def run(self) -> BenchmarkReport:
        """Fit every estimator, resolve each cell's reference, and score it."""
        data = dict(self.task.datasets())
        obs = data[self.obs_key]

        # Pass 1 — fit and predict every estimator, keyed by (rung, complexity).
        predictions: dict[tuple[int, int], Estimate] = {}
        for est in self.estimators:
            predictions[(est.rung, est.complexity)] = est.fit(self.task)(obs)

        oracle = self._oracle_prediction(obs, predictions)

        # Pass 2 — resolve the reference for each cell and score it.
        results: list[CellResult] = []
        for est in self.estimators:
            cell = Cell(self.level, est.rung, est.complexity)
            source = self.reference_rule.source(
                est.rung, axis=self.axis, complexity=est.complexity
            )
            reference, note = self._resolve(source, est, data, predictions, oracle)
            if reference is None:
                results.append(CellResult(cell, source, note=note))
                continue
            prediction = predictions[(est.rung, est.complexity)]
            scores: dict[str, float] = {}
            for scorer in self.scorers:
                for name, value in scorer(prediction, reference).items():
                    scores[f"{scorer.name}.{name}"] = value
            results.append(CellResult(cell, source, scores=scores))

        return BenchmarkReport(
            task=self.task.name,
            domain=self.task.domain,
            axis=self.axis,
            results=results,
        )

    def _oracle_prediction(
        self, obs: object, predictions: Mapping[tuple[int, int], Estimate]
    ) -> Estimate | None:
        """The rung-2 reference posterior, explicit or inferred."""
        if self.reference is not None:
            return self.reference.fit(self.task)(obs)
        rung2 = sorted(
            (e for e in self.estimators if e.rung == 2),
            key=lambda e: e.complexity,
        )
        if rung2:
            first = rung2[0]
            return predictions[(first.rung, first.complexity)]
        return None

    def _resolve(
        self,
        source: ReferenceSource,
        est: Estimator,
        data: Mapping[str, object],
        predictions: Mapping[tuple[int, int], Estimate],
        oracle: Estimate | None,
    ) -> tuple[object | None, str]:
        """Map a :class:`ReferenceSource` to a concrete reference object.

        Returns ``(reference, note)``; a ``None`` reference means the cell
        is not scored, with ``note`` explaining why.
        """
        if source is ReferenceSource.NONE:
            return None, "generative rung: not scored against a reference"
        if source is ReferenceSource.KNOWN_THETA:
            return self._require(data, self.truth_key)
        if source is ReferenceSource.SIMULATOR_OUTPUT:
            return self._require(data, self.state_key)
        if source is ReferenceSource.ORACLE_POSTERIOR:
            if oracle is None:
                return None, "no oracle (rung-2) estimator available"
            return oracle, ""
        if source is ReferenceSource.PREVIOUS_RUNG:
            return self._lookup(predictions, est.rung - 1, est.complexity)
        if source is ReferenceSource.SIMPLER_STEP:
            return self._lookup(predictions, est.rung, est.complexity - 1)
        if source is ReferenceSource.BEST_VALIDATED_BELOW:
            below = [rc for rc in predictions if rc[0] < est.rung]
            if not below:
                return None, "no validated rung below this one"
            best = max(below)
            return predictions[best], ""
        return None, f"unhandled reference source {source!r}"

    @staticmethod
    def _require(data: Mapping[str, object], key: str) -> tuple[object | None, str]:
        if key not in data:
            return None, f"reference key {key!r} absent from task.datasets()"
        return data[key], ""

    @staticmethod
    def _lookup(
        predictions: Mapping[tuple[int, int], Estimate], rung: int, complexity: int
    ) -> tuple[object | None, str]:
        key = (rung, complexity)
        if key not in predictions:
            return None, f"no estimator at rung {rung}, complexity {complexity}"
        return predictions[key], ""
