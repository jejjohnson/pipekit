"""The benchmark runner: fill cube cells, resolve references, score.

:class:`Benchmark` realises ``benchmark_ladder.md`` §4.2 (the ``Benchmark``
dataclass) and §4.3 (the reference rule). It is deliberately carrier- and
scorer-agnostic: it fits each estimator, resolves the reference for that
cell via :class:`~pipekit_evaluate.benchmark.reference.ReferenceRule`, and
delegates the actual critique to the supplied :class:`Scorer`s. Its
:class:`BenchmarkReport` converts into the tagged, content-hashed
:class:`~pipekit_evaluate.report.EvaluationReport` that a plain
:func:`~pipekit_evaluate.report.evaluate` call returns. Writing a
``pipekit_experiment.Run`` per cell is left to a later slice.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pipekit_evaluate.benchmark.cube import Cell
from pipekit_evaluate.benchmark.protocols import Estimate, Estimator, Scorer, Task
from pipekit_evaluate.benchmark.reference import (
    Axis,
    ReferenceRule,
    ReferenceSource,
)
from pipekit_evaluate.report import EvaluationReport, ReportEntry
from pipekit_evaluate.taxonomy import ScorerTags, tags_of


@dataclass(frozen=True)
class CellResult:
    """The scored outcome for one cube cell.

    Attributes:
        cell: The ``(level, rung, complexity)`` cell.
        reference: The cell's primary reference source.
        scores: Named metric values (namespaced by scorer) against the
            primary reference, empty when the cell is not scored or its
            primary reference was unavailable.
        note: Human-readable explanation of every reference that could not
            be resolved; empty when all of them were.
        by_reference: The scores against every reference that resolved,
            primary included, keyed by source. When only ``scores`` is
            given, it is filled in as ``{reference: scores}``.
    """

    cell: Cell
    reference: ReferenceSource
    scores: Mapping[str, float] = field(default_factory=dict)
    note: str = ""
    by_reference: Mapping[ReferenceSource, Mapping[str, float]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.by_reference and self.scores:
            object.__setattr__(self, "by_reference", {self.reference: self.scores})


@dataclass(frozen=True)
class BenchmarkReport:
    """The collected results of a benchmark run.

    Attributes:
        task: The task name.
        domain: The task domain.
        axis: The axis the references were resolved along.
        results: One :class:`CellResult` per estimator.
        entries: Every scored value as a tagged :class:`ReportEntry` naming
            its cell and reference, in scoring order; filled by
            :meth:`Benchmark.run`.
    """

    task: str
    domain: str
    axis: Axis
    results: Sequence[CellResult]
    entries: tuple[ReportEntry, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serialisable form that keeps every reference and note.

        Returns:
            ``{"task", "domain", "axis", "cells"}``, where ``cells`` maps
            each cell label to its ``rung_name``, primary ``reference``,
            ``scores`` by reference source (``{source: {metric: value}}``)
            and ``note``.
        """
        return {
            "task": self.task,
            "domain": self.domain,
            "axis": self.axis.value,
            "cells": {
                r.cell.label: {
                    "rung_name": r.cell.rung_name,
                    "reference": r.reference.value,
                    "scores": {
                        source.value: dict(scores)
                        for source, scores in r.by_reference.items()
                    },
                    "note": r.note,
                }
                for r in self.results
            },
        }

    def to_report(
        self,
        *,
        model_ref: str | None = None,
        dataset_ref: str | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> EvaluationReport:
        """The run as an :class:`EvaluationReport`, one entry per scored value.

        Each entry carries its cell label and reference source, so
        ``report.filter(reference="known_theta")`` or
        ``report.filter(lens="probabilistic")`` cuts across the cube.

        Args:
            model_ref: What was evaluated, recorded on the report.
            dataset_ref: What it was evaluated on, recorded on the report.
            config: Settings recorded on the report and in its hash, beside
                the run's ``task``, ``domain`` and ``axis``.

        Returns:
            The report; its config holds ``task``, ``domain`` and ``axis``
            unless ``config`` sets them.
        """
        run = {"task": self.task, "domain": self.domain, "axis": self.axis.value}
        return EvaluationReport(
            self.entries,
            model_ref=model_ref,
            dataset_ref=dataset_ref,
            config={**run, **(config or {})},
        )


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
            estimator among ``estimators`` with the highest complexity (the
            richest kernel, such as NUTS rather than OI) is used.
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
        self._check_unique_cells()
        self._check_unique_scorers()
        tags = {scorer.name: tags_of(scorer) for scorer in self.scorers}
        data = dict(self.task.datasets())
        if self.obs_key not in data:
            raise KeyError(
                f"Benchmark: obs_key {self.obs_key!r} not in task.datasets() "
                f"(available keys: {sorted(data)})."
            )
        obs = data[self.obs_key]

        # Pass 1 — fit and predict every estimator, keyed by (rung, complexity).
        predictions: dict[tuple[int, int], Estimate] = {}
        for est in self.estimators:
            predictions[(est.rung, est.complexity)] = est.fit(self.task)(obs)

        oracle = self._oracle_prediction(obs, predictions)

        # Pass 2 — resolve each cell's references and score against each. Process
        # in ascending cell order so a cell is only considered "validated" (and
        # thus usable as a BEST_VALIDATED_BELOW reference) once it has itself
        # resolved its primary reference. ``validated`` maps a cell to its
        # prediction.
        validated: dict[tuple[int, int], Estimate] = {}
        by_cell: dict[tuple[int, int], CellResult] = {}
        entries: list[ReportEntry] = []
        for est in sorted(self.estimators, key=lambda e: (e.rung, e.complexity)):
            key = (est.rung, est.complexity)
            cell = Cell(self.level, est.rung, est.complexity)
            primary = self.reference_rule.source(
                est.rung, axis=self.axis, complexity=est.complexity
            )
            # An unscored cell has no sources; resolving its NONE primary
            # still gives the note that says why.
            sources = self.reference_rule.sources(
                est.rung, axis=self.axis, complexity=est.complexity
            ) or (primary,)
            by_reference: dict[ReferenceSource, dict[str, float]] = {}
            notes: list[str] = []
            for source in sources:
                reference, note = self._resolve(
                    source, est, data, predictions, oracle, validated
                )
                if reference is None:
                    # The primary's note reads as before; a further
                    # reference's note says which reference it is about.
                    notes.append(
                        note if source is primary else f"{source.value}: {note}"
                    )
                    continue
                by_reference[source] = self._score(
                    predictions[key], reference, tags, cell, source, entries
                )
            by_cell[key] = CellResult(
                cell,
                primary,
                scores=by_reference.get(primary, {}),
                note="; ".join(notes),
                by_reference=by_reference,
            )
            if primary in by_reference:
                validated[key] = predictions[key]

        # Emit one result per estimator, preserving the caller's order.
        results = [by_cell[(est.rung, est.complexity)] for est in self.estimators]
        return BenchmarkReport(
            task=self.task.name,
            domain=self.task.domain,
            axis=self.axis,
            results=results,
            entries=tuple(entries),
        )

    def _score(
        self,
        prediction: Estimate,
        reference: object,
        tags: Mapping[str, ScorerTags],
        cell: Cell,
        source: ReferenceSource,
        entries: list[ReportEntry],
    ) -> dict[str, float]:
        """Every scorer's values against one reference, namespaced by scorer.

        Each value is also appended to ``entries`` as a tagged
        :class:`ReportEntry` for :meth:`BenchmarkReport.to_report`.
        """
        scores: dict[str, float] = {}
        for scorer in self.scorers:
            for name, value in scorer(prediction, reference).items():
                scores[f"{scorer.name}.{name}"] = value
                entries.append(
                    ReportEntry(
                        scorer.name,
                        name,
                        value,
                        tags[scorer.name],
                        cell=cell.label,
                        reference=source.value,
                    )
                )
        return scores

    def _check_unique_scorers(self) -> None:
        """Reject two scorers with one name.

        Scores are namespaced by scorer name, so two scorers sharing one
        would overwrite each other's values in a cell.
        """
        names = [scorer.name for scorer in self.scorers]
        dups = sorted({n for n in names if names.count(n) > 1})
        if dups:
            raise ValueError(
                f"duplicate scorer names {dups}; each scorer needs its own name"
            )

    def _check_unique_cells(self) -> None:
        """Reject two estimators occupying the same cube cell.

        Each ``(rung, complexity)`` cell must be filled by exactly one
        estimator; otherwise predictions would silently overwrite and
        results would collide (including in :meth:`BenchmarkReport.to_dict`).
        """
        seen: set[tuple[int, int]] = set()
        dups: set[tuple[int, int]] = set()
        for est in self.estimators:
            key = (est.rung, est.complexity)
            if key in seen:
                dups.add(key)
            seen.add(key)
        if dups:
            raise ValueError(
                "duplicate benchmark cells (rung, complexity): "
                f"{sorted(dups)}; each cell must be filled by exactly one estimator"
            )

    def _oracle_prediction(
        self, obs: object, predictions: Mapping[tuple[int, int], Estimate]
    ) -> Estimate | None:
        """The rung-2 reference posterior, explicit or inferred."""
        if self.reference is not None:
            # If the explicit oracle is itself one of the estimators, reuse its
            # already-fitted prediction rather than running it a second time —
            # a slow/stochastic oracle (MCMC, 4D-Var) would otherwise yield a
            # different posterior than the one reported for its own cell.
            for est in self.estimators:
                if est is self.reference:
                    return predictions[(est.rung, est.complexity)]
            return self.reference.fit(self.task)(obs)
        # No explicit oracle: the inferred oracle is the *most* sophisticated
        # rung-2 kernel (highest complexity) — e.g. NUTS / Strong-4D-Var, not
        # the cheap OI/MAP baseline at the bottom of the rung-2 ladder.
        rung2 = [e for e in self.estimators if e.rung == 2]
        if rung2:
            best = max(rung2, key=lambda e: e.complexity)
            return predictions[(best.rung, best.complexity)]
        return None

    def _resolve(
        self,
        source: ReferenceSource,
        est: Estimator,
        data: Mapping[str, object],
        predictions: Mapping[tuple[int, int], Estimate],
        oracle: Estimate | None,
        validated: Mapping[tuple[int, int], Estimate],
    ) -> tuple[object | None, str]:
        """Map a :class:`ReferenceSource` to a concrete reference object.

        Returns ``(reference, note)``; a ``None`` reference means the cell
        is not scored, with ``note`` explaining why. ``validated`` holds the
        cells resolved so far, used by ``BEST_VALIDATED_BELOW``.
        """
        if source is ReferenceSource.NONE:
            if self.axis is Axis.COMPLEXITY and est.complexity <= 0:
                return (
                    None,
                    "simplest step on its rung: no simpler step to compare against",
                )
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
            # Stay on the estimator's own complexity slice: comparing rung 6 to
            # a lower rung at a *different* complexity would co-vary both axes
            # at once (the framework walks one axis at a time).
            below = [
                rc for rc in validated if rc[0] < est.rung and rc[1] == est.complexity
            ]
            if not below:
                return None, "no validated rung below this one on the same complexity"
            best = max(below)  # highest validated rung on this complexity slice
            return validated[best], ""
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
