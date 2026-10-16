"""The evaluation report: every score of a run, tagged and content-hashed.

:func:`evaluate` runs a list of scorers on one prediction and one
reference and collects their values into an :class:`EvaluationReport`.
Each :class:`ReportEntry` carries its scorer's place in the
``Unit x Lens x Stage`` taxonomy, so a report can be filtered by axis
(``report.filter(lens="structural")``), compared with an earlier one
(``report.diff(baseline)``) and identified by a content hash that ignores
when it was made. A benchmark-ladder run becomes the same report through
:meth:`~pipekit_evaluate.benchmark.BenchmarkReport.to_report`, with each
entry naming its cell and reference.
"""

from __future__ import annotations

import importlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any

from pipekit.hashing import sha256_hex, stable_json

from pipekit_evaluate.benchmark.protocols import Scorer
from pipekit_evaluate.benchmark.reference import ReferenceSource
from pipekit_evaluate.taxonomy import Lens, ScorerTags, Stage, Unit, tags_of


@dataclass(frozen=True)
class ReportEntry:
    """One named value from one scorer.

    Attributes:
        scorer: The scorer's name; it may not contain ``.`` or ``:``, which
            separate the parts of :attr:`key`.
        metric: The value's name within the scorer's output.
        value: The value.
        tags: The scorer's taxonomy tags.
        cell: The benchmark cell label (``"L4/r5/c0"``) for a ladder run.
        reference: The reference source (``"known_theta"``) for a ladder
            run.
    """

    scorer: str
    metric: str
    value: float
    tags: ScorerTags = field(default_factory=ScorerTags)
    cell: str | None = None
    reference: str | None = None

    def __post_init__(self) -> None:
        _check_name(self.scorer)
        value = float(self.value)
        # -0.0 == 0.0, so diff() sees no change between them; storing +0.0
        # keeps the content hash in agreement with diff().
        object.__setattr__(self, "value", 0.0 if value == 0.0 else value)

    @property
    def key(self) -> str:
        """The entry's identifier in its report.

        ``"<scorer>.<metric>"``, prefixed by ``"<cell>:<reference>:"`` for a
        ladder run, as in ``"L4/r5/c0:known_theta:rmse.value"``. Unambiguous
        because neither a cell label, a reference source nor a scorer name
        has a ``:``, and a scorer name has no ``.``: the scorer ends at the
        first dot after the prefix, and the metric name may contain more.
        """
        parts = [p for p in (self.cell, self.reference) if p is not None]
        return ":".join([*parts, f"{self.scorer}.{self.metric}"])

    def to_record(self) -> dict[str, Any]:
        """A flat, JSON-serialisable record of the entry."""
        return {
            "scorer": self.scorer,
            "metric": self.metric,
            "value": self.value,
            "unit": self.tags.unit.value,
            "lens": self.tags.lens.value,
            "stages": sorted(s.value for s in self.tags.stages),
            "differentiable": self.tags.differentiable,
            "cell": self.cell,
            "reference": self.reference,
        }


@dataclass(frozen=True)
class EvaluationReport:
    """The tagged scores of one evaluation run.

    Attributes:
        entries: One entry per scorer value; keys are unique.
        model_ref: What was evaluated (a registry hash, a URI), if known.
        dataset_ref: What it was evaluated on, if known.
        config: Free-form settings of the run, kept as a read-only JSON
            snapshot taken at construction: mappings at every depth are
            read-only, lists become tuples, and a value that is not JSON (a
            ``Path``, an object) is stored as its ``repr``. Changing the
            caller's config afterwards changes neither the report nor its
            hash.
        timestamp: When the report was made (UTC). Not part of
            :attr:`content_hash`.
    """

    entries: tuple[ReportEntry, ...]
    model_ref: str | None = None
    dataset_ref: str | None = None
    config: Mapping[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    _config_json: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        seen: set[str] = set()
        dups: set[str] = set()
        for entry in entries:
            if entry.key in seen:
                dups.add(entry.key)
            seen.add(entry.key)
        if dups:
            raise ValueError(
                f"duplicate report entries {sorted(dups)}; scorer names must "
                "be unique within a report"
            )
        object.__setattr__(self, "entries", entries)
        # The canonical JSON is the snapshot: the hash and to_dict read it,
        # so mutating a nested value the caller still holds cannot change a
        # report that has already been hashed.
        # Thawing first lets a frozen config (from filter or replace) hash
        # exactly as the plain config it was made from.
        config_json = stable_json(_thaw(self.config))
        object.__setattr__(self, "_config_json", config_json)
        object.__setattr__(self, "config", _freeze(json.loads(config_json)))

    @property
    def values(self) -> dict[str, float]:
        """Every value by entry key."""
        return {e.key: e.value for e in self.entries}

    @property
    def content_hash(self) -> str:
        """A sha256 over the entries, the references and the config.

        Entry order and the timestamp do not change it, so two runs that
        score the same values on the same references hash equal.
        """
        payload = {
            "entries": [
                e.to_record() for e in sorted(self.entries, key=lambda e: e.key)
            ],
            "model_ref": self.model_ref,
            "dataset_ref": self.dataset_ref,
            "config": json.loads(self._config_json),
        }
        return sha256_hex(stable_json(payload))

    def filter(
        self,
        *,
        unit: Unit | str | None = None,
        lens: Lens | str | None = None,
        stage: Stage | str | None = None,
        scorer: str | None = None,
        cell: str | None = None,
        reference: ReferenceSource | str | None = None,
    ) -> EvaluationReport:
        """The entries that match every given axis value.

        Args:
            unit: Keep entries of this unit.
            lens: Keep entries of this lens.
            stage: Keep entries whose scorer runs at this stage.
            scorer: Keep entries of the scorer with this name.
            cell: Keep entries of this benchmark cell label.
            reference: Keep entries scored against this reference source.

        Returns:
            A report with the matching entries and this report's
            references, config and timestamp.
        """
        unit = None if unit is None else Unit(unit)
        lens = None if lens is None else Lens(lens)
        stage = None if stage is None else Stage(stage)
        reference = None if reference is None else ReferenceSource(reference).value
        kept = tuple(
            e
            for e in self.entries
            if (unit is None or e.tags.unit is unit)
            and (lens is None or e.tags.lens is lens)
            and (stage is None or e.tags.runs_at(stage))
            and (scorer is None or e.scorer == scorer)
            and (cell is None or e.cell == cell)
            and (reference is None or e.reference == reference)
        )
        return replace(self, entries=kept)

    def diff(
        self, baseline: EvaluationReport
    ) -> dict[str, tuple[float | None, float | None]]:
        """The entries whose value differs from ``baseline``'s.

        Args:
            baseline: The report to compare against, such as last week's.

        Returns:
            ``{key: (baseline value, this value)}`` for every key whose
            value changed, with ``None`` on the side that lacks the key.
            Two NaNs count as equal, so a report's diff against itself is
            empty.
        """
        before, after = baseline.values, self.values
        changes: dict[str, tuple[float | None, float | None]] = {}
        for key in sorted(before.keys() | after.keys()):
            old, new = before.get(key), after.get(key)
            if not _same(old, new):
                changes[key] = (old, new)
        return changes

    def to_records(self) -> list[dict[str, Any]]:
        """One flat record per entry, in entry order."""
        return [e.to_record() for e in self.entries]

    def to_dict(self) -> dict[str, Any]:
        """A strict-JSON form, with the content hash.

        A non-finite float (in a value or in the config) is written as the
        string ``"NaN"``, ``"Infinity"`` or ``"-Infinity"``, so the result
        passes ``json.dumps(..., allow_nan=False)``; :meth:`to_records` and
        :meth:`to_pandas` keep the floats.
        """
        return _strict_json(
            {
                "entries": self.to_records(),
                "model_ref": self.model_ref,
                "dataset_ref": self.dataset_ref,
                "config": json.loads(self._config_json),
                "timestamp": self.timestamp.isoformat(),
                "content_hash": self.content_hash,
            }
        )

    def to_pandas(self) -> Any:
        """One row per entry, as a ``pandas.DataFrame``.

        Raises:
            ImportError: If pandas is not installed.
        """
        try:
            pd = importlib.import_module("pandas")
        except ImportError as err:
            raise ImportError(
                "EvaluationReport.to_pandas needs pandas; install it with "
                "`pip install pandas`"
            ) from err
        return pd.DataFrame.from_records(self.to_records())


def evaluate(
    prediction: Any,
    reference: Any,
    scorers: Iterable[Scorer],
    *,
    stage: Stage | str | None = None,
    model_ref: str | None = None,
    dataset_ref: str | None = None,
    config: Mapping[str, Any] | None = None,
) -> EvaluationReport:
    """Score one prediction against one reference with every scorer.

    Args:
        prediction: What is evaluated, in whatever carrier the scorers take.
        reference: What it is compared with.
        scorers: The scorers; their names must be unique.
        stage: Run only the scorers tagged for this stage; all of them
            when ``None``.
        model_ref: What was evaluated, recorded on the report.
        dataset_ref: What it was evaluated on, recorded on the report.
        config: Settings recorded on the report and in its hash.

    Returns:
        The report, one entry per value each scorer returned.

    Raises:
        ValueError: If two scorers share a name, or a name contains ``.`` or
            ``:``; checked before any scorer runs, because either would let
            two scorers' values share a key.
    """
    scorers = list(scorers)
    _check_scorer_names(scorers)
    stage = None if stage is None else Stage(stage)
    entries: list[ReportEntry] = []
    for scorer in scorers:
        tags = tags_of(scorer)
        if stage is not None and not tags.runs_at(stage):
            continue
        for metric, value in scorer(prediction, reference).items():
            entries.append(ReportEntry(scorer.name, metric, value, tags))
    return EvaluationReport(
        tuple(entries),
        model_ref=model_ref,
        dataset_ref=dataset_ref,
        config=config or {},
    )


def _check_name(name: str) -> None:
    reserved = sorted(set(name) & set(".:"))
    if reserved:
        raise ValueError(
            f"scorer name {name!r} contains {reserved}; '.' and ':' separate "
            "the parts of a report key"
        )


def _check_scorer_names(scorers: Sequence[Scorer]) -> None:
    names = [scorer.name for scorer in scorers]
    for name in names:
        _check_name(name)
    dups = sorted({name for name in names if names.count(name) > 1})
    if dups:
        raise ValueError(
            f"duplicate scorer names {dups}; each scorer needs its own name"
        )


def _freeze(value: Any) -> Any:
    """Make decoded JSON read-only at every depth."""
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def _thaw(value: Any) -> Any:
    """Turn mappings and sequences back into the dicts and lists JSON encodes."""
    if isinstance(value, Mapping):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_thaw(v) for v in value]
    return value


def _strict_json(value: Any) -> Any:
    """Replace non-finite floats with the strings JSON has no number for."""
    if isinstance(value, float) and not math.isfinite(value):
        return (
            "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
        )
    if isinstance(value, dict):
        return {k: _strict_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_strict_json(v) for v in value]
    return value


def _same(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is b
    return a == b or (math.isnan(a) and math.isnan(b))
