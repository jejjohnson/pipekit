"""Tests for the evaluation report (geostack Report 14, gate A3).

The running example is Report 14's stage 1: a Gaussian plume, the same
plume displaced by 6 cells, and the flat field at the plume's mean, scored
through three lenses.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import replace

import numpy as np
import pytest
from pipekit_evaluate import (
    EvaluationReport,
    Lens,
    MetricScorer,
    ReportEntry,
    ScorerTags,
    Stage,
    evaluate,
)


def rmse(prediction: np.ndarray, reference: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - reference) ** 2)))


def phase(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """The circular shift that best aligns the two, and the residual RMSE."""
    errors = [rmse(np.roll(prediction, -s), reference) for s in range(len(reference))]
    shift = int(np.argmin(errors))
    return {"shift": shift, "residual": errors[shift]}


def spectrum_gap(prediction: np.ndarray, reference: np.ndarray) -> float:
    """sum_k |P_pred - P_ref| / sum_k P_ref over k > 0; blind to phase."""
    p_pred = np.abs(np.fft.rfft(prediction))[1:] ** 2
    p_ref = np.abs(np.fft.rfft(reference))[1:] ** 2
    return float(np.sum(np.abs(p_pred - p_ref)) / np.sum(p_ref))


SCORERS = [
    MetricScorer(rmse),
    MetricScorer(phase, lens="structural", stages={"final"}),
    MetricScorer(spectrum_gap, lens="spectral", stages={"final", "monitoring"}),
]


@pytest.fixture
def plume() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, sigma, shift = 40, 3.0, 6
    cells = np.arange(n)
    distance = np.minimum(abs(cells - n // 2), n - abs(cells - n // 2))
    truth = 50.0 * np.exp(-(distance**2) / (2 * sigma**2))
    return truth, np.roll(truth, shift), np.full(n, truth.mean())


@pytest.fixture
def report(plume: tuple[np.ndarray, np.ndarray, np.ndarray]) -> EvaluationReport:
    truth, shifted, _ = plume
    return evaluate(shifted, truth, SCORERS, model_ref="h-shifted")


def test_evaluate_collects_every_tagged_value(report: EvaluationReport) -> None:
    values = report.values
    assert set(values) == {
        "rmse.value",
        "phase.shift",
        "phase.residual",
        "spectrum_gap.value",
    }
    assert round(values["rmse.value"], 2) == 20.50
    # the structural and spectral lenses see past the double penalty
    assert values["phase.shift"] == 6
    assert values["phase.residual"] == pytest.approx(0.0, abs=1e-12)
    assert values["spectrum_gap.value"] == pytest.approx(0.0, abs=1e-12)
    by_key = {e.key: e for e in report.entries}
    assert by_key["phase.shift"].tags.lens is Lens.STRUCTURAL
    assert report.model_ref == "h-shifted"


def test_the_flat_field_has_no_power_at_any_wavenumber(
    plume: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    truth, _, flat = plume
    flat_report = evaluate(flat, truth, SCORERS)
    assert flat_report.values["spectrum_gap.value"] == pytest.approx(1.0, rel=1e-12)


# --------------------------------------------------------------------------- #
# Gate A3: the content hash
# --------------------------------------------------------------------------- #
def test_content_hash_is_equal_for_two_runs(
    plume: tuple[np.ndarray, np.ndarray, np.ndarray], report: EvaluationReport
) -> None:
    truth, shifted, _ = plume
    again = evaluate(shifted, truth, list(reversed(SCORERS)), model_ref="h-shifted")
    # a different timestamp and entry order, the same content
    assert again.timestamp >= report.timestamp
    assert again.entries != report.entries
    assert again.content_hash == report.content_hash
    assert len(report.content_hash) == 64


def test_content_hash_changes_with_one_value(report: EvaluationReport) -> None:
    first, *rest = report.entries
    changed = replace(report, entries=(replace(first, value=first.value + 1e-9), *rest))
    assert changed.content_hash != report.content_hash


@pytest.mark.parametrize(
    "change",
    [{"model_ref": "h-other"}, {"dataset_ref": "d1"}, {"config": {"window": 9}}],
)
def test_content_hash_covers_references_and_config(
    report: EvaluationReport, change: dict[str, object]
) -> None:
    assert replace(report, **change).content_hash != report.content_hash


def test_content_hash_ignores_the_timestamp(report: EvaluationReport) -> None:
    from datetime import UTC, datetime

    later = replace(report, timestamp=datetime(2030, 1, 1, tzinfo=UTC))
    assert later.content_hash == report.content_hash


# --------------------------------------------------------------------------- #
# Gate A3: filter and diff
# --------------------------------------------------------------------------- #
def test_filter_by_lens_returns_exactly_those_entries(
    report: EvaluationReport,
) -> None:
    structural = report.filter(lens="structural")
    assert set(structural.values) == {"phase.shift", "phase.residual"}
    assert all(e.tags.lens is Lens.STRUCTURAL for e in structural.entries)
    assert structural.model_ref == report.model_ref


def test_filter_by_stage_unit_and_scorer(report: EvaluationReport) -> None:
    assert set(report.filter(stage=Stage.MONITORING).values) == {"spectrum_gap.value"}
    assert report.filter(stage="training").entries == ()
    assert len(report.filter(unit="field").entries) == 4
    assert set(report.filter(scorer="rmse").values) == {"rmse.value"}
    assert report.filter(lens="pointwise", stage="final").values.keys() == {
        "rmse.value"
    }


def test_diff_against_itself_is_empty(report: EvaluationReport) -> None:
    assert report.diff(report) == {}


def test_diff_reports_each_changed_value_from_the_baseline(
    plume: tuple[np.ndarray, np.ndarray, np.ndarray], report: EvaluationReport
) -> None:
    truth, _, flat = plume
    baseline = evaluate(flat, truth, SCORERS[:2])
    changes = report.diff(baseline)
    assert set(changes) == {
        "rmse.value",
        "phase.shift",
        "phase.residual",
        "spectrum_gap.value",
    }
    before, after = changes["rmse.value"]
    assert (round(before or 0, 2), round(after or 0, 2)) == (15.62, 20.50)
    # absent from the baseline
    assert changes["spectrum_gap.value"][0] is None


def test_diff_treats_two_nans_as_equal() -> None:
    entry = ReportEntry("m", "value", math.nan)
    report = EvaluationReport((entry,))
    assert report.diff(EvaluationReport((entry,))) == {}


# --------------------------------------------------------------------------- #
# Running and serialising
# --------------------------------------------------------------------------- #
def test_evaluate_runs_only_the_scorers_of_a_stage(
    plume: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    truth, shifted, _ = plume
    monitoring = evaluate(shifted, truth, SCORERS, stage="monitoring")
    assert set(monitoring.values) == {"spectrum_gap.value"}


def test_duplicate_scorer_names_are_rejected(
    plume: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    truth, shifted, _ = plume
    with pytest.raises(ValueError, match="duplicate scorer names"):
        evaluate(shifted, truth, [MetricScorer(rmse), MetricScorer(rmse)])


def test_duplicate_scorer_names_are_rejected_before_any_scorer_runs() -> None:
    # disjoint metric keys would never collide as entries, but would merge
    # two scorers into one namespace
    calls: list[str] = []

    def first(p: object, r: object) -> dict[str, float]:
        calls.append("first")
        return {"a": 1.0}

    def second(p: object, r: object) -> dict[str, float]:
        calls.append("second")
        return {"b": 2.0}

    scorers = [MetricScorer(first, name="m"), MetricScorer(second, name="m")]
    with pytest.raises(ValueError, match=r"duplicate scorer names \['m'\]"):
        evaluate(None, None, scorers)
    assert calls == []


def test_entries_with_one_key_are_rejected() -> None:
    entry = ReportEntry("m", "value", 1.0)
    with pytest.raises(ValueError, match="duplicate report entries"):
        EvaluationReport((entry, entry))


# --------------------------------------------------------------------------- #
# The config snapshot
# --------------------------------------------------------------------------- #
def test_config_is_snapshotted_at_construction(report: EvaluationReport) -> None:
    window = {"sizes": [9]}
    snap = replace(report, config={"window": window})
    before = snap.content_hash
    window["sizes"].append(17)  # the caller mutates what it still holds
    assert snap.to_dict()["config"] == {"window": {"sizes": [9]}}
    assert snap.content_hash == before


def test_config_is_read_only_at_every_depth(report: EvaluationReport) -> None:
    snap = replace(report, config={"window": {"sizes": [9]}})
    assert snap.config["window"]["sizes"] == (9,)
    with pytest.raises(TypeError):
        snap.config["window"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        snap.config["window"]["sizes"] = [17]  # type: ignore[index]
    with pytest.raises(AttributeError):
        snap.config["window"]["sizes"].append(17)


def test_a_frozen_config_hashes_like_the_plain_one(report: EvaluationReport) -> None:
    snap = replace(report, config={"window": {"sizes": [9, 17]}, "seed": 0})
    # filter and replace pass the frozen config back through the constructor
    assert snap.filter().content_hash == snap.content_hash
    assert snap.filter().to_dict()["config"] == snap.to_dict()["config"]


def test_signed_zero_hashes_like_zero() -> None:
    plus = EvaluationReport((ReportEntry("m", "bias", 0.0),))
    minus = EvaluationReport((ReportEntry("m", "bias", -0.0),))
    assert minus.diff(plus) == {}
    assert minus.content_hash == plus.content_hash
    assert math.copysign(1.0, minus.entries[0].value) == 1.0


def test_to_dict_serialises_a_non_json_config(report: EvaluationReport) -> None:
    from pathlib import PurePosixPath

    snap = replace(report, config={"root": PurePosixPath("/data")})
    payload = json.loads(json.dumps(snap.to_dict()))
    assert payload["config"] == {"root": "PurePosixPath('/data')"}
    assert payload["content_hash"] == snap.content_hash


def test_to_dict_is_json_serialisable(report: EvaluationReport) -> None:
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["content_hash"] == report.content_hash
    assert payload["entries"][1] == {
        "scorer": "phase",
        "metric": "shift",
        "value": 6.0,
        "unit": "field",
        "lens": "structural",
        "stages": ["final"],
        "differentiable": False,
    }


def test_report_entry_defaults_to_untagged() -> None:
    entry = ReportEntry("m", "value", 1)
    assert entry.tags == ScorerTags()
    assert type(entry.value) is float


def test_to_pandas_has_one_row_per_entry(report: EvaluationReport) -> None:
    pytest.importorskip("pandas")
    frame = report.to_pandas()
    assert list(frame["metric"]) == [e.metric for e in report.entries]


def test_to_pandas_names_the_missing_dependency(
    report: EvaluationReport, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "pandas", None)
    with pytest.raises(ImportError, match="needs pandas"):
        report.to_pandas()


# --------------------------------------------------------------------------- #
# Unambiguous keys and strict JSON
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["a.b", "a:b"])
def test_scorer_names_cannot_hold_a_key_separator(name: str) -> None:
    # ("a.b", "c") and ("a", "b.c") would otherwise share the key "a.b.c"
    with pytest.raises(ValueError, match="separate the parts of a report key"):
        ReportEntry(name, "c", 1.0)
    with pytest.raises(ValueError, match="separate the parts of a report key"):
        evaluate(None, None, [MetricScorer(lambda p, r: 1.0, name=name)])


def test_metric_names_may_hold_dots() -> None:
    report = EvaluationReport((ReportEntry("a", "b.c", 1.0),))
    assert report.values == {"a.b.c": 1.0}


def test_to_dict_is_strict_json_for_non_finite_values() -> None:
    entries = tuple(
        ReportEntry("m", name, value)
        for name, value in [("nan", math.nan), ("up", math.inf), ("down", -math.inf)]
    )
    report = EvaluationReport(entries, config={"floor": -math.inf})
    payload = report.to_dict()
    text = json.dumps(payload, allow_nan=False)
    assert [e["value"] for e in json.loads(text)["entries"]] == [
        "NaN",
        "Infinity",
        "-Infinity",
    ]
    assert payload["config"] == {"floor": "-Infinity"}
    # the records keep the floats
    assert math.isnan(report.to_records()[0]["value"])
