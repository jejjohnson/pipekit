"""Tests for `pipekit.control` — Group D."""

from __future__ import annotations

import pytest
from pipekit import (
    Branch,
    Coalesce,
    Const,
    Identity,
    Lambda,
    Retry,
    Sequential,
    Switch,
    Try,
)


# -----------------------------------------------------------------------------
# Branch
# -----------------------------------------------------------------------------


def test_branch_true_path(Scale_cls):
    op = Branch(predicate=lambda x: x > 0, if_true=Scale_cls(2.0))
    assert op(3.0) == 6.0


def test_branch_false_path_defaults_to_identity(Scale_cls):
    op = Branch(predicate=lambda x: x > 0, if_true=Scale_cls(2.0))
    # Negative input → false branch (Identity) → unchanged
    assert op(-3.0) == -3.0


def test_branch_explicit_false_path(Scale_cls, AddConst_cls):
    op = Branch(
        predicate=lambda x: x > 0,
        if_true=Scale_cls(2.0),
        if_false=AddConst_cls(100.0),
    )
    assert op(3.0) == 6.0
    assert op(-3.0) == 97.0


def test_branch_rejects_non_operator_if_true():
    with pytest.raises(TypeError, match="if_true must be an Operator"):
        Branch(predicate=bool, if_true=lambda x: x)  # type: ignore[arg-type]


def test_branch_rejects_non_operator_if_false():
    with pytest.raises(TypeError, match="if_false must be an Operator"):
        Branch(predicate=bool, if_true=Identity(), if_false="nope")  # type: ignore[arg-type]


def test_branch_is_forbid_in_yaml():
    assert Branch.forbid_in_yaml is True


def test_branch_config_includes_branches(Scale_cls):
    op = Branch(predicate=lambda x: x > 0, if_true=Scale_cls(2.0))
    cfg = op.get_config()
    assert cfg["if_true"]["class"] == "Scale"
    assert cfg["if_false"]["class"] == "Identity"


# -----------------------------------------------------------------------------
# Switch
# -----------------------------------------------------------------------------


def test_switch_matched_case(Scale_cls, AddConst_cls):
    op = Switch(
        key=lambda x: "scale" if x > 0 else "add",
        cases={"scale": Scale_cls(2.0), "add": AddConst_cls(1.0)},
    )
    assert op(3.0) == 6.0
    assert op(-3.0) == -2.0


def test_switch_default_when_no_case_matches(Scale_cls):
    op = Switch(
        key=lambda x: "unknown",
        cases={"scale": Scale_cls(2.0)},
        default=Const(42),
    )
    assert op(3.0) == 42


def test_switch_default_defaults_to_identity(Scale_cls):
    op = Switch(key=lambda x: "no_match", cases={"scale": Scale_cls(2.0)})
    assert op(3.0) == 3.0


def test_switch_rejects_non_operator_case():
    with pytest.raises(TypeError, match=r"cases\['x'\] must be an Operator"):
        Switch(key=str, cases={"x": lambda x: x})  # type: ignore[dict-item]


def test_switch_rejects_non_operator_default():
    with pytest.raises(TypeError, match="default must be an Operator"):
        Switch(key=str, cases={}, default="nope")  # type: ignore[arg-type]


def test_switch_is_forbid_in_yaml():
    assert Switch.forbid_in_yaml is True


# -----------------------------------------------------------------------------
# Try
# -----------------------------------------------------------------------------


def test_try_primary_succeeds(Scale_cls):
    op = Try(primary=Scale_cls(2.0), fallback=Const(-1), on=(ValueError,))
    assert op(3.0) == 6.0


def test_try_fallback_runs_on_caught_exception():
    raiser = Lambda(lambda x: (_ for _ in ()).throw(ValueError("boom")))
    op = Try(primary=raiser, fallback=Const(42), on=(ValueError,))
    assert op(3.0) == 42


def test_try_uncaught_exception_propagates():
    raiser = Lambda(lambda x: (_ for _ in ()).throw(KeyError("nope")))
    op = Try(primary=raiser, fallback=Const(42), on=(ValueError,))
    with pytest.raises(KeyError):
        op(3.0)


def test_try_requires_at_least_one_exception_type():
    with pytest.raises(ValueError, match="must list at least one"):
        Try(primary=Identity(), fallback=Identity(), on=())


def test_try_rejects_non_exception_class():
    with pytest.raises(TypeError, match="must be exception classes"):
        Try(primary=Identity(), fallback=Identity(), on=(int,))  # type: ignore[arg-type]


def test_try_rejects_non_operator_primary():
    with pytest.raises(TypeError, match="primary must be an Operator"):
        Try(primary=lambda x: x, fallback=Identity(), on=(ValueError,))  # type: ignore[arg-type]


def test_try_rejects_non_operator_fallback():
    with pytest.raises(TypeError, match="fallback must be an Operator"):
        Try(primary=Identity(), fallback=lambda x: x, on=(ValueError,))  # type: ignore[arg-type]


def test_try_is_not_forbid_in_yaml():
    """Try holds only operators + exception types — both picklable."""
    assert Try.forbid_in_yaml is False


def test_try_config_serialises_exception_names():
    op = Try(primary=Identity(), fallback=Identity(), on=(ValueError, KeyError))
    cfg = op.get_config()
    assert cfg["on"] == ["ValueError", "KeyError"]


# -----------------------------------------------------------------------------
# Coalesce
# -----------------------------------------------------------------------------


def test_coalesce_first_passing_source_wins(Scale_cls):
    op = Coalesce(
        sources=[Const(-1), Const(0), Scale_cls(2.0)],
        is_ok=lambda r: r > 0,
    )
    # Const(-1) -> -1 (fails); Const(0) -> 0 (fails); Scale(2) on 3 -> 6 (passes)
    assert op(3.0) == 6.0


def test_coalesce_raises_when_all_fail():
    op = Coalesce(sources=[Const(-1), Const(0)], is_ok=lambda r: r > 0)
    with pytest.raises(RuntimeError, match="none of 2 sources"):
        op(3.0)


def test_coalesce_short_circuits_on_first_success(Scale_cls):
    """Later sources should not be invoked once one succeeds."""
    calls: list[str] = []

    def make_source(name, value):
        def fn(_):
            calls.append(name)
            return value

        return Lambda(fn, name=name)

    op = Coalesce(
        sources=[make_source("a", 5), make_source("b", 10)],
        is_ok=lambda r: r > 0,
    )
    assert op(0) == 5
    assert calls == ["a"]


def test_coalesce_requires_at_least_one_source():
    with pytest.raises(ValueError, match="at least one source"):
        Coalesce(sources=[], is_ok=lambda r: True)


def test_coalesce_rejects_non_operator_source():
    with pytest.raises(TypeError, match=r"sources\[0\] must be an Operator"):
        Coalesce(sources=[lambda x: x], is_ok=lambda r: True)  # type: ignore[list-item]


def test_coalesce_is_forbid_in_yaml():
    assert Coalesce.forbid_in_yaml is True


# -----------------------------------------------------------------------------
# Retry
# -----------------------------------------------------------------------------


def test_retry_succeeds_on_first_attempt(Scale_cls):
    op = Retry(op=Scale_cls(2.0), attempts=3, base_delay=0.0, on=(IOError,))
    assert op(3.0) == 6.0


def test_retry_eventually_succeeds(monkeypatch):
    """Two failures, then success — final value should bubble out."""
    counter = {"n": 0}

    def flaky(_):
        counter["n"] += 1
        if counter["n"] < 3:
            raise ConnectionError("transient")
        return "ok"

    op = Retry(
        op=Lambda(flaky),
        attempts=5,
        base_delay=0.0,
        on=(ConnectionError,),
    )
    # No-op sleep
    monkeypatch.setattr(op, "_sleep", lambda s: None)
    assert op("input") == "ok"
    assert counter["n"] == 3


def test_retry_exhausts_attempts(monkeypatch):
    """After all attempts fail, the last exception should propagate."""

    def always_fails(_):
        raise ConnectionError("nope")

    op = Retry(
        op=Lambda(always_fails),
        attempts=3,
        base_delay=0.0,
        on=(ConnectionError,),
    )
    monkeypatch.setattr(op, "_sleep", lambda s: None)
    with pytest.raises(ConnectionError, match="nope"):
        op("input")


def test_retry_uncaught_exception_propagates_immediately():
    def wrong_exc(_):
        raise KeyError("nope")

    op = Retry(
        op=Lambda(wrong_exc),
        attempts=5,
        base_delay=0.0,
        on=(ConnectionError,),
    )
    with pytest.raises(KeyError):
        op("input")


def test_retry_uses_exponential_backoff(monkeypatch):
    """Sleep should be base_delay * 2 ** (attempt - 1)."""
    sleeps: list[float] = []

    def flaky(_):
        raise ConnectionError("transient")

    op = Retry(
        op=Lambda(flaky),
        attempts=4,
        base_delay=0.5,
        on=(ConnectionError,),
    )
    monkeypatch.setattr(op, "_sleep", sleeps.append)
    with pytest.raises(ConnectionError):
        op("x")
    # 3 sleeps before the 4th and final attempt: 0.5, 1.0, 2.0
    assert sleeps == [0.5, 1.0, 2.0]


def test_retry_rejects_invalid_args():
    with pytest.raises(TypeError, match="op must be an Operator"):
        Retry(op="nope", attempts=3, base_delay=0.0, on=(IOError,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="attempts must be >= 1"):
        Retry(op=Identity(), attempts=0, base_delay=0.0, on=(IOError,))
    with pytest.raises(ValueError, match="base_delay must be >= 0"):
        Retry(op=Identity(), attempts=3, base_delay=-1.0, on=(IOError,))
    with pytest.raises(ValueError, match="at least one exception"):
        Retry(op=Identity(), attempts=3, base_delay=0.0, on=())


def test_retry_is_not_forbid_in_yaml():
    """Retry holds only an operator + exception types — both picklable."""
    assert Retry.forbid_in_yaml is False


# -----------------------------------------------------------------------------
# Composition smoke test
# -----------------------------------------------------------------------------


def test_control_flow_composes_into_sequential(Scale_cls, AddConst_cls):
    """Branch/Switch/Try are all Operators; they compose into Sequential."""
    pipe = Sequential(
        [
            Branch(
                predicate=lambda x: x > 0,
                if_true=Scale_cls(2.0),
                if_false=AddConst_cls(100.0),
            ),
            AddConst_cls(1.0),
        ]
    )
    assert pipe(3.0) == 7.0  # (3 * 2) + 1
    assert pipe(-3.0) == 98.0  # (-3 + 100) + 1


def test_coalesce_raising_source_propagates_by_default():
    """Without `on`, a raising source is an error, not a fall-through."""
    boom = Lambda(lambda x: (_ for _ in ()).throw(OSError("io")), name="boom")
    op = Coalesce(sources=[boom, Const(1)], is_ok=lambda r: True)
    with pytest.raises(OSError):
        op(0)


def test_coalesce_on_tolerates_listed_exceptions():
    boom = Lambda(lambda x: (_ for _ in ()).throw(OSError("io")), name="boom")
    op = Coalesce(sources=[boom, Const(7)], is_ok=lambda r: True, on=(OSError,))
    assert op(0) == 7


def test_coalesce_on_does_not_swallow_unlisted_exceptions():
    boom = Lambda(lambda x: (_ for _ in ()).throw(KeyError("k")), name="boom")
    op = Coalesce(sources=[boom, Const(1)], is_ok=lambda r: True, on=(OSError,))
    with pytest.raises(KeyError):
        op(0)


def test_coalesce_all_sources_raising_reports_runtime_error():
    boom = Lambda(lambda x: (_ for _ in ()).throw(OSError("io")), name="boom")
    op = Coalesce(sources=[boom], is_ok=lambda r: True, on=(OSError,))
    with pytest.raises(RuntimeError, match="1 raised"):
        op(0)


def test_coalesce_rejects_non_exception_on_entries():
    with pytest.raises(TypeError):
        Coalesce(sources=[Const(1)], is_ok=lambda r: True, on=("oops",))
