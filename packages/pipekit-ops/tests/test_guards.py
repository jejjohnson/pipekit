"""The opt-in guard on data moving between NumPy and JAX."""

from __future__ import annotations

import contextvars
import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import pytest
from pipekit_ops import (
    BackendError,
    BackendWarning,
    Lambda,
    Map,
    guard,
    guard_level,
    report_move,
)


MOVE = ("Smooth", "JAX data copied to the host")


def test_off_by_default():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        report_move(*MOVE)
    assert guard_level() == "off"


def test_warn_names_the_step_at_the_callers_line():
    with guard("warn"), pytest.warns(BackendWarning, match=r"Smooth: JAX data") as got:
        report_move(*MOVE)
    assert os.path.samefile(got[0].filename, __file__)


def test_raise_is_a_backend_error():
    with guard("raise"), pytest.raises(BackendError, match=r"Smooth: JAX data"):
        report_move(*MOVE)


def test_off_silences_an_outer_guard():
    with guard("raise"), guard("off"):
        report_move(*MOVE)


def test_nesting_restores_the_level_after_an_exception():
    with guard("warn"):
        with pytest.raises(RuntimeError), guard("raise"):
            raise RuntimeError
        assert guard_level() == "warn"
    assert guard_level() == "off"


def test_unknown_level():
    with pytest.raises(ValueError, match="not one of"), guard("loud"):
        pass


@pytest.mark.parametrize("executor", ["serial", "thread", "async"])
def test_holds_inside_pipekits_executors(executor):
    levels = Map(Lambda(lambda _: guard_level()), executor=executor)
    with guard("raise"):
        assert levels([1, 2, 3]) == ["raise"] * 3


def test_a_thread_pool_of_your_own_needs_the_context():
    with guard("raise"), ThreadPoolExecutor(1) as pool:
        assert pool.submit(guard_level).result() == "off"
        context = contextvars.copy_context()
        assert pool.submit(context.run, guard_level).result() == "raise"
