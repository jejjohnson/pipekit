"""`jit_report`: which steps of a pipeline are jittable."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
from pipekit_ops import (
    BackendError,
    Branch,
    Cache,
    Graph,
    Histogram,
    Identity,
    Input,
    Lambda,
    Operator,
    RateLimit,
    Sequential,
    Sink,
    Snapshot,
    StatefulOperator,
    Tap,
    Timeout,
    guard,
    jit_report,
)


jax = pytest.importorskip("jax")
jnp = jax.numpy


class HostSort(Operator):
    backends = ("numpy",)

    def _apply(self, x):
        return np.sort(x)


class Undeclared(Operator):
    def _apply(self, x):
        return jnp.asarray(np.sort(np.asarray(x)))


class Stores(Operator):
    def _apply(self, x):
        self.seen = x.mean()
        return x


class Declared(Stores):
    traceable = False


DOUBLE = Lambda(lambda x: 2 * x)
SIGN = Lambda(lambda x: x if x[0] > 0 else -x)


def verdicts(report):
    return [(step.name, step.jittable) for step in report.steps]


@pytest.mark.parametrize(
    ("op", "reason"),
    [
        (HostSort(), "BackendError: HostSort runs on numpy only"),
        (SIGN, "TracerBoolConversionError"),
        (Undeclared(), "TracerArrayConversionError"),
        (Stores(), "it stores a traced value"),
        (Declared(), "Declared declares traceable = False"),
        (Cache(DOUBLE), "Cache declares traceable"),
        (Sink(lambda x: None), "Sink declares traceable"),
        (Snapshot().at("a"), "declares traceable"),
        (Histogram(bins=2).at("a"), "declares traceable"),
        (Timeout(DOUBLE, seconds=60.0), "Timeout declares traceable"),
        (RateLimit(DOUBLE, calls=9, period=60.0), "RateLimit declares traceable"),
        (Branch(lambda x: True, Cache(DOUBLE), DOUBLE), "Cache declares traceable"),
    ],
)
def test_each_reason(op, reason):
    (step,) = jit_report(op, jnp.arange(3.0)).steps
    assert not step.jittable
    assert reason in step.reason


def test_every_step_of_a_pipeline():
    pipe = DOUBLE | HostSort() | SIGN | Stores() | Identity()
    report = jit_report(pipe, jnp.asarray([3.0, 1.0]))
    assert verdicts(report) == [
        ("[0] Lambda", True),
        ("[1] HostSort", False),
        ("[2] Lambda", False),
        ("[3] Stores", False),
        ("[4] Identity", True),
    ]
    assert str(report).splitlines()[0] == "[0] Lambda: jittable"
    with pytest.raises(BackendError, match=r"\[1\] HostSort.*; \[2\] Lambda.*; \[3\]"):
        report.require()


def test_what_passes_require_runs_under_jit():
    pipe = DOUBLE | DOUBLE
    x = jnp.arange(3.0)
    jit_report(pipe, x).require()
    np.testing.assert_array_equal(jax.jit(pipe)(x), pipe(x))


def test_nested_pipelines_and_graphs_are_one_row_per_step():
    x = Input("x")
    graph = Graph(inputs={"x": x}, outputs={"y": SIGN(DOUBLE(x))})
    pipe = Sequential([Sequential([DOUBLE, SIGN]), graph])
    report = jit_report(pipe, jnp.arange(3.0))
    assert verdicts(report) == [
        ("[0][0] Lambda", True),
        ("[0][1] Lambda", False),
        ("[1][0] Lambda", True),
        ("[1][1] Lambda", False),
    ]


def test_a_graph_of_two_inputs():
    a, b = Input("a"), Input("b")
    total = Lambda(lambda p, q: p + q)(DOUBLE(a), b)
    graph = Graph(inputs={"a": a, "b": b}, outputs={"y": total})
    report = jit_report(graph, jnp.ones(2), jnp.ones(2))
    assert report.jittable and len(report.steps) == 2


def test_a_stateful_pipeline_threads_its_state():
    class Count(StatefulOperator):
        def _apply(self, x, state):
            return x + state, state + 1

    pipe = DOUBLE | Count() | Count()
    report = jit_report(pipe, jnp.ones(2), jnp.asarray(1))
    assert report.jittable
    assert [step.name for step in report.steps] == [
        "[0] Lambda",
        "[1] Count",
        "[2] Count",
    ]


def test_the_pipeline_keeps_nothing_and_the_guard_is_quiet():
    op = Stores()
    with guard("raise"):
        jit_report(op | HostSort(), jnp.arange(3.0))
    assert not hasattr(op, "seen")


def test_a_tap_that_keeps_nothing_is_jittable():
    seen = []
    with pytest.warns(UserWarning, match="trace time"):
        report = jit_report(Tap(lambda x: seen.append(x.shape)), jnp.ones(2))
    assert report.jittable
    assert seen == [(2,), (2,)]


def test_without_jax():
    code = (
        "import sys; sys.modules['jax'] = None\n"
        "from pipekit_ops import Identity, jit_report\n"
        "try:\n    jit_report(Identity(), 1.0)\n"
        "except ImportError as exc:\n    print(exc)\n"
    )
    run = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
    assert run.stdout.strip() == "jit_report needs JAX: pip install jax"
