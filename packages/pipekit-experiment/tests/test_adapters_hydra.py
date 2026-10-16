"""Tests for `pipekit_experiment.adapters.hydra`.

When hydra-core / hydra-zen are absent the adapter raises a clean
ImportError at method invocation. Functional tests are gated with
``pytest.importorskip``.
"""

from __future__ import annotations

import pytest
from pipekit import Const, Identity, Operator
from pipekit_experiment.adapters.hydra import HydraConfigLoader


def test_import_error_when_hydra_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "hydra.utils":
            raise ImportError("simulated absence")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="hydra-core"):
        HydraConfigLoader.from_hydra_cfg({"_target_": "anything"})


def test_to_hydra_cfg_import_error_when_hydra_zen_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "hydra_zen":
            raise ImportError("simulated absence")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="hydra-zen"):
        HydraConfigLoader.to_hydra_cfg(Identity())


def test_to_hydra_cfg_rejects_non_operator():
    with pytest.raises(TypeError):
        HydraConfigLoader.to_hydra_cfg(42)  # type: ignore[arg-type]


def test_from_hydra_cfg_round_trip_with_hydra():
    """End-to-end test when hydra is actually installed."""
    pytest.importorskip("hydra")
    cfg = {
        "_target_": f"{Const.__module__}.{Const.__name__}",
        "value": 7,
    }
    op = HydraConfigLoader.from_hydra_cfg(cfg)
    assert isinstance(op, Const)
    assert op() == 7


def test_from_hydra_cfg_rejects_non_operator_target():
    pytest.importorskip("hydra")
    cfg = {"_target_": "builtins.dict"}
    with pytest.raises(TypeError, match="Operator"):
        HydraConfigLoader.from_hydra_cfg(cfg)


def test_to_yaml_round_trip():
    """to_yaml → from_yaml reconstructs an Operator with the same config."""
    pytest.importorskip("hydra_zen")
    pytest.importorskip("omegaconf")
    op: Operator = Const(99)
    text = HydraConfigLoader.to_yaml(op)
    reconstructed = HydraConfigLoader.from_yaml(text)
    assert isinstance(reconstructed, Const)
    assert reconstructed() == 99


# ── from_yaml_sandboxed ──────────────────────────────────────────────────────
class Halve(Operator):
    """x ↦ x / 2, counting constructions."""

    built = 0

    def __init__(self, divisor: float = 2.0) -> None:
        type(self).built += 1
        self.divisor = divisor

    def _apply(self, x: float) -> float:
        return x / self.divisor


HALVE = f"{__name__}.Halve"


@pytest.fixture
def allow():
    """Register classes for one test, and always unregister them."""
    from pipekit import register, unregister

    registered: list[type[Operator]] = []

    def _allow(*classes: type[Operator]) -> None:
        for cls in classes:
            register(cls)
            registered.append(cls)

    yield _allow
    for cls in registered:
        unregister(cls)


def test_sandboxed_yaml_builds_a_registered_pipeline(allow):
    pytest.importorskip("yaml")
    from pipekit import Sequential

    allow(Sequential, Halve)
    text = f"""
_target_: pipekit.Sequential
operators:
  - _target_: {HALVE}
  - _target_: {HALVE}
    divisor: 4.0
"""
    pipe = HydraConfigLoader.from_yaml_sandboxed(text)
    assert isinstance(pipe, Sequential)
    assert pipe(16.0) == 2.0


def test_sandboxed_yaml_refuses_an_unregistered_step_and_builds_nothing(allow):
    pytest.importorskip("yaml")
    from pipekit import Sequential

    allow(Sequential)
    Halve.built = 0
    text = f"""
_target_: pipekit.Sequential
operators:
  - _target_: {HALVE}
"""
    with pytest.raises(PermissionError, match="Halve"):
        HydraConfigLoader.from_yaml_sandboxed(text)
    assert Halve.built == 0


def test_sandboxed_yaml_refuses_a_callable_that_is_not_an_operator(allow):
    """`instantiate` would call os.system; the sandbox never calls a target."""
    pytest.importorskip("yaml")
    text = "_target_: os.system\ncommand: echo pwned\n"
    with pytest.raises(PermissionError, match="not an Operator class"):
        HydraConfigLoader.from_yaml_sandboxed(text)


def test_sandboxed_yaml_never_imports_a_module():
    pytest.importorskip("yaml")
    text = "_target_: some_module_nobody_imported.Thing\n"
    with pytest.raises(PermissionError, match="not in a loaded module"):
        HydraConfigLoader.from_yaml_sandboxed(text)


def test_sandboxed_yaml_refuses_positional_args_and_partials(allow):
    pytest.importorskip("yaml")
    allow(Halve)
    with pytest.raises(ValueError, match="_args_"):
        HydraConfigLoader.from_yaml_sandboxed(f"_target_: {HALVE}\n_args_: [3]\n")
    with pytest.raises(ValueError, match="_partial_"):
        HydraConfigLoader.from_yaml_sandboxed(f"_target_: {HALVE}\n_partial_: true\n")


def test_sandboxed_yaml_refuses_a_closure_even_if_registered(allow):
    pytest.importorskip("yaml")
    from pipekit import Lambda

    allow(Lambda)
    with pytest.raises(RuntimeError, match="forbid_in_yaml"):
        HydraConfigLoader.from_yaml_sandboxed("_target_: pipekit.Lambda\nfn: abs\n")


def test_sandboxed_yaml_does_not_resolve_interpolations(allow):
    """`${oc.env:...}` stays a literal string: the sandbox reads no environment."""
    pytest.importorskip("yaml")
    allow(Const)
    op = HydraConfigLoader.from_yaml_sandboxed(
        "_target_: pipekit.Const\nvalue: ${oc.env:HOME}\n"
    )
    assert op() == "${oc.env:HOME}"


def test_sandboxed_yaml_reads_exception_classes(allow):
    pytest.importorskip("yaml")
    from pipekit import Try

    allow(Try, Identity, Halve)
    text = f"""
_target_: pipekit.Try
primary: {{_target_: {HALVE}}}
fallback: {{_target_: pipekit.Identity}}
on:
  - {{_target_: hydra.utils.get_class, path: builtins.ZeroDivisionError}}
"""
    op = HydraConfigLoader.from_yaml_sandboxed(text)
    assert op.on == (ZeroDivisionError,)


def test_sandboxed_yaml_builds_a_graph_through_from_config(allow):
    pytest.importorskip("yaml")
    from pipekit import Graph

    allow(Graph, Halve)
    text = f"""
_target_: pipekit.Graph.from_config
config:
  inputs: [x]
  operators:
    - _target_: {HALVE}
  nodes:
    - {{operator: 0, parents: [{{input: x}}]}}
    - {{operator: 0, parents: [{{node: 0}}]}}
  outputs:
    quarter: {{node: 1}}
"""
    graph = HydraConfigLoader.from_yaml_sandboxed(text)
    assert isinstance(graph, Graph)
    assert graph(x=8.0) == {"quarter": 2.0}
