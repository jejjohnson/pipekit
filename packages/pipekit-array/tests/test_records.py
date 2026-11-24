"""Records cross JAX's transforms as pytrees, whichever is imported first."""

from __future__ import annotations

import dataclasses
import subprocess
import sys

import pytest
from pipekit_array.records import register_record


_JAX_LAST = """
import dataclasses, sys
from pipekit_array.records import register_record

@register_record
@dataclasses.dataclass(frozen=True)
class Trend:
    slope: float

assert "jax" not in sys.modules
import jax
assert jax.tree.leaves(Trend(2.0)) == [2.0]
"""


def test_a_record_crosses_jit_and_its_static_field_is_not_traced() -> None:
    jax = pytest.importorskip("jax")

    @dataclasses.dataclass(frozen=True)
    class Trend:
        slope: float
        unit: str

    register_record(Trend, meta_fields=("unit",))

    @jax.jit
    def double(trend: Trend) -> Trend:
        assert isinstance(trend.slope, jax.core.Tracer)
        assert trend.unit == "K"
        return dataclasses.replace(trend, slope=2.0 * trend.slope)

    out = double(Trend(1.5, "K"))
    assert (float(out.slope), out.unit) == (3.0, "K")
    assert jax.tree.leaves(out) == [3.0]


def test_a_record_defined_before_jax_is_imported() -> None:
    pytest.importorskip("jax")
    subprocess.run([sys.executable, "-c", _JAX_LAST], check=True)


def test_a_record_without_jax_is_itself() -> None:
    @dataclasses.dataclass(frozen=True)
    class Trend:
        slope: float

    assert register_record(Trend) is Trend
    assert Trend(1.0).slope == 1.0
