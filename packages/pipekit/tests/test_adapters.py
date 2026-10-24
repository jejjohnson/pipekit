"""Tests for `pipekit.adapters` — the named-array adapter registry (#100)."""

from __future__ import annotations

import dataclasses
import sys
import types
from collections.abc import Iterator
from typing import Any

import pytest
from pipekit import (
    MissingAdapterError,
    NamedArrayAdapter,
    Operator,
    Sequential,
    Signature,
    adapters as adapters_module,
    get_adapter,
    register_adapter,
    signature_of,
    unregister_adapter,
)


@dataclasses.dataclass(frozen=True)
class Toy:
    """A toy carrier: a nested list with named dims."""

    data: Any
    dims: tuple[str, ...]


class SubToy(Toy):
    """A subclass, to check MRO lookup."""


def _shape(data: Any) -> tuple[int, ...]:
    shape: list[int] = []
    while isinstance(data, list):
        shape.append(len(data))
        data = data[0] if data else None
    return tuple(shape)


class ToyAdapter:
    """A minimal adapter for `Toy`."""

    def dims(self, x: Toy) -> tuple[str, ...]:
        return x.dims

    def data(self, x: Toy) -> Any:
        return x.data

    def labels(self, x: Toy, dim: str) -> tuple[Any, ...] | None:
        return None

    def isel(self, x: Toy, indexers: dict[str, Any]) -> Toy:
        raise NotImplementedError

    def rewrap(self, x: Toy, data: Any, dims: tuple[str, ...]) -> Toy:
        return type(x)(data, dims)

    def check_compatible(self, *xs: Toy) -> tuple[Toy, ...]:
        dims = {x.dims for x in xs}
        if len(dims) > 1:
            raise ValueError(f"dims differ: {sorted(dims)}")
        return xs

    def signature(self, x: Toy) -> Signature:
        return Signature(dims=tuple(zip(x.dims, _shape(x.data), strict=True)))


class OtherAdapter(ToyAdapter):
    """A second adapter, to check replacement and closest-wins."""


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Each test sees an empty registry; the real one is restored after."""
    monkeypatch.setattr(adapters_module, "_REGISTRY", {})
    yield


def test_register_and_unregister() -> None:
    toy = Toy([[1.0, 2.0]], ("y", "x"))
    assert get_adapter(toy) is None
    adapter = ToyAdapter()
    register_adapter(Toy, adapter)
    assert get_adapter(toy) is adapter
    unregister_adapter(Toy)
    assert get_adapter(toy) is None
    # unregistering what is not registered is a no-op
    unregister_adapter(Toy)


def test_subclass_gets_its_bases_adapter_and_closest_wins() -> None:
    base, closer = ToyAdapter(), OtherAdapter()
    register_adapter(Toy, base)
    sub = SubToy([1.0], ("x",))
    assert get_adapter(sub) is base
    register_adapter(SubToy, closer)
    assert get_adapter(sub) is closer
    assert get_adapter(Toy([1.0], ("x",))) is base


def test_reregistering_replaces() -> None:
    first, second = ToyAdapter(), OtherAdapter()
    register_adapter(Toy, first)
    register_adapter(Toy, second)
    assert get_adapter(Toy([1.0], ("x",))) is second


def test_unnamed_values_have_no_adapter() -> None:
    assert get_adapter([1.0, 2.0]) is None
    assert get_adapter(3.0) is None
    assert get_adapter(None) is None


def test_register_rejects_non_class_and_non_adapter() -> None:
    with pytest.raises(TypeError, match="needs a class"):
        register_adapter(Toy([1.0], ("x",)), ToyAdapter())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="does not satisfy"):
        register_adapter(Toy, object())  # type: ignore[arg-type]


def test_toy_adapter_satisfies_the_protocol() -> None:
    assert isinstance(ToyAdapter(), NamedArrayAdapter)
    assert not isinstance(object(), NamedArrayAdapter)


def test_known_family_without_adapter_names_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An xarray object with no adapter is never treated as a raw array."""
    fake = types.ModuleType("xarray.core.dataarray")
    data_array = type("DataArray", (), {"__module__": "xarray.core.dataarray"})
    fake.DataArray = data_array  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "xarray.core.dataarray", fake)
    with pytest.raises(MissingAdapterError, match=r"pipekit-array\[xarray\]"):
        get_adapter(data_array())
    assert issubclass(MissingAdapterError, TypeError)
    subclass = type("MyArray", (data_array,), {"__module__": "myapp.arrays"})
    with pytest.raises(MissingAdapterError, match=r"myapp\.arrays\.MyArray"):
        get_adapter(subclass())


def test_other_types_of_a_known_library_pass_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An xr.Dataset is not a carrier: no adapter, no error, signature passes it on."""
    fake = types.ModuleType("xarray.core.dataset")
    dataset = type("Dataset", (), {"__module__": "xarray.core.dataset"})
    fake.Dataset = dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "xarray.core.dataset", fake)
    value = dataset()
    assert get_adapter(value) is None
    assert signature_of(value) is value


def test_check_compatible_returns_the_carriers() -> None:
    adapter = ToyAdapter()
    a, b = Toy([1.0], ("x",)), Toy([2.0], ("x",))
    assert adapter.check_compatible(a, b) == (a, b)
    with pytest.raises(ValueError, match="dims differ"):
        adapter.check_compatible(a, Toy([2.0], ("y",)))


def test_signature_of() -> None:
    register_adapter(Toy, ToyAdapter())
    toy = Toy([[1.0, 2.0, 3.0]], ("y", "x"))
    assert signature_of(toy) == Signature(dims=(("y", 1), ("x", 3)))
    sig = Signature(dims=(("x", 4),))
    assert signature_of(sig) is sig
    assert signature_of(None) is None
    # other values pass through, for libraries with their own signature types
    marker = object()
    assert signature_of(marker) is marker


class Halve(Operator):
    """Drops the last dim, to check summaries start from a carrier."""

    def _apply(self, x: Any) -> Any:
        return x

    def compute_output_signature(self, input_signature: Any) -> Any:
        if input_signature is None:
            return None
        last = input_signature.dims[-1][0]
        return input_signature.drop_dims((last,))


def test_summary_and_signatures_start_from_a_carrier() -> None:
    register_adapter(Toy, ToyAdapter())
    toy = Toy([[1.0, 2.0, 3.0]], ("y", "x"))
    pipe = Sequential([Halve()])
    assert pipe.compute_output_signature(toy) == Signature(dims=(("y", 1),))
    assert "(y=1, x=3)" in pipe.summary(toy)
    # the base Operator default reads the carrier too
    assert Operator().compute_output_signature(toy) == signature_of(toy)
