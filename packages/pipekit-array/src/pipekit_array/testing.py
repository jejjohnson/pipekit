"""Contract checks for named-array adapters and `KernelOp` (gates PK-K1, PK-K4).

A carrier library proves its adapter in its own test suite with one call,
instead of copying tests::

    from pipekit_array.testing import check_adapter, check_kernel_op

    def test_adapter_contract():
        check_adapter(GeoArrayAdapter(), make_geoarray)

    def test_ndvi_on_every_carrier():
        check_kernel_op(ndvi, [(nir_np, red_np), (nir_jax, red_jax), (nir_da, red_da)])

- `check_adapter(adapter, make_carrier)` — every dim-level function (and
  the adapter's fast paths, which are what the functions call) on the
  carrier equals the generic path on a `DimArray` with the same data,
  dims and labels; the protocol methods behave.
- `check_kernel_op(op, calls)` — the same `KernelOp` on each carrier family
  gives equal data, equal dims, and the signature
  ``op.compute_output_signature`` predicts (PK-K1, PK-K4).

Both raise `AssertionError` naming the function and the carrier type.
They need NumPy, and nothing from pytest.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from pipekit import NamedArrayAdapter, Signature, get_adapter

from pipekit_array import dims as _dims
from pipekit_array.dimarray import DimArray
from pipekit_array.kernel import KernelOp


__all__ = ["DIMS", "LABELS", "check_adapter", "check_kernel_op", "values_of"]

#: the layout `check_adapter` builds carriers in: two leading dims and a grid
DIMS: tuple[str, ...] = ("time", "band", "y", "x")
#: labels `check_adapter` gives the band dim
LABELS: dict[str, tuple[Any, ...]] = {"band": ("B2", "B4", "B8")}

MakeCarrier = Callable[[np.ndarray, tuple[str, ...], dict[str, tuple[Any, ...]]], Any]


def values_of(x: Any) -> np.ndarray:
    """A named carrier's data on the host, computing it if lazy."""
    adapter = get_adapter(x)
    data = x if adapter is None else adapter.data(x)
    compute = getattr(data, "compute", None)
    if callable(compute) and type(data).__module__.startswith("dask."):
        data = compute()
    return np.asarray(data)


def _dims_of(x: Any) -> tuple[str, ...]:
    adapter = get_adapter(x)
    assert adapter is not None, f"{type(x).__name__} has no registered adapter"
    return tuple(adapter.dims(x))


def _cases() -> list[tuple[str, Callable[[Any], Any]]]:
    """Each dim-level function, on the `DIMS` layout; outputs are named carriers."""
    return [
        ("isel(time=0)", lambda c: _dims.isel(c, time=0)),
        ("isel(x=slice)", lambda c: _dims.isel(c, x=slice(1, 3))),
        ("sel(band=B8)", lambda c: _dims.sel(c, band="B8")),
        ("sel(band=[B8, B2])", lambda c: _dims.sel(c, band=["B8", "B2"])),
        ("transpose", lambda c: _dims.transpose(c, ("band", "time", "y", "x"))),
        ("reduce(mean, time)", lambda c: _dims.reduce(c, "mean", "time")),
        (
            "reduce(max, (time, band))",
            lambda c: _dims.reduce(c, "max", ("time", "band")),
        ),
        (
            "reduce(mean, time, where)",
            lambda c: _dims.reduce(
                c, "mean", "time", where=np.array([True, False, True])
            ),
        ),
        (
            "argreduce(argmax, time)",
            lambda c: _dims.argreduce(_dims.sel(c, band="B8"), "argmax", "time"),
        ),
        (
            "take_along(time)",
            lambda c: _dims.take_along(
                c, _dims.argreduce(_dims.sel(c, band="B8"), "argmax", "time"), "time"
            ),
        ),
        (
            "map_over(time)",
            lambda c: _dims.map_over(
                c, lambda item: _dims.reduce(item, "sum", "band"), "time"
            ),
        ),
        (
            "stack(run)",
            lambda c: _dims.stack(
                [_dims.isel(c, time=0), _dims.isel(c, time=1)], "run"
            ),
        ),
        (
            "stack_items(run)",
            lambda c: _dims.stack_items(
                [_dims.isel(c, time=0), _dims.isel(c, time=1)], "run"
            ),
        ),
        (
            "concat(time)",
            lambda c: _dims.concat(
                [_dims.isel(c, time=slice(0, 1)), _dims.isel(c, time=slice(1, None))],
                "time",
            ),
        ),
        (
            "combine",
            lambda c: _dims.combine(
                lambda a, b: a - b, _dims.sel(c, band="B8"), _dims.sel(c, band="B4")
            ),
        ),
    ]


def check_adapter(
    adapter: NamedArrayAdapter,
    make_carrier: MakeCarrier,
    *,
    rtol: float = 1e-12,
    seed: int = 0,
) -> None:
    """Assert that ``adapter`` and its fast paths agree with the reference adapter.

    Args:
        adapter: The adapter under test; it must be the one registered for
            the carriers ``make_carrier`` builds.
        make_carrier: ``(data, dims, labels) -> carrier``, building a carrier
            in the `DIMS` layout (``time band y x``, band labels `LABELS`)
            from a float64 NumPy array of shape (3, 3, 4, 5). Put the data on
            the backend under test inside it (``jnp.asarray``, a dask array).
        rtol: Tolerance for reductions; selections compare exactly.
        seed: For the random data.

    Raises:
        AssertionError: On the first mismatch, naming the function and the
            carrier type.
    """
    rng = np.random.default_rng(seed)
    data = rng.random((3, 3, 4, 5))
    carrier = make_carrier(data, DIMS, dict(LABELS))
    reference = DimArray(data, DIMS, dict(LABELS))
    name = type(carrier).__name__
    registered = get_adapter(carrier)
    # the same type: the functions dispatch through the registered instance
    assert type(registered) is type(adapter), (
        f"{name}: get_adapter returns {type(registered).__name__}, "
        f"not the adapter under test {type(adapter).__name__}"
    )
    _check_protocol(adapter, carrier, data, name)
    lazy = _dims.backend_name(adapter.data(carrier)) == "dask"
    for label, fn in _cases():
        if lazy and label.endswith("where)"):
            continue  # dask's reductions take no where= (reduce raises TypeError)
        expected = fn(reference)
        actual = fn(carrier)
        got = _dims_of(actual)
        want = _dims_of(expected)
        assert got == want, f"{label} on {name}: dims {got}, expected {want}"
        exact = not label.startswith(("reduce", "map_over"))
        a, e = values_of(actual), values_of(expected)
        if exact:
            np.testing.assert_array_equal(a, e, err_msg=f"{label} on {name}")
        else:
            np.testing.assert_allclose(a, e, rtol=rtol, err_msg=f"{label} on {name}")


def _check_protocol(
    adapter: NamedArrayAdapter, carrier: Any, data: np.ndarray, name: str
) -> None:
    assert isinstance(adapter, NamedArrayAdapter), f"{name}: not a NamedArrayAdapter"
    assert tuple(adapter.dims(carrier)) == DIMS, f"{name}: dims {adapter.dims(carrier)}"
    np.testing.assert_array_equal(values_of(carrier), data, err_msg=f"{name}: data")
    labels = adapter.labels(carrier, "band")
    assert labels is not None and tuple(labels) == LABELS["band"], (
        f"{name}: band labels {labels}"
    )
    sig = adapter.signature(carrier)
    assert isinstance(sig, Signature), f"{name}: signature is {type(sig).__name__}"
    assert sig.matches(" ".join(DIMS)), f"{name}: signature {sig} does not match {DIMS}"
    assert dict(sig.dims) == dict(zip(DIMS, data.shape, strict=True)), (
        f"{name}: sizes {sig}"
    )
    checked = adapter.check_compatible(carrier, carrier)
    assert len(checked) == 2, (
        f"{name}: check_compatible returned {len(checked)} carriers"
    )
    selected = adapter.isel(carrier, {"time": 0, "band": slice(1, None)})
    assert tuple(adapter.dims(selected)) == ("band", "y", "x"), f"{name}: isel dims"
    assert tuple(adapter.labels(selected, "band") or ()) == LABELS["band"][1:], (
        f"{name}: isel did not slice the band labels"
    )
    rewrapped = adapter.rewrap(carrier, data * 2.0, DIMS)
    np.testing.assert_array_equal(
        values_of(rewrapped), data * 2.0, err_msg=f"{name}: rewrap"
    )


def check_kernel_op(op: KernelOp, calls: Sequence[Any], *, rtol: float = 1e-6) -> None:
    """Assert that one `KernelOp` agrees across carrier families (PK-K1, PK-K4).

    Args:
        op: The operator under test.
        calls: The inputs of each call, one entry per carrier family: a
            carrier for a one-input kernel, or a tuple of inputs. The first
            entry is the reference.
        rtol: Relative tolerance on the outputs' values.

    Raises:
        AssertionError: Outputs differ in dims or values, or a one-input
            kernel's output dims differ from ``op.compute_output_signature``.
    """
    if not calls:
        raise ValueError("check_kernel_op needs at least one call.")
    results = []
    for entry in calls:
        inputs = entry if isinstance(entry, tuple) else (entry,)
        out = op(*inputs)
        outs = out if isinstance(out, tuple) else (out,)
        results.append((inputs, outs))
    _, ref_outs = results[0]
    for inputs, outs in results:
        name = "/".join(type(x).__name__ for x in inputs)
        assert len(outs) == len(ref_outs), (
            f"{name}: {len(outs)} outputs, not {len(ref_outs)}"
        )
        for k, (o, r) in enumerate(zip(outs, ref_outs, strict=True)):
            got, want = _dims_of(o), _dims_of(r)
            assert got == want, f"output {k} on {name}: dims {got}, expected {want}"
            np.testing.assert_allclose(
                values_of(o), values_of(r), rtol=rtol, err_msg=f"output {k} on {name}"
            )
        if len(inputs) == 1:
            _check_signature(op, inputs[0], outs, name)


def _check_signature(op: KernelOp, x: Any, outs: tuple[Any, ...], name: str) -> None:
    adapter = get_adapter(x)
    assert adapter is not None
    predicted = op.compute_output_signature(adapter.signature(x))
    predicted = predicted if isinstance(predicted, tuple) else (predicted,)
    for k, (p, o) in enumerate(zip(predicted, outs, strict=True)):
        got = tuple(d for d, _ in p.dims)
        assert got == _dims_of(o), (
            f"output {k} on {name}: compute_output_signature gives {got}, "
            f"the output has {_dims_of(o)}"
        )
        out_adapter = get_adapter(o)
        if out_adapter is None:
            continue
        actual = out_adapter.signature(o)
        # every size and the dtype the prediction states must be the output's
        for (d, want), (_, have) in zip(p.dims, actual.dims, strict=True):
            assert want is None or want == have, (
                f"output {k} on {name}: compute_output_signature gives {d}={want}, "
                f"the output has {d}={have}"
            )
        assert p.dtype is None or p.dtype == actual.dtype, (
            f"output {k} on {name}: compute_output_signature gives dtype "
            f"{p.dtype}, the output has {actual.dtype}"
        )
