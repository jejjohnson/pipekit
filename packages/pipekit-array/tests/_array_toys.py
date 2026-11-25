"""Toy functions and markers of the tests of plain arrays and of `elementwise`."""

from __future__ import annotations

import importlib.util

import pytest


# the module under test needs both, and so do its tests
pytest.importorskip("array_api_compat")
strict = pytest.importorskip("array_api_strict")

import numpy as np
from pipekit_array.arrays import (
    floating_inputs,
    like,
)
from pipekit_array.kernels import elementwise


SECOND = strict.Device("device1")


def _missing(library: str) -> bool:
    return importlib.util.find_spec(library) is None


needs_jax = pytest.mark.skipif(_missing("jax"), reason="JAX is not installed")
needs_torch = pytest.mark.skipif(_missing("torch"), reason="torch is not installed")


class Unknown:
    """An array-like of no known library, as a pandas Series is."""

    shape = (2,)
    ndim = 1
    dtype = np.dtype(np.float64)

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        return np.zeros(2)


def affine(values, slope, offset=0.0):
    """The toy function: sin(slope · values) + offset, through the backend."""
    xp, (values, slope, offset) = floating_inputs(values, slope, offset)
    return xp.sin(slope * values) + offset


@elementwise()
def density(pressure, temperature, *, constant=287.0):
    """Density = p / (R T)."""
    xp, (p, t) = floating_inputs(pressure, temperature)
    return p / (like(constant, t, xp) * t)


@elementwise(n_outputs=3)
def to_xyz(lon, lat, degrees=False):
    """Unit vectors; z does not depend on the longitude."""
    xp, (lon, lat) = floating_inputs(lon, lat)
    if degrees:
        lon, lat = lon * xp.pi / 180.0, lat * xp.pi / 180.0
    return xp.cos(lat) * xp.cos(lon), xp.cos(lat) * xp.sin(lon), xp.sin(lat)
