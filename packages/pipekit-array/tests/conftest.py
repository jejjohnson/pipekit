"""Shared test fixtures for pipekit-array — per-backend dispatch.

The headline helper is ``_make_array(backend, shape, seed)``: a
single factory that produces a deterministic random array of the
requested backend, skipping the test if that backend isn't
installed. Used by parametrized tests that fan out across
numpy / jax / torch.

See `docs/design/architecture.md` §8 for the test-matrix rationale.
"""

from __future__ import annotations

from typing import Any

import pytest


BACKENDS = ("numpy", "jax", "torch")


@pytest.fixture(params=BACKENDS)
def backend(request: pytest.FixtureRequest) -> str:
    """Parametrized backend label — fans out tests across numpy / jax / torch."""
    return request.param


def _make_array(backend: str, shape: tuple[int, ...] = (4, 8), seed: int = 0) -> Any:
    """Produce a deterministic random array on the requested backend.

    Skips the test via `pytest.importorskip` if the backend (or its
    namespace dispatch shim for torch) isn't installed.
    """
    if backend == "numpy":
        np = pytest.importorskip("numpy")
        rng = np.random.default_rng(seed)
        return rng.standard_normal(shape).astype(np.float32)

    if backend == "jax":
        jax = pytest.importorskip("jax")
        jnp = pytest.importorskip("jax.numpy")
        key = jax.random.key(seed)
        return jax.random.normal(key, shape).astype(jnp.float32)

    if backend == "torch":
        torch = pytest.importorskip("torch")
        pytest.importorskip("array_api_compat")  # required for torch dispatch
        gen = torch.Generator().manual_seed(seed)
        return torch.randn(*shape, generator=gen, dtype=torch.float32)

    raise ValueError(f"Unknown backend: {backend!r}. Choose from {BACKENDS}.")


@pytest.fixture
def make_array():
    """Fixture wrapper around ``_make_array`` for test use."""
    return _make_array
