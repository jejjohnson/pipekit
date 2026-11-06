"""pipekit — the pipekit workspace in one install.

This distribution holds no code. It installs the seven packages of the
workspace, each imported under its own name:

| Distribution | Import | Use it for |
|---|---|---|
| ``pipekit-ops`` | ``pipekit_ops`` | ``Operator``, ``Graph`` and ready-made operators |
| ``pipekit-array`` | ``pipekit_array`` | Named-array and Array API operators |
| ``pipekit-cycle`` | ``pipekit_cycle`` | Time stepping and data assimilation |
| ``pipekit-experiment`` | ``pipekit_experiment`` | Model registry and run tracking |
| ``pipekit-evaluate`` | ``pipekit_evaluate`` | Benchmarks and scoring |
| ``pipekit-train`` | ``pipekit_train`` | Training pipelines |
| ``pipekit-jax`` | ``pipekit_jax`` | Equinox models and diffrax forward models |

    import pipekit_ops as pk
    import pipekit_cycle as pc

Before 0.2.0, `pipekit` was the core package. It is now `pipekit-ops`
(import `pipekit_ops`); asking this module for one of its names raises an
`ImportError` that says where it went.
"""

from __future__ import annotations


__version__ = "0.1.5"  # x-release-please-version

__all__ = ["__version__"]


def __getattr__(name: str) -> object:
    """Point a name that moved to `pipekit_ops` at its new home."""
    import pipekit_ops

    if name in pipekit_ops.__all__:
        raise ImportError(
            f"pipekit.{name} moved to pipekit_ops in pipekit 0.2.0: "
            f"use `from pipekit_ops import {name}`."
        )
    raise AttributeError(f"module 'pipekit' has no attribute {name!r}")
