"""Pytest configuration for adapter tests.

Simulate a multi-device CPU mesh so the Equinox sharding tests (issue #15)
can run without GPUs. The flag is only honoured at JAX backend start-up, so
it is set here (before any test imports JAX). If JAX was already initialised
single-device earlier in the session, the sharding tests skip gracefully on
their ``len(jax.devices()) < 2`` guard rather than failing.
"""

from __future__ import annotations

import os


os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
