"""Backend adapters for `pipekit-train`.

Each adapter is one module — gated behind its backend's optional
extra (declared in this package's ``pyproject.toml``):

- ``pipekit-train[equinox]``   → ``adapters.equinox`` — the reference
  implementation (Equinox + Optax + Orbax + Grain).
- ``pipekit-train[numpyro]``   → ``adapters.numpyro_svi`` and
  ``adapters.numpyro_mcmc`` — Bayesian backends (SVI / NUTS); their
  shared seam (`NumpyroTask`, `NumpyroPredictiveOp`) lives in
  ``adapters.bayes``.
- ``pipekit-train[blackjax]``  → ``adapters.blackjax`` — sampler-library
  backend (NUTS in v1), bridged to NumPyro models via ``adapters.bayes``.
- ``pipekit-train[lightning]`` → ``adapters.lightning`` (v0.2 scaffold).
- ``pipekit-train[keras]``     → ``adapters.keras`` (v0.3 scaffold).

Backend-agnostic helpers (callback dispatch, the optax optimiser
builder) live in ``adapters._common``. The Lightning and Keras modules
ship with a ``run`` that raises ``NotImplementedError`` at first call
so a CLI can branch on what's installed without crashing on import;
they follow in v0.2 / v0.3 per ``docs/design/boundaries.md`` §14.

See ``docs/design/api/adapters.md``.
"""
