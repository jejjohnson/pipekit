"""The backends that `TrainingLoop` runs, one area each, behind its extra.

- ``pipekit-train[equinox]``: `pipekit_train.backends.equinox`, the reference
  implementation (Equinox, Optax, Orbax and Grain).
- ``pipekit-train[numpyro]``: `pipekit_train.backends.numpyro`, SVI and NUTS
  (`run_svi`, `run_mcmc`).
- ``pipekit-train[blackjax]``: `pipekit_train.backends.blackjax`, NUTS from
  BlackJAX on NumPyro models.
- ``pipekit-train[keras]``: `pipekit_train.backends.keras`, a v0.3 scaffold whose
  ``run`` raises ``NotImplementedError``, so a CLI can branch on what is
  installed without crashing on import.

NumPyro and BlackJAX share the seam in ``_bayes`` (`NumpyroTask`,
`NumpyroPredictiveOp`), and every backend shares the callback dispatch and the
optax optimiser builder in ``_common``. Importing this package imports no backend.

See ``docs/design/api/adapters.md``.
"""
