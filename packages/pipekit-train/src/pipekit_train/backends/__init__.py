"""The backends that `TrainingLoop` runs, one area each, behind its extra.

- ``pipekit-train[equinox]``: `pipekit_train.backends.equinox`, the reference
  implementation (Equinox, Optax, Orbax and Grain).
- ``pipekit-train[numpyro]``: `pipekit_train.backends.numpyro`, SVI and NUTS
  (`run_svi`, `run_mcmc`).
- ``pipekit-train[blackjax]``: `pipekit_train.backends.blackjax`, NUTS from
  BlackJAX on NumPyro models.
- ``pipekit-train[keras]``: `pipekit_train.backends.keras`, Keras 3's ``compile``
  and ``fit`` on JAX (`KerasModelOp`, `KerasTask`). Importing it imports no
  Keras, so a CLI can branch on what is installed without crashing on import.

NumPyro and BlackJAX share the seam in ``_bayes`` (`NumpyroTask`,
`NumpyroPredictiveOp`), every backend shares the callback dispatch and the
optax optimiser builder in ``_common``, and the Equinox and Keras backends read
their batches from ``_batches``. Importing this package imports no backend.

See ``docs/design/api/adapters.md``.
"""
