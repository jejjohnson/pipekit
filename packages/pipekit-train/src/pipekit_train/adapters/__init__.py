"""Backend adapters for `pipekit-train`.

Each adapter is one module — gated behind its backend's optional
extra (declared in this package's ``pyproject.toml``):

- ``pipekit-train[equinox]``   → ``adapters.equinox`` (v0.1 reference).
- ``pipekit-train[lightning]`` → ``adapters.lightning`` (v0.2 scaffold).
- ``pipekit-train[keras]``     → ``adapters.keras`` (v0.3 scaffold).

In v0.1 the Equinox adapter is the reference implementation; the
Lightning and Keras modules ship with a ``run`` that raises
``NotImplementedError`` at first call so a CLI can branch on what's
installed without crashing on import. Lightning and Keras follow in
v0.2 / v0.3 per ``docs/design/boundaries.md`` §14.

See ``docs/design/api/adapters.md``.
"""
