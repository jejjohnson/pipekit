"""Backend adapters for `pipekit-train`.

Each adapter is one module — gated behind its backend's optional
extra (declared in this package's ``pyproject.toml``):

- ``pipekit-train[equinox]``   → ``adapters.equinox`` (v0.1 reference; stub).
- ``pipekit-train[lightning]`` → ``adapters.lightning`` (v0.2 scaffold).
- ``pipekit-train[keras]``     → ``adapters.keras`` (v0.3 scaffold).

In the v0.0 scaffold all three modules exist with a ``run`` function
that raises ``NotImplementedError`` at first call — so a CLI can
branch on what's installed without crashing on import. The Equinox
adapter's body lands first (v0.1); Lightning and Keras follow in
v0.2 / v0.3 per ``docs/design/boundaries.md`` §14.

See ``docs/design/api/adapters.md``.
"""
