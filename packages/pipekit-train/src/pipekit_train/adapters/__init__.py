"""Backend adapters for `pipekit-train`.

Each adapter is one module — gated behind its backend's optional
extra (declared in this package's ``pyproject.toml``):

- ``pipekit-train[equinox]``   → ``adapters.equinox`` (v0.1 reference; design only).
- ``pipekit-train[lightning]`` → ``adapters.lightning`` (v0.2 scaffold).
- ``pipekit-train[keras]``     → ``adapters.keras`` (v0.3 scaffold).

Importing an adapter module without its underlying tool raises a
clean ``NotImplementedError`` at first use (when ``run`` is called),
not at import time — so a CLI can branch on what's installed.

See ``docs/design/api/adapters.md``.
"""
