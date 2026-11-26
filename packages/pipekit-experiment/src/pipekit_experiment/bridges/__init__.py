"""Bridges from pipekit-experiment to external tools, one module per tool.

The word is pipekit-array's, for code that works with another library. Each
bridge is gated behind its tool's optional extra (declared in
the package's ``pyproject.toml``):

- ``pipekit-experiment[dvc]``      → `dvc.DVCDatasetVersioning`
- ``pipekit-experiment[hydra]``    → `hydra.HydraConfigLoader`
- ``pipekit-experiment[metaflow]`` → `metaflow.MetaflowStepAdapter`

Importing a bridge without its underlying tool raises a
clean `ImportError` at first use, not at import time — so users can
``from pipekit_experiment.bridges import hydra as hydra_bridge``
in a CLI that branches on what's installed.

See master plan Report 12, Part 3.
"""
