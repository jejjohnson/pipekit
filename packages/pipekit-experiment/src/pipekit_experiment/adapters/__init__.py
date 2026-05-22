"""External-tool adapters for pipekit-experiment.

Each adapter is gated behind its tool's optional extra (declared in
the package's ``pyproject.toml``):

- ``pipekit-experiment[dvc]``      → `dvc.DVCDatasetVersioning`
- ``pipekit-experiment[hydra]``    → `hydra.HydraConfigLoader`
- ``pipekit-experiment[metaflow]`` → `metaflow.MetaflowStepAdapter`

Importing the adapter module without its underlying tool raises a
clean `ImportError` at first use, not at import time — so users can
``from pipekit_experiment.adapters import hydra as hydra_adapter``
in a CLI that branches on what's installed.

See master plan Report 12, Part 3.
"""
