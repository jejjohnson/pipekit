"""Gate G0-3: pipekit-ops declares no dependency and imports only the stdlib."""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path


PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

#: Run in a fresh interpreter: the modules `import pipekit_ops` adds.
PROBE = """
import json, sys
before = set(sys.modules)
import pipekit_ops
print(json.dumps(sorted(set(sys.modules) - before)))
"""

#: Allowed top-level names outside ``sys.stdlib_module_names``: the package
#: itself, and ``__mp_main__``, multiprocessing's alias of ``__main__``.
ALLOWED = {"pipekit_ops", "__mp_main__"}


def test_pyproject_declares_no_dependencies():
    meta = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    assert meta["project"]["dependencies"] == []


def test_import_loads_only_stdlib_modules():
    run = subprocess.run(
        [sys.executable, "-c", PROBE], check=True, capture_output=True, text=True
    )
    added = json.loads(run.stdout)
    outside = sorted(
        name
        for name in added
        if name.split(".")[0] not in {*sys.stdlib_module_names, *ALLOWED}
    )
    assert "pipekit_ops" in added
    assert not outside, f"importing pipekit_ops loaded non-stdlib modules: {outside}"
