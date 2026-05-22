"""DVC adapter — dataset content-hash tracking.

DVC is fundamentally about data versioning, not metrics. This adapter
focuses on **dataset content-hash tracking** via the DVC CLI:

- `add(path)` — stage ``path`` into DVC's content-addressed store,
  producing a ``.dvc`` pointer file.
- `pull(path)` — fetch the data referenced by ``path.dvc`` from the
  configured remote.
- `hash(path)` — return the content hash recorded in ``path.dvc``.

The adapter doesn't import the ``dvc`` Python package — it shells out
to the ``dvc`` CLI. That keeps the dependency surface minimal: any
DVC version installed on the system works. The ``[dvc]`` extra
declares the Python package for environments that prefer in-process
APIs; future versions can switch to the Python API without changing
the public interface here.

Behind ``pipekit-experiment[dvc]`` extra.

See master plan Report 12, section 3.2.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any


class DVCDatasetVersioning:
    """Track datasets via the DVC CLI.

    Args:
        repo: Path to the DVC-initialised git repository. Defaults to
            the current working directory.
        dvc_executable: Override the ``dvc`` binary location (useful
            in CI / virtualenv setups).

    Raises:
        RuntimeError: at first use if the ``dvc`` CLI isn't on PATH.
    """

    def __init__(
        self,
        repo: str | Path = ".",
        dvc_executable: str = "dvc",
    ) -> None:
        self.repo = Path(repo)
        self.dvc_executable = dvc_executable

    def _check(self) -> None:
        if shutil.which(self.dvc_executable) is None:
            raise RuntimeError(
                f"DVC executable {self.dvc_executable!r} not found on PATH. "
                "Install with `pip install pipekit-experiment[dvc]` and "
                "ensure the `dvc` CLI is available."
            )

    def _run(self, *args: str, capture: bool = True) -> subprocess.CompletedProcess:
        self._check()
        return subprocess.run(
            [self.dvc_executable, *args],
            cwd=self.repo,
            check=True,
            capture_output=capture,
            text=True,
        )

    def add(self, path: str | Path) -> Path:
        """Stage ``path`` into DVC. Returns the ``.dvc`` pointer file path."""
        self._run("add", str(path))
        pointer = Path(str(path) + ".dvc")
        return pointer

    def pull(self, path: str | Path) -> None:
        """Fetch the data referenced by ``path`` (or ``path.dvc``)."""
        self._run("pull", str(path))

    def push(self, path: str | Path) -> None:
        """Send the data referenced by ``path`` to the remote."""
        self._run("push", str(path))

    def hash(self, path: str | Path) -> str:
        """Return the content hash recorded for ``path``.

        Reads the ``.dvc`` pointer file and extracts the ``md5`` /
        ``hash`` field. Cheap (no DVC subprocess) once the pointer
        exists.

        Raises:
            FileNotFoundError: if no ``.dvc`` pointer is found.
            ValueError: if the pointer file is malformed.
        """
        pointer_path = self._pointer_path(path)
        try:
            text = pointer_path.read_text()
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"No .dvc pointer for {path!r} (looked at {pointer_path})."
            ) from e
        # DVC pointer files are small YAMLs. Avoid pulling in PyYAML —
        # the hash lives in a top-level ``outs:`` list entry, which
        # YAML renders as ``- md5: <hash>`` or ``- hash: <hash>``.
        for line in text.splitlines():
            stripped = line.strip().lstrip("-").strip()
            if stripped.startswith("md5:"):
                return stripped.split(":", 1)[1].strip()
            if stripped.startswith("hash:"):
                return stripped.split(":", 1)[1].strip()
        raise ValueError(
            f"Could not find a content hash in {pointer_path}; "
            "expected an `md5:` or `hash:` line."
        )

    def status(self, path: str | Path | None = None) -> str:
        """Return the textual ``dvc status`` output."""
        args = ["status"]
        if path is not None:
            args.append(str(path))
        return self._run(*args).stdout

    def _pointer_path(self, path: str | Path) -> Path:
        p = Path(path)
        target = p if p.suffix == ".dvc" else Path(str(p) + ".dvc")
        if not target.is_absolute():
            target = self.repo / target
        return target

    def record_hash_in(self, path: str | Path, sink: dict[str, Any]) -> str:
        """Convenience: stash the dataset hash into ``sink`` and return it.

        Pairs nicely with `TrainingArtifact` construction::

            meta: dict[str, str] = {}
            dvc.record_hash_in("data/train/", meta)
            artifact = TrainingArtifact(..., dataset_hash=meta["data/train/"], ...)
        """
        h = self.hash(path)
        sink[str(path)] = h
        return h
