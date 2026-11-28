"""Run the code blocks of pipekit-array's docs pages and check what they print.

Each page's ``python`` blocks run in order, as one script. A block that is
followed, before the next block, by "This prints `…`" or by "This prints"
and a ``text`` block must print exactly that.
"""

from __future__ import annotations

import contextlib
import io
import re
import sys
import types
import warnings
from pathlib import Path

import pytest


pytest.importorskip("numpy")
pytest.importorskip("jax")
pytest.importorskip("xarray")
pytest.importorskip("sklearn")
pytest.importorskip("einx")
pytest.importorskip("array_api_compat")

DOCS = Path(__file__).resolve().parents[3] / "docs"
PAGES = [
    "tour-array.md",
    "guides/carriers-and-backends.md",
    "guides/library-on-pipekit-array.md",
    "background/array-api.md",
    "tutorials/padded-batches.md",
    "tutorials/sklearn-named-dims.md",
]

_BLOCK = re.compile(r"```python\n(.*?)```", flags=re.S)
_INLINE = re.compile(r"This prints `([^`]*)`")
_FENCED = re.compile(r"This prints\s*\n\s*\n```text\n(.*?)```", flags=re.S)


def _blocks(text: str) -> list[tuple[str, str | None]]:
    """Each python block, and the output the prose after it expects."""
    matches = list(_BLOCK.finditer(text))
    out = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        after = text[m.end() : end]
        fenced, inline = _FENCED.search(after), _INLINE.search(after)
        if fenced and (not inline or fenced.start() < inline.start()):
            expected = _lines(fenced.group(1))
        elif inline:
            expected = _lines(inline.group(1))
        else:
            expected = None
        out.append((m.group(1), expected))
    return out


def _lines(text: str) -> str:
    """Text without trailing spaces on its lines, which editors strip."""
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


@pytest.mark.parametrize("page", PAGES)
def test_page_blocks_run_and_print_what_the_page_says(
    page: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = DOCS / page
    if not path.exists():
        pytest.skip("docs not available (installed package, not a checkout)")
    blocks = _blocks(path.read_text())
    assert blocks, f"{page} has no python blocks"
    # A real module, so dataclasses and pickling resolve the page's classes.
    module = types.ModuleType(f"docs_{path.stem.replace('-', '_')}")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    # A page may turn on JAX's 64-bit mode, as a test suite does; undo it.
    jax = pytest.importorskip("jax")
    previous = jax.config.jax_enable_x64
    try:
        _run(page, blocks, module)
    finally:
        jax.config.update("jax_enable_x64", previous)


def _run(
    page: str, blocks: list[tuple[str, str | None]], module: types.ModuleType
) -> None:
    for i, (code, expected) in enumerate(blocks):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                exec(compile(code, f"{page}[block {i}]", "exec"), module.__dict__)
            except Exception as err:  # pragma: no cover - surfaced with context
                raise AssertionError(f"{page} block {i} failed:\n{code}") from err
        if expected is not None:
            printed = _lines(stdout.getvalue())
            assert printed == expected, f"{page} block {i} printed:\n{printed}"
