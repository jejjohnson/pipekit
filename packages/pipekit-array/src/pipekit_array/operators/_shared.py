"""What the operators share: dtypes of reductions, signatures, lists of names."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pipekit_ops import Signature


#: reductions that keep the input's dtype whatever it is
#: reductions whose output dtype is always the input's
_DTYPE_PRESERVING = frozenset({"max", "min", "nanmax", "nanmin"})


#: reductions that keep a float input's dtype (integers and bools promote)
_FLOAT_PRESERVING = frozenset(
    {
        "sum",
        "prod",
        "nansum",
        "nanprod",
        "mean",
        "nanmean",
        "median",
        "nanmedian",
        "std",
        "nanstd",
        "var",
        "nanvar",
    }
)


#: reductions that return booleans
_BOOLEAN = frozenset({"all", "any"})


def _reduced_dtype(fn: str, dtype: str | None) -> str | None:
    """The dtype a reduction returns, when it is known without running it."""
    if dtype is None:
        return None
    if fn in _DTYPE_PRESERVING:
        return dtype
    if fn in _BOOLEAN:
        return "bool"
    if fn in _FLOAT_PRESERVING and "float" in dtype:
        return dtype
    return None


def _signature(value: Any) -> Signature | None:
    return value if isinstance(value, Signature) else None


def _names(dim: str | Sequence[str]) -> str | tuple[str, ...]:
    """One dim name as is, several as a tuple."""
    return dim if isinstance(dim, str) else tuple(dim)


def _names_config(dim: str | tuple[str, ...]) -> str | list[str]:
    return dim if isinstance(dim, str) else list(dim)
