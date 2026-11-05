"""Read an operator's signature to see which optional keywords it takes."""

from __future__ import annotations

import inspect
from typing import Any

from pipekit import Lambda, Operator


def accepts_keyword(target: Any, name: str) -> bool:
    """Return whether calling ``target`` with ``name=`` reaches a parameter.

    Args:
        target: An operator or a plain callable. `Lambda` forwards every keyword,
            so for it whatever it wraps decides, however deeply nested.
        name: The keyword to look for.

    Returns:
        ``True`` if the signature has that parameter or ``**kwargs``. ``False``
        when the signature can't be read, as for some C functions.

    Examples:
        >>> accepts_keyword(lambda x, key=None: x, "key")
        True
        >>> accepts_keyword(Lambda(lambda x: x), "key")
        False
        >>> accepts_keyword(Lambda(Lambda(lambda x, key=None: x)), "key")
        True
    """
    if isinstance(target, Lambda):
        return accepts_keyword(target.fn, name)
    if isinstance(target, Operator) and type(target).__call__ is Operator.__call__:
        function = target._apply
    else:
        function = target
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        or (
            parameter.name == name
            and parameter.kind is not inspect.Parameter.POSITIONAL_ONLY
        )
        for parameter in parameters
    )
