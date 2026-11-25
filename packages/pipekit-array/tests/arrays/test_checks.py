"""`pipekit_array.arrays`: checks that raise at once, or when compiled code runs."""

from __future__ import annotations

import importlib.util
import subprocess
import sys

import pytest


pytest.importorskip("array_api_compat")
strict = pytest.importorskip("array_api_strict")

import numpy as np
from pipekit_array.arrays import check_values, require_floating
from pipekit_ops import BackendError


def _missing(library: str) -> bool:
    return importlib.util.find_spec(library) is None


needs_jax = pytest.mark.skipif(_missing("jax"), reason="JAX is not installed")
needs_torch = pytest.mark.skipif(_missing("torch"), reason="torch is not installed")
needs_equinox = pytest.mark.skipif(
    _missing("equinox"), reason="equinox is not installed"
)
SECOND = strict.Device("device1")


class LibraryError(Exception):
    """The exception of a library."""


def positive(x, xp):
    return check_values(x, xp.any(x <= 0), "'x' must be positive")


def arrays(values):
    """The values as each kind of concrete array, with its namespace."""
    found = [
        (np.asarray(values), np),
        (strict.asarray(values), strict),
        (strict.asarray(values, device=SECOND), strict),
    ]
    if not _missing("jax"):
        import jax.numpy as jnp

        found.append((jnp.asarray(values), jnp))
    return found


def test_a_concrete_array_comes_back_as_it_is() -> None:
    for x, xp in arrays([1.0, 2.0]):
        assert positive(x, xp) is x


def test_a_concrete_array_raises_and_the_message_ends_with_it() -> None:
    for x, xp in arrays([1.0, -2.0]):
        with pytest.raises(ValueError, match=r"(?s)^'x' must be positive, got .*-2\."):
            positive(x, xp)


@needs_jax
def test_the_message_is_the_same_on_numpy_and_jax() -> None:
    import jax.numpy as jnp

    for x in (np.asarray([1.5, -2.0]), jnp.asarray([1.5, -2.0]), [1.5, -2.0]):
        with pytest.raises(ValueError) as caught:
            check_values(x, True, "no")
        assert str(caught.value).startswith("no, got array([ 1.5, -2. ]")


def test_a_library_raises_its_own_error() -> None:
    with pytest.raises(LibraryError, match="too small"):
        check_values(np.ones(2), True, "too small", error=LibraryError)
    with pytest.raises(LibraryError, match="floating"):
        require_floating(np.arange(2), "x", error=LibraryError)


@needs_equinox
def test_under_jit_the_compiled_code_raises() -> None:
    import jax
    import jax.numpy as jnp

    compiled = jax.jit(lambda x: jnp.log(positive(x, jnp)))
    np.testing.assert_allclose(compiled(jnp.ones(2)), np.zeros(2))
    with pytest.raises(Exception, match="'x' must be positive"):
        compiled(jnp.asarray([1.0, -1.0]))


@needs_equinox
def test_under_jit_a_traced_condition_on_a_numpy_array() -> None:
    import jax
    import jax.numpy as jnp

    table = np.ones(2)
    compiled = jax.jit(lambda k: check_values(table, k < 0, "'k' must be >= 0") * k)
    np.testing.assert_array_equal(compiled(jnp.asarray(2)), [2.0, 2.0])
    with pytest.raises(Exception, match="'k' must be >= 0"):
        compiled(jnp.asarray(-2))


@needs_jax
def test_a_trace_without_equinox_names_it(monkeypatch) -> None:
    import jax
    import jax.numpy as jnp

    monkeypatch.setitem(sys.modules, "equinox", None)
    with pytest.raises(ImportError, match="pip install 'equinox'"):
        jax.jit(lambda x: positive(x, jnp))(jnp.ones(2))


def test_a_concrete_check_imports_neither_jax_nor_equinox() -> None:
    code = (
        "import sys, numpy as np\n"
        "from pipekit_array.arrays import check_values, require_floating\n"
        "check_values(np.ones(2), False, ''), require_floating(np.ones(2), 'x')\n"
        "assert not {'jax', 'equinox', 'torch'} & set(sys.modules)\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_require_floating_takes_real_floating_arrays() -> None:
    for x, _ in arrays([1.0, 2.0]):
        require_floating(x, "x")
    require_floating(1.0, "x")
    require_floating(np.ones(2, dtype=np.float32), "x")


@pytest.mark.parametrize("dtype", ["int64", "bool", "complex128"])
def test_require_floating_names_the_argument_and_the_dtype(dtype) -> None:
    for xp in (np, strict):
        x = xp.zeros(2, dtype=getattr(xp, dtype))
        with pytest.raises(ValueError, match=rf"'x' must have a floating .*{dtype}"):
            require_floating(x, "x")


@needs_jax
def test_require_floating_under_a_trace() -> None:
    import jax
    import jax.numpy as jnp

    with pytest.raises(ValueError, match="got int32"):
        jax.jit(lambda x: require_floating(x, "x"))(jnp.zeros(2, dtype=jnp.int32))


def test_require_floating_refuses_what_is_no_array() -> None:
    with pytest.raises(BackendError):
        require_floating(object(), "x")


@needs_torch
def test_a_tensor_that_needs_a_gradient() -> None:
    import torch

    x = torch.tensor([1.0, -2.0], requires_grad=True)
    require_floating(x, "x")
    with pytest.raises(ValueError, match=r"got array\(\[ 1\., -2\.\]"):
        positive(x, torch)
