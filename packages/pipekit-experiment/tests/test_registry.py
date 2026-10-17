"""Tests for `pipekit_experiment.registry`."""

from __future__ import annotations

import pytest
from pipekit import Const, Identity, Operator
from pipekit_experiment import LocalModelRegistry, ModelRegistry


def test_local_registry_creates_root(tmp_path):
    LocalModelRegistry(tmp_path / "models")
    assert (tmp_path / "models").exists()
    assert (tmp_path / "models" / "_tags").exists()


def test_local_registry_store_and_load_round_trip(tmp_path):
    reg = LocalModelRegistry(tmp_path)
    h = reg.store(Const(42))
    loaded = reg.load(h)
    assert isinstance(loaded, Const)
    assert loaded() == 42


def test_local_registry_store_with_weights(tmp_path):
    reg = LocalModelRegistry(tmp_path)
    h = reg.store(Identity(), weights=b"\x01\x02\x03")
    assert reg.load_weights(h) == b"\x01\x02\x03"


def test_local_registry_load_weights_returns_none_when_absent(tmp_path):
    reg = LocalModelRegistry(tmp_path)
    h = reg.store(Identity())
    assert reg.load_weights(h) is None


def test_local_registry_hash_is_stable_for_identical_models(tmp_path):
    reg = LocalModelRegistry(tmp_path)
    h1 = reg.store(Const(1))
    # Re-storing the same op produces the same hash (idempotent).
    h2 = reg.store(Const(1))
    assert h1 == h2


def test_local_registry_hash_differs_for_different_configs(tmp_path):
    reg = LocalModelRegistry(tmp_path)
    assert reg.store(Const(1)) != reg.store(Const(2))


def test_local_registry_weights_affect_hash(tmp_path):
    reg = LocalModelRegistry(tmp_path)
    h1 = reg.store(Const(1), weights=b"\x00")
    h2 = reg.store(Const(1), weights=b"\x01")
    assert h1 != h2


def test_local_registry_tag_resolution(tmp_path):
    reg = LocalModelRegistry(tmp_path)
    h = reg.store(Const(1), name="v1")
    assert reg.resolve_tag("v1") == h
    assert reg.load("v1")() == 1


def test_local_registry_tag_force_required_to_overwrite(tmp_path):
    reg = LocalModelRegistry(tmp_path)
    h1 = reg.store(Const(1), name="prod")
    h2 = reg.store(Const(2))
    with pytest.raises(FileExistsError):
        reg.tag(h2, "prod")
    reg.tag(h2, "prod", force=True)
    assert reg.resolve_tag("prod") == h2
    del h1


def test_local_registry_load_unknown_raises(tmp_path):
    reg = LocalModelRegistry(tmp_path)
    with pytest.raises(KeyError):
        reg.load("nonexistent")


def test_local_registry_list_by_tag(tmp_path):
    reg = LocalModelRegistry(tmp_path)
    h_alpha = reg.store(Const(1), tags={"family": "alpha"})
    h_beta = reg.store(Const(2), tags={"family": "beta"})
    alphas = reg.list(tags={"family": "alpha"})
    assert alphas == [h_alpha]
    assert h_beta not in alphas


def test_local_registry_store_rejects_non_operator(tmp_path):
    reg = LocalModelRegistry(tmp_path)
    with pytest.raises(TypeError):
        reg.store(object())  # type: ignore[arg-type]


def test_local_registry_satisfies_protocol(tmp_path):
    reg = LocalModelRegistry(tmp_path)
    assert isinstance(reg, ModelRegistry)


def test_local_registry_tag_unknown_hash_raises(tmp_path):
    reg = LocalModelRegistry(tmp_path)
    with pytest.raises(KeyError):
        reg.tag("0" * 64, "tag")


def test_local_registry_round_trips_operator_state(tmp_path):
    """Loaded operator should behave identically to the original."""
    reg = LocalModelRegistry(tmp_path)
    original: Operator = Const(99)
    h = reg.store(original)
    loaded = reg.load(h)
    assert loaded() == original()


def test_local_registry_rejects_path_traversal_refs(tmp_path):
    """Refs and tag names must not escape the registry root."""
    reg = LocalModelRegistry(tmp_path / "models")
    secret = tmp_path / "secret"
    secret.mkdir()
    (secret / "operator.json").write_text('{"class": "Const", "config": {"value": 1}}')
    h = reg.store(Const(1))
    for bad in ("../secret", "..", "a/b", "a\\b", "", "_tags"):
        with pytest.raises(ValueError):
            reg.load(bad)
        with pytest.raises(ValueError):
            reg.tag(h, bad)
        with pytest.raises(ValueError):
            reg.resolve_tag(bad)


def test_local_registry_restore_merges_tags_instead_of_wiping(tmp_path):
    """Re-storing identical content without tags must keep earlier tags."""
    reg = LocalModelRegistry(tmp_path)
    h = reg.store(Const(1), tags={"family": "alpha"})
    h2 = reg.store(Const(1))  # same content, no tags kwarg
    assert h == h2
    assert reg.list(tags={"family": "alpha"}) == [h]
    # New tags merge with (not replace) the old ones.
    reg.store(Const(1), tags={"stage": "prod"})
    assert reg.list(tags={"family": "alpha", "stage": "prod"}) == [h]


def test_local_registry_empty_weights_normalized_to_absent(tmp_path):
    """b'' and None hash identically, so they must store identically too."""
    reg = LocalModelRegistry(tmp_path)
    h_none = reg.store(Const(1))
    h_empty = reg.store(Const(1), weights=b"")
    assert h_none == h_empty
    assert reg.load_weights(h_empty) is None


def test_local_registry_writes_leave_no_tmp_files(tmp_path):
    reg = LocalModelRegistry(tmp_path)
    reg.store(Const(1), name="v1", tags={"a": 1})
    leftovers = [p for p in tmp_path.rglob("*.tmp")]
    assert leftovers == []


def test_s3_registry_functional_round_trip_via_memory_fs():
    """Exercise the fsspec backend end-to-end on the in-memory filesystem."""
    pytest.importorskip("fsspec")
    from fsspec.implementations.memory import MemoryFileSystem
    from pipekit_experiment import S3ModelRegistry

    MemoryFileSystem.store.clear()
    reg = S3ModelRegistry("memory://registry/")
    h = reg.store(Const(7), name="prod", tags={"family": "alpha"})
    assert reg.load(h)() == 7
    assert reg.load("prod")() == 7
    assert reg.resolve_tag("prod") == h
    assert reg.load_weights(h) is None
    assert reg.list(tags={"family": "alpha"}) == [h]
    with pytest.raises(FileExistsError):
        reg.tag(h, "prod")
    with pytest.raises(ValueError):
        reg.load("../escape")
    # Re-store without tags keeps the earlier tag metadata (parity with local).
    reg.store(Const(7))
    assert reg.list(tags={"family": "alpha"}) == [h]


def test_s3_registry_weights_round_trip_via_memory_fs():
    pytest.importorskip("fsspec")
    from fsspec.implementations.memory import MemoryFileSystem
    from pipekit_experiment import S3ModelRegistry

    MemoryFileSystem.store.clear()
    reg = S3ModelRegistry("memory://registry-weights/")
    h = reg.store(Identity(), weights=b"\x01\x02")
    assert reg.load_weights(h) == b"\x01\x02"
    assert reg.load_weights(reg.store(Identity())) is None


def test_s3_registry_import_error_without_fsspec(monkeypatch):
    """S3ModelRegistry constructor raises a clean error when fsspec absent."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "fsspec":
            raise ImportError("simulated absence")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    from pipekit_experiment import S3ModelRegistry

    with pytest.raises(ImportError, match="fsspec"):
        S3ModelRegistry("s3://x/")


def test_hash_framing_unchanged():
    """`pipeline_hash` reproduces the registry's original framing exactly."""
    import hashlib

    from pipekit import Identity
    from pipekit.hashing import pipeline_hash, stable_json

    op = Identity()
    h = hashlib.sha256()
    h.update(stable_json(op.state).encode("utf-8"))
    h.update(b"\x00")
    h.update(b"weights")
    assert pipeline_hash(op, b"weights") == h.hexdigest()

    h2 = hashlib.sha256()
    h2.update(stable_json(op.state).encode("utf-8"))
    h2.update(b"\x00")
    assert pipeline_hash(op) == h2.hexdigest()


def test_store_returns_pipeline_hash(tmp_path):
    """What the registry stores under is what `pipeline_hash` computes."""
    from pipekit import Identity
    from pipekit.hashing import pipeline_hash

    registry = LocalModelRegistry(tmp_path)
    op = Identity() | Identity()
    assert registry.store(op) == pipeline_hash(op)
    assert registry.store(op, weights=b"w") == pipeline_hash(op, b"w")


# ---------------------------------------------------------------------
# Gate E1 — registry round-trip: a loaded operator has the same content
# hash and bitwise-equal outputs; tampered content fails to load.
# ---------------------------------------------------------------------


@pytest.fixture(params=["local", "s3"])
def registry_and_writer(request, tmp_path):
    """A registry of each backend, and a raw writer into its storage."""
    if request.param == "local":
        reg = LocalModelRegistry(tmp_path / "models")

        def write_raw(data: bytes, *parts: str) -> None:
            reg.root.joinpath(*parts).write_bytes(data)

        return reg, write_raw
    fsspec = pytest.importorskip("fsspec")
    from fsspec.implementations.memory import MemoryFileSystem
    from pipekit_experiment import S3ModelRegistry

    MemoryFileSystem.store.clear()
    reg = S3ModelRegistry("memory://registry-e1/")

    def write_raw(data: bytes, *parts: str) -> None:
        with fsspec.open("memory://registry-e1/" + "/".join(parts), "wb") as f:
            f.write(data)

    return reg, write_raw


def _sequential() -> Operator:
    pytest.importorskip("numpy")
    pytest.importorskip("pipekit_array")
    from pipekit import Cache, Fanout
    from pipekit.qc import AssertShape
    from pipekit_array import MeanScalar

    return (
        Identity()
        | AssertShape((4, 3))
        | Cache(MeanScalar(axis=0, keepdims=True))
        | Fanout({"mean": MeanScalar(), "columns": Identity()})
    )


def _graph() -> Operator:
    pytest.importorskip("numpy")
    pytest.importorskip("pipekit_array")
    from pipekit import Graph, Input
    from pipekit.qc import AssertShape
    from pipekit_array import MeanScalar

    x = Input("x")
    checked = AssertShape((4, 3))(x)
    columns = MeanScalar(axis=0, keepdims=True)(checked)
    overall = MeanScalar()(columns)
    return Graph(inputs={"x": x}, outputs={"columns": columns, "mean": overall})


def _assert_bitwise_equal(actual, expected) -> None:
    import numpy as np

    assert actual.keys() == expected.keys()
    for key in expected:
        a, e = np.asarray(actual[key]), np.asarray(expected[key])
        assert a.dtype == e.dtype and a.shape == e.shape, key
        assert a.tobytes() == e.tobytes(), key


@pytest.mark.parametrize("build", [_sequential, _graph], ids=["sequential", "graph"])
@pytest.mark.parametrize("weights", [None, b"\x00\x01opaque"], ids=["bare", "weights"])
def test_registry_round_trip_keeps_hash_and_outputs(
    registry_and_writer, build, weights
):
    import numpy as np
    from pipekit.hashing import pipeline_hash

    reg, _ = registry_and_writer
    op = build()
    x = np.arange(12, dtype=np.float64).reshape(4, 3) / 7.0

    h = reg.store(op, name="e1", weights=weights)
    for ref in (h, "e1"):
        loaded = reg.load(ref)
        assert pipeline_hash(loaded, reg.load_weights(ref)) == h
        _assert_bitwise_equal(loaded(x), op(x))


@pytest.mark.parametrize("target", ["operator.json", "weights.bin"])
def test_registry_load_rejects_tampered_content(registry_and_writer, target):
    from pipekit_experiment import HashMismatchError

    reg, write_raw = registry_and_writer
    h = reg.store(_sequential(), name="e1", weights=b"weights")
    if target == "operator.json":
        # A valid state for a different pipeline: it would load silently.
        tampered = _sequential()
        tampered.operators[1] = type(tampered.operators[1])((5, 3))
        from pipekit.hashing import stable_json

        write_raw(stable_json(tampered.state).encode(), h, target)
    else:
        write_raw(b"other weights", h, target)

    for ref in (h, "e1"):
        with pytest.raises(HashMismatchError, match="corrupted"):
            reg.load(ref)
        with pytest.raises(HashMismatchError, match="corrupted"):
            reg.load_weights(ref)


class _Drifting(Operator):
    """An operator whose state does not round-trip (``n`` grows on rebuild)."""

    def __init__(self, n: int = 0) -> None:
        self.n = n + 1

    def _apply(self, x):
        return x


def test_registry_load_rejects_operator_that_rebuilds_to_another_hash(
    registry_and_writer,
):
    from pipekit_experiment import HashMismatchError

    reg, _ = registry_and_writer
    h = reg.store(_Drifting())
    # The stored bytes are intact, so the weights still load ...
    assert reg.load_weights(h) is None
    # ... but the rebuilt operator is not the one that was stored.
    with pytest.raises(HashMismatchError, match="round-trip"):
        reg.load(h)


def test_hash_mismatch_error_is_a_value_error():
    from pipekit_experiment import HashMismatchError

    assert issubclass(HashMismatchError, ValueError)
