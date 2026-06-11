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


def test_hash_payload_digest_unchanged():
    """The shared hashing helpers reproduce the pre-refactor framing."""
    import hashlib

    from pipekit.hashing import stable_json
    from pipekit_experiment.registry import _hash_payload

    config = {"class": "Identity", "config": {}}
    h = hashlib.sha256()
    h.update(stable_json(config).encode("utf-8"))
    h.update(b"\x00")
    h.update(b"weights")
    assert _hash_payload(config, b"weights") == h.hexdigest()

    h2 = hashlib.sha256()
    h2.update(stable_json(config).encode("utf-8"))
    h2.update(b"\x00")
    assert _hash_payload(config, None) == h2.hexdigest()
