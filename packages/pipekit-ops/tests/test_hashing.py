"""Tests for the shared stable-hash primitives (`pipekit_ops.hashing`)."""

from __future__ import annotations

import hashlib

from pipekit_ops.hashing import sha256_hex, stable_json, stable_repr


def test_stable_json_sorts_keys() -> None:
    assert stable_json({"b": 1, "a": 2}) == stable_json({"a": 2, "b": 1})


def test_stable_json_falls_back_to_repr() -> None:
    class _Odd:
        def __repr__(self) -> str:
            return "<odd>"

    assert "<odd>" in stable_json({"x": _Odd()})


def test_stable_repr_handles_raising_repr() -> None:
    class _Broken:
        def __repr__(self) -> str:
            raise RuntimeError("no repr")

    assert stable_repr(_Broken()) == "<unrepr _Broken>"


def test_sha256_hex_mixes_str_and_bytes() -> None:
    expected = hashlib.sha256(b"abc\x00def").hexdigest()
    assert sha256_hex("abc", b"\x00", "def") == expected


def test_sha256_hex_framing_matters() -> None:
    assert sha256_hex("ab", "c") == sha256_hex("a", "bc")
    assert sha256_hex("ab", b"\x00", "c") != sha256_hex("a", b"\x00", "bc")


def test_cache_key_digest_framing() -> None:
    """Cache keys are ``sha256(repr(x) ‖ 0x00 ‖ pipeline_hash(inner))`` (P4)."""
    from pipekit_ops import Cache, Identity, pipeline_hash

    value, inner = [1, 2], Identity()
    h = hashlib.sha256()
    h.update(repr(value).encode("utf-8"))
    h.update(b"\x00")
    h.update(pipeline_hash(inner).encode("utf-8"))
    assert Cache(inner)._make_key(value) == h.hexdigest()
