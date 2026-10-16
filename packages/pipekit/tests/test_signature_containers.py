"""Containers report the signature of what they return."""

from __future__ import annotations

from pipekit import (
    AsyncMap,
    BatchedMap,
    Branch,
    Cache,
    Coalesce,
    Fanout,
    Identity,
    Operator,
    ProcessMap,
    Profile,
    Retry,
    Signature,
    Switch,
    ThreadMap,
    Try,
)
from pipekit.signature import common_signature


class DropTime(Operator):
    """Drops the time dimension from the signature."""

    def _apply(self, x):
        return x

    def compute_output_signature(self, signature):
        return signature.drop_dims(("time",))


CUBE = Signature(dims=(("time", 12), ("x", 4)), dtype="float32")
IMAGE = Signature(dims=(("x", 4),), dtype="float32")


def test_wrappers_report_their_inner_operator():
    assert Cache(DropTime()).compute_output_signature(CUBE) == IMAGE
    retried = Retry(DropTime(), attempts=2, base_delay=0.0, on=(KeyError,))
    assert retried.compute_output_signature(CUBE) == IMAGE
    assert Profile().wrap(DropTime()).compute_output_signature(CUBE) == IMAGE


def test_alternatives_report_a_signature_only_when_they_agree():
    agree = Try(primary=DropTime(), fallback=DropTime(), on=(KeyError,))
    differ = Try(primary=DropTime(), fallback=Identity(), on=(KeyError,))
    assert agree.compute_output_signature(CUBE) == IMAGE
    assert differ.compute_output_signature(CUBE) is None

    def always(_):
        return True

    branch = Branch(always, if_true=DropTime(), if_false=Identity())
    assert branch.compute_output_signature(CUBE) is None
    switch = Switch(key=always, cases={True: DropTime()}, default=DropTime())
    assert switch.compute_output_signature(CUBE) == IMAGE
    coalesce = Coalesce(sources=[DropTime(), DropTime()], is_ok=always)
    assert coalesce.compute_output_signature(CUBE) == IMAGE


def test_dict_and_list_outputs_report_no_signature():
    assert Fanout({"a": Identity()}).compute_output_signature(CUBE) is None
    for op in (
        ThreadMap(Identity()),
        ProcessMap(Identity()),
        AsyncMap(Identity()),
        BatchedMap(Identity(), batch_size=2),
    ):
        assert op.compute_output_signature(CUBE) is None


def test_common_signature():
    assert common_signature(IMAGE, IMAGE) == IMAGE
    assert common_signature(IMAGE, CUBE) is None
    assert common_signature() is None
