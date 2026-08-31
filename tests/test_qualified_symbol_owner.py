# -*- coding: utf-8 -*-
"""`A::x` has to answer about `A`, not about whichever `x` sorts first.

Members are indexed under their leaf name, so two classes declaring the same
member were one entity to a caller. `BuffersPolicyDB::ping_` and
`MutexBuffersPolicyDB::ping_` both returned the mutex one, and the only way to
tell them apart was to already know the line -- which is what resolve is for.

A qualifier that matches nothing is the other half: silently answering about a
different owner is worse than saying the owner has no such member.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ascendc_codemap_mcp.service.control import status
from ascendc_codemap_mcp.service.query import query

_CANDIDATES = (
    Path(r"d:\PR-review\TEST\ops-transformer\attention\flash_attention_score_grad"),
    Path(r"d:\TEST\ops-transformer\attention\flash_attention_score_grad"),
)
_REL_UO = Path(".ascendc-codemap/arch35/FlashAttentionScoreGrad.arch35.uo")


def _operator() -> Path | None:
    for root in _CANDIDATES:
        if (root / _REL_UO).is_file():
            return root
    return None


FAG = _operator()

pytestmark = pytest.mark.skipif(FAG is None, reason="FAG arch35 .uo missing")


def _head(symbol: str) -> str:
    status(project=str(FAG), architecture="arch35")
    payload = query(project=str(FAG), architecture="arch35", symbol=symbol)
    text = str((payload.get("data") or {}).get("text") or "")
    return "\n".join([ln for ln in text.splitlines() if ln.strip()][:6])


@pytest.mark.parametrize(
    ("symbol", "want_file"),
    [
        ("BuffersPolicyDB::ping_", "common/op_kernel/buffers_policy.h"),
        ("MutexBuffersPolicyDB::ping_", "cube_api/mutex_buffers_policy.h"),
        ("BuffersPolicySingleBuffer::buffer_", "common/op_kernel/buffers_policy.h"),
        ("MutexBuffersPolicySingleBuffer::buffer_", "cube_api/mutex_buffers_policy.h"),
        ("BuffersPolicy3buff::a_", "common/op_kernel/buffers_policy.h"),
        ("MutexBuffersPolicy3buff::a_", "cube_api/mutex_buffers_policy.h"),
    ],
)
def test_a_qualified_member_resolves_inside_its_own_class(symbol: str, want_file: str) -> None:
    head = _head(symbol)
    assert want_file in head, f"{symbol} landed elsewhere:\n{head}"


def test_an_owner_with_no_such_member_says_so() -> None:
    head = _head("Nonexistent::ping_")
    assert "No member of `Nonexistent`" in head, head


@pytest.mark.parametrize(
    ("symbol", "want_file"),
    [
        ("ReduceSinkVF::vregSrc", "vf_post_reduce_sink.h"),
        ("DsAbsReduceMaxVF64::pregFullExe", "vf_ds_abs_reduce_max.h"),
        ("Transdata::pregFullExe", "vf_transdata.h"),
    ],
)
def test_a_function_qualifies_its_own_registers(symbol: str, want_file: str) -> None:
    """A register declared in a VF function has no class to be a member of.

    A class-only owner lookup answered `ReduceSinkVF::vregSrc` with another
    function's register, and then reported that `ReduceSinkVF` has no `vregSrc`
    -- about a register it declares on its own second line.
    """
    head = _head(symbol)
    assert want_file in head, f"{symbol} landed elsewhere:\n{head}"
    assert "No member of" not in head, head


@pytest.mark.parametrize(
    ("symbol", "scopes"),
    [
        ("ping_", ("MutexBuffersPolicyDB", "BuffersPolicyDB")),
        ("buffer_", ("MutexBuffersPolicySingleBuffer", "BuffersPolicySingleBuffer")),
        ("vregSrc", ("BroadcastSubMulVF64",)),
    ],
)
def test_a_name_shared_by_several_scopes_says_which_one_it_answered(
    symbol: str, scopes: tuple[str, ...]
) -> None:
    head = _head(symbol)
    assert "is declared in" in head, head
    assert "This card is" in head, head
    for scope in scopes:
        assert scope in head, f"{scope} missing from:\n{head}"


@pytest.mark.parametrize(
    "symbol",
    [
        "BuffersPolicyDB",      # a class is not declared inside itself
        "Buffer",
        "deterMaxRound",        # a tiling field's identity spans layers, not owners
        "isBn2MultiBlk",
        "ReduceSinkVF",
        "BuffersPolicyDB::ping_",  # already qualified; nothing to disambiguate
    ],
)
def test_an_unambiguous_name_is_not_told_it_is_ambiguous(symbol: str) -> None:
    assert "is declared in" not in _head(symbol), symbol


def test_a_name_that_matches_nothing_is_not_dressed_up_as_a_hit() -> None:
    """`MutexBuffersPolicy` names no entity; the substring fallback found a field."""
    head = _head("MutexBuffersPolicy")
    assert "Nothing is named `MutexBuffersPolicy`" in head, head
    assert "search pattern=MutexBuffersPolicy" in head, head


def test_owner_is_recovered_from_the_enclosing_type_when_scope_is_blank() -> None:
    """The build leaves `scope` empty for whole headers; the TYPE span still knows."""
    from ascendc_codemap_mcp.engine.query.sql import UoSqlQuery

    engine = UoSqlQuery(str(FAG / _REL_UO))
    mutex = "op_kernel/arch35/cube_api/mutex_buffers_policy.h"
    assert engine.owner_of(mutex, 136) == "MutexBuffersPolicyDB"
    assert engine.owner_of(mutex, 70) == "MutexBuffersPolicySingleBuffer"
    assert engine.owner_of(mutex, 1) == ""
