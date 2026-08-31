# -*- coding: utf-8 -*-
"""A tier that the declaration does not fix is a fact, not a gap.

`MutexBuffer<bufferType, syncType> ping_` has a memory space, but only once
the enclosing class template is instantiated. The card dropped `memory_space`
whenever it was UNKNOWN, so "the index never worked it out" and "the source
genuinely does not say" rendered identically -- and a reader filled the
silence by reading the tier off the wrapped LocalTensor, which is UB for a
buffer that is usually L1.

The parameter is read off the enclosing `template <...>` header, not guessed
from argument position: the first argument of `LocalTensor<T>` is an element
type and the first argument of `MutexBuffer<bufferType, …>` is a tier.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ascendc_codemap_mcp.engine.query.bundle import _resource_identity
from ascendc_codemap_mcp.engine.query.sql import _enclosing_template_header
from ascendc_codemap_mcp.engine.semantics.ascendc_storage import (
    template_parameter_types,
    tier_template_parameter,
)
from ascendc_codemap_mcp.service.control import status
from ascendc_codemap_mcp.service.query import query

_CANDIDATES = (
    Path(r"d:\PR-review\TEST\ops-transformer\attention\flash_attention_score_grad"),
    Path(r"d:\TEST\ops-transformer\attention\flash_attention_score_grad"),
)
_REL_UO = Path(".ascendc-codemap/arch35/FlashAttentionScoreGrad.arch35.uo")
FAG = next((r for r in _CANDIDATES if (r / _REL_UO).is_file()), None)

_HEADER = "template <BufferType bufferType, SyncType syncType = SyncType::INNER_CORE_SYNC>"


def test_template_header_parses_into_declared_parameter_types() -> None:
    got = template_parameter_types(_HEADER)
    assert got == {"bufferType": "BufferType", "syncType": "SyncType"}


def test_type_parameters_carry_no_tier() -> None:
    assert template_parameter_types("template <typename T, class U>") == {}
    assert template_parameter_types("not a template") == {}


def test_the_tier_parameter_is_the_one_declared_with_a_tier_type() -> None:
    decl = "MutexBuffer<bufferType, syncType> ping_;"
    assert tier_template_parameter(decl, _HEADER) == ("bufferType", "BufferType")


def test_an_element_type_parameter_is_not_a_tier() -> None:
    """Position alone cannot tell a tier from an element type."""
    assert tier_template_parameter("LocalTensor<T> x;", "template <typename T>") is None


def test_a_non_tier_scalar_parameter_is_not_a_tier() -> None:
    assert tier_template_parameter("Foo<n> x;", "template <uint32_t n>") is None


def test_tposition_parameters_count_as_tiers() -> None:
    assert tier_template_parameter(
        "TQue<pos, 1> q;", "template <TPosition pos>"
    ) == ("pos", "TPosition")


def test_enclosing_template_header_walks_up_to_the_class() -> None:
    text = {
        73: _HEADER,
        74: "class MutexBuffersPolicyDB {",
        75: "public:",
        135: "private:",
        136: "    MutexBuffer<bufferType, syncType> ping_;",
    }
    assert _enclosing_template_header(text, 136) == _HEADER


def test_a_plain_class_has_no_template_header() -> None:
    text = {10: "class Plain {", 12: "    LocalTensor<half> t_;"}
    assert _enclosing_template_header(text, 12) == ""


def test_identity_states_the_parametric_tier_instead_of_dropping_it() -> None:
    data = {"memory_space": "UNKNOWN", "type_name": "MutexBuffer"}
    bare = dict(_resource_identity("BUFFER", data))
    assert "memory_space" not in bare

    named = dict(
        _resource_identity("BUFFER", data, parametric_tier=("bufferType", "BufferType"))
    )
    assert named["memory_space"] == "set by template parameter bufferType (BufferType)"


@pytest.mark.skipif(FAG is None, reason="FAG arch35 .uo missing")
@pytest.mark.parametrize(
    "kw",
    [
        {"symbol": "ping_"},
        {"file": "op_kernel/arch35/cube_api/mutex_buffers_policy.h", "line": 136},
    ],
)
def test_both_ways_of_asking_name_the_tier_parameter(kw: dict) -> None:
    status(project=str(FAG), architecture="arch35")
    payload = query(project=str(FAG), architecture="arch35", **kw)
    text = str((payload.get("data") or {}).get("text") or "")
    assert "set by template parameter bufferType (BufferType)" in text, text
