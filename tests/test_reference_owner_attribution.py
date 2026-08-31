# -*- coding: utf-8 -*-
"""A reference to another class's same-named member has to say so.

`resolve ProcessSink` seeds on `S1S2PostRegbase::ProcessSink` and listed one
reference: `nz_post.h:249`. That line is `ProcessSink()` inside
`FlashAttentionScoreGradNzPost::Process`, calling NzPost's own version, and 254
is NzPost's definition of it. Read plainly, the card claimed a link between two
unrelated templates.

Dropping such rows is the wrong trade. `KernelBase::Init` calling
`preSfmg.Init()` sits in a class that also declares `Init` and is a real
reference; losing a true caller undoes the completeness the card just promised.
So they stay, attributed.
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


def _references(symbol: str) -> list[str]:
    status(project=str(FAG), architecture="arch35")
    text = str((query(project=str(FAG), architecture="arch35", symbol=symbol).get("data") or {}).get("text") or "")
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("**References**"):
            return [
                row[2:]
                for row in lines[index + 1 :]
                if row.startswith("- ")
            ]
    return []


def test_a_reference_inside_another_owners_version_is_attributed() -> None:
    rows = [r for r in _references("ProcessSink") if "nz_post.h" in r]
    assert rows, "the nz_post reference disappeared"
    assert "FlashAttentionScoreGradNzPost" in rows[0], rows[0]
    assert "declares its own ProcessSink" in rows[0], rows[0]


def test_attribution_does_not_drop_the_rows() -> None:
    """A common leaf name must not shrink the list; every caller still shows."""
    rows = _references("Init")
    assert len(rows) >= 6, rows
    assert any("presfmg_regbase.h" in r for r in rows), rows


def test_a_reference_in_a_class_without_its_own_version_carries_no_note() -> None:
    rows = [r for r in _references("Init") if "presfmg_regbase.h:39" in r]
    assert rows, "expected the presfmg reference"
    assert "declares its own" not in rows[0], rows[0]


def test_an_out_of_class_definition_still_names_its_class() -> None:
    """Class bodies do not span out-of-line member definitions."""
    from ascendc_codemap_mcp.engine.query.sql import UoSqlQuery

    engine = UoSqlQuery(str(FAG / _REL_UO))
    nz = "op_kernel/arch35/flash_attention_score_grad_nz_post.h"
    assert engine.owner_of(nz, 249) == "FlashAttentionScoreGradNzPost"
    assert engine.owner_of(nz, 255) == "FlashAttentionScoreGradNzPost"
