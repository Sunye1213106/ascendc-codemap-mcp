# -*- coding: utf-8 -*-
"""C6 Review-20: dialect searches against FAG when a snapshot exists."""
from __future__ import annotations

from pathlib import Path

import pytest

from ascendc_codemap_mcp.service.control import status
from ascendc_codemap_mcp.service.query import query

FAG = Path(
    r"d:\TEST\ops-transformer\attention\flash_attention_score_grad"
)
FAG_UO = FAG / r".ascendc-codemap\arch35\FlashAttentionScoreGrad.arch35.uo"

# Dialect-first phrases. True names are only the useful-locator check.
REVIEW20: list[tuple[str, tuple[str, ...], str]] = [
    ("Q1 empty tensor", ("IsEmptyTensor", "emptyTensor"), "IsEmptyTensor"),
    ("Q2 TND zero seq", ("isSeqExistZero", "actual_seq_qlen"), "isSeqExistZero"),
    ("Q3 dropout", ("keepProb|dropMask", "IsDrop"), "dropMask"),
    ("Q4 s1Inner", ("s1Inner", "S1TemplateNum"), "s1Inner"),
    ("Q5 coreNum", ("GetCoreNumAic", "coreNum"), "GetCoreNumAic"),
    ("Q6 preSfmg", ("enablePreSfmg", "Presfmg"), "enablePreSfmg"),
    ("Q7 post tiling", ("DoPostTiling", "opPost"), "DoPostTiling"),
    ("Q8 sink", ("dsink", "hasSink"), "dsink"),
    ("Q9 NzOut", ("isNzOut|IsNzOut", "isExceedL2Cache"), "IsNzOut"),
    ("Q10 queryGm", ("queryGm", "DataCopy.*queryGm"), "queryGm"),
    ("Q11 L1 multi-buffer", ("4buff|3buff", "MutexBuffersPolicy"), "MutexBuffersPolicy"),
    ("Q12 double buffer", ("double buffer", "MutexBuffersPolicyDB"), "DB"),
    ("Q13 L1 buffers", ("L1",), "L1"),
    ("Q14 hard event", ("HardEvent", "SetFlag"), "HardEvent"),
    ("Q15 GM to UB", ("DataCopyPad", "DataCopy"), "DataCopy"),
    ("Q16 keepProb", ("keepProb",), "keepProb"),
    ("Q17 DataCopyPad", ("DataCopy(Pad)?",), "DataCopy"),
    ("Q18 L1 selector", ("QL1BuffSelector", "KL1BuffSelector"), "BuffSelector"),
    ("Q19 qL1Tensor", ("qL1Tensor",), "qL1Tensor"),
    ("Q20 SetFlag", ("SetFlag", "WaitFlag"), "SetFlag"),
]


def _search(name: str) -> str:
    payload = query(
        project=str(FAG),
        architecture="arch35",
        operation="search",
        name=name,
        limit=20,
    )
    return str((payload.get("data") or {}).get("text") or "")


@pytest.mark.skipif(not FAG_UO.is_file(), reason="FAG arch35 .uo missing")
def test_review20_searches_to_first_useful_locator() -> None:
    status(project=str(FAG), architecture="arch35")
    zero = 0
    over_budget = 0
    evidence_calls = 0
    rows: list[dict[str, object]] = []
    for qid, phrases, needle in REVIEW20:
        found_at = 0
        text = ""
        for i, phrase in enumerate(phrases, start=1):
            text = _search(phrase)
            if needle.lower() in text.lower() and "0 matches" not in text[:20]:
                found_at = i
                break
        if found_at == 0:
            zero += 1
        elif found_at > 2:
            over_budget += 1
        rows.append(
            {
                "id": qid,
                "searches": found_at or len(phrases),
                "hit": found_at > 0,
                "preview": text[:160],
            }
        )
    assert evidence_calls == 0
    assert zero <= 4, f"too many zero-hit Review-20 items: {rows}"
    assert over_budget <= 4, f"too many >2 search Review-20 items: {rows}"
    hit_rate = sum(1 for r in rows if r["hit"]) / len(rows)
    assert hit_rate >= 0.7


def _resolve(**kwargs) -> str:
    payload = query(project=str(FAG), architecture="arch35", operation="resolve", **kwargs)
    return str((payload.get("data") or {}).get("text") or "")


@pytest.mark.skipif(not FAG_UO.is_file(), reason="FAG arch35 .uo missing")
def test_review20_resolve_file_line_anchors_set_split_axis() -> None:
    status(project=str(FAG), architecture="arch35")
    text = _resolve(
        file="op_host/arch35/flash_attention_score_grad_tiling_common_regbase.cpp",
        line=1673,
    )
    assert "SetSplitAxis" in text
    assert "1673|" in text or "isBn2MultiBlk" in text
    assert "Used by" not in text
    assert "Controls" not in text


@pytest.mark.skipif(not FAG_UO.is_file(), reason="FAG arch35 .uo missing")
def test_review20_resolve_assignments_and_compiled() -> None:
    status(project=str(FAG), architecture="arch35")
    assigns = _resolve(symbol="isBn2MultiBlk")
    assert "Assignments" in assigns or "SetSplitAxis" in assigns
    compiled = _resolve(symbol="hasRope")
    assert "Compiled" in compiled
    assert "legal" in compiled.lower()


@pytest.mark.skipif(not FAG_UO.is_file(), reason="FAG arch35 .uo missing")
def test_review20_cal_deter_max_loop_num_span_is_stable() -> None:
    status(project=str(FAG), architecture="arch35")
    path = "op_kernel/arch35/flash_attention_score_grad_kernel_deter.h"
    spans = []
    texts = []
    for loc in (642, 668, 680, 702, 598):
        payload = query(
            project=str(FAG),
            architecture="arch35",
            operation="resolve",
            file=path,
            line=loc,
        )
        data = payload.get("data") or {}
        spans.append((int(data.get("unit_start") or 0), int(data.get("unit_end") or 0)))
        texts.append(str(data.get("text") or ""))
    assert len(set(spans)) == 1
    start, end = spans[0]
    assert start <= 598
    assert end >= 732
    for text in texts:
        assert "CalDeterMaxLoopNum" in text
        assert "598|" in text


@pytest.mark.skipif(not FAG_UO.is_file(), reason="FAG arch35 .uo missing")
def test_review20_shared_registry_header_is_readable() -> None:
    status(project=str(FAG), architecture="arch35")
    payload = query(
        project=str(FAG),
        architecture="arch35",
        operation="resolve",
        file="../common/op_host/fia_tiling_templates_registry.h",
        line=94,
    )
    data = payload.get("data") or {}
    text = str(data.get("text") or "")
    start = int(data.get("unit_start") or 0)
    end = int(data.get("unit_end") or 0)
    assert end > start
    assert text.count("|") >= 8
    assert "DoTilingImpl" in text or "FiaTiling" in text
