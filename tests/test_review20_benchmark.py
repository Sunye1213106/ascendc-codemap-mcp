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
