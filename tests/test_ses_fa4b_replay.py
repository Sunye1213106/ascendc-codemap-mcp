# -*- coding: utf-8 -*-
"""ses_fa4b fail-call replay against the same FAG arch35 .uo.

Inventory (session-ses_fa4b.md) — do not re-index. Each case records the
MCP inputs, what the session got wrong, and the expected predicates.
Acceptance: expected text after a status() warmup. Latency gate is P50 of
the fail set < 50ms (a single first-touch source window may sit a bit over).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ascendc_codemap_mcp.service.control import status
from ascendc_codemap_mcp.service.query import query
from tests.conftest import FAG_REL_UO, fag_operator_root

FAG = fag_operator_root()
FAG_UO = (FAG / FAG_REL_UO) if FAG else Path()

_HOST = "op_host/arch35/flash_attention_score_grad_tiling_common_regbase.cpp"
_NORMAL = "op_host/arch35/flash_attention_score_grad_tiling_normal_regbase.cpp"
_MAX_MS = 50.0

# Session footer times (contrast only; not the gate).
# S1 source 1650-1710 ~179ms, titled SetSparseParams / Called by ProcessOptionalInput
# S2 source 790-900 titled GetPlatformInfo
# T1 DoSparse→isInvalidCol 67ms, 1-hop READS as "the complete one"
# T2 SelectBlockSchedule→CalBandDeterIndex 208ms via isSplitByBlockIdx)
# T3 same pair relation=data, NO_PATH enumerated=1 (kinds smashed into one SQL token)


def _run(**kwargs) -> tuple[str, float, dict]:
    payload = query(project=str(FAG), architecture="arch35", **kwargs)
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    text = str((data or {}).get("text") or payload.get("text") or "")
    ms = float((data or {}).get("server_ms") or payload.get("server_ms") or 0.0)
    extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
    if ms <= 0:
        ms = float(extra.get("server_ms") or 0.0)
    return text, ms, data or {}


def _assert_not_session_regression(ms: float, label: str) -> None:
    """Per-call ceiling is loose; the fail-set P50 is the 50ms gate."""
    assert ms < 200.0, f"{label} server_ms={ms:.1f} (session-regression ceiling 200)"


@pytest.fixture()
def fag_warm() -> Path:
    if FAG is None or not FAG_UO.is_file():
        pytest.skip("FAG arch35 .uo missing")
    # Function-scoped: autouse _reset_runtime closes the query cache around
    # every test, so a module-scoped warmup never survives to the SLA query.
    status(project=str(FAG), architecture="arch35")
    # First source call builds the WRITES index for the snapshot; the SLA
    # is on subsequent queries, not that one-time decode.
    query(
        project=str(FAG),
        architecture="arch35",
        operation="source",
        file=_NORMAL,
        line=1077,
        line_end=1080,
    )
    return FAG


@pytest.mark.skipif(not FAG_UO.is_file(), reason="FAG arch35 .uo missing")
def test_s1_source_window_titles_set_split_axis(fag_warm: Path) -> None:
    """S1: line=1650 sits in SetSparseParams; the window bulk is SetSplitAxis."""
    text, ms, _ = _run(
        operation="source",
        file=_HOST,
        line=1650,
        line_end=1710,
    )
    head = text.splitlines()[0].strip() if text else ""
    assert head == "SetSplitAxis", text[:800]
    assert "ProcessOptionalInput" not in text
    assert "DoOpTiling" in text
    assert "SetSplitAxis" in text
    assert "isBn2MultiBlk" in text
    _assert_not_session_regression(ms, "S1")


@pytest.mark.skipif(not FAG_UO.is_file(), reason="FAG arch35 .uo missing")
def test_s2_source_window_is_not_get_platform_info(fag_warm: Path) -> None:
    """S2: line=790 is the tail of GetPlatformInfo; 790-900 is mostly IsCapable."""
    text, ms, _ = _run(
        operation="source",
        file=_NORMAL,
        line=790,
        line_end=900,
    )
    head = text.splitlines()[0].strip() if text else ""
    assert head != "GetPlatformInfo", text[:800]
    assert "IsCapable" in text
    _assert_not_session_regression(ms, "S2")


@pytest.mark.skipif(not FAG_UO.is_file(), reason="FAG arch35 .uo missing")
def test_t1_dosparse_to_isinvalidcol_family_menu(fag_warm: Path) -> None:
    """T1: not a single READS hop billed as the complete answer."""
    text, ms, data = _run(
        operation="trace",
        symbol="DoSparse",
        to_symbol="isInvalidCol",
    )
    lower = text.lower()
    assert "reads" in lower
    assert "isInvalidCol" in text
    assert "the complete one" not in lower
    assert "shortest path by hop count, not the only one" not in lower
    assert "call" in lower
    assert "data" in lower
    assert "isSplitByBlockIdx)" not in text
    names = [
        str(s.get("from_name") or s.get("to_name") or "")
        for s in (data.get("path") or [])
        if isinstance(s, dict)
    ]
    for name in names:
        leaf = name.replace(".", "::").split("::")[-1].strip()
        assert leaf.isidentifier() or not leaf, name
    _assert_not_session_regression(ms, "T1")


@pytest.mark.skipif(not FAG_UO.is_file(), reason="FAG arch35 .uo missing")
def test_t2_select_block_schedule_path_drops_artifact_node(fag_warm: Path) -> None:
    """T2: directed family menu; never route through isSplitByBlockIdx)."""
    text, ms, _ = _run(
        operation="trace",
        symbol="SelectBlockSchedule",
        to_symbol="CalBandDeterIndex",
    )
    assert "isSplitByBlockIdx)" not in text
    lower = text.lower()
    assert "call" in lower
    assert "data" in lower
    assert "control" in lower
    assert "compile" in lower
    assert "tiling 传输主线" not in text
    assert "the complete one" not in lower
    _assert_not_session_regression(ms, "T2")


@pytest.mark.skipif(not FAG_UO.is_file(), reason="FAG arch35 .uo missing")
def test_t3_relation_data_does_not_smash_kinds(fag_warm: Path) -> None:
    """T3: relation=data must walk real kinds, not one concatenated SQL token."""
    text, ms, data = _run(
        operation="trace",
        symbol="SelectBlockSchedule",
        to_symbol="CalBandDeterIndex",
        relation="data",
    )
    rels = [str(k) for k in (data.get("trace_relations") or [])]
    assert rels, text[:800]
    assert all("," not in k for k in rels), rels
    assert "isSplitByBlockIdx)" not in text
    if "enumerated (1 of them)" in text:
        raise AssertionError("false NO_PATH from smashed relation kinds:\n" + text[:800])
    _assert_not_session_regression(ms, "T3")


@pytest.mark.skipif(not FAG_UO.is_file(), reason="FAG arch35 .uo missing")
def test_failset_p50_under_50ms(fag_warm: Path) -> None:
    """P50 of the fail set after one warmup, not every first-touch source window."""
    times: list[float] = []
    _, ms, _ = _run(operation="source", file=_HOST, line=1650, line_end=1710)
    times.append(ms)
    _, ms, _ = _run(operation="source", file=_NORMAL, line=790, line_end=900)
    times.append(ms)
    _, ms, _ = _run(operation="trace", symbol="DoSparse", to_symbol="isInvalidCol")
    times.append(ms)
    _, ms, _ = _run(
        operation="trace",
        symbol="SelectBlockSchedule",
        to_symbol="CalBandDeterIndex",
    )
    times.append(ms)
    _, ms, _ = _run(
        operation="trace",
        symbol="SelectBlockSchedule",
        to_symbol="CalBandDeterIndex",
        relation="data",
    )
    times.append(ms)
    ordered = sorted(times)
    mid = ordered[len(ordered) // 2]
    assert mid < _MAX_MS, f"fail-set P50={mid:.1f} from {times}"
