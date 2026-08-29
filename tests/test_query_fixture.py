# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

from ascendc_codemap_mcp.engine.store.reader import read_meta
from ascendc_codemap_mcp.service.control import status
from ascendc_codemap_mcp.service.identity import make_id, parse_id, snapshot_id
from ascendc_codemap_mcp.service.query import evidence, query
from ascendc_codemap_mcp.service import runtime


from tests.conftest import write_uo_fixture


def _write_fixture(op: Path, *, arch: str = "arch35") -> Path:
    return write_uo_fixture(op, arch=arch)


def test_codemap_id_parse(tmp_path: Path) -> None:
    assert parse_id("flash_attention_score_grad@arch35") == (
        "flash_attention_score_grad",
        "arch35",
    )
    assert parse_id("p:a91f42::flash_attention_score_grad@arch35") == (
        "flash_attention_score_grad",
        "arch35",
    )
    assert parse_id("p:a91f42/flash_attention_score_grad@arch35") == (
        "flash_attention_score_grad",
        "arch35",
    )
    assert parse_id("nope") is None
    assert make_id("toy_op", "arch35") == "toy_op@arch35"
    canonical = make_id("toy_op", "arch35", project=tmp_path)
    assert canonical.startswith("p:")
    assert canonical.endswith("::toy_op@arch35")


def test_status_reports_identity_and_freshness(tmp_path: Path, monkeypatch) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    product = _write_fixture(op)
    monkeypatch.setattr(
        "ascendc_codemap_mcp.engine.store.writer.detect_source_revision",
        lambda root: "abc123",
    )
    monkeypatch.setattr(
        "ascendc_codemap_mcp.service.freshness.inspect_git_changes",
        lambda *a, **k: {
            "git_ok": True,
            "rows": [],
            "worktree_dirty": False,
            "worktree_fingerprint": "",
            "head_sha": "abc123",
            "base_sha": "abc123",
        },
    )
    st = status(project=str(op), architecture="arch35")
    assert st["ok"] is True
    assert st["indexed"] is True
    assert st["freshness"] == "fresh"
    assert st["codemap"]["alias"] == "toy_op@arch35"
    assert st["codemap"]["id"].endswith("::toy_op@arch35")
    assert st["codemap"]["snapshot_id"].startswith("cm:")
    meta = read_meta(product)
    assert st["snapshot_id"] == snapshot_id(product, meta)


def test_status_stale_when_head_moved(tmp_path: Path, monkeypatch) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    _write_fixture(op)
    monkeypatch.setattr(
        "ascendc_codemap_mcp.engine.store.writer.detect_source_revision",
        lambda root: "fff999",
    )
    monkeypatch.setattr(
        "ascendc_codemap_mcp.service.freshness.inspect_git_changes",
        lambda *a, **k: {
            "git_ok": True,
            "rows": [("M", "op_host/tiling.cpp")],
            "worktree_dirty": False,
            "worktree_fingerprint": "",
            "head_sha": "fff999",
            "base_sha": "abc123",
        },
    )
    st = status(project=str(op), architecture="arch35")
    assert st["freshness"] == "stale"
    assert int(st["changed_files"] or 0) >= 1


def test_query_name_card_from_fixture(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    _write_fixture(op)
    st = status(project=str(op), architecture="arch35")
    assert st["indexed"] is True
    payload = query(project=str(op), architecture="arch35", symbol="IsPse")
    assert payload.get("ok") is True
    data = payload.get("data") or payload
    assert data.get("shape") in {"name", "index"} or (payload.get("data") or {}).get("cards") or payload.get("verdict")
    assert int((payload.get("data") or {}).get("count") or payload.get("count") or 0) >= 0
    assert payload.get("codemap", {}).get("alias") == "toy_op@arch35"
    assert payload.get("evidence")
    assert payload["evidence"][0]["id"].startswith("span:")
    assert "coverage" in payload
    typed = query(codemap_id="toy_op@arch35", symbol="IsPse")
    assert typed.get("ok") is True


def test_query_by_codemap_id_after_status(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    _write_fixture(op)
    status(project=str(op), architecture="arch35")
    payload = query(codemap_id="toy_op@arch35", symbol="IsPse")
    assert payload.get("ok") is True


def test_query_nl_symbol_is_invalid(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    _write_fixture(op)
    payload = query(
        project=str(op),
        architecture="arch35",
        symbol="who writes IsPse",
    )
    assert payload.get("ok") is False
    assert payload.get("error_code") == "INVALID_QUERY"
    assert payload.get("legal_filters") or (payload.get("data") or {}).get("legal_filters")


def test_query_find_without_kind_is_invalid(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    _write_fixture(op)
    payload = query(
        project=str(op),
        architecture="arch35",
        operation="find",
        callee="SyncAll",
    )
    assert payload.get("error_code") == "INVALID_QUERY"


def test_evidence_id_roundtrip(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    _write_fixture(op)
    card = query(project=str(op), architecture="arch35", symbol="IsPse")
    ev_id = card["evidence"][0]["id"]
    around = evidence(codemap_id="toy_op@arch35", evidence_id=ev_id)
    assert around.get("ok") is True
    data = around.get("data") or {}
    assert data.get("shape") == "around"


def test_query_requires_identity(monkeypatch) -> None:
    monkeypatch.delenv("ASCENDC_CODEMAP_PROJECT", raising=False)
    monkeypatch.delenv("ASCENDC_CODEMAP_ARCHITECTURE", raising=False)
    payload = query(symbol="IsPse")
    assert payload.get("ok") is False
    assert payload.get("error_code") in {
        "PROJECT_REQUIRED",
        "ARCHITECTURE_MISSING_IN_RUN_STATE",
        "CODEMAP_NOT_REGISTERED",
        "INVALID_CODEMAP_ID",
    }


def test_query_cache_is_bounded(tmp_path: Path) -> None:
    runtime.cache.close_all()
    runtime.cache.max_open = 2
    try:
        for i in range(3):
            op = tmp_path / f"op{i}"
            op.mkdir()
            _write_fixture(op)
            query(project=str(op), architecture="arch35", symbol="IsPse")
        stats = runtime.cache.stats()
        assert stats["cache_size"] <= 2
    finally:
        runtime.cache.max_open = 4
        runtime.cache.close_all()


def test_query_during_write_is_building(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    _write_fixture(op)
    st = status(project=str(op), architecture="arch35")
    with runtime.locks.write(st["codemap"]["id"]):
        payload = query(codemap_id="toy_op@arch35", symbol="IsPse")
    assert payload.get("state") == "building"
    assert (payload.get("codemap") or {}).get("freshness") == "building"
