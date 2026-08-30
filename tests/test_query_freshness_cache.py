# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

from ascendc_codemap_mcp.service.freshness import compute
from tests.conftest import write_uo_fixture


def test_query_freshness_does_not_hash_the_worktree(monkeypatch, tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    write_uo_fixture(op)

    monkeypatch.setattr(
        "ascendc_codemap_mcp.service.freshness.probe_operator_git",
        lambda _project: {
            "git_ok": True,
            "head": "abc123",
            "dirty": False,
            "changed_files": 0,
        },
    )
    info = compute(op, meta={"source_revision": "abc123", "schema": "codemap-uo/v3"})
    assert info["freshness"] == "fresh"
    assert info["dirty"] is False


def test_freshness_probe_is_reused_within_ttl(monkeypatch, tmp_path: Path) -> None:
    from ascendc_codemap_mcp.service import freshness as freshness_mod

    op = tmp_path / "toy_op"
    op.mkdir()
    write_uo_fixture(op)
    freshness_mod.reset_probe_cache()
    n = {"calls": 0}

    def fake_probe(_project: Path) -> dict:
        n["calls"] += 1
        return {"git_ok": True, "head": "abc123", "dirty": False, "changed_files": 0}

    monkeypatch.setattr(freshness_mod, "_probe_operator_git_uncached", fake_probe)
    compute(op, meta={"source_revision": "abc123", "schema": "codemap-uo/v3"})
    compute(op, meta={"source_revision": "abc123", "schema": "codemap-uo/v3"})
    assert n["calls"] == 1
