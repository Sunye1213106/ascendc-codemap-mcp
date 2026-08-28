# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

from ascendc_codemap_mcp.tools import doctor, status


def test_doctor_fail_closed_without_cann(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "ascendc_codemap_mcp.engine.paths.cann_root",
        lambda explicit=None: None,
    )
    monkeypatch.setattr(
        "ascendc_codemap_mcp.engine.paths.require_cann_ready",
        lambda explicit=None: (None, ["CANN root not found"]),
    )
    monkeypatch.setattr(
        "ascendc_codemap_mcp.engine.paths.explain",
        lambda: "cann_root: NOT FOUND",
    )
    op = tmp_path / "missing_op"
    payload = doctor(project=str(op), architecture="arch35")
    assert payload["ok"] is False
    assert any("CANN" in i or "not found" in i.lower() for i in payload["issues"])
    joined = "\n".join(payload["next_steps"])
    assert "hiascend.com" in joined
    assert "cann-extract" in joined
    assert "linux-x86_64" in joined


def test_missing_cann_layout_mentions_download(monkeypatch) -> None:
    from ascendc_codemap_mcp.engine import paths

    monkeypatch.setattr(paths, "cann_root", lambda explicit=None: None)
    text = "\n".join(paths.cann_layout_issues(None))
    assert "cann-extract" in text
    assert "hiascend.com" in text
    assert "_cann" in text


def test_doctor_requires_architecture(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "ascendc_codemap_mcp.engine.paths.require_cann_ready",
        lambda explicit=None: (None, []),
    )
    monkeypatch.setattr("ascendc_codemap_mcp.engine.paths.explain", lambda: "")
    op = tmp_path / "op"
    op.mkdir()
    payload = doctor(project=str(op), architecture="")
    assert payload["ok"] is False
    assert any("architecture" in i.lower() for i in payload["issues"])


def test_status_unindexed(tmp_path: Path) -> None:
    op = tmp_path / "op"
    op.mkdir()
    payload = status(project=str(op), architecture="arch35")
    assert payload["ok"] is True
    assert payload["indexed"] is False
