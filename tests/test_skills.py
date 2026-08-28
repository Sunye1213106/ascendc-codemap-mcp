# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

from ascendc_codemap_mcp.install.skills import bundled_root, install_for


def test_bundled_skills_exist() -> None:
    root = bundled_root()
    assert (root / "index-operator" / "SKILL.md").is_file()
    assert (root / "update-operator" / "SKILL.md").is_file()
    assert (root / "query-codemap" / "SKILL.md").is_file()
    update = (root / "update-operator" / "SKILL.md").read_text(encoding="utf-8")
    assert "codemap_update" in update
    assert "/uo-update" not in update
    text = (root / "query-codemap" / "SKILL.md").read_text(encoding="utf-8")
    assert "pilot_cli" not in text
    assert "/uo-init" not in text
    assert "codemap_explore" in text
    assert "codemap_id" in text
    index = (root / "index-operator" / "SKILL.md").read_text(encoding="utf-8")
    assert "codemap_index" in index
    assert "cann-extract" in index
    assert "next_steps" in index


def test_install_skills_under_fake_home(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(Path, "home", lambda *args, **kwargs: tmp_path)
    out = install_for("Cursor")
    assert out["ok"] is True
    skill = tmp_path / ".cursor" / "skills" / "ascendc-codemap-query-codemap" / "SKILL.md"
    assert skill.is_file()
    body = skill.read_text(encoding="utf-8")
    assert "codemap_explore" in body
    assert "ascendc-codemap-mcp query" in body
