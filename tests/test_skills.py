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
    assert "update_operator" in update
    assert "/uo-update" not in update
    text = (root / "query-codemap" / "SKILL.md").read_text(encoding="utf-8")
    assert "pilot_cli" not in text
    assert "/uo-init" not in text
    assert "query_codemap" in text


def test_install_skills_under_fake_home(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(Path, "home", lambda *args, **kwargs: tmp_path)
    out = install_for("Cursor")
    assert out["ok"] is True
    skill = tmp_path / ".cursor" / "skills" / "ascendc-codemap-query-codemap" / "SKILL.md"
    assert skill.is_file()
    assert "query_codemap" in skill.read_text(encoding="utf-8")
