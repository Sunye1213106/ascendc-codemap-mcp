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
    assert "codemap_status" not in update
    text = (root / "query-codemap" / "SKILL.md").read_text(encoding="utf-8")
    assert "pilot_cli" not in text
    assert "/uo-init" not in text
    assert "codemap_query" in text
    assert "codemap_id" in text
    assert "dim_names" in text or "**Dims**" in text
    assert "UNKNOWN" in text
    assert "legal_key_count" in text or "operation" in text
    assert "InitBuffer" in text
    assert "codemap_explore" not in text
    assert "overview" not in text or "Do not call overview" in text
    index = (root / "index-operator" / "SKILL.md").read_text(encoding="utf-8")
    assert "codemap_index" in index
    assert "cann-extract" in index
    assert "next_steps" in index
    assert "codemap_status" not in index


def test_install_skills_under_fake_home(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(Path, "home", lambda *args, **kwargs: tmp_path)
    out = install_for("Cursor")
    assert out["ok"] is True
    skill = tmp_path / ".cursor" / "skills" / "ascendc-codemap-query-codemap" / "SKILL.md"
    assert skill.is_file()
    alias = tmp_path / ".cursor" / "skills" / "query-codemap" / "SKILL.md"
    assert alias.is_file()
    body = skill.read_text(encoding="utf-8")
    assert "codemap_query" in body
    assert "codemap_explore" not in body
    assert "ascendc-codemap-mcp query" in body
    assert "**Dims**" in body
    assert "UNKNOWN" in body


def test_install_opencode_skills_under_xdg(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    out = install_for("OpenCode")
    assert out["ok"] is True
    skill = (
        tmp_path
        / "xdg"
        / "opencode"
        / "skills"
        / "ascendc-codemap-query-codemap"
        / "SKILL.md"
    )
    assert skill.is_file()
    agents = tmp_path / "xdg" / "opencode" / "AGENTS.md"
    assert "ascendc-codemap-mcp" in agents.read_text(encoding="utf-8")
