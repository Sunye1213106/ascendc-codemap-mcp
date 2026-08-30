# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

from ascendc_codemap_mcp.install.skills import bundled_root, install_for, uninstall_for


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
    assert "search name" in text
    assert "在 dim 查询上忽略" not in text
    assert "InitBuffer" in text
    assert "codemap_explore" not in text
    assert "overview" not in text or "Do not call overview" in text
    assert "去 evidence" not in text
    assert "语义问题去" not in text
    assert "logical unit" in text
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


def test_uninstall_for_deletes_empty_agents_md(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    install_for("OpenCode")
    agents = tmp_path / "xdg" / "opencode" / "AGENTS.md"
    assert agents.is_file()
    out = uninstall_for("OpenCode")
    assert out["ok"] is True
    assert not agents.exists()
    removed = out["removed"]
    assert any(Path(p).name == "ascendc-codemap-query-codemap" for p in removed)


def test_uninstall_for_deletes_already_empty_agents_md(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    agents = tmp_path / "xdg" / "opencode" / "AGENTS.md"
    agents.parent.mkdir(parents=True)
    agents.write_text("", encoding="utf-8")
    uninstall_for("OpenCode")
    assert not agents.exists()


def test_leftover_skill_folders_after_opencode_only_uninstall(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda *args, **kwargs: tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    install_for("OpenCode")
    install_for("Cursor")
    uninstall_for("OpenCode")
    from ascendc_codemap_mcp.install.skills import leftover_skill_folders

    leftover = leftover_skill_folders()
    names = {p.name for p in leftover}
    assert "ascendc-codemap-query-codemap" in names
    assert any(".cursor" in p.parts for p in leftover)
    assert not any(p.parts[-3:] == ("opencode", "skills", "ascendc-codemap-query-codemap") for p in leftover)
