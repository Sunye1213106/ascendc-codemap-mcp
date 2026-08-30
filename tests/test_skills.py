# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

from ascendc_codemap_mcp.install.skills import (
    bundled_root,
    canonical_query_skill,
    install_for,
    uninstall_for,
)


def test_query_skill_has_single_owner() -> None:
    canonical = canonical_query_skill().read_text(encoding="utf-8")
    bundled = (bundled_root() / "query-codemap" / "SKILL.md").read_text(encoding="utf-8")
    assert canonical == bundled
    assert "Unknown → search" in canonical
    assert "operation=find" not in canonical
    assert "find kind" not in canonical


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
    assert "search works like regex" in text
    assert "kind=" in text
    assert "UNKNOWN" not in text
    assert "COMPLETE" not in text
    assert "find kind" not in text
    assert "FTS" not in text
    assert "codemap_evidence" not in text
    assert "codemap_explore" not in text
    assert "logical unit" not in text
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
    assert "search works like regex" in body
    assert "UNKNOWN" not in body


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
