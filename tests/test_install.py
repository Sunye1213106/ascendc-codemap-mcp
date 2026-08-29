# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path

from ascendc_codemap_mcp.constants import MCP_MARK_BEGIN, PRODUCT_NAME
from ascendc_codemap_mcp.install import claude, codex, cursor, opencode


def test_cursor_upsert_and_uninstall(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "mcp.json"
    path.write_text('{"mcpServers": {"other": {"command": "x"}}}', encoding="utf-8")
    monkeypatch.setattr(cursor, "config_path", lambda: path)
    cmd = ["python", "-m", "ascendc_codemap_mcp"]
    env = {"PYTHONUNBUFFERED": "1"}
    assert cursor.install(cmd, env)["ok"] is True
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "other" in data["mcpServers"]
    entry = data["mcpServers"][PRODUCT_NAME]
    assert entry["command"] == "python"
    assert entry["args"] == ["-m", "ascendc_codemap_mcp"]
    assert cursor.uninstall(cmd)["ok"] is True
    data = json.loads(path.read_text(encoding="utf-8"))
    assert PRODUCT_NAME not in data["mcpServers"]
    assert "other" in data["mcpServers"]


def test_cursor_does_not_clobber_foreign_entry(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps({"mcpServers": {PRODUCT_NAME: {"command": "someone-else"}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(cursor, "config_path", lambda: path)
    out = cursor.install(["python", "-m", "ascendc_codemap_mcp"], {})
    assert out["ok"] is False
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["mcpServers"][PRODUCT_NAME]["command"] == "someone-else"


def test_claude_mcp_servers(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / ".claude.json"
    monkeypatch.setattr(claude, "config_path", lambda: path)
    cmd = ["python", "-m", "ascendc_codemap_mcp"]
    assert claude.install(cmd, {})["ok"] is True
    data = json.loads(path.read_text(encoding="utf-8"))
    assert PRODUCT_NAME in data["mcpServers"]


def test_codex_marker_block(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "config.toml"
    path.write_text("model = \"gpt\"\n", encoding="utf-8")
    monkeypatch.setattr(codex, "config_path", lambda: path)
    cmd = ["C:\\py\\python.exe", "-m", "ascendc_codemap_mcp"]
    assert codex.install(cmd, {})["ok"] is True
    text = path.read_text(encoding="utf-8")
    assert MCP_MARK_BEGIN in text
    assert "ASCENDC_CODEMAP_CANN_ROOT" in text
    assert "model = \"gpt\"" in text
    assert codex.uninstall(cmd)["ok"] is True
    text = path.read_text(encoding="utf-8")
    assert MCP_MARK_BEGIN not in text
    assert "model = \"gpt\"" in text


def test_opencode_local_command_array(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "opencode.json"
    monkeypatch.setattr(opencode, "config_path", lambda: path)
    cmd = ["python", "-m", "ascendc_codemap_mcp"]
    assert opencode.install(cmd, {"PYTHONUNBUFFERED": "1"})["ok"] is True
    data = json.loads(path.read_text(encoding="utf-8"))
    entry = data["mcp"][PRODUCT_NAME]
    assert entry["type"] == "local"
    assert entry["command"] == cmd
    assert entry["enabled"] is True


def test_normalize_hosts() -> None:
    from ascendc_codemap_mcp.install import normalize_hosts

    assert normalize_hosts(None) is None
    assert normalize_hosts([]) is None
    assert normalize_hosts(["all"]) is None
    assert normalize_hosts(["opencode"]) == ["opencode"]
    assert normalize_hosts(["uninstall-opencode"]) == ["opencode"]
    assert normalize_hosts(["OpenCode,cursor"]) == ["opencode", "cursor"]
    try:
        normalize_hosts(["nope"])
    except ValueError as exc:
        assert "unknown host" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_run_install_host_opencode_only(tmp_path: Path, monkeypatch) -> None:
    from ascendc_codemap_mcp.install import run_install, skills as skill_install

    def boom(*_a, **_k):
        raise AssertionError("unexpected host")

    oc = tmp_path / "opencode.json"
    monkeypatch.setattr(opencode, "config_path", lambda: oc)
    monkeypatch.setattr(cursor, "install", boom)
    monkeypatch.setattr(claude, "install", boom)
    monkeypatch.setattr(codex, "install", boom)
    monkeypatch.setattr(skill_install, "install_for", lambda *_a, **_k: None)
    shared = {"called": False}
    monkeypatch.setattr(
        skill_install,
        "install_shared",
        lambda *_a, **_k: shared.__setitem__("called", True),
    )
    assert run_install(hosts=["opencode"]) == 0
    data = json.loads(oc.read_text(encoding="utf-8"))
    assert PRODUCT_NAME in data["mcp"]
    assert shared["called"] is False


def test_run_uninstall_opencode_skills_without_mcp(
    tmp_path: Path, monkeypatch
) -> None:
    from ascendc_codemap_mcp.install import run_uninstall, skills as skill_install

    def boom(*_a, **_k):
        raise AssertionError("unexpected host")

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(opencode, "config_path", lambda: tmp_path / "missing.json")
    monkeypatch.setattr(cursor, "uninstall", boom)
    monkeypatch.setattr(claude, "uninstall", boom)
    monkeypatch.setattr(codex, "uninstall", boom)
    monkeypatch.setattr(
        skill_install,
        "uninstall_shared",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("shared")),
    )
    skill_install.install_for("OpenCode")
    skill = (
        tmp_path
        / "xdg"
        / "opencode"
        / "skills"
        / "ascendc-codemap-query-codemap"
        / "SKILL.md"
    )
    assert skill.is_file()
    assert run_uninstall(hosts=["opencode"]) == 0
    assert not skill.exists()


def test_opencode_home_uses_xdg(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert opencode.home() == tmp_path / "xdg" / "opencode"


def test_cli_install_unknown_host() -> None:
    from ascendc_codemap_mcp.cli import main

    assert main(["install", "--host", "nope"]) == 2
