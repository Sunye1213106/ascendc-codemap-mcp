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
