# -*- coding: utf-8 -*-
"""Install / uninstall MCP entries and skills for four coding agents."""
from __future__ import annotations

import sys
from pathlib import Path

from ascendc_codemap_mcp.constants import PRODUCT_NAME
from ascendc_codemap_mcp.install import claude, codex, cursor, opencode, skills as skill_install


def mcp_command() -> list[str]:
    return [sys.executable, "-m", "ascendc_codemap_mcp"]


def mcp_env() -> dict[str, str]:
    # Pin the agent to this checkout even if another site-packages copy exists.
    src = str(Path(__file__).resolve().parents[2])
    return {
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONPATH": src,
    }


def run_install(*, dry_run: bool = False) -> int:
    cmd = mcp_command()
    env = mcp_env()
    reports = [
        cursor.install(cmd, env, dry_run=dry_run),
        claude.install(cmd, env, dry_run=dry_run),
        codex.install(cmd, env, dry_run=dry_run),
        opencode.install(cmd, env, dry_run=dry_run),
    ]
    for row in reports:
        mark = "plan" if dry_run else ("ok" if row.get("ok") else "skip")
        print(f"{row.get('client')}: {mark} {row.get('path') or ''} {row.get('detail') or ''}".rstrip())
        if row.get("ok") and not dry_run:
            skill_install.install_for(str(row.get("client") or ""), dry_run=False)
    if not dry_run:
        skill_install.install_shared(dry_run=False)
    print(f"{PRODUCT_NAME}: restart your coding agent to load MCP tools.")
    return 0


def run_uninstall(*, dry_run: bool = False) -> int:
    cmd = mcp_command()
    reports = [
        cursor.uninstall(cmd, dry_run=dry_run),
        claude.uninstall(cmd, dry_run=dry_run),
        codex.uninstall(cmd, dry_run=dry_run),
        opencode.uninstall(cmd, dry_run=dry_run),
    ]
    for row in reports:
        mark = "plan" if dry_run else ("ok" if row.get("ok") else "skip")
        print(f"{row.get('client')}: {mark} {row.get('path') or ''} {row.get('detail') or ''}".rstrip())
        if row.get("ok") and not dry_run:
            skill_install.uninstall_for(str(row.get("client") or ""), dry_run=False)
    if not dry_run:
        skill_install.uninstall_shared(dry_run=False)
    return 0
