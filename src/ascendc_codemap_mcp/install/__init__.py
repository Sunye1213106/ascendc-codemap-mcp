# -*- coding: utf-8 -*-
"""Install / uninstall MCP entries and skills for four coding agents."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from ascendc_codemap_mcp.constants import PRODUCT_NAME
from ascendc_codemap_mcp.install import claude, codex, cursor, opencode, skills as skill_install

KNOWN_HOSTS = ("cursor", "claude", "codex", "opencode")

_CLIENTS: dict[str, Any] = {
    "cursor": cursor,
    "claude": claude,
    "codex": codex,
    "opencode": opencode,
}


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


def normalize_hosts(hosts: list[str] | None) -> list[str] | None:
    """Return host keys to touch, or None for every known host.

    Accepts ``all``, comma-separated values, and ``uninstall-opencode`` spellings.
    """
    if not hosts:
        return None
    out: list[str] = []
    for raw in hosts:
        text = str(raw).strip()
        if text.lower().startswith("uninstall-"):
            text = text[len("uninstall-") :]
        for part in text.split(","):
            key = part.strip().lower()
            if not key:
                continue
            if key == "all":
                return None
            if key not in _CLIENTS:
                known = ", ".join((*KNOWN_HOSTS, "all"))
                raise ValueError(f"unknown host {key!r}; expected {known}")
            if key not in out:
                out.append(key)
    return out or None


def _selected_keys(hosts: list[str] | None) -> tuple[str, ...]:
    selected = normalize_hosts(hosts)
    if selected is None:
        return KNOWN_HOSTS
    return tuple(selected)


def run_install(*, dry_run: bool = False, hosts: list[str] | None = None) -> int:
    cmd = mcp_command()
    env = mcp_env()
    selected = normalize_hosts(hosts)
    keys = _selected_keys(hosts)
    reports = [_CLIENTS[key].install(cmd, env, dry_run=dry_run) for key in keys]
    for row in reports:
        mark = "plan" if dry_run else ("ok" if row.get("ok") else "skip")
        print(f"{row.get('client')}: {mark} {row.get('path') or ''} {row.get('detail') or ''}".rstrip())
        if row.get("ok") and not dry_run:
            skill_install.install_for(str(row.get("client") or ""), dry_run=False)
    if not dry_run and (selected is None or "codex" in selected):
        skill_install.install_shared(dry_run=False)
    print(f"{PRODUCT_NAME}: restart your coding agent to load MCP tools.")
    return 0


def run_uninstall(*, dry_run: bool = False, hosts: list[str] | None = None) -> int:
    cmd = mcp_command()
    selected = normalize_hosts(hosts)
    keys = _selected_keys(hosts)
    reports = [_CLIENTS[key].uninstall(cmd, dry_run=dry_run) for key in keys]
    for row in reports:
        mark = "plan" if dry_run else ("ok" if row.get("ok") else "skip")
        print(f"{row.get('client')}: {mark} {row.get('path') or ''} {row.get('detail') or ''}".rstrip())
        if not dry_run:
            # Skills are ours even when the MCP entry is already gone.
            skill_install.uninstall_for(str(row.get("client") or ""), dry_run=False)
    if not dry_run and (selected is None or "codex" in selected):
        skill_install.uninstall_shared(dry_run=False)
    return 0
