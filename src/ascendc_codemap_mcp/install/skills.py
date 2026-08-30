# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ascendc_codemap_mcp.constants import (
    AGENTS_MARK_BEGIN,
    AGENTS_MARK_END,
    SKILL_NAMES,
)
from ascendc_codemap_mcp.install.opencode import home as opencode_home

_AGENTS_BODY = """# AscendC CodeMap MCP

Use MCP server `ascendc-codemap-mcp`. Identity: `codemap_discover` then `codemap.id`, or project+architecture on `codemap_query`. Missing → `codemap_doctor` / `codemap_index`. Stale/dirty → `codemap_update`.

Unknown → `codemap_query operation=search name=`. Known or file:line → `operation=resolve`. Query reads the snapshot only.

Query reads the `.uo` snapshot only. Architecture: see the package `docs/ARCHITECTURE.md`.
"""


def bundled_root() -> Path:
    here = Path(__file__).resolve()
    root = here.parents[1] / "skills"
    if (root / "index-operator" / "SKILL.md").is_file():
        return root
    raise FileNotFoundError("bundled AscendC CodeMap skills are missing")


def canonical_query_skill() -> Path:
    """Repo-root ``skills/query-codemap/SKILL.md`` is the single owner."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "skills" / "query-codemap" / "SKILL.md"
        package_copy = parent / "src" / "ascendc_codemap_mcp" / "skills" / "query-codemap" / "SKILL.md"
        if candidate.is_file() and package_copy.is_file():
            return candidate
    return bundled_root() / "query-codemap" / "SKILL.md"


def sync_query_skill() -> str:
    """Copy the canonical query skill into the package bundle. Returns the text."""
    src = canonical_query_skill()
    body = src.read_text(encoding="utf-8")
    dest = bundled_root() / "query-codemap" / "SKILL.md"
    if dest.resolve() != src.resolve():
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(body, encoding="utf-8")
    return body


def _skill_folder_names() -> tuple[str, ...]:
    names: list[str] = []
    for name in SKILL_NAMES:
        names.append(f"ascendc-codemap-{name}")
        names.append(name)
    return tuple(names)


def _copy_skills(dest: Path) -> None:
    src = bundled_root()
    dest.mkdir(parents=True, exist_ok=True)
    query_body = sync_query_skill()
    for name in SKILL_NAMES:
        body = query_body if name == "query-codemap" else (src / name / "SKILL.md").read_text(encoding="utf-8")
        for folder in (f"ascendc-codemap-{name}", name):
            target = dest / folder / "SKILL.md"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body, encoding="utf-8")


def _remove_skills(dest: Path) -> list[str]:
    removed: list[str] = []
    if not dest.is_dir():
        return removed
    import shutil

    for folder in _skill_folder_names():
        path = dest / folder
        if path.is_dir():
            shutil.rmtree(path)
            removed.append(str(path))
    return removed


def _upsert_agents(path: Path) -> None:
    block = f"{AGENTS_MARK_BEGIN}\n{_AGENTS_BODY.rstrip()}\n{AGENTS_MARK_END}\n"
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    start = existing.find(AGENTS_MARK_BEGIN)
    end = existing.find(AGENTS_MARK_END)
    if start >= 0 and end > start:
        end_at = end + len(AGENTS_MARK_END)
        while end_at < len(existing) and existing[end_at] in "\r\n":
            end_at += 1
        text = existing[:start] + block + existing[end_at:]
    else:
        text = existing
        if text and not text.endswith("\n"):
            text += "\n"
        if text:
            text += "\n"
        text += block
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _remove_agents(path: Path) -> None:
    if not path.is_file():
        return
    existing = path.read_text(encoding="utf-8")
    start = existing.find(AGENTS_MARK_BEGIN)
    end = existing.find(AGENTS_MARK_END)
    if start < 0 or end < start:
        if not existing.strip():
            path.unlink()
        return
    end_at = end + len(AGENTS_MARK_END)
    while end_at < len(existing) and existing[end_at] in "\r\n":
        end_at += 1
    remaining = existing[:start] + existing[end_at:]
    if remaining.strip():
        path.write_text(remaining, encoding="utf-8")
    else:
        path.unlink()


def _client_skill_dir(client: str) -> Path | None:
    home = Path.home()
    if client == "Cursor":
        return home / ".cursor" / "skills"
    if client == "Claude Code":
        return home / ".claude" / "skills"
    if client == "Codex":
        base = Path(os.environ.get("CODEX_HOME") or (home / ".codex"))
        return base / "skills"
    if client == "OpenCode":
        return opencode_home() / "skills"
    return None


def _client_agents_path(client: str) -> Path | None:
    home = Path.home()
    if client == "Codex":
        base = Path(os.environ.get("CODEX_HOME") or (home / ".codex"))
        return base / "AGENTS.md"
    if client == "OpenCode":
        return opencode_home() / "AGENTS.md"
    return None


def install_for(client: str, *, dry_run: bool = False) -> dict[str, Any]:
    dest = _client_skill_dir(client)
    if dest is None or dry_run:
        return {"ok": dest is not None, "client": client, "path": str(dest or "")}
    _copy_skills(dest)
    agents = _client_agents_path(client)
    if agents is not None:
        _upsert_agents(agents)
    return {"ok": True, "client": client, "path": str(dest)}


def uninstall_for(client: str, *, dry_run: bool = False) -> dict[str, Any]:
    dest = _client_skill_dir(client)
    if dest is None or dry_run:
        return {
            "ok": dest is not None,
            "client": client,
            "path": str(dest or ""),
            "removed": [],
        }
    removed = _remove_skills(dest)
    agents = _client_agents_path(client)
    if agents is not None:
        _remove_agents(agents)
    return {"ok": True, "client": client, "path": str(dest), "removed": removed}


def install_shared(*, dry_run: bool = False) -> None:
    if dry_run:
        return
    dest = Path.home() / ".agents" / "skills"
    _copy_skills(dest)


def uninstall_shared(*, dry_run: bool = False) -> list[str]:
    if dry_run:
        return []
    return _remove_skills(Path.home() / ".agents" / "skills")


def leftover_skill_folders() -> list[Path]:
    dests: list[Path] = []
    for client in ("Cursor", "Claude Code", "Codex", "OpenCode"):
        dest = _client_skill_dir(client)
        if dest is not None:
            dests.append(dest)
    dests.append(Path.home() / ".agents" / "skills")
    found: list[Path] = []
    seen: set[Path] = set()
    for dest in dests:
        key = dest.resolve() if dest.exists() else dest
        if key in seen:
            continue
        seen.add(key)
        if not dest.is_dir():
            continue
        for folder in _skill_folder_names():
            path = dest / folder
            if path.is_dir():
                found.append(path)
    return found
