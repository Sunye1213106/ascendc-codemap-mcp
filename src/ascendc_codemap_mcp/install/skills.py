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

Use MCP server `ascendc-codemap-mcp`. Identity: `codemap_discover` then `codemap.id` (`p:<workspace>::op@arch`), or pass project+architecture on `codemap_query`. Missing → `codemap_doctor` / `codemap_index`. Stale/dirty → `codemap_update`.

```text
代码语义（谁写谁读 / 为什么走这条路 / 改了影响谁）  → 一次 codemap_query
已知文件 + 精确源码细节（卡片没列出的行）          → targeted Read
```

不知道确切 ident：先 `codemap_query operation=find name=<片段>`（子串，或用 `*` `?` 通配）拿 **Names**；miss 时同一次调用会按驼峰 / `_` / 数字 / 缩写再搜。再 `resolve` 其中真名。不要对 ident / 调用点做 grep。`resolve` 已有 `symbol=` 时多余的 `name=` 会被忽略。

CLI fallback (same engine): `ascendc-codemap-mcp query --codemap-id ID --symbol <ident>`.
Query text is already Read. Empty resolve lists **Dims**. UNKNOWN means the ident is absent on this operator — use those Dims. AMBIGUOUS is multiple definition bodies, each with its own Source — pick one. **Used at** and **Call sites** on the card are already Read. Kernel API Source is the bottom layer. Do not call overview first. INVALID_QUERY prints `did you mean:` ready-made calls — resend one verbatim.
"""


def bundled_root() -> Path:
    here = Path(__file__).resolve()
    candidates = (
        here.parents[1] / "skills",
        here.parents[3] / "skills",
    )
    for root in candidates:
        if (root / "index-operator" / "SKILL.md").is_file():
            return root
    raise FileNotFoundError("bundled AscendC CodeMap skills are missing")


def _copy_skills(dest: Path) -> None:
    src = bundled_root()
    dest.mkdir(parents=True, exist_ok=True)
    for name in SKILL_NAMES:
        body = (src / name / "SKILL.md").read_text(encoding="utf-8")
        for folder in (f"ascendc-codemap-{name}", name):
            target = dest / folder / "SKILL.md"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body, encoding="utf-8")


def _remove_skills(dest: Path) -> None:
    if not dest.is_dir():
        return
    import shutil

    for name in SKILL_NAMES:
        for folder in (f"ascendc-codemap-{name}", name):
            path = dest / folder
            if path.is_dir():
                shutil.rmtree(path)


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
        return
    end_at = end + len(AGENTS_MARK_END)
    while end_at < len(existing) and existing[end_at] in "\r\n":
        end_at += 1
    path.write_text(existing[:start] + existing[end_at:], encoding="utf-8")


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
        return {"ok": dest is not None, "client": client, "path": str(dest or "")}
    _remove_skills(dest)
    agents = _client_agents_path(client)
    if agents is not None:
        _remove_agents(agents)
    return {"ok": True, "client": client, "path": str(dest)}


def install_shared(*, dry_run: bool = False) -> None:
    if dry_run:
        return
    dest = Path.home() / ".agents" / "skills"
    _copy_skills(dest)


def uninstall_shared(*, dry_run: bool = False) -> None:
    if dry_run:
        return
    _remove_skills(Path.home() / ".agents" / "skills")
