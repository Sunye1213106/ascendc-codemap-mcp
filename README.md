# AscendC CodeMap MCP

AscendC operator CodeMap for AI coding agents. Indexes one operator (Host / TilingKey / TilingData / Kernel) into a committed `.uo` graph, then answers structural queries over MCP.

This is the operator analogue of [codebase-memory-mcp](https://github.com/DeusData/codebase-memory-mcp): install once, plug into Cursor / OpenCode / Codex / Claude Code, then say **Index this operator**.

CANN headers and libclang are required to **build** a CodeMap. Querying an existing `.uo` does not need CANN.

## Install

```bash
pip install -e .
ascendc-codemap-mcp install
```

Restart the coding agent. Tools appear as MCP server `ascendc-codemap-mcp`.

Windows:

```powershell
pip install -e .
ascendc-codemap-mcp install
```

`install` writes owned MCP entries (and skills) for detected clients. It does not enable YOLO modes or rewrite unrelated servers. `uninstall` removes only owned entries.

## CLI

```text
ascendc-codemap-mcp                 # stdio MCP (default)
ascendc-codemap-mcp install
ascendc-codemap-mcp uninstall
ascendc-codemap-mcp doctor --project <op> --architecture arch35
ascendc-codemap-mcp index  --project <op> --architecture arch35
ascendc-codemap-mcp update --project <op> --architecture arch35
ascendc-codemap-mcp status --project <op> --architecture arch35
ascendc-codemap-mcp query  --project <op> --architecture arch35 [pattern]
```

## MCP tools

- `codemap_doctor` — CANN / libclang / operator path / architecture
- `index_operator` — prepare → extract → analyze → commit (cold build; no `.uo` yet)
- `update_operator` — incremental refresh of an existing `.uo` after source changes
- `codemap_status` — whether `.uo` exists
- `query_codemap` — four shapes: index, identifier, `Dim=V` / `Name=Value`, `file`+`line`

Do not auto-index on MCP connect. Indexing an operator can take minutes.

## Product layout

```text
<operator>/.ascendc-codemap/<arch>/<op>.<arch>.uo
```

## Environment

- `ASCENDC_CODEMAP_CANN_ROOT` (fallback: `ASCEND_CANN_PACKAGE_PATH`, `ASCEND_HOME_PATH`, `CANN_ROOT`)
- `ASCENDC_CODEMAP_CACHE_DIR` (default `~/.cache/ascendc-codemap-mcp`)
- Codex MCP subprocesses only receive names listed in `env_vars`; `install` forwards the CANN/cache variables.

## Clients

| Client | MCP config |
| --- | --- |
| Cursor | `~/.cursor/mcp.json` → `mcpServers` |
| Claude Code | `~/.claude.json` → `mcpServers` |
| Codex | `$CODEX_HOME/config.toml` managed block |
| OpenCode | `opencode.json(c)` → `mcp` / `type: local` |

Cursor and Claude Code get skills only (no Grep-intercepting hooks). Codex and OpenCode also get an `AGENTS.md` section.
