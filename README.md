# AscendC CodeMap MCP

AscendC operator CodeMap for AI coding agents. Indexes one operator (Host / TilingKey / TilingData / Kernel) into a committed `.uo` graph, then answers structural queries over MCP.

This is a **semantic compiler CodeMap**, not a general code-memory store. `.uo` holds what the code *is*. Agent memory, plans, reviews, and ADRs stay in the caller (AscendC-Pilot, your session, etc.).

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

The server uses the official Python MCP SDK and speaks **2026-07-28** (stateless `server/discover`) while still serving handshake-era clients from the same process.

## Identity

After discover/index, pass a stable id instead of an absolute path on every call:

```text
codemap_id   = p:<workspace>/op_name@arch     # canonical, e.g. p:a91f42/flash_attention_score_grad@arch35
alias        = op_name@arch                   # unique in this process; AMBIGUOUS_CODEMAP_ID otherwise
snapshot_id  = cm:<sha256 prefix>             # committed graph identity (digest/revision, not path/mtime)
```

## CLI

```text
ascendc-codemap-mcp                 # stdio MCP (default)
ascendc-codemap-mcp serve --transport streamable-http --port 8765
ascendc-codemap-mcp install
ascendc-codemap-mcp uninstall
ascendc-codemap-mcp doctor   --project <op> --architecture arch35
ascendc-codemap-mcp discover --project <op>
ascendc-codemap-mcp index    --project <op> --architecture arch35
ascendc-codemap-mcp update   --codemap-id <op>@arch35
ascendc-codemap-mcp status   --codemap-id <op>@arch35
ascendc-codemap-mcp query    --codemap-id <op>@arch35 [pattern]
```

`--project` + `--architecture` still work for bootstrap and scripts.

## MCP tools

Control:

- `codemap_discover` — scan an operator directory; returns `codemap_id`
- `codemap_status` — **freshness** (`fresh` / `dirty` / `stale` / `building` / `blocked` / `incompatible` / `unknown`), revisions, `snapshot_id`
- `codemap_doctor` — CANN / libclang / operator path / architecture
- `codemap_index` — prepare → extract → analyze → commit (cold build)
- `codemap_update` — incremental refresh. `ok` is transport success; read `state` and `updated`

Query:

- `codemap_overview` — launch / dim / tiling-data index
- `codemap_symbol` — one identifier
- `codemap_selection` — `dim` + optional `value`
- `codemap_evidence` — `evidence_id` (`span:...`) plus `expected_snapshot_id`, or `file`+`line`

Compatibility facade: `query_codemap` (and `index_operator` / `update_operator` aliases).

Read tools are marked `readOnlyHint`. Index/update are additive (`destructiveHint=false`, closed world). Query tools publish an `outputSchema` for the shared envelope (`ok`, `codemap`, `verdict`, `layer`, `data`, `evidence`, `coverage`, `next_cursor`) and return `structuredContent`.

`codemap_index` is cancellable between `prepare` / `extract` / `analyze` / `commit`. `codemap_update` is cancellable between `detect` / `plan` / each rebuild layer / `commit`. Clang inside a step still runs to the end of that step. Cancellation is per request, not per CodeMap.

Do not auto-index on MCP connect. Indexing an operator can take minutes. Queries against a map being rebuilt return `freshness=building` rather than a half-written file.

`codemap_evidence` accepts `expected_snapshot_id` from a prior `evidence[].snapshot_id`. A mismatch returns `SNAPSHOT_CHANGED`. `next_cursor` is bound to that snapshot and query; nested edge samples set `coverage.nested_truncated` without a fake continuation.

## Resources and prompts

| Kind | Name | URI / args |
| --- | --- | --- |
| Resource | runtime | `codemap://runtime` |
| Resource template | snapshot | `codemap://map/{codemap_id}` |
| Prompt | `query_operator` | `codemap_id`, optional `focus` |
| Prompt | `build_codemap` | `project`, `architecture` |

Completions for `codemap_id` / `architecture` come from the process registry plus common arch names.

HTTP (optional, same SDK server as stdio):

```text
ascendc-codemap-mcp serve --transport streamable-http --host 127.0.0.1 --port 8765
```

## Product layout

```text
<operator>/.ascendc-codemap/<arch>/<op>.<arch>.uo
```

## Environment

- `ASCENDC_CODEMAP_CANN_ROOT` (fallback: `ASCEND_CANN_PACKAGE_PATH`, `ASCEND_HOME_PATH`, `CANN_ROOT`)
- `ASCENDC_CODEMAP_CACHE_DIR` (default `~/.cache/ascendc-codemap-mcp`)
- `ASCENDC_CODEMAP_PROJECT` / `ASCENDC_CODEMAP_ARCHITECTURE` — default identity when `codemap_id` is used after a discover in this process, or as CLI defaults
- `ASCENDC_CODEMAP_MAX_OPEN` — bounded query-handle LRU (default 4)
- Codex MCP subprocesses only receive names listed in `env_vars`; `install` forwards the CANN/cache variables.

## Clients

| Client | MCP config |
| --- | --- |
| Cursor | `~/.cursor/mcp.json` → `mcpServers` |
| Claude Code | `~/.claude.json` → `mcpServers` |
| Codex | `$CODEX_HOME/config.toml` managed block |
| OpenCode | `opencode.json(c)` → `mcp` / `type: local` |

Cursor and Claude Code get skills only (no Grep-intercepting hooks). Codex and OpenCode also get an `AGENTS.md` section.
