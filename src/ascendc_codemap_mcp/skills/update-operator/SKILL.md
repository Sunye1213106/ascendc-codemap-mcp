---
name: update-operator
description: Incrementally refresh an existing AscendC operator CodeMap after source changes. Use when a .uo already exists and the operator sources moved (git pull, local edits). Do not full re-index.
---

# Update an operator CodeMap

Use MCP server `ascendc-codemap-mcp`. Tool: `codemap_update`.

## When

A `.uo` exists and sources changed. Git pull, branch switch, or local Host/Kernel edits. Discover/`codemap_query` reports `stale` or `dirty`.

Do **not** call `codemap_index` for a refresh. That is a cold rebuild (minutes). `codemap_update` detects the delta and rebuilds only the affected layers.

## Steps

1. Prefer `codemap_id` from discover. Otherwise require `project` + `architecture`. Do not guess architecture.
2. If discover says not indexed, stop and use skill `index-operator`.
3. Call `codemap_update`. Queries during the rebuild see `freshness=building`.
4. Read `state` and `updated`, not only `ok`. `ok=true` with `state=needs_confirmation` means the snapshot did **not** advance — tell the user, then retry with `confirm_scope: true` only after they confirm.
5. Do not patch `.uo` by hand.

## Stop

- `state: failed` — report `error` / failed rebuild action. Do not invent graph edges.
- `state: completed` with `mode: noop` — graph already matches sources; query as-is.
