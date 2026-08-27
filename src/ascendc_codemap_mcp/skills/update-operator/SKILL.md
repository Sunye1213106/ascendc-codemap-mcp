---
name: update-operator
description: Incrementally refresh an existing AscendC operator CodeMap after source changes. Use when a .uo already exists and the operator sources moved (git pull, local edits). Do not full re-index.
---

# Update an operator CodeMap

Use MCP server `ascendc-codemap-mcp`. Tool: `update_operator`.

## When

`codemap_status` says indexed, and sources changed. Git pull, branch switch, or local Host/Kernel edits.

Do **not** call `index_operator` for a refresh. That is a cold rebuild (minutes). `update_operator` detects the delta and rebuilds only the affected layers.

## Steps

1. Require `project` (operator directory, absolute) and `architecture`. Do not guess architecture.
2. Call `codemap_status`. If not indexed, stop and use skill `index-operator`.
3. Call `update_operator`. This can still take a minute when Host/Kernel layers rebuild.
4. If `status` is `blocked` and `needs_scope_review` is true, tell the user, then retry with `confirm_scope: true` only after they confirm.
5. Call `codemap_status` again. Do not patch `.uo` by hand.

## Stop

- `status: fail` — report `error` / failed rebuild action. Do not invent graph edges.
- `status: pass` with `mode: noop` — graph already matches sources; query as-is.
