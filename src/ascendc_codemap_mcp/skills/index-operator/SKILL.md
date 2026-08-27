---
name: index-operator
description: Build an AscendC operator CodeMap. Use when the user asks to index an operator, or query_codemap / codemap_status reports no .uo product.
---

# Index an operator

Use MCP server `ascendc-codemap-mcp`. Tools: `codemap_doctor`, `index_operator`, `codemap_status`.

## When

No committed CodeMap for this operator + architecture. User asked to **build from scratch**.

If a `.uo` already exists and sources changed, use skill `update-operator` instead.

## Steps

1. Require `project` (operator directory, absolute) and `architecture` (e.g. `arch35`). Do not guess architecture from another tree.
2. Call `codemap_doctor` with those arguments. If `ok` is false, stop and report `issues` / `explain` (CANN root, libclang).
3. Call `index_operator`. This can take minutes. Do not call it automatically on session start.
4. Call `codemap_status`. A partial product is valid; residuals stay unresolved. Do not patch `.uo` by hand.

## Stop

- Doctor failed: not an indexing problem until CANN/libclang/operator path are fixed.
- Index failed: report `failed_step` and `error`. Do not invent graph edges.
