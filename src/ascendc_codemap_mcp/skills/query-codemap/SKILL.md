---
name: query-codemap
description: Query an existing AscendC operator CodeMap. Use when asking what is on the graph, who writes/reads a name, whether a Dim=V compiles, or what sits at an evidence site.
---

# Query CodeMap

Use MCP server `ascendc-codemap-mcp`. Prefer typed tools. `query_codemap` is a compatibility facade.

## Before querying

1. `codemap_discover` with the operator `project` if you do not yet have a `codemap.id` (`p:<workspace>/op@arch`; `op@arch` is an alias).
2. `codemap_status(codemap_id)`. Read `freshness`.
3. `freshness=unknown` and `indexed=false` → skill `index-operator`. `stale` or `dirty` → skill `update-operator`. `building` → wait and status again.

Do not Glob/Grep for `.uo`. Do not guess architecture.

## Typed tools

| Intent | Tool |
| --- | --- |
| What can this map answer | `codemap_overview` |
| Identifier definition / writers / readers | `codemap_symbol` (`symbol` is one identifier) |
| Dim legal set / Name=Value compiles | `codemap_selection` (`dim`, optional `value`) |
| Continue from a card | `codemap_evidence` (`evidence_id` from `evidence[].id`, plus `expected_snapshot_id`) |

If `coverage.truncated` and `next_cursor` is set, pass `next_cursor`. Nested neighbor samples may set `nested_truncated` without a cursor — do not invent a page. Do not pass a natural-language sentence as `symbol`.

## Rules

- Quote a construct only after seeing its own `file:line` or `evidence_id`.
- `count: 0` is not "does not exist". Follow `hint` / `canonical` / `text_hits`, then PARTIAL / UNKNOWN.
- List conclusions need totals. If `coverage.truncated` or `count` exceeds listed neighbors, PARTIAL.
- Answer the layer asked. Host produced ≠ template admissible ≠ kernel consumed.
- Keep `evidence_id` (`span:...`) and its `snapshot_id` across turns; pass `expected_snapshot_id` on `codemap_evidence`. Line numbers drift when sources move.

## Output

```text
verdict: ANSWERED | PARTIAL | UNKNOWN
layer: domain | template | host | kernel
span: file:line
coverage: returned/total truncated=...
missing: ...
```
