---
name: query-codemap
description: Query an existing AscendC operator CodeMap. Use when asking what is on the graph, who writes/reads a name, whether a Dim=V compiles, or what sits at file:line.
---

# Query CodeMap

Use MCP `query_codemap` on server `ascendc-codemap-mcp`. Four input shapes only. Do not pass natural-language sentences.

## Before querying

If `codemap_status` says not indexed, stop and use skill `index-operator`. If indexed but sources changed (git pull), use skill `update-operator` before querying. Do not Glob/Grep for `.uo`.

## Four shapes

1. **No pattern** — index (launch / PIPE names).
2. **Identifier** — name / definition / who writes who reads. One card includes `definition`, `host.writers`, `kernel.readers`, `flow`.
3. **`Dim=<name>` or `Name=Value`** — legal set / whether a combo compiles. Read `sel_sites` / `dim_coverage`.
4. **`file` + `line`** — statement window at a site copied from a previous card. Includes enclosing + `impact`.

## Rules

- Pick the shortest shape. Cards are evidence pointers (`file:line`). Quote a construct only after seeing its own line.
- `count: 0` is not "does not exist". Follow `hint` / `canonical` / `text_hits`, then PARTIAL / UNKNOWN.
- List conclusions need totals (`dim_coverage`, `matching_block_count`, `edges.*.count`). If `count` exceeds listed neighbors, PARTIAL.
- Answer the layer asked. Host produced ≠ template admissible ≠ kernel consumed.
- Do not write LLM patches into `.uo`.

## Output

```text
verdict: ANSWERED | PARTIAL | UNKNOWN
layer: domain | template | host | kernel
span: file:line
coverage: dim_coverage=... / count=...
missing: ...
```
