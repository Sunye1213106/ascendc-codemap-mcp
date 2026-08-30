---
name: query-codemap
description: Query an existing AscendC operator CodeMap. Use when asking who writes/reads a name, why a path is taken, which TilingKey/Dim values exist on this operator, or what a change affects.
---

# Query CodeMap

Identity: `codemap_query` with `codemap_id`, or `project` + `architecture`. Missing map → `index-operator`. Stale/dirty → `update-operator`.

Unknown → search
Known or file:line → resolve
Query reads snapshot only

search works like regex search over the indexed source snapshot.
Add kind= to search semantic entities.

search → file:line → resolve(file,line)
known symbol → resolve(symbol)

resolve returns source plus CodeMap semantic context.

```text
ascendc-codemap-mcp query --codemap-id ID --operation search --name L1
ascendc-codemap-mcp query --codemap-id ID --symbol IsRope
ascendc-codemap-mcp query --codemap-id ID --file op_kernel/k.h --line 10
```
