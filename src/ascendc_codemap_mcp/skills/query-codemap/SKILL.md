---
name: query-codemap
description: Query an existing AscendC operator CodeMap. Use when asking who writes/reads a name, why a path is taken, which TilingKey/Dim values exist on this operator, or what a change affects.
---

# Query CodeMap

Identity: `codemap_query` with `codemap_id`, or `project` + `architecture`. Missing map → `index-operator`. Stale/dirty → `update-operator`.

Unknown → search
Known or file:line → resolve
Query reads snapshot only

search works like regex over indexed source lines. `pattern` matches enum
values, strings, macros, comments, and identifiers — not only symbol names.
`name` is a silent alias for `pattern`. `file=` is an optional glob/path
filter; `**` matches zero or more directories (so `op_host/**/*.cpp`
includes `op_host/foo.cpp`). Optional `kind=` restricts entity-kind search
(BUFFER, EVENT, …).

search → file:line → resolve(file,line)
known symbol → resolve(symbol)  (name= is accepted as symbol)

resolve(file,line) returns the enclosing function plus **Fields in this unit**
(Host value definitions / Kernel consumers for tiling fields mentioned there).
resolve(symbol) returns a Symbol Bundle: Host value definitions, Transport,
Kernel consumers, Assignments, Calls / Called by, Compiled legal keys.
Query operations are search and resolve.

```text
ascendc-codemap-mcp query --codemap-id ID --operation search --pattern L1
ascendc-codemap-mcp query --codemap-id ID --operation search --pattern DT_HIFLOAT8
ascendc-codemap-mcp query --codemap-id ID --operation search --pattern isBn2 --file op_host/**
ascendc-codemap-mcp query --codemap-id ID --symbol IsRope
ascendc-codemap-mcp query --codemap-id ID --file op_kernel/k.h --line 10
```
