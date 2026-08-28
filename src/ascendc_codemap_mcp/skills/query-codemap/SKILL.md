---
name: query-codemap
description: Query an existing AscendC operator CodeMap. Use when asking who writes/reads a name, why a path is taken, or what a change affects.
---

# Query CodeMap

Identity: `codemap_discover` then pass `codemap_id`. Missing map → `index-operator`. Stale/dirty → `update-operator`.

```text
代码语义（谁写谁读 / 为什么走这条路 / 改了影响谁）  → codemap_explore
已知文件 + 精确源码细节                            → targeted Read
字面文本 / 正则 / 文档 / 配置                      → grep/read
CodeMap 报 INCOMPLETE                              → 按它给的窗口做 targeted 源码兜底
```

`codemap_explore` 参数：一个 ident、`Dim=V`、`file`+`line`、现象短语，或上一张卡的 `evidence_id`/`cursor`。

CLI（无 MCP 的子 agent / 脚本）走同一引擎：

```text
ascendc-codemap-mcp query --codemap-id ID "<ident|Dim=V|现象>"
ascendc-codemap-mcp query --codemap-id ID --file PATH --line N
```

`completeness=COMPLETE` 才可当答完。`INCOMPLETE` 时只用卡上的 `windows`，不要无边界 grep。
