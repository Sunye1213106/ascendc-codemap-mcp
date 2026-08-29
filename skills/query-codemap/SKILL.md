---
name: query-codemap
description: Query an existing AscendC operator CodeMap. Use when asking who writes/reads a name, why a path is taken, which TilingKey/Dim values exist on this operator, or what a change affects.
---

# Query CodeMap

Identity: `codemap_query` with `codemap_id`, or `project` + `architecture`. Missing map → `index-operator`. Stale/dirty → `update-operator`.

```text
代码语义（谁写谁读 / 为什么走这条路 / 改了影响谁）  → 一次 codemap_query
已知文件 + 精确源码细节（卡片没列出的行）          → targeted Read
```

`operation` 是闭集 enum，缺省 `resolve`：

| operation | 何时 | 选择器 |
|---|---|---|
| `resolve` | 身份：一个 ident 或 file:line | `symbol` / `file`（可无 line） / `file`+`line` / `entity_id` |
| `find` | 集合：所有 call site / 谓词 / **名字发现** | `kind` + `callee` / `literal` / `operator` / `referenced_value`，或单独 `name` |
| `impact` | 改了影响谁 | 同一个 seed |
| `contract` | 跨层契约 | 同一个 seed（必须唯一） |
| `entry` | 入口 / early-return / bailout | `entry_role` |
| `trace` | A 到 B 有没有 typed path | `from_symbol` + `to_symbol` |

**不知道 ident 就先发现，再定位。**只知道片段时先 `find name=`（子串，或用 `*` `?` 通配）。miss 时同一次调用会按驼峰 / `_` / 数字切词，并展开常见缩写（`Buf`↔`Buffer`），直接返回 **Names**。再 `resolve` 其中一个真名。两步都是图查询，不要对 ident / 调用点做 grep。

```text
find name=*Foo*            →  **Names**（精确叶名 > 词法 token > 子串）
find name=BufferNum        →  无此叶名时同一次给出 Buffer 家族
resolve <其中一个真名>      →  一份定义 + **Used at**（其它文件各自 snippet）
file=foo.h                 →  主 Source 切到该文件，不必带 line
```

`resolve` 已有 `symbol=` 时多余的 `name=` 会被忽略，不必重发。截断时收紧 `name=` 或提高 `limit`，不要改用 grep。`INVALID_QUERY` 的 `did you mean:` 是现成调用，照抄一条重发。

返回正文即答案。列出的源码、**Used at**、**Call sites** 视为已经 Read。不要为每个候选再 `resolve` 三次。

**Done：** 正文已有 Host→Kernel Flow，或 **Dims** 列表，或 dim 取值集合。`legal_key_count: 0` 在 dim 查询上忽略。Kernel API（`InitBuffer` / `DataCopyPad` / `SetFlag`）有 Source 就是底层站点，不要再找 Host 写维。

**UNKNOWN：** 该 ident 在本算子不存在。正文 **Dims** 就是本算子维名，改用其中一个。不要重试同一 ident，不要把别的算子的维搬过来。

空 `resolve` 也返回 **Dims**。`find` 返回集合（`total` / `truncated`），多个站点不是歧义。`resolve` / `contract` 的 `AMBIGUOUS` 只表示 **多份定义体**——每个候选带自己文件的 Source，挑一个再 `resolve`。同名的定义 vs 使用点不是歧义，使用点在 **Used at**。

Names 下已有 **Call sites** 表时，不要再发 `find kind=OPERATION callee=…`。

CLI（无 MCP 的子 agent / 脚本）走同一引擎：

```text
ascendc-codemap-mcp query --codemap-id ID --symbol IsRope
ascendc-codemap-mcp query --codemap-id ID --operation find --name '*Foo*'
ascendc-codemap-mcp query --codemap-id ID --operation find --kind OPERATION --callee Foo
ascendc-codemap-mcp query --codemap-id ID --file PATH --line N --line-end M
```
