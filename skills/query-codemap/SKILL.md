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
| `search` | 有词不知 ident：扫源码行 | `name`（短语；可选 `file=`） |
| `find` | 集合：所有 call site / 谓词 / **名字发现** | `kind` + `callee` / `literal` / `operator` / `referenced_value`，或单独 `name` |
| `impact` | 改了影响谁 | 同一个 seed |
| `contract` | 跨层契约 | 同一个 seed（同叶名多 kind 会合并） |
| `entry` | 入口 / early-return / bailout | `entry_role` |
| `trace` | A 到 B 有没有 typed path | `from_symbol` + `to_symbol` |

**不知道 ident 就先 search，再定位。**只知道源码里的词时先 `search name=`（走 `source_fts`，不扩缩写）。`find name=` 只匹配 `entity.name`（子串，或用 `*` `?` 通配）；0 命中就是 0，不会切词扩成 Buffer 家族。再 `resolve` / `contract` 其中一个真名。

```text
有词不知 ident     →  search name=
有 ident           →  resolve / contract
要证明编没编       →  resolve dim= / dim= + value=
kind 站点集合      →  find kind= （OPERATION 按文件轮转）
```

```text
search name=BufferNum     →  path:line + 该行原文（0 = UNKNOWN，改短语）
find name=*Foo*           →  Matches（operator-local / 稀有名优先；glob 不是 exact leaf）
find name=*BufferNum*     →  无此叶名则 0，改 search name=BufferNum
resolve <其中一个真名>      →  Definition + References（file:line）
contract <ident>           →  Host → TilingKey → Kernel（模板宏默认折叠）
file=foo.h                 →  Definition 切到该文件，不必带 line
```

`resolve` 已有 `symbol=` 时多余的 `name=` 会被忽略，不必重发。截断时收紧 `name=` 或提高 `limit`。`INVALID_QUERY` 的 `did you mean:` 是现成调用，照抄一条重发。

Returned Definition spans are usable evidence. If neighboring lines are absent, use `codemap_evidence` or targeted Read. 不要为每个候选再 `resolve` 三次。

**Done：** 正文已有 contract / Definition / **Dims** / dim 取值集合（`value: n`）。`legal_key_count` 是编进包的 key 总数。Kernel API（`InitBuffer` / `DataCopyPad` / `SetFlag`）有 Definition 就是底层站点，不要再找 Host 写维。

**UNKNOWN：** 该 ident 在本算子不存在，或 search 0 命中。正文 **Dims** 就是本算子维名。不要重试同一 ident，不要把别的算子的维搬过来。

空 `resolve` 也返回 **Dims**。`find` / `search` 返回集合（`Matches:`），多个站点不是歧义。`resolve` / `contract` 的 `AMBIGUOUS` 只表示 **不同叶名或多份定义体**。同名的定义 vs 使用点不是歧义，使用点在 **References**。

Matches 下已有 **Call sites** 表时，不要再发 `find kind=OPERATION callee=…`。

CLI（无 MCP 的子 agent / 脚本）走同一引擎：

```text
ascendc-codemap-mcp query --codemap-id ID --symbol IsRope
ascendc-codemap-mcp query --codemap-id ID --operation search --name BufferNum
ascendc-codemap-mcp query --codemap-id ID --operation find --name '*Foo*'
ascendc-codemap-mcp query --codemap-id ID --operation find --kind OPERATION --callee Foo
ascendc-codemap-mcp query --codemap-id ID --file PATH --line N --line-end M
```
