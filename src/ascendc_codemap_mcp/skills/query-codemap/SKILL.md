---
name: query-codemap
description: Query an existing AscendC operator CodeMap. Use when asking who writes/reads a name, why a path is taken, which TilingKey/Dim values exist on this operator, or what a change affects.
---

# Query CodeMap

Identity: `codemap_query` with `codemap_id`, or `project` + `architecture`. Missing map → `index-operator`. Stale/dirty → `update-operator`.

Query reads the `.uo` snapshot only. It compresses facts; it does not re-analyze source.

## Happy path

```text
unknown word   →  search name=
known ident    →  resolve symbol=
```

Advanced: `find` enumerates sites; `trace` proves A→B. `contract` / `impact` / `entry` still exist — prefer `resolve` first.

`operation` is a closed enum, default `resolve`:

| operation | 何时 | 选择器 |
|---|---|---|
| `search` | 有词不知 ident：扫源码行 | `name`（短语；可选 `file=`） |
| `resolve` | 身份：一个 ident 或 file:line | `symbol` / `file`（可无 line） / `file`+`line` |
| `find` | 集合：所有 call site / 谓词 / 名字发现 | `kind` + `callee`，或单独 `name` |
| `trace` | A 到 B 有没有 typed path | `from_symbol` + `to_symbol` |
| `contract` / `impact` / `entry` | 旧卡片；dossier 覆盖前保留 | 同一个 seed |

```text
search name=BufferNum     →  path:line + 该行原文（0 = UNKNOWN，改短语）
resolve <其中一个真名>      →  名字 + file:line + 最小源码窗
find kind=OPERATION callee=InitBuffer
file=foo.h                 →  Definition 切到该文件
```

`find name=` 只匹配 `entity.name`；0 命中就是 0，不会扩成 Buffer 家族。再 `resolve` 其中一个真名。

**UNKNOWN：** 该 ident 在本算子不存在。正文 **Dims** 就是本算子维名。不要重试同一 ident。

空 `resolve` 也返回 **Dims**。`legal_key_count` 是编进包的 key 总数。Kernel API（`InitBuffer` / `DataCopyPad` / `SetFlag`）有 Definition 就是底层站点。

`resolve` 的 Definition 已是完整 logical unit。日常不要再跟 `evidence_id` 拉源码窗。

Do not call overview. 不要为每个候选再 `resolve` 三次。`INVALID_QUERY` 的 `did you mean:` 照抄一条重发。

CLI：

```text
ascendc-codemap-mcp query --codemap-id ID --symbol IsRope
ascendc-codemap-mcp query --codemap-id ID --operation search --name BufferNum
ascendc-codemap-mcp query --codemap-id ID --operation find --kind OPERATION --callee Foo
```
