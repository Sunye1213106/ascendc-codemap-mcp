# Canonical facts next：before / after

对照 [ses_faf0](../../ops-transformer/attention/session-ses_faf0.md)（源码基线）与 [ses_faf2](../../ops-transformer/session-ses_faf2.md)（CodeMap 辅助）。本轮不改 Skill 去强迫 MCP；目标是 **CodeMap-assisted correctness ≥ source baseline**，再在 compiled product / storage lineage / Host→Kernel 上形成 Unique Win。

评测包：`flash_attention_score_grad` arch35 `.uo`。Commit 1 之后旧包只作 before 对照；invariant probe 要求重编后的包。

## 1. 假 identity（using TYPE 当 FIELD / `METHOD constexpr`）

| | Before (faf2 / 旧 .uo) | After |
| --- | --- | --- |
| `QL1BuffSelector::TYPE` | FIELD + 假契约 `get_TYPE` + `PRODUCER_MISSING` | `bind_or_create` 唯一入口；`using TYPE` **禁止** mint FIELD；契约 facet 不对 TYPE/裸 FIELD 打 `PRODUCER_MISSING` |
| `METHOD constexpr` | lexical `if constexpr (` 被当成函数名，吸收大量 CALLS | `_FUNC_DEF_RE` + `is_forbidden_callable_name`；METHOD/FUNCTION 关键词直接拒绝 mint |
| Integrity | 假 identity 能 commit | `KEYWORD_CALLABLE_NAME` / `DUPLICATE_DECLARATION` 进 integrity gate |

**Unique Win：** 否（这是正确性还债，不是相对 grep 的新能力）。

## 2. 重复 CALLS

| | Before | After |
| --- | --- | --- |
| `relation_id` | CALLS 拼 call-site，同一 `(src,dst)` 多条 confirmed 边 | 拓扑唯一 `(kind, src, dst)`；多 site 进 `attrs.sites[]` |
| Integrity | 重复 triple 能 commit | `DUPLICATE_CONFIRMED_TRIPLE` 拦 commit |

**Unique Win：** 否。

## 3. Storage owner / BACKED_BY

| | Before | After |
| --- | --- | --- |
| BACKED_BY 终点 | 392 条指向 `AscendC::LocalTensor` catalog TYPE，0 条 BUFFER | 必须是 TBuf/TQue/InitBuffer 分配的 BUFFER/QUEUE/REGISTER |
| 类型关系 | 复用 BACKED_BY | `INSTANCE_OF` = template specialization / C++ view type |
| 黄金空间 | 变量名猜 UB | `queryGm→GM`、`qL1Buffer→L1`、L0A、VReg→REG 走 `BufferType`/`TPosition`/`allocated` |
| DataCopy | 单端 operand | CallExpr + 双端 operand → `FLOWS_TO via=MemoryTransfer` |

**Unique Win：** **是（storage lineage）** — grep 看不到 GM/UB/L1/L0/REG 物主，这是图上的语义边。旧 .uo 未重编前探针会失败。

## 4. Agent DTO / evidence 冗余

| | Before (faf2) | After |
| --- | --- | --- |
| 日常路径 | resolve 后约 19 次 `codemap_evidence`，snippet 在 seeds/enclosing/hits 重复 | DTO：`name/kind/file/line/summary/source/facets/counts`；`projection=source` 整条 logical unit（`using` / constexpr / assignment），不再 3 行截断 |
| Skill | （历史）语义问题去 evidence | **只删鼓励**；写明 Definition 已是完整 unit。不强迫用 CodeMap |
| `codemap_evidence` | 服务保留 | 服务保留 |

**Unique Win：** 否（降低假动作，不是相对源码的新事实）。若某题仍要 evidence 补窗，标 **当前 CodeMap 尚未形成 semantic advantage**。

## 5. Compiled support（RoPE / DTemplate）

| | Before | After |
| --- | --- | --- |
| `resolve(hasRope)` | 只走 name card，不 join `legal_key`；cover 才要 `dim=` | 在**已持久化** key space 上投影：Host encoding ↔ Kernel dim ↔ `legal`/`variants`；RoPE 反事实 `DTemplate=128, IsRope=1 → legal=no` |
| Query 重解析模板 | 有风险 | **禁止**；只读 `legal_key_dim` |

faf0 用 grep 扫 `IsRope,1` 白名单，源码略胜。这是本应属于 CodeMap 的 Unique Win。

**Unique Win：** **是（compiled product / Host→Kernel）** — 前提是 resolve 一次能闭合；golden 与 faf0 同证明强度、1–3 次 query。

## 6. find 穷尽 + trace COMPLETE

| | Before | After |
| --- | --- | --- |
| `find kind=OPERATION callee=InitBuffer` | 正文 `showing 8 of 129` 且 `next_cursor=null` | envelope：`returned` / `total` / `exhaustive`；未穷尽必有 `next_cursor` |
| `query_trace` | `COMPLETE if steps else COMPLETE`（空 hops 也 COMPLETE） | `COMPLETE ⇒ hops≥1 且每 hop 有 relation`；否则 UNKNOWN |

**Unique Win：** 否（诚实度）。穷尽 call-site 相对 grep 是便利，不是语义优势。

## 7. Unique Win 总表（相对源码基线）

主序：correctness → false facts consumed → proof completeness → calls/tokens。

| 题类 | CodeMap Unique Win | 说明 |
| --- | --- | --- |
| 编进包的 key 是否合法（RoPE × D=128） | **YES** | compiled support facet |
| Host encoding ↔ Kernel specialization | **YES** | 同一张 resolve 卡 |
| Q/K/Dy buffer 物主与 L1/L0/REG | **YES**（重编后） | BACKED_BY → storage object |
| Sync / ABI / 纯找源码 | **NO** | 「只是帮找到源码」≠ Unique Win；faf0/faf2 四道共享题结论基本平手 |
| dropout / 4-buffer 策略 | 当前尚未形成 semantic advantage | 正确性修好后应用 compiled + storage 再评，不靠 Prompt 掩盖 |

活体同题 A/B（同模型、同 revision、同 arch35）需在 **重编 FAG .uo 之后** 新开 session。本报告的 after 列是代码不变量与 fixture/toy compile 验收，不是新 session 的 calls/token 数。

## 测试

- `tests/test_canonical_facts.py`：identity / CALLS sites / BACKED_BY / DTO / find envelope / trace hops / `resolve(hasRope)` golden / toy `compile_codemap` / FAG probe（缺文件 skip）
- Skill：禁止「去 evidence」鼓励语；Definition = 完整 logical unit
