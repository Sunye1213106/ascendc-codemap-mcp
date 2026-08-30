# Architecture Contract

> **Build for truth. Project for usefulness.**
>
> 构建层追求语义真值，Query 层追求信息密度。内部越精确，外部越简单。

**AscendC CodeMap** 是按 operator + architecture 编译得到的、自包含 AscendC 语义索引。构建阶段将 C++ / CANN DSL 编译成唯一、可信、可追溯的语义事实；`.uo` 是唯一产品真值；Query 把这些事实压缩成面向 AI 的上下文；Agent 只负责理解任务和使用事实。

Query 是 semantic projection / compression，不是查库，也不能重新分析代码。

```text
Source + BuildContext
        → Language Frontends
        → Canonical Compiler Facts
        → Analysis IR (HostIR / KernelIR)
        → AscendC Semantic Facts
        → Verified .uo
        → AI Semantic Projection
        → Agent
```

内部可以有 canonical key、USR、provenance、evidence。Agent 默认看不到它们。Internal identity ≠ Agent vocabulary。

## Freeze

- 不新增 public `operation`，不新增 Agent retry / hint 路由，不特化某个算子的抽取器，不把 rg 塞进 MCP。
- 新能力必须先回答 FactKind（下表）和 semantic domain。答不出就不加。
- Query 不读工作区源码，不做 task routing。
- Agent-facing 默认不暴露内部 join / debug 标识。
- 旧 `.uo` 不保证兼容；产品承诺是重编。

## Layers

| Layer | Owns | Does not |
| --- | --- | --- |
| Language Frontends | Clang AST, TPL DSL, macros, build product, `source_line` | Buffer lifetime, impact |
| Canonical Compiler Facts | One record per FactKind: identity, call, write, control, type, macro | A second identity for the same C++ declaration |
| Analysis IR | HostIR SSA / def-use / path condition; KernelIR TilingKey → constexpr branch | Rediscovering Clang declarations |
| AscendC Semantic Facts | Config, Storage, Lifetime, Sync, Compute, ABI | Guessing physical space from names |
| Verified `.uo` | Persist + integrity | Commit because “it looks queryable” |
| AI Semantic Projection | Compress for the model | Re-analyze source, dump the graph, show internal ids |

Clang AST is not `.uo`. HostIR / KernelIR are not `.uo`. They consume layer-2 facts and produce layer-4 facts that get committed.

## Fact Ownership Matrix

One **canonical owner** per FactKind. Multiple **evidence** producers are allowed. There is one `WRITES` edge, not `Clang_WRITES` + `HostIR_WRITES` + `Source_WRITES`.

| FactKind | Owner | Primary | Fallback | Forbidden |
| --- | --- | --- | --- | --- |
| TYPE / nested `Owner::TYPE` | CompilerFacts | Clang TypeDecl / AliasDecl | Bind existing declaration from source | `SRCPOL::{file}::{leaf}` |
| Call / Write / Control | CompilerFacts | Clang CallExpr / assignment | HostIR as extra evidence | A second WRITES relation |
| TPL declaration | TPL frontend | TPL parser | — | Guess via Clang as C++ |
| registration macro | Macro/DSL frontend | Macro facts | Deterministic macro text | Operator-name special cases |
| compiled legal keys | Build product | Compiled product | — | Recompute at query time |
| source snapshot | source index | `source_line` | — | Query `read_text` of the working tree |
| Host SSA / def-use | HostIR | Consume Clang facts | — | Walk AST again for the same write |
| Kernel TilingKey branch | KernelIR | Clang + TPL | — | compile_policy rediscovering the TYPE |
| DataCopy direction | Storage | CallSite + resolved operand + registry `arg_effect` | — | Variable-name guessing |

Truth floor for `mint_*`: frontend > confirmed AST operand + registry > deterministic fallback with provenance > heuristic (candidate only). **Identity unresolved → no confirmed semantic edge.**

Later stages may add evidence. They must not rewrite a canonical identity.

## AI Projection Contract

Agent-facing output must not expose identifiers used only for join, dedup, provenance, or debug:

`entity_id`, `relation_id`, `evidence_id`, USR, canonical key, raw attrs, pass names, trust internals, source hash.

Ambiguity uses human coordinates: name + `file:line`, then `resolve file= line=`. Graph ids live in the database, not in the daily protocol.

Output order:

1. identity for human (name + file:line)
2. semantic answer
3. decisive source evidence
4. compact counts
5. detail only when requested (`projection=locations`)

`projection`:

| Value | Role |
| --- | --- |
| `summary` | Default. ~300–800 tokens. Highest-value 3–5 facets, aggregated. |
| `source` | summary + the smallest statement window that carries the meaning |
| `locations` | Exhaustive sites |

Happy path: unknown symbol → `search`; known → `resolve`. `find` enumerates; `trace` proves A→B. `contract` / `impact` / `entry` stay until a benchmark shows the dossier covers them.

Query snippets come from `.uo` `source_line` / `source_span`. Missing → `source unavailable in snapshot`.

## Semantic domains

Control/Config · Storage/Memory · Lifetime/Sync · Compute · ABI/Product.

Storage model: `Tensor/View --BACKED_BY--> StorageOwner --physical_space--> GM|UB|L1|L0*|REG`. Space comes from `BufferType` / `TPosition` / `QuePosition` / confirmed InitBuffer operands, not from the word `LocalTensor`.

## Hard gates

| Gate | Target |
| --- | --- |
| Ordinary `resolve(summary)` / `search` internal ids | 0 |
| Query working-tree reads | 0 |
| Empty facets | omitted |
| Duplicate semantic fact in one card | 0 |
| Default resolve size | < 800 tokens when possible |
| Confirmed false semantic fact | 0 on commit (integrity) |

Entity/relation counts are an internal implementation detail, not a product success metric.
