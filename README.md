# AscendC CodeMap MCP

面向 AI coding agent 的 AscendC 算子语义图。把一个算子在指定 architecture 下的 **Host / TilingData / TilingKey / Template / Kernel / AscendC API** 编进持久化 `.uo` 图，再通过 MCP 做低延迟查询。

这是**语义编译图**，不是通用代码记忆。`.uo` 只存代码*是什么*。Agent 的计划、review、ADR 留在会话或调用方（例如 AscendC-Pilot）。

对 SFAG 算子的本地实测：

| 操作 | 耗时 |
| --- | --- |
| 首次构建 CodeMap | 约 1 分钟 |
| 单次查询 | 约 20 ms |

一次构建，之后反复查询，避免每次重新 grep 和通读源码。

构建需要 CANN Toolkit 头文件和 LLVM clang。查询已有 `.uo` 不依赖 CANN。环境不会自动配好：先 `codemap_doctor`，按 `next_steps` 下载 `.run` 并用 `cann-extract` 解包。

## 它解决什么

AscendC 算子逻辑通常跨多层传递：

```text
Host → TilingData → TilingKey / Template → Kernel → AscendC API
```

Kernel 里看到的行为，往往由 Host 条件决定，再经 TilingData、TilingKey 和模板参数一路传下来。普通搜索能回答「这个名字出现在哪」，很难直接回答：

- 谁写入了这个 TilingData 字段？哪个 Kernel 读取了它？
- 哪条 TilingKey / Template 控制这条路径？某个模板组合是否合法？
- 这段 Kernel 最终落到哪个 AscendC API？
- Buffer、Queue、Pipe、Event 之间是什么关系？

CodeMap 把这些关系提前提取进图，让 agent 按标识符、Dim、证据位点查询。

## 和 Codebase Memory / CodeGraph

Codebase Memory 与 CodeGraph 是通用代码图：多语言、通用导航。CodeMap 只做 **AscendC 算子领域语义**。

| | Codebase Memory / CodeGraph | AscendC CodeMap |
| --- | --- | --- |
| 目标 | 通用代码理解 | AscendC 算子理解 |
| 解析 | 主要 Tree-sitter | Clang + CANN 编译上下文 |
| 语言 | 多语言 | C++ / AscendC |
| Call graph | 有 | 有 |
| Template 语义 | 通用 | 重点建模 |
| TilingData / TilingKey | — | 有 |
| Host → Kernel 数据流 | — | 有 |
| Buffer / Queue / Pipe / Event | — | 有 |
| Architecture / 编译条件 | 通用 | 按 arch 产品槽 |

他们更擅长「代码在哪、谁调用谁」。CodeMap 更希望回答：**这个算子为什么走到这个 Kernel，以及这个值怎么从 Host 传下来。**

本仓库不把 memory / ADR / 文档检索、默认 Cypher/SQL、impact 分析或可视化做成产品功能。

## 为什么用 Clang

Tree-sitter 适合快速、多语言的语法树。AscendC 大量依赖 C++ 编译语义，例如：

```cpp
template <bool IsPse, typename T>
__aicore__ inline void Process(...)
```

只看语法很难完整恢复模板参数与实例化、类型、宏、architecture 分支、Host 字段写入、TilingKey 选择和 Kernel specialization。因此构建路径是 **libclang + 实际 CANN 环境**，再叠加 AscendC 专用分析。

## 当前建模的语义

**Host / Tiling：** Function、Method、Branch、Predicate、Input、Output、TilingData、Field、Host READ/WRITE、Compile Macro / Variable。

**Template / TilingKey：** TilingKey、Template、Template Argument / Instance、Build Variant、Binding / Specialization / Selection。

**Kernel：** Kernel、Operation、Buffer、Register、Queue、Pipe、Event、AscendC API。

边上常见：

```text
READS / WRITES    CALLS    BINDS    SELECTS    LAUNCHES
GUARDED_BY        FLOWS_TO    SAVES / RESTORES    SIGNALS / AWAITS
```

串起来仍是：Host 条件 → TilingData 字段 → TilingKey / Template → Kernel → AscendC API。

## 身份与新鲜度

索引之后用稳定 id，不要每次贴绝对路径：

```text
codemap_id   = p:<workspace>::op_name@arch     例如 p:a91f42::flash_attention_score_grad@arch35
alias        = op_name@arch                    本进程内唯一时可用；撞车返回 AMBIGUOUS_CODEMAP_ID
snapshot_id  = cm:<digest 前缀>                已提交图的内容身份，不是 path/mtime
```

`codemap_status` 看的是相对当前源码的 **freshness**，不是「磁盘上有没有 `.uo`」：

`fresh` / `dirty` / `stale` / `building` / `blocked` / `incompatible` / `unknown`

读写契约：

- `ok` 只表示这次调用在协议层成功。更新是否发生看 `state` 和 `updated`。
- `codemap_update` 得到 `state=needs_confirmation` 时图**没有**前进，需用户确认后再 `confirm_scope=true`。
- 已有 `.uo` 时再调 `codemap_index` 会 `ALREADY_INDEXED`（`updated=false`），应改用 `codemap_update`。
- 正在构建时，查询返回 `freshness=building`，不会读半写文件。

## 构建环境（Clang + CANN）

查询已有 `.uo` **不需要** CANN / Clang。冷构建（`codemap_index`）需要三者齐备：

1. Python ≥ 3.10 与本仓库（`pip install -e .`）
2. **LLVM 18 clang 可执行文件** + pip `libclang`（只有 Python 绑定不够，TPL 预处理要跑 `clang -E`）
3. **CANN Toolkit 头文件**（解包 `.run` 即可，不必在本机完整安装 Toolkit，也不需要 NPU）

Agent 先跑 doctor，再按返回的 `next_steps` 配环境，不要猜路径：

```bash
pip install -e .
python -m ascendc_codemap_mcp doctor --project <算子目录> --architecture arch35
```

`ok=false` 时执行 `next_steps`，再跑一次 doctor。不要执行 `.run` 安装脚本。

### Clang / libclang

Ubuntu / Debian：

```bash
sudo apt-get update
sudo apt-get install -y clang
python -c "import clang.cindex as c; print(c.__file__)"
clang --version
```

Windows（LLVM 18 与 pip `libclang` 18.x 对齐）：

```powershell
winget install --id LLVM.LLVM --version 18.1.8 -e
python -c "import clang.cindex as c; print(c.__file__)"
clang --version
```

`clang` 不在 PATH 时：

```powershell
$env:CLANG_EXE = "C:\Program Files\LLVM\bin\clang.exe"
```

也可设 `UO_CLANG` / `LLVM_HOME`。

### 下载 CANN Toolkit `.run`

CodeMap 只要 **Toolkit 开发套件**里的头文件。不要下 kernels / nnal 来代替 Toolkit。

1. **先搜本机**，避免重复下载（包很大）：

```powershell
Get-ChildItem -Path "$HOME\Downloads","D:\Downloads","$HOME" -Filter "Ascend-cann-toolkit_*.run" -ErrorAction SilentlyContinue
```

```bash
ls ~/Downloads/Ascend-cann-toolkit_*.run 2>/dev/null
```

2. 没有文件时，打开昇腾社区下载中心（**需要华为账号登录**；社区包的直链通常带签名，**未登录 wget 会失败**）：

- 社区版 CANN：<https://www.hiascend.com/developer/download/community/result?module=cann>
- 软件页：<https://www.hiascend.com/software/cann>
- 入口：<https://www.hiascend.com/developer/download>

3. 页面上选 **CANN Toolkit**，操作系统 **Linux**，架构：

| 本机 | 下载哪个 `.run` |
| --- | --- |
| Windows / Linux x86_64 | `Ascend-cann-toolkit_<version>_linux-x86_64.run` |
| Linux aarch64 | `Ascend-cann-toolkit_<version>_linux-aarch64.run` |

Windows 也下 **linux-x86_64** 包：`cann-extract` 只解出文件树，**不会执行** installer。

版本尽量与算子真实编译环境一致。部分社区版安装文档会给出 `ascend-repo.obs.cn-east-2.myhuaweicloud.com` 的 wget；仅当该版本文档写明 URL 时再用，不要随便抓一个旧包。

已有官方安装（`source set_env.sh`）则把 `ASCEND_HOME_PATH` 指到安装前缀，不必再解 `.run`。

### 解包 CANN（不要跑 installer）

推荐解到本仓库 `_cann/pkg`，doctor 会自动发现，不必设环境变量：

```powershell
$pkg = Join-Path (Get-Location) "_cann\pkg"
python -m ascendc_codemap_mcp cann-extract `
  "D:\Downloads\Ascend-cann-toolkit_<version>_linux-x86_64.run" `
  --dest $pkg
python -m ascendc_codemap_mcp cann-extract --fixup --dest $pkg
```

Linux：

```bash
pkg="$(pwd)/_cann/pkg"
python -m ascendc_codemap_mcp cann-extract \
  ~/Downloads/Ascend-cann-toolkit_<version>_linux-x86_64.run \
  --dest "$pkg"
python -m ascendc_codemap_mcp cann-extract --fixup --dest "$pkg"
```

等价写法：`python scripts/cann_extract.py ...`（checkout 内、尚未 `pip install` 时）。

解完应能看到：

```text
_cann/pkg/
├── cann-metadef/
├── cann-asc-devkit/
├── cann-opbase/
├── cann-npu-runtime/
├── cann-ge-compiler/
└── bisheng/          # 若包内有
```

`ASCENDC_CODEMAP_CANN_ROOT` 指 package **根**（上面这一层），不要指到某个 `include/`。解到别处时设用户级环境变量，只写当前会话会丢。

Windows：

```powershell
[Environment]::SetEnvironmentVariable("ASCENDC_CODEMAP_CANN_ROOT", "<abs-pkg>", "User")
```

Linux：

```bash
echo 'export ASCENDC_CODEMAP_CANN_ROOT=/abs/path/to/_cann/pkg' >> ~/.bashrc
```

`--fixup` 会补 `asc/impl/include` → `asc/include`（官方包不带这个目录，vanilla clang 需要这个 junction）。

## 安装 MCP

```bash
pip install -e .
python -m ascendc_codemap_mcp doctor --project <算子目录> --architecture arch35
python -m ascendc_codemap_mcp install
```

Windows：

```powershell
pip install -e .
python -m ascendc_codemap_mcp install
```

`install` 为已检测到的客户端写入本产品的 MCP 条目和 skills，不改无关 server、不打开 YOLO。`uninstall` 只删除本产品写入的内容。装完后重启 coding agent。工具出现在 MCP server `ascendc-codemap-mcp`。

协议是官方 Python SDK 的 **2026-07-28**（含 stateless `server/discover`），同一进程仍服务握手期客户端。

不要在 MCP 连接时自动 index。冷构建可能要几分钟。

## 构建与更新

```bash
ascendc-codemap-mcp index --project <算子目录> --architecture arch35
```

产物：

```text
<operator>/.ascendc-codemap/<arch>/<op>.<arch>.uo
```

源码变化后增量刷新（优先 `codemap_id`）：

```bash
ascendc-codemap-mcp update --codemap-id <id>
ascendc-codemap-mcp status --codemap-id <id>
```

`index` 在 prepare / extract / analyze / commit 之间可取消。`update` 在 detect / plan / 各层 rebuild / commit 之间可取消。某一步内部的 Clang 会跑完该步。取消是**每次调用一份 token**，不是按 CodeMap 共享。

## 查询

CLI：

```bash
ascendc-codemap-mcp discover --project <算子目录>
ascendc-codemap-mcp query --codemap-id <id> IsPse
```

Agent 用 typed 工具（`symbol` 必须是一个标识符，不要塞自然语言句子）：

| 意图 | 工具 |
| --- | --- |
| 扫目录、拿到 `codemap.id` | `codemap_discover` |
| 新鲜度 | `codemap_status` |
| 这张图能回答什么 | `codemap_overview` |
| 标识符定义 / writers / readers | `codemap_symbol` |
| Dim 合法集 / `Name=Value` | `codemap_selection` |
| 从卡片继续 | `codemap_evidence`（`evidence_id` + `expected_snapshot_id`） |
| 构建前检查 | `codemap_doctor` |
| 冷构建 | `codemap_index` |
| 增量刷新 | `codemap_update` |

兼容别名：`query_codemap`、`index_operator`、`update_operator`。只读工具带 `readOnlyHint`。查询结果走统一 envelope（`ok`、`codemap`、`verdict`、`layer`、`data`、`evidence`、`coverage`、`next_cursor`）和 `structuredContent`。

跟 `evidence[].id`（`span:...`）走，并把当时的 `snapshot_id` 当作 `expected_snapshot_id`；对不上是 `SNAPSHOT_CHANGED`。`coverage.truncated` 且带了 `next_cursor` 再翻页；nested neighbor 样本只标 `nested_truncated`，不是假翻页。`count: 0` 不等于「图上没有」，跟 `hint`。

回答所问的层：Host 写出 ≠ 模板可编译 ≠ Kernel 消费。不要把 LLM 补丁写进 `.uo`，不要对产品图跑原始 SQL/Cypher。

## 资源与传输

| 类型 | 名称 | URI / 参数 |
| --- | --- | --- |
| Resource | runtime | `codemap://runtime` |
| Resource template | 一张图的身份与新鲜度 | `codemap://map/{codemap_id}` |
| Prompt | `query_operator` | `codemap_id`，可选 `focus` |
| Prompt | `build_codemap` | `project`、`architecture` |

`codemap_id` / `architecture` 的补全来自本进程 registry 和常见 arch 名。

默认 stdio。同一 SDK server 也可 HTTP（这是传输，不是远程图服务）：

```bash
ascendc-codemap-mcp serve --transport streamable-http --host 127.0.0.1 --port 8765
```

`install` 覆盖 Cursor、Claude Code、Codex、OpenCode。Cursor / Claude Code 只写 skills；Codex / OpenCode 另写 `AGENTS.md` 片段。

## 环境变量

| 变量 | 作用 |
| --- | --- |
| `ASCENDC_CODEMAP_CANN_ROOT` | 解包后的 CANN 根（还可回退 `ASCEND_CANN_PACKAGE_PATH`、`ASCEND_HOME_PATH`、`CANN_ROOT`）。解到 `<checkout>/_cann/pkg` 时不必设 |
| `CLANG_EXE` / `UO_CLANG` / `LLVM_HOME` | clang 可执行文件；仅 pip `libclang` 不够 |
| `ASCENDC_CODEMAP_CACHE_DIR` | 缓存目录，默认 `~/.cache/ascendc-codemap-mcp` |
| `ASCENDC_CODEMAP_PROJECT` / `ASCENDC_CODEMAP_ARCHITECTURE` | 本进程 discover 之后的默认身份；也作 CLI 默认 |
| `ASCENDC_CODEMAP_MAX_OPEN` | 打开的查询句柄 LRU 上限，默认 4 |

Codex 子进程只转发 `env_vars` 里列出的名字；`install` 会带上 CANN / cache 相关变量。

Windows 可以解 linux-x86_64 `.run` 再构建；查询已有 `.uo` 不依赖 CANN。

## License

MIT
