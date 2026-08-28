---
name: index-operator
description: Build an AscendC operator CodeMap. Use when the user asks to index an operator, or codemap_status / discover reports no .uo product.
---

# Index an operator

Use MCP server `ascendc-codemap-mcp`. Tools: `codemap_doctor`, `codemap_index`, `codemap_status`, `codemap_discover`.

## When

No committed CodeMap for this operator + architecture. User asked to **build from scratch**.

If a `.uo` already exists and sources changed, use skill `update-operator` instead.

## Steps

1. Require `project` (operator directory, absolute) and `architecture` (e.g. `arch35`). Do not guess architecture from another tree.
2. Call `codemap_doctor` with those arguments. If `ok` is false, **do not index**. Execute `next_steps` in order:
   - Install LLVM 18 clang if `clang_exe` is empty (`winget install --id LLVM.LLVM --version 18.1.8 -e` on Windows; `apt-get install clang` on Debian). pip `libclang` is not enough.
   - Search Downloads for `Ascend-cann-toolkit_*_linux-*.run`. If missing, ask the user to log into [昇腾下载中心](https://www.hiascend.com/developer/download/community/result?module=cann) and download **CANN Toolkit** (not kernels/nnal). On Windows still get the `linux-x86_64` `.run`.
   - Unpack without running the installer: `python -m ascendc_codemap_mcp cann-extract <run> --dest <codemap-checkout>/_cann/pkg` then the same command with `--fixup`.
   - Re-run `codemap_doctor`. Stop if it is still not `ok`.
3. Call `codemap_index`. This can take minutes. Do not call it automatically on session start. Other queries against this id return `freshness=building` until it finishes.
4. Read `codemap.id` from the result. Call `codemap_status` on that id. A partial product is valid; residuals stay unresolved. Do not patch `.uo` by hand.

## Stop

- Doctor failed: execute `next_steps`, then doctor again. Not an indexing problem until CANN/clang/operator path are fixed. Do not execute the CANN `.run` installer.
- Index `state=failed`: report `error` / `failed_step`. Do not invent graph edges.
