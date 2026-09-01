---
name: query-codemap
description: Query an existing AscendC operator CodeMap. Use when asking who writes/reads a name, why a path is taken, which TilingKey/Dim values exist on this operator, or what a change affects.
---

# Query CodeMap

Identity on every call: `codemap_id`, or `project` + `architecture`. No map → `index-operator`. Stale → `update-operator`. Query reads the committed snapshot only; it never opens working-tree files.

## Pick the tool by what you already have

| You have | Call | Required |
| --- | --- | --- |
| a string, no idea where it lives | `codemap_search` | `pattern` |
| a name | `codemap_trace` | `symbol` |
| a `file:line` | `codemap_source` | `file`, `line` |

The parameter sets do not overlap: `trace` takes no `file`/`line`, `source` takes no `symbol`. Mixing them **drops the illegal parameters** and keeps the legal ones (`trace` keeps `symbol`, `source` keeps `file`/`line`) — it does not guess which you meant.

**If you know the name, call `codemap_trace(symbol=)` and nothing else.** It already contains the definition body, so you do not need a second call to read the lines.

## What each call returns

### `codemap_search(pattern="DT_HIFLOAT8")`

Regex over indexed source lines. Matches enum values, string literals, macros and comments — not just symbol names. `file=` is an optional glob/path filter (`**` matches zero or more directories, so `op_host/**/*.cpp` includes `op_host/foo.cpp`). `kind=` restricts to an entity kind (BUFFER, EVENT, …).

```text
26 matches · 13 source units — complete
Trace for the full picture (writers, guards, consumers)
  trace symbol=queryType   FIELD

CheckAttentionInDtype  op_host/arch35/..._common_regbase.cpp:178-201
  189:    if (queryType == ge::DT_HIFLOAT8) {
SetSplitAxis  op_host/arch35/..._common_regbase.cpp:1657-1719
  1664:      fBaseParams.queryType == ge::DT_FLOAT8_E4M3FN || ...
```

Hits grouped by enclosing function. Use it to obtain a name, then stop searching and switch to `trace`.

### `codemap_trace(symbol="isBn2MultiBlk")`

The default shape, and the one that answers most questions. One closed card:

```text
IsBn2MultiBlk
TILING_KEY
op_host/arch35/..._common_regbase.cpp:1673

Coverage: 11 name matches (first page only; search pattern=isBn2MultiBlk for all) · the rest complete

**Definition**
1673|      fBaseParams.isBn2MultiBlk =
1674|          bnSparseLimit && (fBaseParams.s1 > BN2_MAX_S || ...

Writes  3 of 3, complete
  SetSplitAxis:1673 = bnSparseLimit && (s1 > BN2_MAX_S || s2 > BN2_MAX_S) && …
      reached by DoOpTiling:817
  SetSplitAxis:1693 = false when fBaseParams.dropMaskOuter && isBn2MultiBlk
      reached by DoOpTiling:817
  DoSparse:1099 = false when (isInvalidCol || isInvalidRow) && splitAxis == BN2
      reached by DoOpTiling:819
Guarded by
  !(DoBn2s2Sparse() && blockOuter >= aicNum)  1099

Compiled
- legal=yes · variants=4617 · dim=IsBn2MultiBlk
- values: {0: 4537, 1: 80}
Kernel specialization
- IsRope: {0: 80, 1: 0}
```

`reached by` gives the call order, which is what settles who overrides whom. `Kernel specialization` gives co-occurrence in the compiled key space: `IsRope: {0: 80, 1: 0}` says no built key has both `IsRope=1` and this flag set. When a card names a TILING_KEY dim, `trace dim=<that name>` lists its built values; `trace dim=*` lists every dim on this operator.

A host `virtual` call is not a single target. `Calls` / `Virtual dispatch` lists the empty base and every override (for example `DoSparse:1110 GetSparseUnpadBlockInfo()` → varlen override at `:953` and empty virtual at `normal_regbase.h:95`). Do not treat the first listed definition as the only body.

### `codemap_trace(symbol="DoOpTiling", to_symbol="DoSparse")`

A short **directed** menu of how two names relate, split by family (call / data / control / compile). At most two paths per family. This is relatedness, not the write-complete chain — for writers still `trace symbol=` the interesting hop.

```text
Path  DoOpTiling → DoSparse

call
**1 hop**
1. DoOpTiling  ..._normal_regbase.cpp:815  —CALLS→  DoSparse  ..._normal_regbase.cpp:1077

data
  no path

control  weak
  no path

compile
  no path

Coverage: directed family walk, not all simple paths; 4 nodes enumerated.
Pick a hop name and trace symbol= for its card.
```

A 1-hop READS edge is not the write chain. A `relation=data` miss means that family has no indexed edge, not that no semantic data flow exists. Do not treat one mixed shortest hop as the full answer.

### `codemap_trace(dim="*")`

Lists every compiled dim and its built-value distribution. Use this when you do not yet know the dim name. A guessed name that is **not** a compile dim (for example a tiling-data field) returns the real dim list instead of an empty answer.

### `codemap_trace(dim="IsRope", value="1")`

The compiled legal key space. **This is the compiler's answer, and it outranks anything you could work out by reading dispatch macros.**

```text
**Dim**
- SplitAxis: {0, 1}
- InputDType: {3, 2}
- DTemplateNum: {192}
- IsBn2MultiBlk: {0}
- IsDNoEqual: {1}
- legal_key_count: 224

**Cross**
- InputDType: {0: 0, 1: 0, 2: 112, 3: 112, 4: 0, 5: 0, 6: 0}
```

Read this as: with `IsRope=1` there are 224 built keys; `D` is always 192; `InputDType` is only 2 or 3, so dtype 1 is not built at all; `SplitAxis` is only 0 or 1. Any question shaped "which combinations are actually supported / legal / built / reachable" must come from here, never from reading source.

### `codemap_source(file="…_normal_regbase.cpp", line=1099)`

Reads indexed source at a location. The card is titled as the **enclosing function** that owns most of the asked window: keep that function when `line` is inside it and it still covers the bulk; retitle to the next function when `line` sits in the previous function's tail. Callers follow that identity. `line_end=` crops the snippet.

```text
DoSparse
op_host/arch35/..._normal_regbase.cpp:1077-1149

Source
1077|  ge::graphStatus FlashAttentionScoreGradTilingNormalRegbase::DoSparse()
...

**Called by**
- DoOpTiling @819
```

The returned source is already read — do not open the file again. You do not need `relation=call` to get callers: that is the default on a function source card.

## Narrowing, when you want it

`relation=` accepts `call`, `data`, `control`, `compile`, comma-separated. Omitting it returns the four-family menu (directed, short). It is an optional narrowing of a `trace` card, not the way to get callers — source and default `trace` already include call edges. A narrowed card names what it left out:

```text
Withheld by relation filter: compile, control. Drop relation= to see every family.
```

## Reading completeness

Every section states its own standing, and the three states are distinct:

| Line | Means |
| --- | --- |
| `Writes  3 of 3, complete` | counted, and there are exactly three |
| `Writes  3 of 11 shown` | more exist; page or narrow |
| `Writes  not computed here` | this card did not compute it — ask the symbol card |

`not computed here` is **not** "there are none". A card that says `no resolved caller` under a body it just printed is a defect, not a fact.

## Stop rules

- Two or three `search` calls per question is normal. More than that means you are searching for something you should already have a name for.
- Three to five `trace` calls per question is normal. More than that means you are re-confirming facts a card already stated.
- Do not call `source` to double-check something `trace` reported. The write, its value, its guard and its caller are all on the card, with line numbers.
- Do not read dispatch code to answer a "which combinations are built" question. Call `trace dim=*` to list dims, then `trace dim=<name>` or `trace dim=<name> value=<v>`.

## CLI

```text
ascendc-codemap-mcp query --codemap-id ID --operation search --pattern DT_HIFLOAT8
ascendc-codemap-mcp query --codemap-id ID --operation search --pattern isBn2 --file op_host/**
ascendc-codemap-mcp query --codemap-id ID --operation trace --symbol IsRope
ascendc-codemap-mcp query --codemap-id ID --operation trace --dim *
ascendc-codemap-mcp query --codemap-id ID --operation trace --symbol DoOpTiling --to-symbol DoSparse
ascendc-codemap-mcp query --codemap-id ID --operation source --file op_kernel/k.h --line 10
```
