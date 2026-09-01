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

The parameter sets do not overlap: `trace` takes no `file`/`line`, `source` takes no `symbol`. Mixing them returns `INVALID_QUERY` with the legal filter list.

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

Coverage: 11 definition sites · the rest complete

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

`reached by` gives the call order, which is what settles who overrides whom. `Kernel specialization` gives co-occurrence in the compiled key space: `IsRope: {0: 80, 1: 0}` says no built key has both `IsRope=1` and this flag set.

### `codemap_trace(symbol="DoOpTiling", to_symbol="DoSparse")`

Shortest relation path between two names — call chain, value propagation, or guard reachability.

```text
Path  DoOpTiling → DoSparse

**1 hop**
1. DoOpTiling  ..._normal_regbase.cpp:815  —CALLS→  DoSparse  ..._normal_regbase.cpp:1077

Coverage: breadth-first over 14 relation kinds; 4 nodes enumerated.
Shortest path by hop count, not the only one.
```

No path is reported as either `NO_PATH` (walked to exhaustion, none exists) or `SEARCH_BUDGET` (hit the bound). They are different answers; do not read the second as the first.

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

Reads indexed source at a location, plus the state changes, branches and tiling fields of that unit. `line_end=` takes a range, which is how you continue a body that was cut.

```text
DoSparse
op_host/arch35/..._normal_regbase.cpp:1077-1149
Coverage: the unit around …:1099 · state changes and fields below are every one
in this unit; definition sites, callers and guards for DoSparse are on its
symbol card — trace symbol=DoSparse

Source
1077|  ge::graphStatus FlashAttentionScoreGradTilingNormalRegbase::DoSparse()
...
```

The returned source is already read — do not open the file again. Callers and definition sites are not computed here; they belong to a name.

## Narrowing, when you want it

`relation=` accepts `call`, `data`, `control`, `compile`, comma-separated. Omitting it walks every family, so **the shortest call is also the complete one**. A narrowed card names what it left out:

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
- Do not read dispatch code to answer a "which combinations are built" question. Call `trace(dim=, value=)`.

## CLI

```text
ascendc-codemap-mcp query --codemap-id ID --operation search --pattern DT_HIFLOAT8
ascendc-codemap-mcp query --codemap-id ID --operation search --pattern isBn2 --file op_host/**
ascendc-codemap-mcp query --codemap-id ID --operation trace --symbol IsRope
ascendc-codemap-mcp query --codemap-id ID --operation trace --symbol DoOpTiling --to-symbol DoSparse
ascendc-codemap-mcp query --codemap-id ID --operation source --file op_kernel/k.h --line 10
```
