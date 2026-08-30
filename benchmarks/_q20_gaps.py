# -*- coding: utf-8 -*-
"""Fill remaining holes with targeted UO queries."""
from __future__ import annotations

import sqlite3
from pathlib import Path

UO = Path(
    r"d:\TEST\ops-transformer\attention\flash_attention_score_grad"
    r"\.ascendc-codemap\arch35\FlashAttentionScoreGrad.arch35.uo"
)
OUT = Path(r"d:\TEST\ascendc-codemap-mcp\benchmarks\_q20_gaps.txt")


def connect():
    c = sqlite3.connect(str(UO))
    c.row_factory = sqlite3.Row
    return c


def fts(c, q, limit=10):
    quoted = '"' + q.replace('"', " ") + '"'
    tot = c.execute("SELECT COUNT(*) FROM source_fts f WHERE f.source_fts MATCH ?", (quoted,)).fetchone()[0]
    rows = c.execute(
        """
        SELECT sl.path, sl.line, sl.text
        FROM source_fts f JOIN source_line sl ON sl.id = f.rowid
        WHERE f.source_fts MATCH ?
        ORDER BY CASE WHEN sl.path LIKE '%/arch35/%' OR sl.path LIKE '%flash_attention_score_grad_tiling.cpp%' THEN 0 ELSE 1 END,
                 sl.path, sl.line
        LIMIT ?
        """,
        (quoted, limit),
    ).fetchall()
    return tot, rows


def win(c, path, line, radius=10):
    leaf = path.replace("\\", "/").split("/")[-1]
    rows = c.execute(
        """
        SELECT path, line, text FROM source_line
        WHERE REPLACE(path,'\\','/') LIKE '%' || ? AND line BETWEEN ? AND ?
        ORDER BY line
        """,
        (leaf, max(1, line - radius), line + radius),
    ).fetchall()
    return rows


def dump(c, out, title, q, limit=8, radius=8):
    tot, rows = fts(c, q, limit)
    out.write(f"\n## {title}  FTS {q!r} total={tot}\n")
    for r in rows:
        path = (r["path"] or "").replace("\\", "/")
        out.write(f"  {path}:{r['line']}: {(r['text'] or '').rstrip()[:160]}\n")
    if rows:
        r0 = rows[0]
        out.write(f"  WINDOW {r0['path']}:{r0['line']}\n")
        for w in win(c, r0["path"], r0["line"], radius):
            out.write(f"    {w['line']}| {(w['text'] or '').rstrip()[:160]}\n")


def main():
    c = connect()
    with OUT.open("w", encoding="utf-8") as out:
        dump(c, out, "Q1 RunEmptyTiling", "RunEmptyTiling", 8, 12)
        dump(c, out, "Q1 RunEmptyTilingRegbase", "RunEmptyTilingRegbase", 6, 15)
        dump(c, out, "Q1 GetShapeSize 0", "GetShapeSize() == 0", 8, 8)
        dump(c, out, "Q1 FAG_EMPTY", "FAG_EMPTY_TILING_KEY", 6, 6)
        dump(c, out, "Q2 isSeqExistZero assign", "isSeqExistZero = true", 6, 12)
        dump(c, out, "Q3 keepProb 1 drop", "keepProb == 1", 10, 12)
        dump(c, out, "Q3 dropMask optional", "dropMask", 6, 4)
        dump(c, out, "Q4 set_s1Inner", "set_s1Inner", 8, 8)
        dump(c, out, "Q4 s1Inner =", "s1Inner =", 8, 8)
        dump(c, out, "Q7 DoPostTiling", "DoPostTiling", 8, 20)
        dump(c, out, "Q7 FP32 post", "DT_FLOAT", 6, 4)
        dump(c, out, "Q9 isNzOut", "isNzOut", 8, 12)
        dump(c, out, "Q9 D 96", "Aligned96", 4, 6)
        dump(c, out, "Q12 GqaDense def", "GqaDense", 8, 15)
        dump(c, out, "Q12 g == 1 swizzle", "g == 1", 8, 8)
        dump(c, out, "Q18 FagTilingWithTemplate", "FagTilingWithTemplate", 6, 8)
        dump(c, out, "Q18 entry GetTiling", "needDeterPrefix", 4, 10)
        dump(c, out, "Q19 SetScheduleMode impl", "SetScheduleMode", 6, 12)
        dump(c, out, "Q19 isBn2MultiBlk =", "isBn2MultiBlk", 6, 12)

        # Q1: who calls RunEmpty from DoTiling
        tot, rows = fts(c, "RunEmptyTilingRegbase", 20)
        out.write("\n## Q1 all RunEmptyTilingRegbase hits\n")
        for r in rows:
            out.write(f"  {(r['path'] or '')}:{r['line']}: {(r['text'] or '').rstrip()[:180]}\n")

        # Q3 host checks involving keepProb in tiling_common
        tot, rows = fts(c, "keepProb", 30)
        out.write("\n## Q3 keepProb hits in op_host arch35 (first 20)\n")
        n = 0
        for r in rows:
            p = (r["path"] or "").replace("\\", "/")
            if "op_host" in p and "arch35" in p:
                out.write(f"  {p}:{r['line']}: {(r['text'] or '').rstrip()[:180]}\n")
                n += 1
                if n >= 20:
                    break

        # Q9 NzOut assignment
        tot, rows = fts(c, "fBaseParams.isNzOut", 10)
        out.write("\n## Q9 fBaseParams.isNzOut\n")
        for r in rows:
            out.write(f"  {(r['path'] or '')}:{r['line']}: {(r['text'] or '').rstrip()[:180]}\n")
            for w in win(c, r["path"], r["line"], 12):
                out.write(f"    {w['line']}| {(w['text'] or '').rstrip()[:160]}\n")
            break

        # Q10 GET_IS_L1
        dump(c, out, "Q10 GET_IS_L1_PRELOAD", "GET_IS_L1_PRELOAD", 4, 14)
        dump(c, out, "Q10 GET_IS_L1_REUSE", "GET_IS_L1_REUSE", 4, 14)
        dump(c, out, "Q10 IS_SMALL_D_PRELOAD", "IS_SMALL_D_PRELOAD", 4, 10)

        # Q14 host policy function
        dump(c, out, "Q14 ChooseBand", "deterBandScheduleMode =", 8, 20)

        # Q17 hasRope NUM192
        dump(c, out, "Q17 hasRope NUM192", "hasRope", 6, 8)
        tot, rows = fts(c, "NUM192", 8)
        out.write("\n## Q17 NUM192 all\n")
        for r in rows:
            out.write(f"  {(r['path'] or '')}:{r['line']}: {(r['text'] or '').rstrip()[:180]}\n")

        # Q6 enablePreSfmg assignment window
        dump(c, out, "Q6 enablePreSfmg assign", "enablePreSfmg =", 6, 16)

        # Q7 DoPostTiling definition
        rows = win(c, "flash_attention_score_grad_tiling_normal_regbase.cpp", 1671, 40)
        out.write("\n## Q7 DoPostTiling body around 1671\n")
        for w in rows:
            out.write(f"    {w['line']}| {(w['text'] or '').rstrip()[:160]}\n")

        # Q18 entry_regbase 217
        rows = win(c, "flash_attention_score_grad_entry_regbase.h", 217, 12)
        out.write("\n## Q18 entry_regbase.h ~217\n")
        for w in rows:
            out.write(f"    {w['line']}| {(w['text'] or '').rstrip()[:160]}\n")

        # Q19 SetScheduleMode around 1933
        rows = win(c, "flash_attention_score_grad_tiling_normal_regbase.cpp", 1933, 10)
        out.write("\n## Q19 SetScheduleMode ~1933\n")
        for w in rows:
            out.write(f"    {w['line']}| {(w['text'] or '').rstrip()[:160]}\n")

        rows = win(c, "flash_attention_score_grad_tiling_common_regbase.cpp", 1673, 15)
        out.write("\n## Q19 isBn2MultiBlk ~1673\n")
        for w in rows:
            out.write(f"    {w['line']}| {(w['text'] or '').rstrip()[:160]}\n")

        # Q13 DETER mapping
        dump(c, out, "Q13 DETER_CAUSAL assign", "DETER_CAUSAL", 10, 14)

        # Q20 GET_TPL window
        rows = win(c, "flash_attention_score_grad_tiling_normal_regbase.cpp", 1892, 12)
        out.write("\n## Q20 GET_TPL_TILING_KEY ~1892\n")
        for w in rows:
            out.write(f"    {w['line']}| {(w['text'] or '').rstrip()[:160]}\n")

        rows = win(c, "flash_attention_score_grad_template_tiling_key.h", 49, 80)
        out.write("\n## Q20 ARGS_DECL 49-130\n")
        for w in rows:
            out.write(f"    {w['line']}| {(w['text'] or '').rstrip()[:160]}\n")

    print("wrote", OUT, OUT.stat().st_size)


if __name__ == "__main__":
    main()
