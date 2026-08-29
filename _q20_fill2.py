# -*- coding: utf-8 -*-
import sqlite3
from pathlib import Path

UO = r"d:\TEST\ops-transformer\attention\flash_attention_score_grad\.ascendc-codemap\arch35\FlashAttentionScoreGrad.arch35.uo"
OUT = Path(r"d:\TEST\ascendc-codemap-mcp\_q20_fill2.txt")
c = sqlite3.connect(UO)
c.row_factory = sqlite3.Row


def win(substr, line, r=25):
    rows = c.execute(
        """SELECT line, text FROM source_line
           WHERE REPLACE(path,'\\','/') LIKE ?
             AND line BETWEEN ? AND ?
           ORDER BY line""",
        ("%" + substr, max(1, line - r), line + r),
    ).fetchall()
    return rows


def dump(out, title, substr, line, r=25):
    out.write(f"\n==== {title} {substr}:{line} ====\n")
    rows = win(substr, line, r)
    if not rows:
        out.write("EMPTY\n")
        return
    for w in rows:
        out.write(f"{w['line']}| {(w['text'] or '').rstrip()[:220]}\n")


with OUT.open("w", encoding="utf-8") as out:
    dump(out, "Q3 keepProbPtr CheckParams", "flash_attention_score_grad_tiling.cpp", 305, 45)
    dump(out, "Q3 keepProb < 1 host", "flash_attention_score_grad_tiling_common_regbase.cpp", 1075, 50)
    dump(out, "Q3 hasDrop dropMask", "flash_attention_score_grad_tiling_common_regbase.cpp", 1163, 55)
    dump(out, "Q3 aclnn notHasDropMask", "aclnn_flash_attention_score_grad.cpp", 252, 20)
    dump(out, "Q3 aclnn dropMask when keepProb", "aclnn_flash_attention_score_grad.cpp", 319, 70)
    dump(out, "Q4 FuzzyForBestSplit", "flash_attention_score_grad_tiling_normal_regbase.cpp", 1906, 40)
    dump(out, "Q4 s1Inner NUM_TWO", "flash_attention_score_grad_tiling_normal_regbase.cpp", 1320, 45)
    dump(out, "Q4 CUBE_BASEM def", "flash_attention_score_grad_block_cube.h", 132, 25)
    dump(out, "Q4 FP32 D>256", "flash_attention_score_grad_tiling_common_regbase.cpp", 860, 30)
    dump(out, "Q7 entry Post", "flash_attention_score_grad_entry_regbase.h", 88, 55)
    dump(out, "Q7 BN2 IMPL", "flash_attention_score_grad_entry_regbase.h", 1, 5)
    dump(out, "Q8 ProcessSinkInfo", "flash_attention_score_grad_tiling_common_regbase.cpp", 999, 50)
    dump(out, "Q8 post dsink", "flash_attention_score_grad_s1s2_bn2gs1s2_post_regbase.h", 1, 5)
    dump(out, "Q13 deter map", "flash_attention_score_grad_tiling_normal_regbase.cpp", 1195, 45)
    dump(out, "Q14 SelectDeterBand", "flash_attention_score_grad_tiling_normal_regbase.cpp", 247, 100)
    dump(out, "Q14 kernel CalBandDeterIndex", "flash_attention_score_grad_kernel_deter.h", 493, 55)
    dump(out, "Q18 InitTilingData", "flash_attention_score_grad_tiling_normal_regbase.cpp", 2184, 80)
    dump(out, "Q18 aliases", "flash_attention_score_grad_template_tiling_key.h", 30, 20)
    dump(out, "Q6 enablePreSfmg assign", "flash_attention_score_grad_tiling_normal_regbase.cpp", 1568, 30)
    dump(out, "Q10 GET_IS_L1_PRELOAD body", "flash_attention_score_grad_common.h", 240, 30)
    dump(out, "Q10 KL1 DyL1", "flash_attention_score_grad_block_cube.h", 62, 35)
    dump(out, "Q1 IsTndAllSeqLenZero", "flash_attention_score_grad_tiling.cpp", 267, 40)

    # fts for CheckDrop / drop mask keepProb error
    quoted = '"dropMask"'
    rows = c.execute(
        """SELECT sl.path, sl.line, sl.text
           FROM source_fts f JOIN source_line sl ON sl.id=f.rowid
           WHERE f.source_fts MATCH ? AND sl.path LIKE '%tiling_common_regbase.cpp%'
           ORDER BY sl.line LIMIT 40""",
        (quoted,),
    ).fetchall()
    out.write("\n==== Q3 dropMask in tiling_common ====\n")
    for r in rows:
        out.write(f"{(r['path'] or '')}:{r['line']}: {(r['text'] or '').rstrip()[:200]}\n")

    quoted = '"keepProb"'
    rows = c.execute(
        """SELECT sl.path, sl.line, sl.text
           FROM source_fts f JOIN source_line sl ON sl.id=f.rowid
           WHERE f.source_fts MATCH ? AND sl.path LIKE '%tiling_common_regbase.cpp%'
           ORDER BY sl.line LIMIT 40""",
        (quoted,),
    ).fetchall()
    out.write("\n==== Q3 keepProb in tiling_common ====\n")
    for r in rows:
        out.write(f"{(r['path'] or '')}:{r['line']}: {(r['text'] or '').rstrip()[:200]}\n")

    quoted = '"GRAPH_PARAM_INVALID"'
    rows = c.execute(
        """SELECT sl.path, sl.line, sl.text
           FROM source_fts f JOIN source_line sl ON sl.id=f.rowid
           WHERE f.source_fts MATCH ? AND sl.text LIKE '%keepProb%'
           ORDER BY sl.line LIMIT 20""",
        (quoted,),
    ).fetchall()
    out.write("\n==== Q3 GRAPH_PARAM keepProb ====\n")
    for r in rows:
        out.write(f"{(r['path'] or '')}:{r['line']}: {(r['text'] or '').rstrip()[:200]}\n")

    quoted = '"must be empty"'
    rows = c.execute(
        """SELECT sl.path, sl.line, sl.text
           FROM source_fts f JOIN source_line sl ON sl.id=f.rowid
           WHERE f.source_fts MATCH ?
           ORDER BY sl.line LIMIT 20""",
        (quoted,),
    ).fetchall()
    out.write("\n==== Q3 must be empty ====\n")
    for r in rows:
        out.write(f"{(r['path'] or '')}:{r['line']}: {(r['text'] or '').rstrip()[:200]}\n")

    quoted = '"dropMask must"'
    try:
        tot = c.execute("SELECT COUNT(*) FROM source_fts f WHERE f.source_fts MATCH ?", (quoted,)).fetchone()[0]
        rows = c.execute(
            """SELECT sl.path, sl.line, sl.text
               FROM source_fts f JOIN source_line sl ON sl.id=f.rowid
               WHERE f.source_fts MATCH ?
               ORDER BY sl.line LIMIT 20""",
            (quoted,),
        ).fetchall()
        out.write(f"\n==== Q3 dropMask must total={tot} ====\n")
        for r in rows:
            out.write(f"{(r['path'] or '')}:{r['line']}: {(r['text'] or '').rstrip()[:200]}\n")
    except Exception as e:
        out.write(f"err {e}\n")

    quoted = '"IsTndAllSeqLenZero"'
    rows = c.execute(
        """SELECT sl.path, sl.line, sl.text
           FROM source_fts f JOIN source_line sl ON sl.id=f.rowid
           WHERE f.source_fts MATCH ?
           ORDER BY sl.line LIMIT 10""",
        (quoted,),
    ).fetchall()
    out.write("\n==== Q2 IsTndAllSeqLenZero ====\n")
    for r in rows:
        out.write(f"{(r['path'] or '')}:{r['line']}: {(r['text'] or '').rstrip()[:200]}\n")

    quoted = '"INVOKE_FAG_GENERAL_S1S2_BN2"'
    rows = c.execute(
        """SELECT sl.path, sl.line, sl.text
           FROM source_fts f JOIN source_line sl ON sl.id=f.rowid
           WHERE f.source_fts MATCH ? AND sl.path LIKE '%entry_regbase%'
           ORDER BY sl.line LIMIT 20""",
        (quoted,),
    ).fetchall()
    out.write("\n==== Q7 INVOKE in entry ====\n")
    for r in rows:
        out.write(f"{(r['path'] or '')}:{r['line']}: {(r['text'] or '').rstrip()[:200]}\n")

    quoted = '"dsinkGm"'
    rows = c.execute(
        """SELECT sl.path, sl.line, sl.text
           FROM source_fts f JOIN source_line sl ON sl.id=f.rowid
           WHERE f.source_fts MATCH ? AND (sl.text LIKE '%SetValue%' OR sl.text LIKE '%DataCopy%' OR sl.text LIKE '%CopyOut%' OR sl.text LIKE '%dsink%')
           ORDER BY sl.path, sl.line LIMIT 30""",
        (quoted,),
    ).fetchall()
    out.write("\n==== Q8 dsinkGm ====\n")
    for r in rows:
        out.write(f"{(r['path'] or '')}:{r['line']}: {(r['text'] or '').rstrip()[:200]}\n")

print("wrote", OUT, OUT.stat().st_size)
