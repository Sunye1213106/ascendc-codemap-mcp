# -*- coding: utf-8 -*-
"""Exact-path remaining windows. No leaf fallback."""
from __future__ import annotations

import sqlite3
from pathlib import Path

UO = Path(
    r"d:\TEST\ops-transformer\attention\flash_attention_score_grad"
    r"\.ascendc-codemap\arch35\FlashAttentionScoreGrad.arch35.uo"
)
OUT = Path(r"d:\TEST\ascendc-codemap-mcp\_q20_fill.txt")


def connect():
    c = sqlite3.connect(str(UO))
    c.row_factory = sqlite3.Row
    return c


def fts(c, q, limit=20, host_only=False):
    quoted = '"' + q.replace('"', " ") + '"'
    tot = c.execute("SELECT COUNT(*) FROM source_fts f WHERE f.source_fts MATCH ?", (quoted,)).fetchone()[0]
    extra = ""
    if host_only:
        extra = "AND (sl.path LIKE '%op_host%' OR sl.path LIKE '%op_api%' OR sl.path LIKE '%op_graph%')"
    rows = c.execute(
        f"""
        SELECT sl.path, sl.line, sl.text
        FROM source_fts f JOIN source_line sl ON sl.id = f.rowid
        WHERE f.source_fts MATCH ? {extra}
        ORDER BY sl.path, sl.line
        LIMIT ?
        """,
        (quoted, limit),
    ).fetchall()
    return tot, rows


def win(c, path_substr, line, radius=12):
    rows = c.execute(
        """
        SELECT path, line, text FROM source_line
        WHERE REPLACE(path,'\\','/') LIKE ?
          AND line BETWEEN ? AND ?
        ORDER BY line
        """,
        ("%" + path_substr.replace("\\", "/"), max(1, line - radius), line + radius),
    ).fetchall()
    return rows


def dump_win(out, title, c, path_substr, line, radius=14):
    out.write(f"\n## {title}  {path_substr}:{line}\n")
    rows = win(c, path_substr, line, radius)
    if not rows:
        out.write("  EMPTY WINDOW\n")
        return
    for w in rows:
        out.write(f"  {w['line']}| {(w['text'] or '').rstrip()[:200]}\n")


def dump_fts(out, title, c, q, limit=15, host_only=False):
    tot, rows = fts(c, q, limit, host_only)
    out.write(f"\n## {title}  FTS {q!r} total={tot} host_only={host_only}\n")
    for r in rows:
        p = (r["path"] or "").replace("\\", "/")
        out.write(f"  {p}:{r['line']}: {(r['text'] or '').rstrip()[:180]}\n")


def legal(c, dims):
    # dims: list of (dim,value)
    q = """
    SELECT COUNT(*) FROM legal_key k
    WHERE k.status='template_admissible'
    """
    args = []
    for i, (d, v) in enumerate(dims):
        q += f"""
        AND EXISTS (
          SELECT 1 FROM legal_key_dim d{i}
          WHERE d{i}.key_id=k.id AND d{i}.dim=? AND d{i}.value=?
        )
        """
        args.extend([d, str(v)])
    return c.execute(q, args).fetchone()[0]


def main():
    c = connect()
    with OUT.open("w", encoding="utf-8") as out:
        # Q1 entry
        dump_fts(out, "Q1 IsEmptyOutput", c, "IsEmptyOutput", 10)
        dump_win(out, "Q1 tiling.cpp around 247", c, "op_host/flash_attention_score_grad_tiling.cpp", 247, 20)
        dump_win(out, "Q1 tiling.cpp around 500", c, "op_host/flash_attention_score_grad_tiling.cpp", 505, 25)
        dump_win(out, "Q1 apt.cpp empty branch", c, "op_kernel/flash_attention_score_grad_apt.cpp", 55, 20)
        dump_win(out, "Q1 empty kernel Process", c, "flash_attention_score_grad_empty_tensor_regbase.h", 28, 40)

        # Q2 more seq zero
        dump_win(out, "Q2 seq loop more", c, "flash_attention_score_grad_tiling_normal_regbase.cpp", 627, 50)
        dump_fts(out, "Q2 sValueZeroUnderTND", c, "sValueZeroUnderTND", 12)
        dump_fts(out, "Q2 tailZeroCount", c, "tailZeroCount", 12, host_only=True)

        # Q3 keepProb host
        dump_fts(out, "Q3 keepProb host", c, "keepProb", 40, host_only=True)
        dump_fts(out, "Q3 dropMaskOuter", c, "dropMaskOuter", 20, host_only=True)
        dump_fts(out, "Q3 keep_prob", c, "keep_prob", 20)
        dump_fts(out, "Q3 dropMask optional input", c, "dropMaskOptional", 15)
        dump_fts(out, "Q3 keepProb < 1", c, "keepProb < 1", 15)

        # Q4 FuzzyForBestSplit / s1CvRatio / FP32 D>256
        dump_fts(out, "Q4 FuzzyForBestSplit", c, "FuzzyForBestSplit", 8)
        dump_win(out, "Q4 DoSplit more", c, "flash_attention_score_grad_tiling_normal_regbase.cpp", 900, 30)
        dump_win(out, "Q4 s1Inner * NUM_TWO", c, "flash_attention_score_grad_tiling_normal_regbase.cpp", 1328, 20)
        dump_win(out, "Q4 GetTilingKey s1Inner/2", c, "flash_attention_score_grad_tiling_normal_regbase.cpp", 1909, 20)
        dump_win(out, "Q4 FP32 D>256 host", c, "flash_attention_score_grad_tiling_common_regbase.cpp", 873, 20)
        dump_fts(out, "Q4 S1CV_RATIO", c, "S1CV_RATIO_DEFAULT", 8)
        dump_fts(out, "Q4 CUBE_BASEM", c, "CUBE_BASEM", 8)

        # Q5 compileInfo aivNum
        dump_win(out, "Q5 GetPlatformInfo", c, "flash_attention_score_grad_tiling_normal_regbase.cpp", 758, 35)
        dump_win(out, "Q5 SetBlockDim", c, "flash_attention_score_grad_tiling_normal_regbase.cpp", 1910, 30)

        # Q6 enablePreSfmg full assign
        dump_win(out, "Q6 enablePreSfmg assign body", c, "flash_attention_score_grad_tiling_normal_regbase.cpp", 1574, 25)
        dump_win(out, "Q6 kernel_base gate", c, "flash_attention_score_grad_kernel_base.h", 686, 8)
        dump_win(out, "Q6 entry Presfmg", c, "flash_attention_score_grad_entry_regbase.h", 45, 20)

        # Q7 BN2 entry vs Post
        dump_fts(out, "Q7 ORIG_DTYPE_QUERY DT_FLOAT", c, "ORIG_DTYPE_QUERY != DT_FLOAT", 8)
        dump_win(out, "Q7 entry Post gate", c, "flash_attention_score_grad_entry_regbase.h", 88, 50)
        dump_fts(out, "Q7 BN2 IMPL", c, "INVOKE_FAG", 12)
        dump_fts(out, "Q7 splitAxis BN2 post", c, "IS_BN2", 8)

        # Q8 kernel dsink
        dump_fts(out, "Q8 ProcessSinkInfo more", c, "ProcessSinkInfo", 8)
        dump_win(out, "Q8 ProcessSinkInfo", c, "flash_attention_score_grad_tiling_common_regbase.cpp", 999, 40)
        dump_fts(out, "Q8 dsinkGm write", c, "dsinkGm", 12)
        dump_fts(out, "Q8 vf_cal_sink", c, "vf_cal_sink", 8)

        # Q9 FP16_C0_SIZE NZ_OUT_MIN
        dump_fts(out, "Q9 FP16_C0_SIZE", c, "FP16_C0_SIZE", 8)
        dump_fts(out, "Q9 NZ_OUT_MIN_S_SIZE", c, "NZ_OUT_MIN_S_SIZE", 8)
        dump_win(out, "Q9 isNzOut full", c, "flash_attention_score_grad_tiling_normal_regbase.cpp", 829, 12)

        # Q10 selectors K/Dy + macros
        dump_win(out, "Q10 QL1/KL1/DyL1", c, "flash_attention_score_grad_block_cube.h", 51, 40)
        dump_win(out, "Q10 GET_IS_L1_PRELOAD def", c, "flash_attention_score_grad_common.h", 240, 30)

        # Q11 safety + performance
        dump_win(out, "Q11 ScheduleSafe", c, "flash_attention_score_grad_tiling_normal_regbase.cpp", 113, 40)
        dump_win(out, "Q11 PerformanceEnough", c, "flash_attention_score_grad_tiling_normal_regbase.cpp", 143, 50)
        dump_win(out, "Q11 host gates 824-856", c, "flash_attention_score_grad_tiling_normal_regbase.cpp", 824, 40)

        # Q12 GQA vs swizzle
        dump_win(out, "Q12 SelectGQADenseSchedule", c, "flash_attention_score_grad_tiling_normal_regbase.cpp", 77, 50)
        dump_win(out, "Q12 SelectBlockSchedule g==1", c, "flash_attention_score_grad_tiling_normal_regbase.cpp", 340, 45)

        # Q13 mapping
        dump_win(out, "Q13 deter type map 1190-1230", c, "flash_attention_score_grad_tiling_normal_regbase.cpp", 1195, 45)

        # Q14 SelectDeterBand
        dump_win(out, "Q14 SelectDeterBand 247", c, "flash_attention_score_grad_tiling_normal_regbase.cpp", 247, 95)
        dump_win(out, "Q14 kernel dispatch 493", c, "flash_attention_score_grad_kernel_deter.h", 493, 55)

        # Q15 legal already have; host emit
        dump_win(out, "Q15 GetDeterSparseType", c, "flash_attention_score_grad_tiling_normal_regbase.cpp", 1210, 30)
        dump_win(out, "Q15 TND causal param", c, "flash_attention_score_grad_tiling_varlen_regbase.cpp", 121, 40)

        # Q16 host encode
        dump_win(out, "Q16 inputDtype encode", c, "flash_attention_score_grad_tiling_common_regbase.cpp", 1718, 30)
        dump_win(out, "Q16 DECL InputDType", c, "flash_attention_score_grad_template_tiling_key.h", 57, 20)

        # Q17 GetDTemplateType full
        dump_win(out, "Q17 GetDTemplateType", c, "flash_attention_score_grad_tiling_common_regbase.cpp", 906, 30)

        # Q18 InitTilingData all branches
        dump_win(out, "Q18 InitTilingData", c, "flash_attention_score_grad_tiling_normal_regbase.cpp", 2184, 80)
        dump_win(out, "Q18 aliases in template_tiling_key", c, "flash_attention_score_grad_template_tiling_key.h", 30, 20)
        dump_win(out, "Q18 NEED_DETER macros", c, "flash_attention_score_grad_common.h", 210, 25)

        # Q19 SyncAll sites via FTS
        dump_fts(out, "Q19 SyncALLCores", c, "SyncALLCores", 30)
        dump_fts(out, "Q19 SyncAll", c, "SyncAll", 20)

        # Q20 SEL count already; consumer
        dump_win(out, "Q20 apt template args", c, "op_kernel/flash_attention_score_grad_apt.cpp", 35, 25)

        # legal extras
        out.write("\n## LEGAL extras\n")
        combos = [
            [("IsEmptyTensor", 1)],
            [("IsTnd", 1), ("IsEmptyTensor", 1)],
            [("IsTnd", 1), ("DeterType", 3), ("SplitAxis", 0), ("IsTndSwizzle", 0)],
            [("IsTnd", 1), ("DeterType", 3), ("IsTndSwizzle", 1)],
            [("IsTnd", 1), ("IsBn2MultiBlk", 1)],
            [("IsNzOut", 1)],
            [("IsNzOut", 1), ("DTemplateNum", 128)],
            [("IsDrop", 1), ("DTemplateNum", 128), ("SplitAxis", 0)],
            [("InputDType", 1), ("DTemplateNum", 768), ("S1TemplateNum", 64)],
            [("InputDType", 1), ("DTemplateNum", 256), ("S1TemplateNum", 64)],
            [("IsRope", 1), ("IsDNoEqual", 1)],
            [("IsRope", 1), ("IsDNoEqual", 0)],
        ]
        for dims in combos:
            n = legal(c, dims)
            out.write(f"  {dims} -> {n}\n")

        # entity existence for agent-first names
        out.write("\n## ENTITY existence\n")
        for name in [
            "BufferNum", "isSeqExistZero", "keepProb", "s1Inner", "coreNum",
            "enablePreSfmg", "DoPostTiling", "dsink", "IsNzOut", "QL1BuffSelector",
            "IsTndDeterSwizzleSupported", "SelectGQADenseSchedule", "deterBandScheduleMode",
            "InputDType", "IsRope", "NEED_DETER", "SetScheduleMode", "GET_TPL_TILING_KEY",
            "IsEmptyTensor", "MutexBuffersPolicy4buff", "FP16_C0_SIZE",
        ]:
            n = c.execute("SELECT COUNT(*) FROM entity WHERE name=?", (name,)).fetchone()[0]
            like = c.execute(
                "SELECT COUNT(*) FROM entity WHERE name LIKE ? COLLATE NOCASE", (f"%{name}%",)
            ).fetchone()[0]
            out.write(f"  exact={name!s} n={n} like={like}\n")

    print("wrote", OUT, OUT.stat().st_size)


if __name__ == "__main__":
    main()
