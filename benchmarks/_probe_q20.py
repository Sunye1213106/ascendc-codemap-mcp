# -*- coding: utf-8 -*-
"""Ideal query traces for the 20-question MCP A/B set, against the real .uo."""
from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path

UO = Path(
    r"d:\TEST\ops-transformer\attention\flash_attention_score_grad"
    r"\.ascendc-codemap\arch35\FlashAttentionScoreGrad.arch35.uo"
)
OUT = Path(r"d:\TEST\ascendc-codemap-mcp\benchmarks\_uo_q20.json")

IDENT_LIKE = r"[A-Za-z_][A-Za-z0-9_]*"


def conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(UO))
    c.row_factory = sqlite3.Row
    return c


def fts(c: sqlite3.Connection, q: str, limit: int = 6) -> dict:
    quoted = '"' + q.replace('"', " ") + '"'
    try:
        rows = c.execute(
            """
            SELECT sl.path, sl.line, sl.text
            FROM source_fts f
            JOIN source_line sl ON sl.id = f.rowid
            WHERE f.source_fts MATCH ?
            ORDER BY sl.path, sl.line
            LIMIT ?
            """,
            (quoted, limit),
        ).fetchall()
        total_row = c.execute(
            """
            SELECT COUNT(*)
            FROM source_fts f
            WHERE f.source_fts MATCH ?
            """,
            (quoted,),
        ).fetchone()
        total = int(total_row[0] if total_row else 0)
    except sqlite3.Error as exc:
        return {"q": q, "error": str(exc), "hits": [], "total": 0}
    return {
        "q": q,
        "total": total,
        "hits": [
            {
                "file": str(r["path"] or "").replace("\\", "/"),
                "line": int(r["line"] or 0),
                "text": (r["text"] or "").strip()[:140],
            }
            for r in rows
        ],
    }


def names(c: sqlite3.Connection, like: str, limit: int = 8) -> list[dict]:
    rows = c.execute(
        """
        SELECT kind, name, file, line_start AS line
        FROM entity
        WHERE name LIKE ? COLLATE NOCASE
        ORDER BY length(name), kind
        LIMIT ?
        """,
        (like, limit),
    ).fetchall()
    return [
        {
            "kind": r["kind"],
            "name": r["name"],
            "file": (r["file"] or "").replace("\\", "/"),
            "line": r["line"],
        }
        for r in rows
    ]


def exact_kinds(c: sqlite3.Connection, name: str, limit: int = 12) -> list[dict]:
    rows = c.execute(
        """
        SELECT kind, name, file, line_start AS line
        FROM entity
        WHERE name = ? OR name LIKE '%::' || ?
        ORDER BY kind, file, line_start
        LIMIT ?
        """,
        (name, name, limit),
    ).fetchall()
    return [
        {
            "kind": r["kind"],
            "name": r["name"],
            "file": (r["file"] or "").replace("\\", "/"),
            "line": r["line"],
        }
        for r in rows
    ]


def edges_for(c: sqlite3.Connection, name: str, limit: int = 12) -> list[dict]:
    rows = c.execute(
        """
        SELECT r.kind AS rel, src.kind AS src_kind, src.name AS src_name,
               src.file AS src_file, src.line_start AS src_line,
               dst.kind AS dst_kind, dst.name AS dst_name,
               dst.file AS dst_file, dst.line_start AS dst_line
        FROM relation r
        JOIN entity src ON src.id = r.src
        JOIN entity dst ON dst.id = r.dst
        WHERE (src.name = ? OR dst.name = ?)
          AND r.kind IN ('WRITES','READS','DERIVES','BINDS','SELECTS','CALLS')
        LIMIT ?
        """,
        (name, name, limit),
    ).fetchall()
    return [
        {
            "rel": r["rel"],
            "src": f"{r['src_kind']} {r['src_name']} {(r['src_file'] or '')}:{r['src_line']}",
            "dst": f"{r['dst_kind']} {r['dst_name']} {(r['dst_file'] or '')}:{r['dst_line']}",
        }
        for r in rows
    ]


def dim_values(c: sqlite3.Connection, dim: str) -> list[dict]:
    rows = c.execute(
        """
        SELECT value, COUNT(*) AS n
        FROM legal_key_dim
        WHERE dim = ?
        GROUP BY value
        ORDER BY n DESC
        """,
        (dim,),
    ).fetchall()
    return [{"value": r["value"], "n": r["n"]} for r in rows]


def combo(c: sqlite3.Connection, filters: dict[str, str], nonempty: bool = True) -> int:
    sql = "SELECT COUNT(DISTINCT k.id) FROM legal_key k"
    args: list[str] = []
    for i, (dim, value) in enumerate(filters.items()):
        a = f"d{i}"
        sql += f" JOIN legal_key_dim {a} ON {a}.key_id = k.id AND {a}.dim = ? AND CAST({a}.value AS TEXT) = ?"
        args.extend([dim, str(value)])
    if nonempty:
        sql += " WHERE IFNULL(k.status,'') != 'empty'"
    try:
        return int(c.execute(sql, args).fetchone()[0])
    except sqlite3.Error:
        sql2 = "SELECT COUNT(DISTINCT k.id) FROM legal_key k"
        args = []
        for i, (dim, value) in enumerate(filters.items()):
            a = f"d{i}"
            sql2 += f" JOIN legal_key_dim {a} ON {a}.key_id = k.id AND {a}.dim = ? AND {a}.value = ?"
            args.extend([dim, value])
        return int(c.execute(sql2, args).fetchone()[0])


def status_counts(c: sqlite3.Connection) -> list[dict]:
    rows = c.execute(
        "SELECT IFNULL(status,'(null)') AS s, COUNT(*) n FROM legal_key GROUP BY 1"
    ).fetchall()
    return [{"status": r["s"], "n": r["n"]} for r in rows]


def main() -> None:
    c = conn()
    questions: dict = {}

    # --- Q1 empty tensor ---
    questions["Q1"] = {
        "agent_tokens": ["IsEmptyTensor", "empty", "numel", "EmptyTensor"],
        "fts": [fts(c, t) for t in ["IsEmptyTensor", "EmptyTensor", "numel == 0", "empty tensor"]],
        "names": {
            "IsEmptyTensor": exact_kinds(c, "IsEmptyTensor"),
            "*Empty*": names(c, "%Empty%", 10),
        },
        "edges": edges_for(c, "IsEmptyTensor", 8),
        "legal": {"IsEmptyTensor=1": combo(c, {"IsEmptyTensor": "1"}), "IsEmptyTensor=0": combo(c, {"IsEmptyTensor": "0"})},
    }

    # --- Q2 TND zero seq ---
    questions["Q2"] = {
        "agent_tokens": ["actual_seq_qlen", "isSeqExistZero", "TND", "S=0"],
        "fts": [fts(c, t) for t in ["isSeqExistZero", "actual_seq_qlen", "seqExistZero", "tndS2ExistZero"]],
        "names": {
            "isSeqExistZero": exact_kinds(c, "isSeqExistZero"),
            "IsTnd": exact_kinds(c, "IsTnd"),
        },
        "edges": edges_for(c, "isSeqExistZero", 8),
    }

    # --- Q3 keepProb dropMask ---
    questions["Q3"] = {
        "agent_tokens": ["keepProb", "dropMask", "IsDrop"],
        "fts": [fts(c, t) for t in ["keepProb", "dropMask", "KeepProb"]],
        "names": {
            "IsDrop": exact_kinds(c, "IsDrop"),
            "keepProb": names(c, "%keepProb%", 8),
            "dropMask": names(c, "%dropMask%", 8),
        },
        "edges": edges_for(c, "IsDrop", 8),
    }

    # --- Q4 s1Inner 64 vs 128 ---
    questions["Q4"] = {
        "agent_tokens": ["s1Inner", "s1Template", "cubeBaseM", "S1TemplateNum"],
        "fts": [fts(c, t) for t in ["s1Inner", "cubeBaseM", "s1BaseSize", "S1TemplateNum"]],
        "names": {
            "s1Inner": exact_kinds(c, "s1Inner"),
            "S1TemplateNum": exact_kinds(c, "S1TemplateNum"),
        },
        "legal_S1": dim_values(c, "S1TemplateNum"),
    }

    # --- Q5 coreNum /2 ---
    questions["Q5"] = {
        "agent_tokens": ["coreNum", "aicNum", "GetCoreNumAiv", "SetBlockDim"],
        "fts": [fts(c, t) for t in ["GetCoreNumAiv", "GetCoreNumAic", "set_coreNum", "SetBlockDim"]],
        "names": {
            "coreNum": exact_kinds(c, "coreNum"),
            "aicNum": exact_kinds(c, "aicNum"),
        },
        "edges_coreNum": edges_for(c, "coreNum", 12),
        "edges_aicNum": edges_for(c, "aicNum", 8),
    }

    # --- Q6 enablePreSfmg ---
    questions["Q6"] = {
        "agent_tokens": ["enablePreSfmg", "Presfmg", "PreSfmg"],
        "fts": [fts(c, t) for t in ["enablePreSfmg", "Presfmg", "preSfmg"]],
        "names": {
            "enablePreSfmg": exact_kinds(c, "enablePreSfmg"),
            "*PreSfmg*": names(c, "%PreSfmg%", 8),
            "*Presfmg*": names(c, "%Presfmg%", 8),
        },
        "edges": edges_for(c, "enablePreSfmg", 10),
    }

    # --- Q7 FP32 Post ---
    questions["Q7"] = {
        "agent_tokens": ["Post", "FP32", "Launch", "BN2"],
        "fts": [fts(c, t) for t in ["RunPost", "isNeedPost", "needPost", "LaunchPost"]],
        "names": {
            "needPost": names(c, "%Post%", 12),
            "*Launch*": names(c, "%LaunchKernel%", 8),
        },
    }

    # --- Q8 sink / dsink ---
    questions["Q8"] = {
        "agent_tokens": ["sink", "dsink", "workspace"],
        "fts": [fts(c, t) for t in ["dsink", "hasSink", "sinkWorkspace"]],
        "names": {
            "sink": names(c, "%sink%", 12),
            "dsink": names(c, "%dsink%", 8),
            "hasSink": exact_kinds(c, "hasSink") or names(c, "%HasSink%", 8),
        },
        "edges": edges_for(c, "dsink", 10) or edges_for(c, "hasSink", 10),
    }

    # --- Q9 NzOut ---
    questions["Q9"] = {
        "agent_tokens": ["NzOut", "IsNzOut", "L2", "exceedL2"],
        "fts": [fts(c, t) for t in ["IsNzOut", "enableNzOut", "isExceedL2Cache"]],
        "names": {"IsNzOut": exact_kinds(c, "IsNzOut")},
        "edges": edges_for(c, "IsNzOut", 10),
        "legal": dim_values(c, "IsNzOut"),
    }

    # --- Q10 L1 buffer dropout ---
    questions["Q10"] = {
        "agent_tokens": ["BufferNum", "4-buffer", "QL1", "dropout", "IS_DROP"],
        "fts": [
            fts(c, t)
            for t in [
                "BufferNum",
                "IS_SMALL_D_PRELOAD",
                "QL1BuffSelector",
                "MutexBuffersPolicy4buff",
                "IS_DROP",
            ]
        ],
        "names": {
            "*BufferNum*": names(c, "%BufferNum%", 8),
            "QL1BuffSelector": exact_kinds(c, "QL1BuffSelector"),
            "IS_SMALL_D_PRELOAD": exact_kinds(c, "IS_SMALL_D_PRELOAD"),
            "MutexBuffersPolicy4buff": exact_kinds(c, "MutexBuffersPolicy4buff"),
        },
        "edges": edges_for(c, "IS_SMALL_D_PRELOAD", 8),
    }

    # --- Q11 TND deter swizzle ---
    questions["Q11"] = {
        "agent_tokens": ["enableSwizzle", "TND", "swizzle", "IsTndDeterSwizzleSupported"],
        "fts": [
            fts(c, t)
            for t in [
                "IsTndDeterSwizzleSupported",
                "enableSwizzle",
                "TND_SWIZZLE_PREFIX_NUM",
            ]
        ],
        "names": {
            "IsTndSwizzle": exact_kinds(c, "IsTndSwizzle"),
            "IsTndDeterSwizzleSupported": exact_kinds(c, "IsTndDeterSwizzleSupported"),
        },
        "legal": {
            "IsTnd=1 IsTndSwizzle=1": combo(c, {"IsTnd": "1", "IsTndSwizzle": "1"}),
            "IsTnd=1 DeterType=2 IsTndSwizzle=1": combo(
                c, {"IsTnd": "1", "DeterType": "2", "IsTndSwizzle": "1"}
            ),
        },
    }

    # --- Q12 GQA dense vs swizzle ---
    questions["Q12"] = {
        "agent_tokens": ["GQA", "g > 1", "dense schedule", "swizzle"],
        "fts": [fts(c, t) for t in ["GqaDense", "gqaDense", "g > 1", "isGqaDense"]],
        "names": {
            "*Gqa*": names(c, "%Gqa%", 10),
            "*GQA*": names(c, "%GQA%", 10),
        },
    }

    # --- Q13 RIGHT_DOWN_CAUSAL ---
    questions["Q13"] = {
        "agent_tokens": ["RIGHT_DOWN_CAUSAL", "S1==S2", "DeterType", "DETER_CAUSAL"],
        "fts": [fts(c, t) for t in ["RIGHT_DOWN_CAUSAL", "DETER_CAUSAL", "isS1S2Same"]],
        "names": {"DeterType": exact_kinds(c, "DeterType")},
        "legal_deter": dim_values(c, "DeterType"),
    }

    # --- Q14 BAND schedule ---
    questions["Q14"] = {
        "agent_tokens": ["BAND", "deterBandScheduleMode", "causal", "dense"],
        "fts": [fts(c, t) for t in ["deterBandScheduleMode", "CalBandDeterIndex", "DISABLED"]],
        "names": {"deterBandScheduleMode": exact_kinds(c, "deterBandScheduleMode")},
        "edges": edges_for(c, "deterBandScheduleMode", 16),
    }

    # --- Q15 TND causal compiled? ---
    questions["Q15"] = {
        "agent_tokens": ["TND", "DETER_CAUSAL", "561003", "legal key"],
        "fts": [fts(c, t) for t in ["CalcleTNDCausalDeterParam", "DETER_CAUSAL"]],
        "legal": {
            "IsTnd=1 DeterType=3": combo(c, {"IsTnd": "1", "DeterType": "3"}),
            "IsTnd=1 DeterType=3 IsTndSwizzle=1": combo(
                c, {"IsTnd": "1", "DeterType": "3", "IsTndSwizzle": "1"}
            ),
            "IsTnd=1 DeterType=3 SplitAxis=0": combo(
                c, {"IsTnd": "1", "DeterType": "3", "SplitAxis": "0"}
            ),
        },
        "tnd_deter_matrix": None,
    }
    matrix = []
    for deter in ("1", "2", "3", "4"):
        for split in ("0", "1", "5"):
            for sw in ("0", "1"):
                n = combo(
                    c,
                    {"IsTnd": "1", "DeterType": deter, "SplitAxis": split, "IsTndSwizzle": sw},
                )
                if n:
                    matrix.append(
                        {"DeterType": deter, "SplitAxis": split, "IsTndSwizzle": sw, "n": n}
                    )
    questions["Q15"]["tnd_deter_matrix"] = matrix

    # --- Q16 FP8 ---
    questions["Q16"] = {
        "agent_tokens": ["FP8", "HIFLOAT8", "InputDType", "561003"],
        "fts": [fts(c, t) for t in ["FP8_E5M2", "HIFLOAT8", "InputDType"]],
        "names": {"InputDType": exact_kinds(c, "InputDType")},
        "legal": dim_values(c, "InputDType"),
    }

    # --- Q17 RoPE D=192 ---
    questions["Q17"] = {
        "agent_tokens": ["hasRope", "dTemplateType", "IsDNoEqual", "IsRope"],
        "fts": [fts(c, t) for t in ["hasRope", "dTemplateType", "NUM192", "dNoEqual"]],
        "names": {
            "IsRope": exact_kinds(c, "IsRope"),
            "IsDNoEqual": exact_kinds(c, "IsDNoEqual"),
            "dNoEqual": exact_kinds(c, "dNoEqual"),
        },
        "legal": {
            "IsRope=1": combo(c, {"IsRope": "1"}),
            "IsRope=1 DTemplateNum=192": combo(c, {"IsRope": "1", "DTemplateNum": "192"}),
            "IsRope=1 DTemplateNum=128": combo(c, {"IsRope": "1", "DTemplateNum": "128"}),
            "IsRope=0 DTemplateNum=192": combo(c, {"IsRope": "0", "DTemplateNum": "192"}),
        },
        "edges": edges_for(c, "IsDNoEqual", 10),
    }

    # --- Q18 TilingData layouts ---
    questions["Q18"] = {
        "agent_tokens": ["TilingData", "NEED_DETER", "isTndSwizzle", "GetTilingData"],
        "fts": [fts(c, t) for t in ["NEED_DETER", "needDeterPrefix", "FagTilingWithTemplate"]],
        "names": {
            "*TilingData*": names(c, "%TilingDataUs1s2%", 10),
            "NEED_DETER": exact_kinds(c, "NEED_DETER"),
        },
    }

    # --- Q19 SyncAll / schedule ---
    questions["Q19"] = {
        "agent_tokens": ["SyncAll", "SetScheduleMode", "isBn2MultiBlk"],
        "fts": [fts(c, t) for t in ["SetScheduleMode", "isBn2MultiBlk", "SyncALLCores"]],
        "names": {
            "SyncAll": exact_kinds(c, "SyncAll") or names(c, "%SyncAll%", 8),
            "isBn2MultiBlk": exact_kinds(c, "isBn2MultiBlk"),
            "IsBn2MultiBlk": exact_kinds(c, "IsBn2MultiBlk"),
        },
        "edges": edges_for(c, "isBn2MultiBlk", 10),
        "sync_ops": names(c, "%SyncALL%", 12),
        "legal": {
            "IsTnd=1 IsBn2MultiBlk=1": combo(c, {"IsTnd": "1", "IsBn2MultiBlk": "1"}),
            "IsTnd=1 SplitAxis=1": combo(c, {"IsTnd": "1", "SplitAxis": "1"}),
        },
    }

    # --- Q20 tiling key new dim ---
    questions["Q20"] = {
        "agent_tokens": ["ASCENDC_TPL_ARGS_DECL", "GET_TPL_TILING_KEY", "SEL"],
        "fts": [fts(c, t) for t in ["ASCENDC_TPL_ARGS_DECL", "GET_TPL_TILING_KEY", "ASCENDC_TPL_ARGS_SEL"]],
        "names": {
            "IsDrop": exact_kinds(c, "IsDrop"),
        },
        "dims": [r[0] for r in c.execute("SELECT DISTINCT dim FROM legal_key_dim ORDER BY 1")],
        "sel_groups": int(
            c.execute("SELECT COUNT(DISTINCT sel_group) FROM legal_key").fetchone()[0]
        ),
        "legal_n": int(c.execute("SELECT COUNT(*) FROM legal_key").fetchone()[0]),
    }

    report = {
        "legal_status": status_counts(c),
        "questions": questions,
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", OUT, "bytes", OUT.stat().st_size)


if __name__ == "__main__":
    main()
