# -*- coding: utf-8 -*-
"""Q6-Q20 actual UO traces. Imports helpers by inlining (standalone)."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

UO = Path(
    r"d:\TEST\ops-transformer\attention\flash_attention_score_grad"
    r"\.ascendc-codemap\arch35\FlashAttentionScoreGrad.arch35.uo"
)
OUT = Path(r"d:\TEST\ascendc-codemap-mcp\_run_q20_b.json")


def connect() -> sqlite3.Connection:
    c = sqlite3.connect(str(UO))
    c.row_factory = sqlite3.Row
    return c


def fts(c: sqlite3.Connection, q: str, limit: int = 8) -> dict:
    quoted = '"' + q.replace('"', " ") + '"'
    try:
        tot = c.execute(
            "SELECT COUNT(*) FROM source_fts f WHERE f.source_fts MATCH ?", (quoted,)
        ).fetchone()[0]
        rows = c.execute(
            """
            SELECT sl.path, sl.line, sl.text
            FROM source_fts f JOIN source_line sl ON sl.id = f.rowid
            WHERE f.source_fts MATCH ?
            ORDER BY
              CASE WHEN sl.path LIKE '%/arch35/%' THEN 0 ELSE 1 END,
              sl.path, sl.line
            LIMIT ?
            """,
            (quoted, limit),
        ).fetchall()
    except sqlite3.Error as exc:
        return {"q": q, "error": str(exc), "total": 0, "hits": []}
    return {
        "q": q,
        "total": int(tot),
        "hits": [
            {
                "file": (r["path"] or "").replace("\\", "/"),
                "line": int(r["line"] or 0),
                "text": (r["text"] or "").rstrip(),
            }
            for r in rows
        ],
    }


def window(c: sqlite3.Connection, path: str, line: int, radius: int = 8) -> list[str]:
    rows = c.execute(
        """
        SELECT line, text FROM source_line
        WHERE REPLACE(path,'\\','/') = REPLACE(?,'\\','/')
          AND line BETWEEN ? AND ?
        ORDER BY line
        """,
        (path, max(1, line - radius), line + radius),
    ).fetchall()
    if not rows:
        leaf = path.replace("\\", "/").split("/")[-1]
        rows = c.execute(
            """
            SELECT line, text FROM source_line
            WHERE REPLACE(path,'\\','/') LIKE '%' || ?
              AND line BETWEEN ? AND ?
            ORDER BY line
            """,
            (leaf, max(1, line - radius), line + radius),
        ).fetchall()
    return [f"{r['line']}| {(r['text'] or '').rstrip()}" for r in rows]


def attach_windows(c: sqlite3.Connection, fts_res: dict, n: int = 2, radius: int = 8) -> dict:
    wins = []
    for h in (fts_res.get("hits") or [])[:n]:
        wins.append({"at": f"{h['file']}:{h['line']}", "window": window(c, h["file"], h["line"], radius)})
    fts_res["windows"] = wins
    return fts_res


def exact(c: sqlite3.Connection, name: str, limit: int = 12) -> list[dict]:
    rows = c.execute(
        """
        SELECT kind, name, file, line_start AS line
        FROM entity WHERE name = ? OR name LIKE '%::' || ?
        ORDER BY kind, file, line_start LIMIT ?
        """,
        (name, name, limit),
    ).fetchall()
    return [
        {"kind": r["kind"], "name": r["name"], "file": (r["file"] or "").replace("\\", "/"), "line": r["line"]}
        for r in rows
    ]


def names(c: sqlite3.Connection, like: str, limit: int = 10) -> list[dict]:
    rows = c.execute(
        """
        SELECT kind, name, file, line_start AS line
        FROM entity WHERE name LIKE ? COLLATE NOCASE
        ORDER BY length(name), kind LIMIT ?
        """,
        (like, limit),
    ).fetchall()
    return [
        {"kind": r["kind"], "name": r["name"], "file": (r["file"] or "").replace("\\", "/"), "line": r["line"]}
        for r in rows
    ]


def edges(c: sqlite3.Connection, name: str, limit: int = 16) -> list[dict]:
    rows = c.execute(
        """
        SELECT r.kind AS rel,
               src.kind AS sk, src.name AS sn, src.file AS sf, src.line_start AS sl,
               dst.kind AS dk, dst.name AS dn, dst.file AS df, dst.line_start AS dl
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
            "src": f"{r['sk']} {r['sn']} {(r['sf'] or '')}:{r['sl']}",
            "dst": f"{r['dk']} {r['dn']} {(r['df'] or '')}:{r['dl']}",
        }
        for r in rows
    ]


def combo(c: sqlite3.Connection, filters: dict[str, str]) -> int:
    sql = "SELECT COUNT(DISTINCT k.id) FROM legal_key k"
    args: list[str] = []
    for i, (dim, value) in enumerate(filters.items()):
        a = f"d{i}"
        sql += (
            f" JOIN legal_key_dim {a} ON {a}.key_id = k.id"
            f" AND {a}.dim = ? AND CAST({a}.value AS TEXT) = ?"
        )
        args.extend([dim, str(value)])
    return int(c.execute(sql, args).fetchone()[0])


def dim_hist(c: sqlite3.Connection, dim: str) -> list[dict]:
    rows = c.execute(
        "SELECT value, COUNT(*) n FROM legal_key_dim WHERE dim=? GROUP BY value ORDER BY n DESC",
        (dim,),
    ).fetchall()
    return [{"value": r["value"], "n": r["n"]} for r in rows]


def main() -> None:
    c = connect()
    report: dict = {}

    report["Q6"] = {
        "start": ["enablePreSfmg=1", "Presfmg", "D=64", "确定性"],
        "fts": [
            attach_windows(c, fts(c, t, 8), 2)
            for t in ["enablePreSfmg", "Presfmg", "preSfmgLimit", "dTemplateType > 64"]
        ],
        "names": {
            "enablePreSfmg": exact(c, "enablePreSfmg"),
            "Presfmg": names(c, "%Presfmg%", 8),
        },
        "edges": edges(c, "enablePreSfmg", 16),
        "kernel_gate": attach_windows(c, fts(c, "dTemplateType > 64", 8), 2, 6),
    }

    report["Q7"] = {
        "start": ["FP32 没有 Post", "FP16 有 Post", "BN2GS1S2/BN2S2 vs BN2"],
        "fts": [
            attach_windows(c, fts(c, t, 8), 2)
            for t in ["opPost", "DoPostTiling", "PostTiling", "isFp32", "needCast"]
        ],
        "names": {"DoPostTiling": exact(c, "DoPostTiling"), "opPost": names(c, "%opPost%", 8)},
        "edges": edges(c, "DoPostTiling", 8),
    }

    report["Q8"] = {
        "start": ["sink 输入", "workspace 变大", "dsink"],
        "fts": [
            attach_windows(c, fts(c, t, 8), 2)
            for t in ["dsink", "sinkWorkspace", "hasSink", "set_dsink"]
        ],
        "names": {"dsink": names(c, "%dsink%", 10), "sink": names(c, "%Sink%", 10)},
        "edges_dsink": edges(c, "dsink", 12),
    }

    report["Q9"] = {
        "start": ["NzOut", "超 L2", "D=96", "D=88"],
        "fts": [
            attach_windows(c, fts(c, t, 8), 2)
            for t in ["IsNzOut", "isExceedL2Cache", "enableNz", "Aligned96", "NzOut"]
        ],
        "names": {"IsNzOut": exact(c, "IsNzOut")},
        "edges": edges(c, "IsNzOut", 10),
        "legal": dim_hist(c, "IsNzOut"),
    }

    report["Q10"] = {
        "start": ["4-buffer", "dropout", "Query L1", "BufferNum"],
        "fts": [
            attach_windows(c, fts(c, t, 8), 2)
            for t in [
                "BufferNum",
                "4buff",
                "QL1BuffSelector",
                "IS_SMALL_D_PRELOAD",
                "MutexBuffersPolicy4buff",
                "IS_DROP",
            ]
        ],
        "names": {
            "QL1BuffSelector": exact(c, "QL1BuffSelector"),
            "KL1BuffSelector": exact(c, "KL1BuffSelector"),
            "DyL1BuffSelector": exact(c, "DyL1BuffSelector"),
            "IS_SMALL_D_PRELOAD": exact(c, "IS_SMALL_D_PRELOAD"),
        },
        "edges": edges(c, "IS_SMALL_D_PRELOAD", 8),
    }

    report["Q11"] = {
        "start": ["TND dense 确定性 swizzle", "有的开有的不开"],
        "fts": [
            attach_windows(c, fts(c, t, 8), 2)
            for t in [
                "IsTndDeterSwizzleSupported",
                "enableSwizzle",
                "TND_SWIZZLE_PREFIX_NUM",
                "isSeqExistZero",
            ]
        ],
        "names": {
            "IsTndDeterSwizzleSupported": exact(c, "IsTndDeterSwizzleSupported"),
            "IsTndSwizzle": exact(c, "IsTndSwizzle"),
        },
        "legal": {
            "IsTnd=1 IsTndSwizzle=1": combo(c, {"IsTnd": "1", "IsTndSwizzle": "1"}),
            "IsTnd=1 DeterType=2 IsTndSwizzle=1": combo(
                c, {"IsTnd": "1", "DeterType": "2", "IsTndSwizzle": "1"}
            ),
            "IsTnd=1 DeterType=3 IsTndSwizzle=1": combo(
                c, {"IsTnd": "1", "DeterType": "3", "IsTndSwizzle": "1"}
            ),
        },
    }

    report["Q12"] = {
        "start": ["GQA dense 确定性调度", "g>1 swizzle"],
        "fts": [
            attach_windows(c, fts(c, t, 8), 2)
            for t in ["GqaDense", "gqaDense", "g > 1", "block swizzle", "g == 1"]
        ],
        "names": {"*Gqa*": names(c, "%Gqa%", 10), "*GQA*": names(c, "%GQA%", 10)},
    }

    report["Q13"] = {
        "start": ["RIGHT_DOWN_CAUSAL", "S1==S2 vs S1!=S2", "确定性类型"],
        "fts": [
            attach_windows(c, fts(c, t, 8), 2)
            for t in ["RIGHT_DOWN_CAUSAL", "DETER_CAUSAL", "isS1S2Same", "DETER_DENSE"]
        ],
        "names": {"DeterType": exact(c, "DeterType")},
        "legal": dim_hist(c, "DeterType"),
        "edges": edges(c, "DeterType", 8),
    }

    report["Q14"] = {
        "start": ["BAND 确定性", "causal/dense", "deterBandScheduleMode"],
        "fts": [
            attach_windows(c, fts(c, t, 8), 2)
            for t in ["deterBandScheduleMode", "CalBandDeterIndex", "CalCausalSwizzleIndex", "CalDenseSwizzleIndex"]
        ],
        "names": {"deterBandScheduleMode": exact(c, "deterBandScheduleMode")},
        "edges": edges(c, "deterBandScheduleMode", 16),
    }

    report["Q15"] = {
        "start": ["TND 确定性 causal 没编", "561003"],
        "fts": [
            attach_windows(c, fts(c, t, 6), 2)
            for t in ["CalcleTNDCausalDeterParam", "DETER_CAUSAL", "561003"]
        ],
        "matrix": [],
    }
    matrix = []
    for deter in ("0", "1", "2", "3", "4"):
        for split in ("0", "1", "5"):
            for sw in ("0", "1"):
                n = combo(
                    c,
                    {"IsTnd": "1", "DeterType": deter, "SplitAxis": split, "IsTndSwizzle": sw},
                )
                if n:
                    matrix.append({"DeterType": deter, "SplitAxis": split, "IsTndSwizzle": sw, "n": n})
    report["Q15"]["matrix"] = matrix
    report["Q15"]["host_can_emit"] = attach_windows(c, fts(c, "DETER_CAUSAL", 8), 2)

    report["Q16"] = {
        "start": ["Host 识别 FP8", "kernel 是否编了", "561003"],
        "fts": [
            attach_windows(c, fts(c, t, 8), 2)
            for t in ["FP8_E5M2", "HIFLOAT8", "InputDType", "FP8_E4M3"]
        ],
        "names": {"InputDType": exact(c, "InputDType")},
        "legal": dim_hist(c, "InputDType"),
        "legal_456": {
            "InputDType=4": combo(c, {"InputDType": "4"}),
            "InputDType=5": combo(c, {"InputDType": "5"}),
            "InputDType=6": combo(c, {"InputDType": "6"}),
        },
        "template_block_dtype": [
            dict(r)
            for r in c.execute(
                """
                SELECT value, COUNT(*) n FROM template_block_dim
                WHERE dim='InputDType' GROUP BY value
                """
            )
        ],
    }

    report["Q17"] = {
        "start": ["RoPE D=128", "模板 D=192", "D 不等"],
        "fts": [
            attach_windows(c, fts(c, t, 8), 2)
            for t in ["hasRope", "NUM192", "dNoEqual", "dTemplateType"]
        ],
        "names": {
            "IsRope": exact(c, "IsRope"),
            "IsDNoEqual": exact(c, "IsDNoEqual"),
            "dNoEqual": exact(c, "dNoEqual"),
        },
        "edges": edges(c, "IsDNoEqual", 12),
        "legal": {
            "IsRope=1": combo(c, {"IsRope": "1"}),
            "IsRope=1 D=192": combo(c, {"IsRope": "1", "DTemplateNum": "192"}),
            "IsRope=1 D=128": combo(c, {"IsRope": "1", "DTemplateNum": "128"}),
            "IsRope=0 D=192": combo(c, {"IsRope": "0", "DTemplateNum": "192"}),
        },
    }

    report["Q18"] = {
        "start": ["TilingData 布局", "TND/确定性/swizzle"],
        "fts": [
            attach_windows(c, fts(c, t, 8), 2)
            for t in ["NEED_DETER", "needDeterPrefix", "FagTilingWithTemplate", "GetTilingData"]
        ],
        "names": {
            "NEED_DETER": exact(c, "NEED_DETER"),
            "TilingData": names(c, "%TilingDataUs1s2%", 8),
        },
    }

    report["Q19"] = {
        "start": ["SyncAll", "SetScheduleMode", "BN2 + multi-block + TND"],
        "fts": [
            attach_windows(c, fts(c, t, 8), 2)
            for t in ["SetScheduleMode", "isBn2MultiBlk", "SyncALLCores", "bnSparseLimit"]
        ],
        "names": {
            "isBn2MultiBlk": exact(c, "isBn2MultiBlk"),
            "IsBn2MultiBlk": exact(c, "IsBn2MultiBlk"),
            "SyncAll": names(c, "SyncAll", 8),
        },
        "sync_files": [
            {"file": r[0], "n": r[1]}
            for r in c.execute(
                """
                SELECT file, COUNT(*) n FROM entity
                WHERE kind='OPERATION' AND name='SyncAll'
                GROUP BY file ORDER BY n DESC
                """
            )
        ],
        "legal": {
            "IsTnd=1 IsBn2MultiBlk=1": combo(c, {"IsTnd": "1", "IsBn2MultiBlk": "1"}),
            "IsTnd=1 SplitAxis=1": combo(c, {"IsTnd": "1", "SplitAxis": "1"}),
        },
        "edges": edges(c, "IsBn2MultiBlk", 8),
    }

    report["Q20"] = {
        "start": ["tiling key 加一个 bool 维"],
        "fts": [
            attach_windows(c, fts(c, t, 4), 1)
            for t in ["ASCENDC_TPL_ARGS_DECL", "GET_TPL_TILING_KEY", "ASCENDC_TPL_ARGS_SEL"]
        ],
        "dims": [r[0] for r in c.execute("SELECT DISTINCT dim FROM legal_key_dim ORDER BY 1")],
        "sel_groups": int(c.execute("SELECT COUNT(DISTINCT sel_group) FROM legal_key").fetchone()[0]),
        "legal_n": int(c.execute("SELECT COUNT(*) FROM legal_key").fetchone()[0]),
        "entry": attach_windows(c, fts(c, "RegbaseFAG", 6), 1, 5),
    }

    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Q6-Q20 written", OUT, "bytes", OUT.stat().st_size)


if __name__ == "__main__":
    main()
