# -*- coding: utf-8 -*-
"""Actually execute 20-question traces against the FAG arch35 .uo."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

UO = Path(
    r"d:\TEST\ops-transformer\attention\flash_attention_score_grad"
    r"\.ascendc-codemap\arch35\FlashAttentionScoreGrad.arch35.uo"
)
OUT = Path(r"d:\TEST\ascendc-codemap-mcp\_run_q20.json")


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
              CASE WHEN sl.path LIKE '%/arch35/%' THEN 0
                   WHEN sl.path LIKE '%/arch22/%' THEN 2
                   ELSE 1 END,
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


def window(c: sqlite3.Connection, path: str, line: int, radius: int = 6) -> list[str]:
    rows = c.execute(
        """
        SELECT line, text FROM source_line
        WHERE path = ? AND line BETWEEN ? AND ?
        ORDER BY line
        """,
        (path, max(1, line - radius), line + radius),
    ).fetchall()
    if not rows:
        # path in DB may use backslash or different spelling
        rows = c.execute(
            """
            SELECT line, text FROM source_line
            WHERE REPLACE(path,'\\','/') LIKE '%' || ? AND line BETWEEN ? AND ?
            ORDER BY line
            """,
            (path.replace("\\", "/").split("/")[-1], max(1, line - radius), line + radius),
        ).fetchall()
    return [f"{r['line']}| {(r['text'] or '').rstrip()}" for r in rows]


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
        {
            "kind": r["kind"],
            "name": r["name"],
            "file": (r["file"] or "").replace("\\", "/"),
            "line": r["line"],
        }
        for r in rows
    ]


def exact(c: sqlite3.Connection, name: str, limit: int = 12) -> list[dict]:
    rows = c.execute(
        """
        SELECT kind, name, file, line_start AS line
        FROM entity
        WHERE name = ? OR name LIKE '%::' || ?
        ORDER BY kind, file, line_start LIMIT ?
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
    out = []
    for r in rows:
        out.append(
            {
                "rel": r["rel"],
                "src": f"{r['sk']} {r['sn']} {(r['sf'] or '')}:{r['sl']}",
                "dst": f"{r['dk']} {r['dn']} {(r['df'] or '')}:{r['dl']}",
            }
        )
    return out


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


def pick_arch35(hits: list[dict]) -> dict | None:
    for h in hits:
        f = h.get("file") or ""
        if "/arch35/" in f or f.endswith("arch35.h"):
            return h
    return hits[0] if hits else None


def attach_windows(c: sqlite3.Connection, fts_res: dict, n: int = 2, radius: int = 7) -> dict:
    wins = []
    for h in (fts_res.get("hits") or [])[:n]:
        wins.append(
            {
                "at": f"{h['file']}:{h['line']}",
                "window": window(c, h["file"], h["line"], radius),
            }
        )
    fts_res["windows"] = wins
    return fts_res


def main() -> None:
    c = connect()
    report: dict = {}

    # ========== Q1 empty tensor ==========
    q1_fts = [
        attach_windows(c, fts(c, t, 8), 2)
        for t in ["IsEmptyTensor", "emptyTensor", "EmptyTensor", "numel"]
    ]
    # follow tiling.cpp emptyTensor and apt.cpp
    empty_cpp = c.execute(
        """
        SELECT path, line, text FROM source_line
        WHERE REPLACE(path,'\\','/') LIKE '%flash_attention_score_grad_tiling.cpp'
          AND line BETWEEN 1 AND 180
        ORDER BY line
        """
    ).fetchall()
    apt = c.execute(
        """
        SELECT path, line, text FROM source_line
        WHERE REPLACE(path,'\\','/') LIKE '%flash_attention_score_grad_apt.cpp'
          AND (text LIKE '%IsEmptyTensor%' OR text LIKE '%EmptyTensor%')
        ORDER BY line LIMIT 20
        """
    ).fetchall()
    report["Q1"] = {
        "start": ["numel=0", "empty", "IsEmptyTensor", "tiling 入口"],
        "fts": q1_fts,
        "tiling_cpp_1_180": [
            f"{r['line']}| {(r['text'] or '').rstrip()}" for r in empty_cpp if r["line"] <= 160
        ],
        "apt_empty_lines": [
            f"{(r['path'] or '')}:{r['line']}| {(r['text'] or '').rstrip()}" for r in apt
        ],
        "names": {
            "IsEmptyTensor": exact(c, "IsEmptyTensor"),
            "EMPTY_TENSOR": names(c, "%EMPTY_TENSOR%", 8),
        },
        "edges": edges(c, "IsEmptyTensor", 12),
        "legal": {
            "IsEmptyTensor=1": combo(c, {"IsEmptyTensor": "1"}),
            "IsEmptyTensor=0": combo(c, {"IsEmptyTensor": "0"}),
        },
    }

    # ========== Q2 TND zero seq ==========
    q2_fts = [
        attach_windows(c, fts(c, t, 8), 2)
        for t in [
            "isSeqExistZero",
            "actual_seq_qlen",
            "seqExistZero",
            "tndExistZero",
            "IsEmptyTensor",
        ]
    ]
    report["Q2"] = {
        "start": ["TND", "actual_seq_qlen", "actual_seq_kvlen", "全 0", "empty tensor"],
        "fts": q2_fts,
        "names": {
            "isSeqExistZero": exact(c, "isSeqExistZero"),
            "IsTnd": exact(c, "IsTnd"),
        },
        "edges_isSeq": edges(c, "isSeqExistZero", 8),
        "legal_tnd": {
            "IsTnd=1": combo(c, {"IsTnd": "1"}),
            "IsTnd=1 IsEmptyTensor=1": combo(c, {"IsTnd": "1", "IsEmptyTensor": "1"}),
        },
    }

    # ========== Q3 keepProb dropMask ==========
    q3_fts = [
        attach_windows(c, fts(c, t, 8), 2, 8)
        for t in ["keepProb", "dropMask", "keepProb == 1", "keepProb >"]
    ]
    report["Q3"] = {
        "start": ["keepProb=1", "dropMask optional input", "参数错误"],
        "fts": q3_fts,
        "names": {
            "keepProb": exact(c, "keepProb") or names(c, "%keepProb%", 8),
            "dropMask": names(c, "%dropMask%", 8),
            "IsDrop": exact(c, "IsDrop"),
        },
        "edges_IsDrop": edges(c, "IsDrop", 10),
        "legal_drop": dim_hist(c, "IsDrop"),
    }

    # ========== Q4 s1Inner 64 vs 128 ==========
    q4_fts = [
        attach_windows(c, fts(c, t, 8), 2)
        for t in ["s1Inner", "cubeBaseM", "s1BaseSize", "S1TemplateNum"]
    ]
    report["Q4"] = {
        "start": ["Host s1Inner=64", "kernel 模板 128", "FP32 D>256"],
        "fts": q4_fts,
        "names": {
            "s1Inner": exact(c, "s1Inner"),
            "S1TemplateNum": exact(c, "S1TemplateNum"),
        },
        "legal_S1": dim_hist(c, "S1TemplateNum"),
        "legal_fp32_d": {
            "InputDType=1 DTemplateNum=256 S1=64": combo(
                c, {"InputDType": "1", "DTemplateNum": "256", "S1TemplateNum": "64"}
            ),
            "InputDType=1 DTemplateNum=256 S1=128": combo(
                c, {"InputDType": "1", "DTemplateNum": "256", "S1TemplateNum": "128"}
            ),
            "InputDType=1 DTemplateNum=768 S1=64": combo(
                c, {"InputDType": "1", "DTemplateNum": "768", "S1TemplateNum": "64"}
            ),
            "InputDType=1 DTemplateNum=768 S1=128": combo(
                c, {"InputDType": "1", "DTemplateNum": "768", "S1TemplateNum": "128"}
            ),
        },
    }

    # ========== Q5 coreNum /2 ==========
    q5_fts = [
        attach_windows(c, fts(c, t, 8), 2)
        for t in ["GetCoreNumAiv", "GetCoreNumAic", "set_coreNum", "SetBlockDim", "coreNum /"]
    ]
    report["Q5"] = {
        "start": ["coreNum=72", "AIC 36", "coreNum/2"],
        "fts": q5_fts,
        "names": {"coreNum": exact(c, "coreNum"), "aicNum": exact(c, "aicNum")},
        "edges_coreNum": edges(c, "coreNum", 14),
        "edges_aicNum": edges(c, "aicNum", 8),
    }

    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Q1-Q5 written", OUT, "bytes", OUT.stat().st_size)


if __name__ == "__main__":
    main()
