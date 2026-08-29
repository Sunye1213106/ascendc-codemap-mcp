# -*- coding: utf-8 -*-
import sqlite3

c = sqlite3.connect(
    r"d:\TEST\ops-transformer\attention\flash_attention_score_grad"
    r"\.ascendc-codemap\arch35\FlashAttentionScoreGrad.arch35.uo"
)
c.row_factory = sqlite3.Row


def fts(q, n=6):
    quoted = '"' + q.replace('"', " ") + '"'
    tot = c.execute(
        "SELECT COUNT(*) FROM source_fts f WHERE f.source_fts MATCH ?", (quoted,)
    ).fetchone()[0]
    rows = c.execute(
        """
        SELECT sl.path, sl.line, sl.text
        FROM source_fts f JOIN source_line sl ON sl.id = f.rowid
        WHERE f.source_fts MATCH ?
        ORDER BY sl.path, sl.line LIMIT ?
        """,
        (quoted, n),
    ).fetchall()
    print(f"\nFTS {q!r} total={tot}")
    for r in rows:
        path = (r["path"] or "").replace("\\", "/")
        print(f"  {path}:{r['line']}: {(r['text'] or '').strip()[:130]}")


for q in [
    "BufferNum",
    "EmptyTensor",
    "IsEmptyTensorValue",
    "opPost",
    "DoPost",
    "postKernel",
    "AIC_CORE",
    "NUM_TWO",
    "coreNum /",
    "Aligned96",
    "dTemplateType ==",
    "enablePreSfmg =",
    "deterSparseType",
]:
    fts(q, 5)

print("\n--- enablePreSfmg writers ---")
for r in c.execute(
    """
    SELECT r.kind, src.kind AS sk, src.name AS sn, src.file, src.line_start,
           dst.kind AS dk, dst.name AS dn, dst.file AS df, dst.line_start AS dl
    FROM relation r
    JOIN entity src ON src.id = r.src
    JOIN entity dst ON dst.id = r.dst
    WHERE (dst.name = 'enablePreSfmg' OR src.name = 'enablePreSfmg')
      AND r.kind IN ('WRITES','READS','DERIVES')
    LIMIT 20
    """
):
    print(
        f"  {r['kind']} {r['sk']} {r['sn']} {r['file']}:{r['line_start']} -> {r['dk']} {r['dn']} {r['df']}:{r['dl']}"
    )

print("\n--- SyncAll OPERATION files ---")
for r in c.execute(
    """
    SELECT file, COUNT(*) n FROM entity
    WHERE kind='OPERATION' AND name='SyncAll'
    GROUP BY file ORDER BY n DESC
    """
):
    print(f"  {r['n']:3} {r['file']}")

print("\n--- IsEmptyTensor entity kinds ---")
for r in c.execute(
    "SELECT kind, file, line_start, substr(data,1,80) FROM entity WHERE name='IsEmptyTensor' LIMIT 8"
):
    print(dict(r))

print("\n--- template_block InputDType domains ---")
print("template_block", c.execute("SELECT COUNT(*) FROM template_block").fetchone()[0])
print("template_block_dim", c.execute("SELECT COUNT(*) FROM template_block_dim").fetchone()[0])
try:
    for r in c.execute(
        """
        SELECT dim, value, COUNT(*) n FROM template_block_dim
        WHERE dim='InputDType' GROUP BY value
        """
    ):
        print(dict(r))
except Exception as e:
    print("tb dim", e)
    print([x[1] for x in c.execute("PRAGMA table_info(template_block_dim)")])
