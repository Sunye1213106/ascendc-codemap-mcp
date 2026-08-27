# -*- coding: utf-8 -*-
"""Extract Host OP_CHECK_IF sites as locatable facts bound to fields/inputs.

Reviewers otherwise Grep these by hand. This pass records ``file:line`` plus a
short guard and attaches ``check_sites`` on matching TILING_FIELD / INPUT /
FIELD names. It does not prove the check is sufficient.
"""
from __future__ import annotations

import re
from pathlib import Path

from ascendc_codemap_mcp.engine.ids import branch_id
from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import Entity, EntityKind
from ascendc_codemap_mcp.engine.ir.relation import RelationKind
from ascendc_codemap_mcp.engine.passes.source_text_cache import read_text
from ascendc_codemap_mcp.engine.passes.tiling_host_writes import (
    _line,
    _mask_non_code,
    _selected_host_files,
)

#: Every fact this pass mints carries it. The evidence is a regex over masked
#: host source plus a name match against known fields, so the honest trust is
#: advisory -- which is what ``ADVISORY_PROVENANCE`` maps this label to. Left
#: unlabelled, these 10k edges read as ``legacy_unknown``, i.e. unclassified.
HOST_CHECK_PROVENANCE = "source_host_check"

_CHECK_RE = re.compile(
    r"\b(?P<macro>OP_CHECK_IF|OP_TILING_CHECK|OPS_CHECK|OP_CHECK)\s*\(\s*(?P<guard>[^,)\n]+)"
)
_IDENT_RE = re.compile(r"\b[A-Za-z_]\w*\b")
_SETTER_RE = re.compile(r"\bset_([A-Za-z_]\w*)")
_NOISE = frozenset(
    {
        "OP_CHECK_IF",
        "OP_TILING_CHECK",
        "OPS_CHECK",
        "OP_CHECK",
        "OP_LOGE",
        "OP_LOGW",
        "GRAPH_FAILED",
        "GRAPH_SUCCESS",
        "return",
        "true",
        "false",
        "nullptr",
        "NULL",
        "if",
        "else",
        "const",
        "static",
        "auto",
        "int",
        "bool",
        "size_t",
        "uint32_t",
        "uint64_t",
        "int32_t",
        "int64_t",
        "this",
        "std",
        "GetDim",
        "GetShape",
        "GetInputShape",
        "GetOptionalInputShape",
    }
)


def enrich_host_checks(
    codemap: CodeMap,
    operator_root: str | Path,
    *,
    architecture: str = "",
) -> CodeMap:
    root = Path(operator_root).expanduser().resolve()
    fields: dict[str, list[Entity]] = {}
    for kind in (EntityKind.TILING_FIELD, EntityKind.INPUT, EntityKind.FIELD):
        for ent in codemap.by_kind(kind):
            name = str(ent.name or "").strip()
            if name:
                fields.setdefault(name, []).append(ent)
            leaf = name.rsplit(".", 1)[-1]
            if leaf and leaf != name:
                fields.setdefault(leaf, []).append(ent)

    paths = _selected_host_files(root, architecture)
    sites = 0
    bound = 0
    for path in paths:
        raw = read_text(path)
        masked = _mask_non_code(raw)
        file = _rel(root, path)
        for match in _CHECK_RE.finditer(masked):
            guard = raw[match.start("guard"):match.end("guard")].strip()
            if not guard:
                continue
            line = _line(raw, match.start())
            macro = match.group("macro")
            short = guard[:120]
            targets: list[Entity] = []
            for name in _guard_symbols(guard):
                targets.extend(fields.get(name) or ())
            if not targets:
                # A check whose guard names nothing already on the map is a
                # locatable comment, not a graph fact. Clang already minted the
                # same sites when it had a containing function / operands.
                continue
            eid = branch_id(
                side="host",
                file=file,
                function="_check",
                guard=short,
                ordinal=line,
            )
            br = codemap.upsert(
                EntityKind.BRANCH,
                short,
                eid=eid,
                attrs={
                    "layer": "host",
                    "predicate": short,
                    "branch_kind": "host_check",
                    "check_macro": macro,
                    "function": "",
                    "provenance": HOST_CHECK_PROVENANCE,
                },
                file=file,
                line=line,
                status="confirmed",
            )
            sites += 1
            site = {
                "file": file,
                "line": line,
                "guard": short,
                "macro": macro,
            }
            seen_ids: set[str] = set()
            for ent in targets:
                if ent.id in seen_ids:
                    continue
                seen_ids.add(ent.id)
                _attach_check(ent, site)
                edge = {"provenance": HOST_CHECK_PROVENANCE, "file": file, "line": line}
                codemap.link(RelationKind.GUARDED_BY, ent.id, br.id, attrs=dict(edge))
                codemap.link(RelationKind.CONTROLS, br.id, ent.id, attrs=dict(edge))
                bound += 1

    closure = dict(codemap.meta.get("kernel_tiling_closure") or {})
    closure["host_check_sites"] = sites
    closure["host_check_bindings"] = bound
    codemap.meta["kernel_tiling_closure"] = closure
    return codemap


def _guard_symbols(guard: str) -> set[str]:
    names = {tok for tok in _IDENT_RE.findall(guard or "") if tok not in _NOISE}
    names.update(_SETTER_RE.findall(guard or ""))
    return names


def _attach_check(ent: Entity, site: dict) -> None:
    rows = ent.attrs.setdefault("check_sites", [])
    if site not in rows:
        rows.append(site)


def _rel(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        try:
            return path.resolve().relative_to(root.parent).as_posix()
        except ValueError:
            return path.as_posix().replace("\\", "/")
