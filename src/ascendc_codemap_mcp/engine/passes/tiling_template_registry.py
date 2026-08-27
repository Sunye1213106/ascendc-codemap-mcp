# -*- coding: utf-8 -*-
"""Project REGISTER_TILING_TEMPLATE_WITH_ARCH into the CodeMap.

Reuses ``anchors.extract_registry`` and ``registry_capable.extract_iscapable``.
Does not re-parse macros and does not overlap ``REGISTER_TILING_FOR_TILINGKEY``.
"""

from __future__ import annotations

import re
from pathlib import Path

from ascendc_codemap_mcp.engine.anchors import extract_registry
from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import EntityKind
from ascendc_codemap_mcp.engine.ir.relation import RelationKind
from ascendc_codemap_mcp.engine.registry_capable import extract_iscapable
from ascendc_codemap_mcp.engine.passes.source_text_cache import read_text
from ascendc_codemap_mcp.engine.source_layout import selected_host_files, selected_kernel_files

_REGISTER_TILING_DEFAULT_RE = re.compile(
    r"REGISTER_TILING_DEFAULT\s*\(\s*([A-Za-z_:][A-Za-z0-9_:]*)\s*\)"
)


def enrich_tiling_template_registry(
    codemap: CodeMap,
    operator_root: str | Path,
    *,
    architecture: str = "",
) -> CodeMap:
    import time as _time

    from ascendc_codemap_mcp.engine.timing import log as _tlog

    root = Path(operator_root).expanduser().resolve()
    op_name = str(codemap.op_name or "").strip()
    t0 = _time.perf_counter()
    hits = extract_registry(root, op_name) if op_name else []
    t_reg = _time.perf_counter() - t0
    count = 0
    t_cap0 = _time.perf_counter()
    capable_by_file: dict[Path, list] = {}
    for hit in hits:
        cls = str(hit.get("class") or "")
        if not cls:
            continue
        abs_file = Path(str(hit.get("file") or ""))
        try:
            rel = str(abs_file.resolve().relative_to(root)).replace("\\", "/")
        except ValueError:
            rel = str(hit.get("file") or "").replace("\\", "/")
        line = int(hit.get("line") or 0)
        cap_file = ""
        cap_line = 0
        try:
            if abs_file.is_file():
                found = capable_by_file.get(abs_file)
                if found is None:
                    found = extract_iscapable(abs_file)
                    capable_by_file[abs_file] = found
                found = [p for p in found if cls in p.class_name]
            else:
                found = []
        except Exception:
            found = []
        if found:
            cap_file = str(found[0].file or rel)
            cap_line = int(found[0].line or 0)
            try:
                cap_path = Path(cap_file)
                cap_file = str(cap_path.resolve().relative_to(root)).replace("\\", "/")
            except (ValueError, OSError):
                cap_file = cap_file.replace("\\", "/")
        eid = f"TILINGTPLREG::{rel}::{line}::{cls}"
        codemap.upsert(
            EntityKind.PREDICATE,
            f"REGISTER_TILING_TEMPLATE_{cls}",
            eid=eid,
            attrs={
                "predicate_role": "tiling_template_registry",
                "class": cls,
                "priority": int(hit.get("priority") or 0),
                "arch_expr": str(hit.get("arch_expr") or ""),
                "op": str(hit.get("op") or op_name),
                "architecture": architecture,
                "is_capable_file": cap_file,
                "is_capable_line": cap_line,
                "provenance": "source_register_tiling_template",
            },
            file=rel,
            line=line,
            status="confirmed",
        )
        _bind_template_registry(codemap, eid, cls, architecture)
        count += 1
    t_cap = _time.perf_counter() - t_cap0
    defaults = _emit_register_tiling_defaults(codemap, root, architecture)
    _tlog(
        f"{_time.perf_counter() - t0:7.3f}s  tiling_template_registry  "
        f"hits={count} extract_registry={t_reg:.3f}s iscapable={t_cap:.3f}s"
    )
    meta = dict(codemap.meta.get("tiling_template_registry") or {})
    meta["count"] = count
    meta["register_tiling_default"] = defaults
    codemap.meta["tiling_template_registry"] = meta
    return codemap


def _emit_register_tiling_defaults(
    codemap: CodeMap, root: Path, architecture: str
) -> int:
    count = 0
    seen: set[Path] = set()
    for path in list(selected_host_files(root, architecture)) + list(
        selected_kernel_files(root, architecture)
    ):
        key = path.resolve()
        if key in seen or not path.is_file():
            continue
        seen.add(key)
        raw = read_text(path)
        try:
            rel = path.resolve().relative_to(root).as_posix()
        except ValueError:
            rel = path.as_posix().replace("\\", "/")
        for match in _REGISTER_TILING_DEFAULT_RE.finditer(raw):
            cls = match.group(1)
            line = raw.count("\n", 0, match.start()) + 1
            eid = f"TILINGDEFAULT::{rel}::{line}::{cls}"
            codemap.upsert(
                EntityKind.PREDICATE,
                "REGISTER_TILING_DEFAULT",
                eid=eid,
                attrs={
                    "predicate_role": "tiling_default_registry",
                    "class": cls.split("::")[-1],
                    "architecture": architecture,
                    "provenance": "source_register_tiling_default",
                },
                file=rel,
                line=line,
                status="confirmed",
            )
            _bind_template_registry(
                codemap, eid, cls.split("::")[-1], architecture
            )
            count += 1
    return count


def _bind_template_registry(
    codemap: CodeMap, predicate_id: str, class_name: str, architecture: str
) -> None:
    """Hang a template-registry PREDICATE off the type it names, or the ARCH.

    ``REGISTER_TILING_TEMPLATE_WITH_ARCH`` and ``REGISTER_TILING_DEFAULT``
    already record the class they register. The packed-key sibling pass
    (``tiling_registration``) emits SELECTS to TILING_DATA for the same
    fact; this pass stopped at minting the PREDICATE, so the three registry
    sites were degree-0. Prefer TILING_DATA / TYPE of that class; fall back
    to ARCH because the macro is architecture-gated (``arch_expr``).
    """
    short = str(class_name or "").split("::")[-1]
    if not short:
        return
    target = None
    for kind in (EntityKind.TILING_DATA, EntityKind.TYPE):
        hits = list(codemap.by_name(short, kind=kind)) or list(
            codemap.by_name(class_name, kind=kind)
        )
        if hits:
            target = hits[0]
            break
    if target is not None:
        codemap.link(
            RelationKind.SELECTS,
            predicate_id,
            target.id,
            attrs={"provenance": "source_register_tiling_template", "class": short},
            status="confirmed",
        )
        return
    for arch in codemap.by_kind(EntityKind.ARCH):
        if architecture and arch.name != architecture:
            continue
        codemap.link(
            RelationKind.AVAILABLE_ON,
            predicate_id,
            arch.id,
            attrs={"provenance": "source_register_tiling_template", "class": short},
            status="confirmed",
        )
