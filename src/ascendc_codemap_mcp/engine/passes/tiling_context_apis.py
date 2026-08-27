# -*- coding: utf-8 -*-
"""Project a frozen CANN ``TilingContext`` host-API catalog onto the CodeMap.

Host IR already walked these call sites during extract. This pass does not
re-lex sources and does not mint every ``context_->`` method. The catalog is
intentionally tiny: ``SetScheduleMode`` (hang / batch vs stream) and
``SetBlockDim`` (same path, cheap).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from ascendc_codemap_mcp.engine.ids import operation_site_id
from ascendc_codemap_mcp.engine.ir.codemap import CodeMap, _unique_named
from ascendc_codemap_mcp.engine.ir.entity import EntityKind
from ascendc_codemap_mcp.engine.ir.relation import RelationKind
from ascendc_codemap_mcp.engine.passes.kernel_scan import norm_file

# Frozen CANN gert::TilingContext methods. Not every host setter.
TILING_CONTEXT_HOST_APIS = frozenset({"SetScheduleMode", "SetBlockDim"})
# Frozen CANN platform-info APIs. Separate catalog from TilingContext setters.
PLATFORM_HOST_APIS = frozenset({"GetCoreNumAiv", "GetCoreNumAic", "GetCurNpuArch"})
_API_CATALOGS = (
    (TILING_CONTEXT_HOST_APIS, "cann_tiling_context", "host_tiling_context_api"),
    (PLATFORM_HOST_APIS, "cann_platform", "host_platform_api"),
)
_MAX_SITES_PER_API = 32
_MAX_TOTAL = 128


def _site_get(site: Any, name: str, default: Any = "") -> Any:
    if isinstance(site, dict):
        return site.get(name, default)
    return getattr(site, name, default)


def _callee_short(site: Any) -> str:
    raw = str(_site_get(site, "callee") or "")
    return raw.split("::")[-1].strip()


def enrich_tiling_context_apis(
    codemap: CodeMap,
    operator_root: str | Path,
    *,
    architecture: str = "",
    host_ir: Any = None,
) -> CodeMap:
    """Mint locatable host OPERATION nodes for catalog TilingContext / platform calls."""
    if host_ir is None:
        return codemap
    root = Path(operator_root).expanduser().resolve() if operator_root else Path()
    root_s = str(root) if operator_root else ""
    per_api: dict[str, int] = defaultdict(int)
    minted = 0
    catalog_of: dict[str, str] = {}
    for names, _catalog, _prov in _API_CATALOGS:
        for name in names:
            catalog_of[name] = _catalog
    api_names = set(catalog_of)
    for site in list(getattr(host_ir, "call_sites", None) or []):
        if minted >= _MAX_TOTAL:
            break
        callee = _callee_short(site)
        if callee not in api_names:
            continue
        if per_api[callee] >= _MAX_SITES_PER_API:
            continue
        file = str(_site_get(site, "file") or "")
        line = int(_site_get(site, "line") or 0)
        if not file or line <= 0:
            continue
        nfile = norm_file(file, root_s)
        column = int(_site_get(site, "column") or 0)
        oid = operation_site_id(
            file=nfile, line=line, column=column, callee=callee, root=root_s
        )
        args_raw = _site_get(site, "args") or ()
        args = [str(a) for a in (args_raw if not isinstance(args_raw, str) else (args_raw,))]
        catalog = catalog_of[callee]
        provenance = next(
            prov for names, _cat, prov in _API_CATALOGS if callee in names
        )
        caller = str(_site_get(site, "caller") or "")
        attrs = {
            "callee": callee,
            "layer": "host",
            "catalog": catalog,
            "function": caller,
            "receiver": str(_site_get(site, "receiver") or ""),
            "receiver_type": str(_site_get(site, "receiver_type") or ""),
            "args": args,
            "argument": args[0] if args else "",
            "architecture": architecture,
            "provenance": provenance,
            "column": column,
        }
        op = codemap.upsert(
            EntityKind.OPERATION,
            callee,
            eid=oid,
            attrs=attrs,
            file=nfile,
            line=line,
            status="confirmed",
        )
        owner = _unique_named(
            codemap, caller, (EntityKind.FUNCTION, EntityKind.METHOD)
        )
        if owner is None and caller:
            same_file = [
                ent
                for ent in codemap.by_name(caller, kind=EntityKind.FUNCTION)
                if str(ent.file or "").replace("\\", "/") == nfile
            ]
            if len(same_file) == 1:
                owner = same_file[0]
        if owner is not None:
            codemap.link(
                RelationKind.CALLS,
                owner.id,
                op.id,
                attrs={"provenance": provenance, "file": nfile, "line": line},
            )
        per_api[callee] += 1
        minted += 1
    codemap.meta["tiling_context_apis"] = {
        "count": minted,
        "by_callee": dict(per_api),
        "catalog": sorted(TILING_CONTEXT_HOST_APIS),
        "platform_catalog": sorted(PLATFORM_HOST_APIS),
    }
    return codemap
