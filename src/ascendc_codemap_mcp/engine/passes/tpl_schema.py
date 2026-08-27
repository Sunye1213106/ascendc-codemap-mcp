# -*- coding: utf-8 -*-
"""TplSchemaPass — ASCENDC_TPL ARGS_DECL + ARGS_SEL into CodeMap + TG D blobs.

Stores selection groups as TEMPLATE entities (not one entity per legal key).
Legal packed-key space D goes into ``context['tg_views']`` view blobs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ascendc_codemap_mcp.engine.ir.codemap import CodeMap
from ascendc_codemap_mcp.engine.ir.entity import EntityKind
from ascendc_codemap_mcp.engine.ir.relation import RelationKind
from ascendc_codemap_mcp.engine.materialize_tiling import (
    build_template_blocks,
    expand_legal_with_groups,
)
from ascendc_codemap_mcp.engine.canonical_tpl_projection import TPL_VIEW_NAMES
from ascendc_codemap_mcp.engine.tpl_dsl import TplSchema, parse_file, parse_tpl_corpus, schema_construct_macros


def run(codemap: CodeMap, *, context: dict[str, Any] | None = None) -> CodeMap:
    ctx = context if context is not None else {}
    try:
        schema, header = _resolve_schema(codemap, ctx)
        if schema is not None and schema.dims:
            _upsert_tpl_macros(codemap, ctx, header, schema)
        if schema is None or not schema.dims:
            codemap.meta["tpl_schema_pass"] = "v1-missing"
            return codemap

        header_ref = _portable_header_ref(header, ctx)
        _upsert_dims(codemap, schema, header_ref)
        _upsert_sel_groups(codemap, schema, header_ref)
        existing = ctx.get("tg_views")
        if not isinstance(existing, dict):
            existing = {}
        # Views require ARGS_SEL TEMPLATE facts so commit can rebuild them.
        # Dims-only schemas must not stamp a D that project_tpl_views_from_codemap
        # returns as {}.
        if not schema.selections or not any(schema.selections):
            for name in TPL_VIEW_NAMES:
                existing.pop(name, None)
            ctx["tg_views"] = existing
            codemap.meta["tpl_schema_pass"] = "v1-decl-only"
            codemap.meta["tpl_schema"] = {
                "op_tag": schema.op_tag,
                "dim_count": len(schema.dims),
                "args_sel_group_count": 0,
                "legal_key_count": 0,
                "header": header_ref,
            }
            return codemap
        views = _build_tpl_views(schema, header_ref)
        existing.update(views)
        ctx["tg_views"] = existing
        ctx["tpl_schema"] = schema

        codemap.meta["tpl_schema_pass"] = "v1"
        codemap.meta["tpl_schema"] = {
            "op_tag": schema.op_tag,
            "dim_count": len(schema.dims),
            "args_sel_group_count": len(schema.selections),
            "legal_key_count": int(views["tiling/exhaustive_key_space.yaml"]["legal_key_count"]),
            "header": header_ref,
        }
        codemap.meta["args_sel_group_count"] = len(schema.selections)
        codemap.meta["legal_key_count"] = int(
            views["tiling/exhaustive_key_space.yaml"]["legal_key_count"]
        )
        return codemap
    finally:
        # Source-contract packing / selected-arch DECL is the schema.
        # A later glob of sibling-arch ARGS_DECL must not expand it.
        from ascendc_codemap_mcp.engine.passes.source_contract import reconcile_source_declared_tiling_keys

        reconcile_source_declared_tiling_keys(codemap)


def _kernel_args_from_ctx(ctx: dict[str, Any], path: Path) -> list[str]:
    clang_ctx = ctx.get("clang_ctx")
    if clang_ctx is not None and hasattr(clang_ctx, "kernel_args"):
        try:
            return list(clang_ctx.kernel_args(dtype_variant=None, source_path=path))
        except TypeError:
            try:
                return list(clang_ctx.kernel_args(dtype_variant=None))
            except Exception:
                return []
        except Exception:
            return []
    raw = ctx.get("kernel_args")
    return list(raw) if isinstance(raw, (list, tuple)) else []


def _preprocess_pick_decl(ctx: dict[str, Any], root: Path, arch: str) -> Path | None:
    """Prefer the ARGS_DECL file clang.exe actually included from the kernel entry."""
    from ascendc_codemap_mcp.engine.clang_cmd import clang_preprocess
    from ascendc_codemap_mcp.engine.source_layout import (
        list_tpl_decl_candidates,
        pick_kernel_entry,
        selected_kernel_files,
        tpl_decl_candidates_from_preprocess,
        tpl_decl_files,
    )

    if len(list_tpl_decl_candidates(root, arch)) < 2:
        return None

    files = list(selected_kernel_files(root, arch))
    entry = pick_kernel_entry(files, arch) if files else None
    if entry is None:
        return None
    args = _kernel_args_from_ctx(ctx, Path(entry))
    if not args:
        return None
    stdout = clang_preprocess(entry, args)
    if not stdout:
        return None
    hits = tpl_decl_candidates_from_preprocess(stdout, root)
    if not hits:
        return None
    ranked = tpl_decl_files(root, arch)
    prefer = {p.resolve() for p in ranked}
    for hit in hits:
        if hit.resolve() in prefer or "tiling_key" in hit.name.lower():
            return hit.resolve()
    return hits[0].resolve()


def _resolve_schema(
    codemap: CodeMap, ctx: dict[str, Any]
) -> tuple[TplSchema | None, Path | None]:
    header = _find_header(codemap, ctx)
    op_root = str(ctx.get("op_root") or "").strip()
    arch = str(ctx.get("architecture") or codemap.architecture or "")
    if op_root and arch:
        try:
            picked = _preprocess_pick_decl(ctx, Path(op_root), arch)
        except Exception:
            picked = None
        if picked is not None and picked.is_file():
            header = picked
    paths: list[Path] = []
    if header is not None and header.is_file():
        paths.append(header)
    op_root = str(ctx.get("op_root") or "").strip()
    arch = str(ctx.get("architecture") or codemap.architecture or "")
    if op_root:
        try:
            from ascendc_codemap_mcp.engine.source_layout import tpl_sel_files

            for sel in tpl_sel_files(Path(op_root), arch):
                if sel.is_file() and sel.resolve() not in {p.resolve() for p in paths}:
                    paths.append(sel)
        except Exception:
            pass
    if paths:
        try:
            return parse_tpl_corpus(paths), paths[0]
        except Exception as exc:  # noqa: BLE001
            if header is not None and header.is_file():
                try:
                    return parse_file(header), header
                except Exception:
                    pass
            codemap.meta["tpl_schema_parse_error"] = str(exc)[:240]
            return None, header

    dsl = str(ctx.get("tiling_key_dsl") or "")
    if dsl.strip():
        from ascendc_codemap_mcp.engine.tpl_dsl import parse_args_decl, parse_args_sel

        schema = parse_args_decl(dsl)
        schema.selections = parse_args_sel(dsl)
        return schema, None
    return None, None


def _upsert_tpl_macros(
    codemap: CodeMap,
    ctx: dict[str, Any],
    header: Path | None,
    schema: TplSchema,
) -> int:
    """Mint the TPL construct names the parsed schema actually uses."""
    names = schema_construct_macros(schema)
    locate_paths = _construct_locate_paths(ctx, header)
    fallback_ref = _portable_header_ref(header, ctx) if header is not None else ""
    minted = 0
    for name in sorted(names):
        site, line = _first_token_site(name, locate_paths)
        file_ref = _portable_header_ref(site, ctx) if site is not None else fallback_ref
        codemap.upsert(
            EntityKind.MACRO,
            name,
            eid=f"SRCTPLMACRO::{file_ref or fallback_ref}::{name}",
            attrs={
                "layer": "tpl",
                "provenance": "source_tpl_macro",
                "coverage_hint": "template_match",
            },
            file=file_ref or fallback_ref,
            line=line,
            status="confirmed",
        )
        minted += 1
    if minted:
        codemap.meta["tpl_macro_count"] = minted
    return minted


def _construct_locate_paths(ctx: dict[str, Any], header: Path | None) -> list[Path]:
    """Header + already-selected TPL/host files. Names come from the schema."""
    paths: list[Path] = []
    seen: set[Path] = set()

    def _add(path: Path | None) -> None:
        if path is None or not path.is_file():
            return
        resolved = path.expanduser().resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        paths.append(resolved)

    _add(header)
    op_root = str(ctx.get("op_root") or "").strip()
    arch = str(ctx.get("architecture") or "")
    if op_root:
        root = Path(op_root).expanduser().resolve()
        try:
            from ascendc_codemap_mcp.engine.source_layout import selected_host_files, tpl_sel_files

            for sel in tpl_sel_files(root, arch):
                _add(sel)
            for host in selected_host_files(root, arch):
                _add(host)
        except Exception:
            pass
    return paths


def _first_token_site(name: str, paths: list[Path]) -> tuple[Path | None, int]:
    token = str(name or "")
    if not token:
        return None, 0
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        pos = text.find(token)
        if pos >= 0:
            return path, text.count("\n", 0, pos) + 1
    return None, 0


def _portable_header_ref(header: Path | None, ctx: dict[str, Any]) -> str:
    """Return a machine-independent source reference for a discovered TPL header.

    UO products are portable artifacts.  Absolute build-machine paths must not
    leak into entity ``file`` fields, TG views or metadata because downstream
    source freshness checks resolve evidence relative to the operator checkout.
    The canonical in-tree form matches the other source passes:
    ``<operator>/op_kernel/...``.
    """
    if header is None:
        return ""
    resolved = header.expanduser().resolve()
    op_root = str(ctx.get("op_root") or "").strip()
    if op_root:
        root = Path(op_root).expanduser().resolve()
        try:
            return resolved.relative_to(root.parent).as_posix()
        except ValueError:
            pass
    # An explicitly supplied external header is unusual but still must not make
    # a committed .uo host-specific.  Preserve a stable basename rather than an
    # absolute path; parse-time resolution has already happened above.
    return resolved.name


def _find_header(codemap: CodeMap, ctx: dict[str, Any]) -> Path | None:
    explicit = ctx.get("tiling_key_header") or ctx.get("tpl_header")
    op_root = str(ctx.get("op_root") or "").strip()
    if explicit:
        p = Path(str(explicit)).expanduser()
        if p.is_file():
            if op_root:
                try:
                    p.resolve().relative_to(Path(op_root).expanduser().resolve())
                    return p.resolve()
                except ValueError:
                    pass
            else:
                return p.resolve()

    if op_root:
        root = Path(op_root).expanduser().resolve()
        arch = str(ctx.get("architecture") or codemap.architecture or "")
        try:
            from ascendc_codemap_mcp.engine.source_layout import select_tpl_decl_header

            hit = select_tpl_decl_header(root, arch)
            if hit is not None and hit.is_file():
                return hit.resolve()
        except Exception:
            pass
        try:
            from ascendc_codemap_mcp.engine.op_spec import discover

            spec = discover(root, arch_dir=arch or None)
            header = spec.tiling_key_header
            if header and Path(header).is_file():
                resolved = Path(header).resolve()
                try:
                    resolved.relative_to(root)
                    return resolved
                except ValueError:
                    pass
        except Exception:
            pass
        # Fallback glob under op_kernel
        hits = sorted(root.glob("op_kernel/**/*template_tiling_key.h"))
        arch = str(codemap.architecture or "")
        if arch:
            prefer = [h for h in hits if arch in h.as_posix()]
            if prefer:
                return prefer[0].resolve()
        if hits:
            return hits[0].resolve()

    for ent in codemap.by_kind(EntityKind.FILE):
        name = str(ent.name or "").replace("\\", "/")
        if name.endswith("template_tiling_key.h") and op_root:
            # FILE names are often op-relative with op folder prefix
            root = Path(op_root).expanduser().resolve()
            cand = root / Path(name).name
            if cand.is_file():
                return cand.resolve()
            # strip leading "<op>/"
            parts = name.split("/", 1)
            if len(parts) == 2:
                cand = root / parts[1]
                if cand.is_file():
                    return cand.resolve()
            for hit in root.glob(f"**/{Path(name).name}"):
                return hit.resolve()
    return None


def _upsert_dims(codemap: CodeMap, schema: TplSchema, header_ref: str) -> None:
    shift = 0
    for order, dim in enumerate(schema.dims):
        domain = [str(v) for v in dim.value_domain]
        if str(dim.kind).upper() == "BOOL":
            from ascendc_codemap_mcp.engine.tpl_dsl import canonicalize_bool_token

            domain = [canonicalize_bool_token(v) for v in domain]
        bit_lo = int(dim.bit_lo or shift)
        bit_hi = int(dim.bit_hi or (bit_lo + max(int(dim.bw), 1) - 1))
        if not dim.bit_hi and not dim.bit_lo:
            bit_lo = shift
            bit_hi = shift + max(int(dim.bw), 1) - 1
        key = codemap.upsert(
            EntityKind.TILING_KEY,
            dim.name,
            attrs={
                "layer": "tiling",
                "source_declared": True,
                "decl_kind": dim.kind,
                "kind_tpl": dim.kind,
                "bit_width": int(dim.bw),
                "bw": int(dim.bw),
                "bit_offset": bit_lo,
                "bit_lo": bit_lo,
                "bit_hi": bit_hi,
                "bit_end": bit_hi,
                "decl_order": order,
                "allowed_values": domain,
                "value_domain": domain,
                "provenance": "source_tpl_args_decl",
                "bw_token": str(dim.bw_token or ""),
            },
            file=header_ref,
            status="confirmed",
        )
        _bind_dim_constructs(codemap, schema, key, dim)
        shift += max(int(dim.bw), 1)


def _bind_macro(
    codemap: CodeMap,
    src_id: str,
    name: str,
    *,
    provenance: str,
    extra: dict[str, Any] | None = None,
) -> None:
    token = str(name or "").strip()
    if not token:
        return
    for macro in codemap.by_name(token, kind=EntityKind.MACRO):
        attrs = {"provenance": provenance}
        if extra:
            attrs.update(extra)
        codemap.link(
            RelationKind.BINDS,
            src_id,
            macro.id,
            attrs=attrs,
            status="confirmed",
        )


def _bind_dim_constructs(codemap: CodeMap, schema: TplSchema, key, dim) -> None:
    has_sel = bool(schema.selections and any(schema.selections))
    _bind_macro(codemap, key.id, "ASCENDC_TPL_ARGS_DECL", provenance="source_tpl_schema_construct")
    _bind_macro(codemap, key.id, "GET_TPL_TILING_KEY", provenance="source_tpl_schema_construct")
    if has_sel:
        _bind_macro(codemap, key.id, "ASCENDC_TPL_ARGS_SEL", provenance="source_tpl_schema_construct")
        _bind_macro(codemap, key.id, "ASCENDC_TPL_SEL", provenance="source_tpl_schema_construct")
    kind = str(dim.kind or "").upper()
    if kind:
        _bind_macro(
            codemap,
            key.id,
            f"ASCENDC_TPL_{kind}_DECL",
            provenance="source_tpl_schema_construct",
        )
        if has_sel and kind != "KERNEL_TYPE":
            _bind_macro(
                codemap,
                key.id,
                f"ASCENDC_TPL_{kind}_SEL",
                provenance="source_tpl_schema_construct",
            )
    token = str(dim.bw_token or "").strip()
    if token:
        _bind_macro(
            codemap,
            key.id,
            token,
            provenance="source_tpl_bw_macro",
            extra={"bw_token": token},
        )
    if kind == "UINT" and dim.vals:
        marker = str(dim.vals[0])
        if "UI_LIST" in marker:
            _bind_macro(codemap, key.id, "ASCENDC_TPL_UI_LIST", provenance="source_tpl_schema_construct")
        if "UI_RANGE" in marker:
            _bind_macro(codemap, key.id, "ASCENDC_TPL_UI_RANGE", provenance="source_tpl_schema_construct")


def _upsert_sel_groups(codemap: CodeMap, schema: TplSchema, header_ref: str) -> None:
    blocks = build_template_blocks(schema)
    for block in blocks:
        tpl = codemap.upsert(
            EntityKind.TEMPLATE,
            block.name,
            attrs={
                "layer": "template",
                "tpl_role": "args_sel_group",
                "sel_group_index": block.sel_group_index,
                "fixed_fields": dict(block.fixed_fields),
                "field_domains": {k: list(v) for k, v in block.field_domains.items()},
                "product_count": int(block.product_count),
                "provenance": "source_tpl_args_sel",
            },
            file=header_ref,
            line=int(block.line_start or 0),
            line_end=int(block.line_start or 0),
            status="confirmed",
        )
        for dim_name in list(block.fixed_fields) + list(block.field_domains):
            keys = codemap.by_name(dim_name, kind=EntityKind.TILING_KEY)
            if not keys:
                continue
            codemap.link(
                RelationKind.BINDS,
                tpl.id,
                keys[0].id,
                attrs={
                    "provenance": "source_tpl_args_sel",
                    "sel_group_index": block.sel_group_index,
                    "fixed": dim_name in block.fixed_fields,
                },
                status="confirmed",
            )


def _build_tpl_views(schema: TplSchema, header_ref: str) -> dict[str, Any]:
    blocks = [b.to_dict() for b in build_template_blocks(schema)]
    fallback = {d.name: (list(d.value_domain) or ["0"])[0] for d in schema.dims}
    rows: list[dict[str, Any]] = []
    for idx, (gi, dims) in enumerate(expand_legal_with_groups(schema)):
        full = {name: str(dims.get(name, fallback[name])) for name in fallback}
        try:
            key = int(schema.encode_tiling_key(full))
        except (ValueError, KeyError):
            continue
        rows.append(
            {
                "index": idx,
                "tiling_key": key,
                "tiling_key_hex": f"0x{key:016x}",
                "dims": full,
                "sel_group_id": f"ARGS_SEL_{gi}",
                "status": "template_admissible",
            }
        )

    dims_doc = [
        {
            "name": d.name,
            "kind": d.kind,
            "bw": int(d.bw),
            "bit_lo": int(d.bit_lo),
            "bit_hi": int(d.bit_hi),
            "value_domain": [str(v) for v in d.value_domain],
        }
        for d in schema.dims
    ]
    selections = []
    for gi, group in enumerate(schema.selections):
        selections.append(
            {
                "sel_group_index": gi,
                "sels": [
                    {
                        "name": str(s.get("name")),
                        "kind": str(s.get("kind")),
                        "vals": list(s.get("vals") or []),
                    }
                    for s in group
                ],
            }
        )

    return {
        "tiling/tpl_schema.yaml": {
            "schema": "uo-tpl-schema/v1",
            "op_tag": schema.op_tag,
            "header": header_ref,
            "dims": dims_doc,
            "selections": selections,
        },
        "tiling/template_blocks.yaml": {
            "schema": "uo-template-blocks/v1",
            "blocks": blocks,
            "count": len(blocks),
        },
        "tiling/exhaustive_key_space.yaml": {
            "schema": "uo-exhaustive-key-space/v1",
            "legal_key_count": len(rows),
            "legal_key_index": "tiling/legal_key_index.jsonl",
            "template_blocks": blocks,
            "header": header_ref,
            "status": "template_admissible",
        },
        "tiling/legal_key_index.jsonl": {
            "schema": "uo-legal-key-index/v1",
            "count": len(rows),
            "rows": rows,
        },
    }
