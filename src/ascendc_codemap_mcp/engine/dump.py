# -*- coding: utf-8 -*-
"""uo-dump — debug/export surface for CodeMap ``.uo`` view_blobs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

# Short aliases → view_blob names stored in the ``.uo`` product.
VIEW_ALIASES: dict[str, str] = {
    "manifest": "manifest.yaml",
    "quality": "quality.yaml",
    "operator": "operator.yaml",
    "operator_graph": "ir/operator_graph.yaml",
    "graph": "ir/operator_graph.yaml",
    "tilingdata": "views/tilingdata.yaml",
    "kernel": "views/kernel.yaml",
    "call_graph": "views/call_graph.yaml",
    "legal_keys": "tiling/legal_key_index.jsonl",
    "key_space": "tiling/key_space.yaml",
    "exhaustive_key_space": "tiling/exhaustive_key_space.yaml",
    "coverage_model": "tiling/coverage_model.yaml",
    "variables": "tiling/variables.yaml",
    "constraints": "tiling/constraints.yaml",
    "host_derivation": "ir/host_derivation.yaml",
    "tg_host_view": "ir/tg_host_view.yaml",
    "integrity": "checks/integrity.yaml",
    "artifact_hashes": "checks/artifact_hashes.yaml",
    "legal_key_index": "tiling/legal_key_index.jsonl",
    "summary": "summary",
    "host": "ir/tg_host_view.yaml",
    "macros": "macros",
    "templates": "templates",
    "arch": "arch",
}


def resolve_view_name(view: str) -> str:
    text = str(view or "").strip().replace("\\", "/")
    if not text:
        raise ValueError("view name required")
    if text in VIEW_ALIASES:
        return VIEW_ALIASES[text]
    if text.endswith(".yaml") or text.endswith(".yml") or text.endswith(".jsonl"):
        return text
    if "/" in text and not text.endswith(".yaml"):
        return text + ".yaml"
    if text + ".yaml" in VIEW_ALIASES.values():
        return text + ".yaml"
    return VIEW_ALIASES.get(text, text)


def _resolve_db_or_uo(uo_root: Path, *, architecture: str = "") -> tuple[str, Path]:
    """Return ('uo', path). Production path is ``.uo`` only."""
    if uo_root.is_file() and uo_root.suffix == ".uo":
        return "uo", uo_root
    from ascendc_codemap_mcp.engine.store.reader import find_uo_product

    found = find_uo_product(
        uo_root if uo_root.is_dir() else uo_root.parent,
        architecture=architecture,
    )
    if found is not None and found.suffix == ".uo":
        return "uo", found
    raise FileNotFoundError(
        f"missing CodeMap product (.uo) under {uo_root}; "
        "expected .ascendc-codemap/<arch>/<op>.<arch>.uo"
    )


def dump_view(
    uo_root: str | Path,
    view: str,
    *,
    out: str | Path | None = None,
) -> dict[str, Any]:
    """Load one view from ``.uo`` and optionally write ``out``."""
    root = Path(uo_root).expanduser().resolve()
    _kind, product = _resolve_db_or_uo(root)
    name = resolve_view_name(view)
    return _dump_from_uo(product, name, out=out)


def _dump_from_uo(
    uo_path: Path,
    name: str,
    *,
    out: str | Path | None = None,
) -> dict[str, Any]:
    from ascendc_codemap_mcp.engine.query.engine import CodeMapQuery
    from ascendc_codemap_mcp.engine.store.reader import list_views, load_production_view, read_codemap, read_meta

    meta = read_meta(uo_path)
    cm = read_codemap(uo_path)
    q = CodeMapQuery(codemap=cm, path=str(uo_path))

    if name in {"summary", "summary.yaml"}:
        payload: Any = q.summary()
    elif name in {"macros", "macros.yaml"}:
        payload = {"entities": [e.to_dict() for e in cm.by_kind("MACRO")]}
    elif name in {"templates", "templates.yaml"}:
        payload = {
            "templates": [e.to_dict() for e in cm.by_kind("TEMPLATE")],
            "instances": [e.to_dict() for e in cm.by_kind("TEMPLATE_INSTANCE")],
        }
    elif name in {"arch", "arch.yaml"}:
        payload = {
            "arch": [e.to_dict() for e in cm.by_kind("ARCH")],
            "build_variant": [e.to_dict() for e in cm.by_kind("BUILD_VARIANT")],
        }
    elif name in {"kernel", "views/kernel.yaml"}:
        payload = {"kernels": [e.to_dict() for e in cm.by_kind("KERNEL")]}
    elif name in {"ir/tg_host_view.yaml", "tg_host_view", "host"}:
        payload = load_production_view(uo_path, "ir/tg_host_view.yaml") or load_production_view(
            uo_path, "tg_host_view"
        )
        if payload is None:
            payload = {
                "schema": "tg-host-view/v1",
                "fields": [
                    e.to_dict()
                    for e in cm.by_kind("FIELD") + cm.by_kind("TILING_FIELD")
                ],
                "source": {"product": str(uo_path), "meta": meta},
            }
    else:
        payload = load_production_view(uo_path, name)
        if payload is None:
            # Fall back to full codemap dict slices.
            if name in {"graph", "ir/operator_graph.yaml", "operator_graph"}:
                payload = cm.to_dict()
            else:
                available = list_views(uo_path)
                raise KeyError(
                    f"view not found in .uo: {name}; available={available[:40]}"
                )

    if out is not None:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(
                payload,
                allow_unicode=True,
                sort_keys=True,
                default_flow_style=False,
            ),
            encoding="utf-8",
        )
    return {
        "ok": True,
        "view": name,
        "product": str(uo_path),
        "out": str(out) if out else "",
        "payload": payload,
    }


def dump_all_views(
    uo_root: str | Path, *, out_dir: str | Path | None = None
) -> dict[str, Any]:
    """Materialize every stored view under out_dir."""
    root = Path(uo_root).expanduser().resolve()
    _kind, product = _resolve_db_or_uo(root)
    target = Path(out_dir).expanduser().resolve() if out_dir else root
    written: list[str] = []
    from ascendc_codemap_mcp.engine.store.reader import list_views, load_production_view, read_codemap

    for name in list_views(product):
        payload = load_production_view(product, name)
        if payload is None:
            continue
        path = target / (name if "/" in name or name.endswith(".yaml") else f"{name}.yaml")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=True, default_flow_style=False),
            encoding="utf-8",
        )
        written.append(name)
    cm = read_codemap(product)
    summary_path = target / "summary.yaml"
    summary_path.write_text(
        yaml.safe_dump(cm.summary(), allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )
    written.append("summary.yaml")
    return {"ok": True, "out_dir": target.as_posix(), "written": written, "product": str(product)}


def dump_path_query(uo_file: Path, start: str, end: str) -> dict[str, Any]:
    from ascendc_codemap_mcp.engine.query.engine import open_codemap_query

    q = open_codemap_query(uo_file)
    return {"ok": True, "path": q.find_path(start, end)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m uo_init.dump",
        description="Dump CodeMap views from .uo",
    )
    parser.add_argument("target", nargs="?", default="", help=".uo file or view name")
    parser.add_argument("view", nargs="?", default="", help="view name when target is .uo")
    parser.add_argument("--uo-root", default="", help="UO root or .uo path")
    parser.add_argument("--out", default="", help="output file path")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--host", action="store_true")
    parser.add_argument("--kernel", action="store_true")
    parser.add_argument("--macros", action="store_true")
    parser.add_argument("--templates", action="store_true")
    parser.add_argument("--arch", action="store_true")
    parser.add_argument("--symbol", default="", help="dump symbol by name")
    parser.add_argument("--path", nargs=2, metavar=("FROM", "TO"), help="find_path FROM TO")
    parser.add_argument(
        "--materialize-tg",
        action="store_true",
        help="backfill TPL/D + tg_host_view/operator_graph view_blobs into .uo",
    )
    parser.add_argument("--header", default="", help="explicit template_tiling_key.h for --materialize-tg")
    parser.add_argument("--op-root", default="", help="operator source root for --materialize-tg")
    parser.add_argument("--format", default="yaml", choices=("yaml", "json"))
    args = parser.parse_args(argv)

    # Support: uo-dump file.uo --summary
    uo = Path(args.uo_root).expanduser().resolve() if args.uo_root else Path.cwd()
    target = Path(args.target) if args.target else Path()
    if args.target and (target.suffix == ".uo" or target.is_file()):
        uo = target.expanduser().resolve()
        view = args.view
    else:
        view = args.target or args.view

    if args.summary:
        view = "summary"
    elif args.host:
        view = "host"
    elif args.kernel:
        view = "kernel"
    elif args.macros:
        view = "macros"
    elif args.templates:
        view = "templates"
    elif args.arch:
        view = "arch"

    if args.materialize_tg:
        try:
            from ascendc_codemap_mcp.engine.tg_projection import backfill_from_source

            kind, product = _resolve_db_or_uo(uo)
            if kind != "uo":
                raise FileNotFoundError("--materialize-tg requires a .uo product")
            from ascendc_codemap_mcp.engine.source_layout import is_product_architecture

            if (
                is_product_architecture(product.parent.name)
                and product.parent.parent.name == ".ascendc-codemap"
            ):
                op_root = product.parent.parent.parent
            else:
                op_root = product.parent.parent.parent
            if args.op_root:
                op_root = Path(args.op_root).expanduser().resolve()
            result = backfill_from_source(
                op_root,
                uo_path=product,
                tiling_key_header=args.header or None,
            )
            print(json.dumps(result, ensure_ascii=False))
            return 0 if result.get("ok") else 1
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"ok": False, "error": str(exc)[:400]}, ensure_ascii=False))
            return 1

    if args.path:
        try:
            kind, product = _resolve_db_or_uo(uo)
            if kind != "uo":
                raise FileNotFoundError("find_path requires a .uo product")
            result = dump_path_query(product, args.path[0], args.path[1])
            _emit(result, args.format)
            return 0
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"ok": False, "error": str(exc)[:400]}, ensure_ascii=False))
            return 1

    if args.symbol:
        try:
            from ascendc_codemap_mcp.engine.query.engine import open_codemap_query

            kind, product = _resolve_db_or_uo(uo)
            q = open_codemap_query(product if kind == "uo" else uo)
            _emit({"ok": True, "symbol": args.symbol, "hits": q.find_symbol(args.symbol)}, args.format)
            return 0
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"ok": False, "error": str(exc)[:400]}, ensure_ascii=False))
            return 1

    if args.list:
        try:
            kind, product = _resolve_db_or_uo(uo)
            if kind == "uo":
                from ascendc_codemap_mcp.engine.store.reader import list_views

                print(json.dumps({"ok": True, "views": list_views(product)}, ensure_ascii=False))
            return 0
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"ok": False, "error": str(exc)[:400]}, ensure_ascii=False))
            return 1

    try:
        if args.all or view in {"all", "--all"}:
            result = dump_all_views(uo, out_dir=args.out or None)
            print(json.dumps({k: v for k, v in result.items() if k != "payload"}, ensure_ascii=False))
            return 0
        if not view:
            parser.error("view required (or pass --summary/--all/--list)")
        result = dump_view(uo, view, out=args.out or None)
        if args.out:
            print(json.dumps({k: v for k, v in result.items() if k != "payload"}, ensure_ascii=False))
        else:
            _emit(result.get("payload"), args.format)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": str(exc)[:400]}, ensure_ascii=False))
        return 1


def _emit(payload: Any, fmt: str) -> None:
    if fmt == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        yaml.safe_dump(
            payload,
            sys.stdout,
            allow_unicode=True,
            sort_keys=True,
            default_flow_style=False,
        )


if __name__ == "__main__":
    raise SystemExit(main())
