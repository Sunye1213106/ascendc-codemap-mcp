# -*- coding: utf-8 -*-
"""ascendc-codemap-mcp CLI. No args → stdio MCP server."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from ascendc_codemap_mcp.constants import PRODUCT_NAME, SERVER_VERSION


def _print(payload: dict[str, Any]) -> int:
    print(json.dumps(payload, ensure_ascii=False, default=str))
    return 0 if payload.get("ok", True) else 1


def main(argv: list[str] | None = None) -> int:
    try:
        return _main(argv)
    finally:
        try:
            from ascendc_codemap_mcp.service import runtime

            runtime.shutdown()
        except Exception:  # noqa: BLE001
            pass


def _main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        from ascendc_codemap_mcp.server import serve

        return serve()
    if args[0] in {"-h", "--help", "help"}:
        print(
            f"{PRODUCT_NAME} {SERVER_VERSION}\n"
            "  (no args)              stdio MCP server\n"
            "  serve [--transport stdio|streamable-http] [--host HOST] [--port N]\n"
            "  install [--dry-run] [--host opencode|cursor|claude|codex|all]\n"
            "  uninstall [--dry-run] [--host opencode|cursor|claude|codex|all]\n"
            "  doctor    --project DIR --architecture ARCH\n"
            "  cann-extract [.run] --dest DIR [--fixup] [--list]\n"
            "  discover  [--project DIR] [--architecture ARCH]\n"
            "  index     --project DIR --architecture ARCH\n"
            "  update    (--codemap-id ID | --project DIR --architecture ARCH) [--confirm-scope]\n"
            "  query     (--codemap-id ID | --project DIR --architecture ARCH) [--operation OP] [--symbol S] [--file F --line N]\n"
            "  status    (--codemap-id ID | --project DIR --architecture ARCH)\n"
        )
        return 0
    cmd = args[0]
    rest = args[1:]
    if cmd == "serve":
        parser = argparse.ArgumentParser(prog=f"{PRODUCT_NAME} serve")
        parser.add_argument(
            "--transport",
            default="stdio",
            choices=("stdio", "streamable-http"),
        )
        parser.add_argument("--host", default="127.0.0.1")
        parser.add_argument("--port", type=int, default=8765)
        parser.add_argument("--path", default="/mcp", dest="streamable_http_path")
        ns = parser.parse_args(rest)
        from ascendc_codemap_mcp.server import serve

        if ns.transport == "stdio":
            return serve(transport="stdio")
        return serve(
            transport="streamable-http",
            host=ns.host,
            port=ns.port,
            streamable_http_path=ns.streamable_http_path,
            stateless_http=True,
            json_response=True,
        )
    if cmd in {"install", "uninstall"}:
        from ascendc_codemap_mcp.install import normalize_hosts, run_install, run_uninstall

        parser = argparse.ArgumentParser(prog=f"{PRODUCT_NAME} {cmd}")
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument(
            "--host",
            action="append",
            default=[],
            help="opencode, cursor, claude, codex, or all (repeatable / comma-separated)",
        )
        parser.add_argument(
            "platform",
            nargs="?",
            default="",
            help="same as --host; default is all coding agents",
        )
        ns = parser.parse_args(rest)
        raw = [*(ns.host or [])]
        if ns.platform:
            raw.append(ns.platform)
        try:
            hosts = normalize_hosts(raw or None)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        if cmd == "install":
            return run_install(dry_run=ns.dry_run, hosts=hosts)
        return run_uninstall(dry_run=ns.dry_run, hosts=hosts)
    if cmd in {"cann-extract", "extract-cann"}:
        from ascendc_codemap_mcp.cann_extract import main as extract_main

        return extract_main(rest)
    if cmd in {"doctor", "index", "update", "query", "status", "discover"}:
        parser = argparse.ArgumentParser(prog=f"{PRODUCT_NAME} {cmd}")
        parser.add_argument("--project", default="")
        parser.add_argument("--architecture", default="")
        parser.add_argument("--codemap-id", default="", dest="codemap_id")
        parser.add_argument("pattern", nargs="?", default="")
        from ascendc_codemap_mcp.engine.query.typed import OPERATIONS

        parser.add_argument("--operation", default="resolve", choices=OPERATIONS)
        parser.add_argument("--symbol", default="")
        parser.add_argument("--name", default="")
        parser.add_argument("--kind", default="")
        parser.add_argument("--layer", default="")
        parser.add_argument("--callee", default="")
        parser.add_argument("--referenced-symbol", default="", dest="referenced_symbol")
        parser.add_argument("--referenced-value", default="", dest="referenced_value")
        parser.add_argument("--literal", default="")
        parser.add_argument("--operator", default="")
        parser.add_argument("--dim", default="")
        parser.add_argument("--value", default="")
        parser.add_argument("--file", default="")
        parser.add_argument("--line", type=int, default=0)
        parser.add_argument("--line-end", type=int, default=0)
        parser.add_argument("--confirm-scope", action="store_true")
        parser.add_argument("--limit", type=int, default=8)
        parser.add_argument("--cursor", default="")
        ns = parser.parse_args(rest)
        from ascendc_codemap_mcp.service import control
        from ascendc_codemap_mcp.service import query as query_mod
        from ascendc_codemap_mcp import tools as tool_impl

        if cmd == "doctor":
            return _print(
                tool_impl.doctor(project=ns.project, architecture=ns.architecture)
            )
        if cmd == "discover":
            return _print(
                control.discover(project=ns.project, architecture=ns.architecture)
            )
        if cmd == "index":
            return _print(
                tool_impl.index_operator(
                    project=ns.project, architecture=ns.architecture
                )
            )
        if cmd == "update":
            return _print(
                tool_impl.update_operator(
                    project=ns.project,
                    architecture=ns.architecture,
                    confirm_scope=bool(ns.confirm_scope),
                    codemap_id=ns.codemap_id,
                )
            )
        if cmd == "status":
            return _print(
                tool_impl.status(
                    project=ns.project,
                    architecture=ns.architecture,
                    codemap_id=ns.codemap_id,
                )
            )
        return _print(
            query_mod.query(
                project=ns.project,
                architecture=ns.architecture,
                operation=ns.operation,
                symbol=str(ns.symbol or ns.pattern or ""),
                name=ns.name,
                file=ns.file,
                line=ns.line,
                line_end=ns.line_end,
                kind=ns.kind,
                layer=ns.layer,
                callee=ns.callee,
                referenced_symbol=ns.referenced_symbol,
                referenced_value=ns.referenced_value,
                literal=ns.literal,
                operator=ns.operator,
                dim=ns.dim,
                value=ns.value,
                codemap_id=ns.codemap_id,
                limit=ns.limit,
                cursor=ns.cursor,
            )
        )
    print(f"unknown command: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
