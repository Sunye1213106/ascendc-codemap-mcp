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
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        from ascendc_codemap_mcp.server import serve

        return serve()
    if args[0] in {"-h", "--help", "help"}:
        print(
            f"{PRODUCT_NAME} {SERVER_VERSION}\n"
            "  (no args)              stdio MCP server\n"
            "  serve                  stdio MCP server\n"
            "  install [--dry-run]    configure Cursor / Claude Code / Codex / OpenCode\n"
            "  uninstall [--dry-run]  remove owned MCP entries and skills\n"
            "  doctor  --project DIR --architecture ARCH\n"
            "  index   --project DIR --architecture ARCH\n"
            "  update  --project DIR --architecture ARCH [--confirm-scope]\n"
            "  query   --project DIR --architecture ARCH [pattern] [--file F --line N]\n"
            "  status  --project DIR --architecture ARCH\n"
        )
        return 0
    cmd = args[0]
    rest = args[1:]
    if cmd == "serve":
        from ascendc_codemap_mcp.server import serve

        return serve()
    if cmd == "install":
        from ascendc_codemap_mcp.install import run_install

        dry = "--dry-run" in rest
        return run_install(dry_run=dry)
    if cmd == "uninstall":
        from ascendc_codemap_mcp.install import run_uninstall

        dry = "--dry-run" in rest
        return run_uninstall(dry_run=dry)
    if cmd in {"doctor", "index", "update", "query", "status"}:
        parser = argparse.ArgumentParser(prog=f"{PRODUCT_NAME} {cmd}")
        parser.add_argument("--project", default="")
        parser.add_argument("--architecture", default="")
        parser.add_argument("pattern", nargs="?", default="")
        parser.add_argument("--file", default="")
        parser.add_argument("--line", type=int, default=0)
        parser.add_argument("--line-end", type=int, default=0)
        parser.add_argument("--confirm-scope", action="store_true")
        ns = parser.parse_args(rest)
        from ascendc_codemap_mcp import tools as tool_impl

        if cmd == "doctor":
            return _print(
                tool_impl.doctor(project=ns.project, architecture=ns.architecture)
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
                )
            )
        if cmd == "status":
            return _print(
                tool_impl.status(project=ns.project, architecture=ns.architecture)
            )
        try:
            return _print(
                tool_impl.query_codemap(
                    project=ns.project,
                    architecture=ns.architecture,
                    pattern=ns.pattern,
                    file=ns.file,
                    line=ns.line,
                    line_end=ns.line_end,
                )
            )
        except Exception as exc:  # noqa: BLE001
            return _print({"ok": False, "error": str(exc)[:500]})
    print(f"unknown command: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
