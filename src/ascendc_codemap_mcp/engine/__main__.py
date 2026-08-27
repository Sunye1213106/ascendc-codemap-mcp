# -*- coding: utf-8 -*-
"""``python -m uo_init`` entrypoints (dump / locate helpers)."""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(
            "usage: python -m uo_init <dump|locate> ...\n"
            "  dump     dump CodeMap views from .uo\n"
            "  locate   source locator over the .uo product\n"
            "Also: python -m uo_init.dump <file.uo> --summary|--host|--path A B"
        )
        return 0 if args else 2
    cmd = args[0]
    rest = args[1:]
    if cmd == "dump":
        from ascendc_codemap_mcp.engine.dump import main as dump_main

        return dump_main(rest)
    if cmd == "locate":
        return _locate_main(rest)
    print(f"unknown command: {cmd}", file=sys.stderr)
    return 2


def _locate_main(argv: list[str]) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(prog="python -m uo_init locate")
    parser.add_argument("query")
    parser.add_argument("--uo-root", default="")
    parser.add_argument("--kind", default="", help="comma-separated node kinds")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--mode",
        choices=("search", "dim", "branch", "field"),
        default="search",
    )
    args = parser.parse_args(argv)
    from pathlib import Path

    from ascendc_codemap_mcp.engine.source_locator import open_locator

    uo = Path(args.uo_root).expanduser().resolve() if args.uo_root else Path.cwd()
    loc = open_locator(uo)
    kinds = [k for k in str(args.kind or "").split(",") if k.strip()]
    if args.mode == "dim":
        rows = loc.locate_dim(args.query, limit=args.limit)
    elif args.mode == "branch":
        rows = loc.locate_branch(args.query, limit=args.limit)
    elif args.mode == "field":
        rows = loc.locate_field(args.query, limit=args.limit)
    else:
        rows = loc.locate(args.query, kinds=kinds or None, limit=args.limit)
    print(
        json.dumps(
            {
                "ok": True,
                "count": len(rows),
                "locations": [r.to_dict() for r in rows],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
