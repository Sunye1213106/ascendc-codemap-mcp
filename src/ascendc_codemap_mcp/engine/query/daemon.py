# -*- coding: utf-8 -*-
"""Long-lived uo-query process: one SQLite connection, many hops.

Agent CLI still looks like ``acp uo-query --project … <ident>``. The client
talks to this process over localhost so each hop skips interpreter + import +
SQLite open.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path

from ascendc_codemap_mcp.engine.query.sql import UoSqlQuery
from ascendc_codemap_mcp.engine.query_client import _port_file
from ascendc_codemap_mcp.engine.store.reader import find_uo_product

_HOST = "127.0.0.1"


def _write_endpoint(product: Path, port: int) -> None:
    path = _port_file(product)
    payload = {
        "port": port,
        "pid": os.getpid(),
        "mtime_ns": int(product.stat().st_mtime_ns),
        "product": str(product),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _unlink_endpoint(product: Path) -> None:
    path = _port_file(product)
    try:
        path.unlink()
    except OSError:
        pass


def _handle(q: UoSqlQuery, req: dict) -> dict:
    op = str(req.get("op") or "query")
    if op == "ping":
        return {"ok": True, "op": "ping", "pid": os.getpid()}
    if op == "shutdown":
        return {"ok": True, "op": "shutdown"}
    payload = q.agent_query(
        pattern=str(req.get("pattern") or ""),
        file=str(req.get("file") or ""),
        line=int(req.get("line") or 0),
        line_end=int(req.get("line_end") or 0),
        limit=int(req.get("limit") or 8),
    )
    payload["engine"] = "uo_init.uo_query"
    payload["daemon"] = True
    return payload


def serve(product: Path, *, architecture: str = "") -> int:
    os.environ["UO_QUERY_DAEMON"] = "1"
    q = UoSqlQuery(product)
    # Warm the connection and leaf index before accepting clients.
    q.agent_query()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((_HOST, 0))
    sock.listen(16)
    sock.settimeout(1800.0)
    port = int(sock.getsockname()[1])
    _write_endpoint(product, port)
    try:
        while True:
            try:
                conn, _addr = sock.accept()
            except TimeoutError:
                break
            except OSError:
                break
            with conn:
                chunks: list[bytes] = []
                conn.settimeout(60.0)
                try:
                    while True:
                        piece = conn.recv(1 << 16)
                        if not piece:
                            break
                        chunks.append(piece)
                        if b"\n" in piece:
                            break
                    line = b"".join(chunks).split(b"\n", 1)[0]
                    req = json.loads(line.decode("utf-8")) if line else {"op": "ping"}
                    if not isinstance(req, dict):
                        req = {"op": "ping"}
                    reply = _handle(q, req)
                    conn.sendall(
                        (json.dumps(reply, ensure_ascii=False, default=str) + "\n").encode(
                            "utf-8"
                        )
                    )
                    if str(req.get("op") or "") == "shutdown":
                        break
                except Exception as exc:  # noqa: BLE001
                    try:
                        conn.sendall(
                            (
                                json.dumps({"ok": False, "error": str(exc)[:300]}) + "\n"
                            ).encode("utf-8")
                        )
                    except OSError:
                        pass
    finally:
        _unlink_endpoint(product)
        try:
            q.close()
        except Exception:  # noqa: BLE001
            pass
        sock.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="uo-query-daemon")
    parser.add_argument("--product", default="")
    parser.add_argument("--project", default="")
    parser.add_argument("--architecture", default="")
    parser.add_argument("--op-name", default="")
    args = parser.parse_args(argv)
    product = Path(args.product).expanduser() if args.product else None
    if product is None or not product.is_file():
        root = Path(args.project).expanduser().resolve()
        found = find_uo_product(
            root, op_name=str(args.op_name or ""), architecture=str(args.architecture or "")
        )
        if found is None:
            print("missing .uo product", file=sys.stderr)
            return 2
        product = Path(found)
    return serve(product, architecture=str(args.architecture or ""))


if __name__ == "__main__":
    raise SystemExit(main())
