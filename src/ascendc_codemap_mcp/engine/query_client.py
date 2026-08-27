# -*- coding: utf-8 -*-
"""Stdlib-only client for the uo-query daemon.

Agent-facing CLI stays the four existing shapes. This module never imports
``uo_init.query.sql`` so a hop can skip the 200 ms engine import when a
daemon is already holding the SQLite connection.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_IDLE_ENV = "UO_QUERY_NODAEMON"
_HOST = "127.0.0.1"


def _port_file(product: Path) -> Path:
    return product.with_name(product.name + ".queryd")


def _read_endpoint(product: Path) -> dict[str, Any] | None:
    path = _port_file(product)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    try:
        mtime = int(product.stat().st_mtime_ns)
    except OSError:
        return None
    if int(data.get("mtime_ns") or 0) != mtime:
        return None
    port = int(data.get("port") or 0)
    if port <= 0:
        return None
    return data


def _send(port: int, payload: dict[str, Any], *, timeout: float = 30.0) -> dict[str, Any]:
    raw = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
    with socket.create_connection((_HOST, port), timeout=timeout) as sock:
        sock.sendall(raw)
        chunks: list[bytes] = []
        sock.settimeout(timeout)
        while True:
            piece = sock.recv(1 << 16)
            if not piece:
                break
            chunks.append(piece)
            if b"\n" in piece:
                break
    line = b"".join(chunks).split(b"\n", 1)[0]
    if not line:
        raise ConnectionError("empty daemon reply")
    out = json.loads(line.decode("utf-8"))
    if not isinstance(out, dict):
        raise ConnectionError("daemon reply is not an object")
    return out


def ping(product: Path) -> bool:
    ep = _read_endpoint(product)
    if ep is None:
        return False
    try:
        reply = _send(int(ep["port"]), {"op": "ping"}, timeout=2.0)
    except (OSError, json.JSONDecodeError, ConnectionError):
        return False
    return bool(reply.get("ok"))


def _spawn_daemon(product: Path, *, architecture: str = "") -> None:
    env = os.environ.copy()
    env["UO_QUERY_DAEMON"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [
        sys.executable,
        "-m",
        "uo_init.query.daemon",
        "--product",
        str(product),
    ]
    if architecture:
        cmd.extend(["--architecture", architecture])
    kwargs: dict[str, Any] = {
        "env": env,
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
    }
    if os.name == "nt":
        kwargs["creationflags"] = (
            getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        )
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen(cmd, **kwargs)


def ensure_daemon(
    product: Path,
    *,
    architecture: str = "",
    wait_s: float = 8.0,
) -> dict[str, Any] | None:
    if ping(product):
        return _read_endpoint(product)
    _spawn_daemon(product, architecture=architecture)
    deadline = time.perf_counter() + wait_s
    while time.perf_counter() < deadline:
        if ping(product):
            return _read_endpoint(product)
        time.sleep(0.05)
    return _read_endpoint(product) if ping(product) else None


def try_agent_query(
    product: Path,
    *,
    pattern: str = "",
    file: str = "",
    line: int = 0,
    line_end: int = 0,
    limit: int = 8,
    architecture: str = "",
) -> dict[str, Any] | None:
    """Return a query payload from the daemon, or None to fall back in-process."""
    if str(os.environ.get(_IDLE_ENV) or "").strip() in {"1", "true", "yes"}:
        return None
    ep = _read_endpoint(product)
    if ep is not None:
        try:
            reply = _send(
                int(ep["port"]),
                {
                    "op": "query",
                    "pattern": pattern,
                    "file": file,
                    "line": int(line or 0),
                    "line_end": int(line_end or 0),
                    "limit": int(limit or 8),
                },
                timeout=60.0,
            )
        except (OSError, json.JSONDecodeError, ConnectionError):
            reply = None
        else:
            if reply.get("error") and not reply.get("shape"):
                reply = None
            if reply is not None:
                return reply
    ep = ensure_daemon(product, architecture=architecture)
    if ep is None:
        return None
    try:
        reply = _send(
            int(ep["port"]),
            {
                "op": "query",
                "pattern": pattern,
                "file": file,
                "line": int(line or 0),
                "line_end": int(line_end or 0),
                "limit": int(limit or 8),
            },
            timeout=60.0,
        )
    except (OSError, json.JSONDecodeError, ConnectionError):
        return None
    if reply.get("error") and not reply.get("shape"):
        return None
    return reply
