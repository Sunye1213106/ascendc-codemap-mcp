# -*- coding: utf-8 -*-
"""Typed CodeMap queries. Engine ``agent_query`` stays; this is the contract."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

from ascendc_codemap_mcp.constants import SERVER_VERSION
from ascendc_codemap_mcp.service import evidence as evidence_mod
from ascendc_codemap_mcp.service.envelope import (
    STATE_BUILDING,
    VERDICT_ANSWERED,
    VERDICT_PARTIAL,
    VERDICT_UNKNOWN,
    envelope,
    fail,
)
from ascendc_codemap_mcp.service.identity import (
    CodemapRef,
    is_ref,
    public_handle,
    resolve,
)
from ascendc_codemap_mcp.service import runtime

_KIND_LAYER = {
    "TILING_KEY": "template",
    "TEMPLATE": "template",
    "TEMPLATE_ARG": "template",
    "TEMPLATE_INSTANCE": "template",
    "BUILD_VARIANT": "template",
    "TILING_FIELD": "host",
    "TILING_DATA": "host",
    "FUNCTION": "host",
    "METHOD": "host",
    "FIELD": "host",
    "INPUT": "host",
    "OUTPUT": "host",
    "KERNEL": "kernel",
    "PIPE": "kernel",
    "BUFFER": "kernel",
    "OPERATION": "kernel",
    "EVENT": "kernel",
    "QUEUE": "kernel",
    "REGISTER": "kernel",
    "CONTRACT": "host",
    "PREDICATE": "host",
    "BRANCH": "host",
    "COMPILE_VAR": "template",
}

_PAGE_KEYS = (
    "cards",
    "sel_sites",
    "nearby",
    "template_blocks",
    "dim_names",
    "tiling_data_names",
    "phases",
    "neighbors",
    "hits",
    "text_hits",
)


def _meta(product: Path) -> dict[str, Any]:
    from ascendc_codemap_mcp.engine.store.reader import read_meta

    try:
        return dict(read_meta(product))
    except Exception:  # noqa: BLE001
        return {}


def _need_ref(
    *,
    codemap_id: str = "",
    project: str = "",
    architecture: str = "",
    require_indexed: bool = True,
) -> CodemapRef | dict[str, Any]:
    return resolve(
        codemap_id=codemap_id,
        project=project,
        architecture=architecture,
        registry=runtime.registry,
        require_indexed=require_indexed,
    )


def _building(ref: CodemapRef) -> dict[str, Any]:
    product = ref.product
    meta = _meta(product) if product is not None and product.is_file() else {}
    from ascendc_codemap_mcp.service.freshness import compute

    info = compute(ref.project, meta=meta, building=True, blocked=runtime.is_blocked(ref.id))
    return envelope(
        ok=True,
        state=STATE_BUILDING,
        updated=False,
        verdict=VERDICT_UNKNOWN,
        codemap=public_handle(ref, meta=meta, freshness_info=info),
        data={},
        coverage={"returned": 0, "total": 0, "truncated": False},
        extra={"error_code": "BUILDING", "error": "CodeMap snapshot is being rebuilt"},
    )


def _infer_layer(payload: dict[str, Any]) -> str:
    shape = str(payload.get("shape") or "")
    if shape == "cover":
        return "template"
    if shape == "index":
        return "host"
    for key in ("cards", "hits", "seeds"):
        rows = payload.get(key)
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            kind = str(rows[0].get("kind") or "")
            if kind in _KIND_LAYER:
                return _KIND_LAYER[kind]
    enc = payload.get("enclosing")
    if isinstance(enc, dict):
        kind = str(enc.get("kind") or "")
        if kind in _KIND_LAYER:
            return _KIND_LAYER[kind]
    return "host"


def _edges_truncated(payload: dict[str, Any]) -> bool:
    edges = payload.get("edges")
    if not isinstance(edges, dict):
        cards = payload.get("cards")
        if isinstance(cards, list):
            for card in cards:
                if isinstance(card, dict) and _edges_truncated(card):
                    return True
        return False
    for bucket in edges.values():
        if not isinstance(bucket, dict):
            continue
        neighbors = bucket.get("neighbors")
        count = int(bucket.get("count") or 0)
        if bucket.get("truncated") or (
            isinstance(neighbors, list) and count > len(neighbors)
        ):
            return True
    return False


def _infer_verdict(payload: dict[str, Any], *, truncated: bool) -> str:
    completeness = str(payload.get("completeness") or "")
    if completeness == "UNKNOWN" or payload.get("empty_reason") == "nl_or_multi_token":
        return VERDICT_UNKNOWN
    if completeness == "INCOMPLETE" or completeness == "AMBIGUOUS":
        return VERDICT_PARTIAL
    if completeness == "COMPLETE":
        return VERDICT_ANSWERED if not truncated else VERDICT_PARTIAL
    if not payload.get("ok"):
        return VERDICT_UNKNOWN
    shape = str(payload.get("shape") or "")
    count = int(payload.get("count") or 0)
    if truncated or payload.get("truncated"):
        return VERDICT_PARTIAL
    if shape == "index":
        return VERDICT_ANSWERED
    if count == 0:
        return VERDICT_PARTIAL if payload.get("hint") else VERDICT_UNKNOWN
    if completeness and completeness not in {"full", "siblings_checked", ""}:
        if not payload.get("answerable"):
            return VERDICT_PARTIAL
    return VERDICT_ANSWERED


class CursorError(ValueError):
    error_code = "CURSOR_INVALID"


def query_fingerprint(
    *,
    engine: str,
    pattern: str = "",
    file: str = "",
    line: int = 0,
    line_end: int = 0,
) -> str:
    blob = json.dumps(
        {
            "e": str(engine or ""),
            "p": str(pattern or ""),
            "f": str(file or ""),
            "l": int(line or 0),
            "le": int(line_end or 0),
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def decode_cursor(cursor: str) -> dict[str, Any]:
    raw = str(cursor or "").strip()
    if not raw:
        return {"o": 0, "s": "", "q": "", "c": ""}
    try:
        padded = raw + ("=" * ((4 - len(raw) % 4) % 4))
        data = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
        if not isinstance(data, dict):
            raise CursorError("cursor is not an object")
        return {
            "o": max(0, int(data.get("o") or 0)),
            "s": str(data.get("s") or ""),
            "q": str(data.get("q") or ""),
            "c": str(data.get("c") or ""),
        }
    except CursorError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise CursorError("invalid cursor") from exc


def encode_cursor(
    offset: int,
    *,
    snapshot: str,
    query: str,
    collection: str = "",
) -> str:
    payload: dict[str, Any] = {
        "s": str(snapshot or ""),
        "q": str(query or ""),
        "o": int(offset),
    }
    if collection:
        payload["c"] = str(collection)
    blob = json.dumps(payload, separators=(",", ":")).encode("ascii")
    return base64.urlsafe_b64encode(blob).decode("ascii").rstrip("=")


def _primary_list(payload: dict[str, Any]) -> tuple[str | None, list[Any]]:
    for key in _PAGE_KEYS:
        rows = payload.get(key)
        if isinstance(rows, list) and rows:
            return key, rows
    return None, []


def paginate(
    payload: dict[str, Any],
    *,
    limit: int,
    offset: int,
    snapshot: str,
    query: str,
) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    page = max(1, min(int(limit or 8), 32))
    key, rows = _primary_list(payload)
    total_hint = int(payload.get("count") or payload.get("total") or 0)
    nested_trunc = _edges_truncated(payload)
    if key is None:
        total = total_hint
        returned = total if not payload.get("truncated") else min(page, total)
        primary_more = bool(payload.get("truncated"))
        truncated = bool(primary_more or nested_trunc)
        coverage = {
            "returned": returned,
            "total": total,
            "truncated": truncated,
            "nested_truncated": nested_trunc and not primary_more,
            "token_budget": 24_000,
        }
        # Nested edge samples are not a continuable collection.
        nxt = (
            encode_cursor(offset + page, snapshot=snapshot, query=query)
            if primary_more
            else None
        )
        return payload, coverage, nxt
    total = max(total_hint, len(rows) + offset)
    window = rows[offset : offset + page] if offset else rows[:page]
    if offset or len(rows) > page:
        payload = dict(payload)
        payload[key] = window
    primary_more = bool(
        (offset + len(window) < len(rows)) or (total_hint > offset + len(window))
    )
    truncated = bool(payload.get("truncated") or nested_trunc or primary_more)
    coverage = {
        "returned": len(window),
        "total": total,
        "truncated": truncated,
        "nested_truncated": nested_trunc and not primary_more,
        "token_budget": 24_000,
    }
    nxt = (
        encode_cursor(offset + len(window), snapshot=snapshot, query=query)
        if primary_more and window
        else None
    )
    return payload, coverage, nxt


def _run_query(
    ref: CodemapRef,
    *,
    pattern: str = "",
    file: str = "",
    line: int = 0,
    line_end: int = 0,
    limit: int = 8,
    cursor: str = "",
    flatten: bool = False,
    engine: str = "",
    expected_snapshot_id: str = "",
    evidence_id: str = "",
) -> dict[str, Any]:
    if runtime.is_building(ref.id) or not runtime.locks.try_read(ref.id):
        return _building(ref)
    product = ref.product
    if product is None or not Path(product).is_file():
        runtime.locks.release_read(ref.id)
        return fail(
            f"no .uo for {ref.id}",
            error_code="CODEMAP_NOT_INDEXED",
        )
    try:
        meta = _meta(product)
        from ascendc_codemap_mcp.service.freshness import compute

        info = compute(
            ref.project,
            meta=meta,
            building=False,
            blocked=runtime.is_blocked(ref.id),
        )
        handle = public_handle(ref, meta=meta, freshness_info=info)
        snapshot = str(handle.get("snapshot_id") or "")
        expected = str(expected_snapshot_id or "").strip()
        if expected and snapshot and expected != snapshot:
            return fail(
                "evidence belongs to a previous CodeMap snapshot; re-resolve",
                error_code="SNAPSHOT_CHANGED",
                extra={
                    "codemap": handle,
                    "expected_snapshot_id": expected,
                    "current_snapshot_id": snapshot,
                },
            )
        path = str(file or "").strip()
        line_n = int(line or 0)
        ev_id = str(evidence_id or "").strip()
        if ev_id and not (path and line_n):
            with runtime.cache.open(product) as query:
                located = evidence_mod.lookup_span(query, ev_id)
            if located is None:
                return fail(
                    f"unknown evidence_id: {ev_id}",
                    error_code="EVIDENCE_NOT_FOUND",
                    extra={"codemap": handle},
                )
            path = str(located.get("file") or "")
            line_n = int(located.get("line") or 0)
            if line_end <= 0:
                line_end = line_n
        if engine == "codemap_evidence" and (not path or line_n <= 0):
            return fail(
                "evidence_id, entity_id, or file+line is required",
                error_code="EVIDENCE_REQUIRED",
                extra={"codemap": handle},
            )
        file = path
        line = line_n
        qfp = query_fingerprint(
            engine=engine,
            pattern=pattern,
            file=file,
            line=line,
            line_end=line_end,
        )
        try:
            decoded = decode_cursor(cursor)
        except CursorError as exc:
            return fail(str(exc), error_code="CURSOR_INVALID")
        if str(cursor or "").strip():
            if not decoded.get("s") or not decoded.get("q"):
                return fail(
                    "cursor is bound to snapshot_id and query; re-run without cursor",
                    error_code="CURSOR_MISMATCH",
                    extra={"codemap": handle},
                )
            if decoded["s"] != snapshot:
                return fail(
                    "cursor snapshot does not match the current CodeMap; re-query",
                    error_code="SNAPSHOT_CHANGED",
                    extra={
                        "codemap": handle,
                        "expected_snapshot_id": decoded["s"],
                        "current_snapshot_id": snapshot,
                    },
                )
            if decoded["q"] != qfp:
                return fail(
                    "cursor belongs to a different query",
                    error_code="CURSOR_MISMATCH",
                    extra={"codemap": handle},
                )
        page = max(1, min(int(limit or 8), 32))
        offset = int(decoded.get("o") or 0)
        fetch_limit = page + offset
        with runtime.cache.open(product) as query:
            payload = query.agent_query(
                pattern=str(pattern or ""),
                file=str(file or ""),
                line=int(line or 0),
                line_end=int(line_end or 0),
                limit=fetch_limit,
            )
        payload, coverage, nxt = paginate(
            payload,
            limit=page,
            offset=offset,
            snapshot=snapshot,
            query=qfp,
        )
        ev = evidence_mod.collect(
            payload, op_root=ref.project, snapshot=snapshot
        )
        truncated = bool(coverage.get("truncated"))
        verdict = _infer_verdict(payload, truncated=truncated)
        layer = _infer_layer(payload)
        extra = {
            "engine": engine or "codemap_query",
            "shape": payload.get("shape"),
            "count": payload.get("count"),
            "version": SERVER_VERSION,
        }
        body = envelope(
            ok=bool(payload.get("ok", True)),
            codemap=handle,
            verdict=verdict,
            layer=layer,
            data=payload,
            evidence=ev,
            coverage=coverage,
            next_cursor=nxt,
            extra=extra if flatten else {"engine": extra["engine"]},
        )
        if flatten:
            skip = set(body)
            for key, value in payload.items():
                if key not in skip:
                    body[key] = value
            if engine:
                body["engine"] = engine
        return body
    finally:
        runtime.locks.release_read(ref.id)


def overview(
    *,
    codemap_id: str = "",
    project: str = "",
    architecture: str = "",
    limit: int = 8,
    cursor: str = "",
) -> dict[str, Any]:
    ref = _need_ref(codemap_id=codemap_id, project=project, architecture=architecture)
    if not is_ref(ref):
        return ref  # type: ignore[return-value]
    return _run_query(ref, limit=limit, cursor=cursor, engine="codemap_overview")


def symbol(
    *,
    codemap_id: str = "",
    project: str = "",
    architecture: str = "",
    symbol: str,
    limit: int = 8,
    cursor: str = "",
) -> dict[str, Any]:
    ref = _need_ref(codemap_id=codemap_id, project=project, architecture=architecture)
    if not is_ref(ref):
        return ref  # type: ignore[return-value]
    name = str(symbol or "").strip()
    if not name:
        return fail("symbol is required", error_code="SYMBOL_REQUIRED")
    return _run_query(
        ref, pattern=name, limit=limit, cursor=cursor, engine="codemap_symbol"
    )


def selection(
    *,
    codemap_id: str = "",
    project: str = "",
    architecture: str = "",
    dim: str = "",
    value: str = "",
    limit: int = 8,
    cursor: str = "",
) -> dict[str, Any]:
    ref = _need_ref(codemap_id=codemap_id, project=project, architecture=architecture)
    if not is_ref(ref):
        return ref  # type: ignore[return-value]
    dim_s = str(dim or "").strip()
    value_s = str(value or "").strip()
    if not dim_s:
        return fail("dim is required (e.g. IsPse)", error_code="DIM_REQUIRED")
    if dim_s.startswith("Dim="):
        pattern = dim_s
    elif "=" in dim_s and not value_s:
        pattern = dim_s
    elif value_s:
        pattern = f"{dim_s}={value_s}"
    else:
        pattern = f"Dim={dim_s}"
    return _run_query(
        ref, pattern=pattern, limit=limit, cursor=cursor, engine="codemap_selection"
    )


def evidence(
    *,
    codemap_id: str = "",
    project: str = "",
    architecture: str = "",
    evidence_id: str = "",
    entity_id: str = "",
    file: str = "",
    line: int = 0,
    line_end: int = 0,
    limit: int = 8,
    cursor: str = "",
    expected_snapshot_id: str = "",
) -> dict[str, Any]:
    ref = _need_ref(codemap_id=codemap_id, project=project, architecture=architecture)
    if not is_ref(ref):
        return ref  # type: ignore[return-value]
    path = str(file or "").strip()
    line_n = int(line or 0)
    ev_id = str(evidence_id or entity_id or "").strip()
    if not ev_id and (not path or line_n <= 0):
        return fail(
            "evidence_id, entity_id, or file+line is required",
            error_code="EVIDENCE_REQUIRED",
        )
    return _run_query(
        ref,
        file=path,
        line=line_n,
        line_end=int(line_end or line_n),
        limit=limit,
        cursor=cursor,
        engine="codemap_evidence",
        expected_snapshot_id=str(expected_snapshot_id or "").strip(),
        evidence_id=ev_id,
    )


def explore(
    *,
    codemap_id: str = "",
    project: str = "",
    architecture: str = "",
    query: str = "",
    pattern: str = "",
    file: str = "",
    line: int = 0,
    line_end: int = 0,
    evidence_id: str = "",
    cursor: str = "",
    limit: int = 8,
    expected_snapshot_id: str = "",
) -> dict[str, Any]:
    """Single agent entry: ident / Dim=V / file:line / phenomenon / evidence page."""
    ev = str(evidence_id or "").strip()
    if ev:
        return evidence(
            codemap_id=codemap_id,
            project=project,
            architecture=architecture,
            evidence_id=ev,
            file=file,
            line=line,
            line_end=line_end,
            limit=limit,
            cursor=cursor,
            expected_snapshot_id=expected_snapshot_id,
        )
    return query_codemap(
        codemap_id=codemap_id,
        project=project,
        architecture=architecture,
        pattern=str(query or pattern or ""),
        file=file,
        line=line,
        line_end=line_end,
        limit=limit,
        cursor=cursor,
    )


def query_codemap(
    *,
    project: str = "",
    architecture: str = "",
    pattern: str = "",
    file: str = "",
    line: int = 0,
    line_end: int = 0,
    codemap_id: str = "",
    limit: int = 8,
    cursor: str = "",
) -> dict[str, Any]:
    """Compatibility facade for the four historical query shapes."""
    ref = _need_ref(
        codemap_id=codemap_id,
        project=project,
        architecture=architecture,
        require_indexed=True,
    )
    if not is_ref(ref):
        return ref  # type: ignore[return-value]
    return _run_query(
        ref,
        pattern=pattern,
        file=file,
        line=line,
        line_end=line_end,
        limit=limit,
        cursor=cursor,
        flatten=True,
        engine="query_codemap",
    )
