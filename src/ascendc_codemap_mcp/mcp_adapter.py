# -*- coding: utf-8 -*-
"""MCP adapter: official SDK owns transport, schema, cancellation, structured results."""
from __future__ import annotations

import asyncio
import atexit
import json
import os
import threading
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any, Literal

from mcp.server import MCPServer
from mcp.server.mcpserver.context import Context
from mcp.server.mcpserver.resources.templates import ResourceSecurity
from mcp.types import (
    CallToolResult,
    Completion,
    PromptReference,
    ResourceTemplateReference,
    TextContent,
    ToolAnnotations,
)

from ascendc_codemap_mcp.constants import SERVER_NAME, SERVER_VERSION
from ascendc_codemap_mcp.engine.query.contract import INSTRUCTIONS
from ascendc_codemap_mcp.service import runtime
from ascendc_codemap_mcp.service.control import (
    doctor as doctor_impl,
    discover as discover_impl,
    index_operator as index_impl,
    status as status_impl,
    update_operator as update_impl,
)
from ascendc_codemap_mcp.service.models import DoctorResult, Envelope
from ascendc_codemap_mcp.service.query import (
    evidence as evidence_impl,
    query as query_impl,
)

INSTRUCTIONS = INSTRUCTIONS  # PublicQueryContract; re-export for tests.

READ = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)
WRITE_ADDITIVE = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=False,
    open_world_hint=False,
)


@asynccontextmanager
async def _lifespan(_server: MCPServer):
    try:
        yield {}
    finally:
        runtime.shutdown()


mcp = MCPServer(
    SERVER_NAME,
    title="AscendC CodeMap",
    description=(
        "AscendC operator CodeMap: Host/Tiling/Kernel semantic graph for coding agents."
    ),
    instructions=INSTRUCTIONS,
    version=SERVER_VERSION,
    lifespan=_lifespan,
)
atexit.register(runtime.shutdown)


def _envelope(payload: dict[str, Any]) -> Envelope:
    return Envelope.model_validate(payload)


_FOLLOWUP_KEYS = (
    "ok",
    "verdict",
    "layer",
    "error",
    "error_code",
    "next_cursor",
    "legal_filters",
    "parsed_tokens",
    "operation",
)
_CODEMAP_FOLLOWUP = ("id", "snapshot_id", "architecture", "alias")


def _agent_followup(payload: dict[str, Any]) -> dict[str, Any]:
    """Exception-oriented: success keeps snapshot/cursor only."""
    out: dict[str, Any] = {}
    handle = payload.get("codemap")
    if isinstance(handle, dict):
        keep = {key: handle[key] for key in _CODEMAP_FOLLOWUP if handle.get(key) not in (None, "")}
        if keep:
            out["codemap"] = keep
            if keep.get("id"):
                out["codemap_id"] = keep["id"]
            if keep.get("snapshot_id"):
                out["snapshot_id"] = keep["snapshot_id"]
    nxt = payload.get("next_cursor")
    if nxt not in (None, ""):
        out["next_cursor"] = nxt
    for key in ("server_ms", "render_ms", "response_chars"):
        if payload.get(key) not in (None, ""):
            out[key] = payload[key]
    failed = (not payload.get("ok", True)) or payload.get("error_code") or payload.get("error")
    if failed:
        for key in _FOLLOWUP_KEYS:
            value = payload.get(key)
            if value in (None, ""):
                continue
            out[key] = value
        if "ok" not in out:
            out["ok"] = False
    return out


def _query_result(payload: dict[str, Any]) -> CallToolResult:
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    text = str((data or {}).get("text") or "")
    followup = _agent_followup(payload)
    if payload.get("error_code") == "INVALID_QUERY":
        hint = str((data or {}).get("hint") or payload.get("error") or "")
        legal = payload.get("legal_filters") or (data or {}).get("legal_filters") or []
        tokens = payload.get("parsed_tokens") or (data or {}).get("parsed_tokens") or []
        text = (
            f"INVALID_QUERY: {hint}\n"
            f"legal filters: {', '.join(str(x) for x in legal) or '(none)'}\n"
            f"parsed tokens: {', '.join(str(x) for x in tokens) or '(none)'}"
        )
        followup["legal_filters"] = list(legal)
        followup["parsed_tokens"] = list(tokens)
    elif not text:
        err = str(payload.get("error_code") or "").strip()
        msg = str(payload.get("error") or "").strip()
        text = f"{err}: {msg}".strip(": ").strip() or json.dumps(followup, ensure_ascii=False)
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structured_content=followup,
    )


_explore_result = _query_result


def _compat_result(payload: dict[str, Any]) -> CallToolResult:
    """Same text, but structured output still satisfies the retired Envelope schema."""
    result = _query_result(payload)
    structured = dict(result.structured_content or {})
    structured.setdefault("ok", bool(payload.get("ok", True)))
    structured["deprecated"] = "use codemap_query(operation=search|resolve)"
    return CallToolResult(content=result.content, structured_content=structured)


DEFAULT_MCP_TOOLS = (
    "codemap_search",
    "codemap_trace",
    "codemap_source",
    "codemap_doctor",
    "codemap_index",
    "codemap_update",
    "codemap_discover",
)


def listed_tool_names() -> set[str] | None:
    raw = str(os.environ.get("ASCENDC_CODEMAP_MCP_TOOLS") or "").strip()
    if raw in {"*", "all"}:
        return None
    if not raw:
        return set(DEFAULT_MCP_TOOLS)
    names: set[str] = set()
    for part in raw.split(","):
        name = part.strip()
        if not name:
            continue
        if name in {"index_operator", "update_operator"} or name.startswith(
            "codemap_"
        ):
            names.add(name)
        else:
            names.add(f"codemap_{name}")
    return names


def _doctor(payload: dict[str, Any]) -> DoctorResult:
    return DoctorResult.model_validate(payload)


def _progress(ctx: Context) -> Callable[[int, int, str], None]:
    loop = asyncio.get_running_loop()

    def on_progress(current: int, total: int, message: str) -> None:
        try:
            asyncio.run_coroutine_threadsafe(
                ctx.report_progress(float(current), float(total), message),
                loop,
            )
        except Exception:  # noqa: BLE001
            return

    return on_progress


async def _run_cancellable(fn: Callable[[threading.Event], dict[str, Any]]) -> dict[str, Any]:
    stop = threading.Event()
    loop = asyncio.get_running_loop()
    fut = loop.run_in_executor(None, lambda: fn(stop))
    try:
        return await fut
    except asyncio.CancelledError:
        stop.set()
        await asyncio.shield(fut)
        raise


async def _notify_maps(ctx: Context) -> None:
    try:
        await ctx.notify_resources_changed()
    except Exception:  # noqa: BLE001
        return


@mcp.tool(title="Discover CodeMaps", annotations=READ, structured_output=True)
def codemap_discover(project: str = "", architecture: str = "") -> Envelope:
    """List CodeMaps under an operator directory (or this process's registry). Returns codemap_id values for later calls."""
    return _envelope(discover_impl(project=project, architecture=architecture))


@mcp.tool(title="CodeMap status", annotations=READ, structured_output=True)
def codemap_status(
    codemap_id: str = "",
    project: str = "",
    architecture: str = "",
) -> Envelope:
    """Freshness of one CodeMap versus current sources. Pass codemap_id, or project AND architecture."""
    return _envelope(
        status_impl(codemap_id=codemap_id, project=project, architecture=architecture)
    )


@mcp.tool(title="Search CodeMap", annotations=READ, structured_output=True)
def codemap_search(
    pattern: str = "",
    codemap_id: str = "",
    project: str = "",
    architecture: str = "",
    file: str = "",
    kind: str = "",
    cursor: str = "",
    limit: int = 20,
) -> CallToolResult:
    """Find a name you do not have yet: regex over the indexed source snapshot. Enum values, string literals, macros, and comments all match, not just symbol names. file= is an optional glob (** matches zero directories); kind= restricts to an entity kind. Returns file:line hits grouped by unit — feed a name to codemap_trace, or a location to codemap_source. Identity is codemap_id, or project+architecture."""
    return _query_result(
        query_impl(
            operation="search",
            codemap_id=codemap_id,
            project=project,
            architecture=architecture,
            pattern=pattern,
            file=file,
            kind=kind,
            cursor=cursor,
            limit=limit,
        )
    )


@mcp.tool(title="Trace CodeMap symbol", annotations=READ, structured_output=True)
def codemap_trace(
    symbol: str = "",
    to_symbol: str = "",
    codemap_id: str = "",
    project: str = "",
    architecture: str = "",
    dim: str = "",
    value: str = "",
    relation: str = "",
    kind: str = "",
    cursor: str = "",
    limit: int = 8,
) -> CallToolResult:
    """Semantic facts for a name you already have. Three shapes: symbol= alone returns one closed card (definition body, every write with its value, its guard and the call that reaches it, reads, kernel consumers, Calls / Called by, compiled legal keys) — this is the default and it is complete, so start here; symbol= plus to_symbol= returns the shortest relation path between them (call chain, value propagation, guard reachability); dim= plus value= returns the compiled legal key space — use it for any "which combinations are actually built" question instead of reading dispatch macros, because it is the compiler's answer and source inference is not. relation= optionally narrows to call, data, control or compile (comma-separated); omit it to get every family, which is what you usually want. Takes no file/line: use codemap_source to read lines."""
    return _query_result(
        query_impl(
            operation="trace",
            codemap_id=codemap_id,
            project=project,
            architecture=architecture,
            symbol=symbol,
            to_symbol=to_symbol,
            dim=dim,
            value=value,
            relation=relation,
            kind=kind,
            cursor=cursor,
            limit=limit,
        )
    )


@mcp.tool(title="Read CodeMap source", annotations=READ, structured_output=True)
def codemap_source(
    file: str = "",
    line: int = 0,
    line_end: int = 0,
    codemap_id: str = "",
    project: str = "",
    architecture: str = "",
    limit: int = 8,
) -> CallToolResult:
    """Read indexed source at a location, with the state changes, branches and tiling fields of that unit. line_end= takes a range, which is how you finish a snippet that was cut. The returned source is already Read — do not open the file again. Callers and definition sites are not computed here; they belong to a name, so ask codemap_trace for those. Takes no symbol."""
    return _query_result(
        query_impl(
            operation="source",
            codemap_id=codemap_id,
            project=project,
            architecture=architecture,
            file=file,
            line=line,
            line_end=line_end,
            limit=limit,
        )
    )


# Superseded by the three above. Sessions that already handshook this name keep
# working; `listed_tool_names` keeps it out of the advertised set.
@mcp.tool(title="Query CodeMap (compat)", annotations=READ, structured_output=True)
def codemap_query(
    operation: Literal["search", "resolve", "trace", "source"] = "resolve",
    codemap_id: str = "",
    project: str = "",
    architecture: str = "",
    name: str = "",
    pattern: str = "",
    symbol: str = "",
    file: str = "",
    line: int = 0,
    line_end: int = 0,
    to_symbol: str = "",
    dim: str = "",
    value: str = "",
    kind: str = "",
    cursor: str = "",
    limit: int = 20,
) -> CallToolResult:
    """Deprecated. Use codemap_search (find a name), codemap_trace (facts about a name), or codemap_source (read lines)."""
    return _query_result(
        query_impl(
            operation=operation,
            codemap_id=codemap_id,
            project=project,
            architecture=architecture,
            name=name,
            pattern=pattern,
            symbol=symbol,
            file=file,
            line=line,
            line_end=line_end,
            to_symbol=to_symbol,
            dim=dim,
            value=value,
            kind=kind,
            cursor=cursor,
            limit=limit,
        )
    )


@mcp.tool(title="CodeMap evidence", annotations=READ, structured_output=True)
def codemap_evidence(
    codemap_id: str,
    evidence_id: str = "",
    entity_id: str = "",
    file: str = "",
    line: int = 0,
    line_end: int = 0,
    limit: int = 8,
    cursor: str = "",
    expected_snapshot_id: str = "",
) -> Envelope:
    """Debug/explicit evidence expansion only. Do not use to read ordinary source; codemap_source does that. Prefer evidence_id. Pass expected_snapshot_id from evidence[].snapshot_id."""
    return _envelope(
        evidence_impl(
            codemap_id=codemap_id,
            evidence_id=evidence_id,
            entity_id=entity_id,
            file=file,
            line=line,
            line_end=line_end,
            limit=limit,
            cursor=cursor,
            expected_snapshot_id=expected_snapshot_id,
        )
    )


@mcp.tool(title="CodeMap doctor", annotations=READ, structured_output=True)
def codemap_doctor(project: str = "", architecture: str = "") -> DoctorResult:
    """Check CANN headers, clang/libclang, and the operator path before indexing. If ok is false, follow next_steps (download Toolkit .run, cann-extract, LLVM 18)."""
    return _doctor(doctor_impl(project=project, architecture=architecture))


@mcp.tool(title="Index operator", annotations=WRITE_ADDITIVE, structured_output=True)
async def codemap_index(project: str, architecture: str, ctx: Context) -> Envelope:
    """Cold-build a CodeMap (prepare → extract → analyze → commit). Minutes. Only when no .uo exists; if one already exists, returns ALREADY_INDEXED and tells you to use codemap_update. Do not call on connect. Cancellable between steps."""
    on_progress = _progress(ctx)

    def work(stop: threading.Event) -> dict[str, Any]:
        return index_impl(
            project=project,
            architecture=architecture,
            should_stop=stop.is_set,
            on_progress=on_progress,
        )

    payload = await _run_cancellable(work)
    await _notify_maps(ctx)
    return _envelope(payload)


@mcp.tool(title="Update CodeMap", annotations=WRITE_ADDITIVE, structured_output=True)
async def codemap_update(
    ctx: Context,
    codemap_id: str = "",
    project: str = "",
    architecture: str = "",
    confirm_scope: bool = False,
) -> Envelope:
    """Incremental refresh after source changes. Prefer codemap_id. Read state and updated; do not treat ok as rebuild success. Cancellable between detect/plan/layer/commit."""
    on_progress = _progress(ctx)

    def work(stop: threading.Event) -> dict[str, Any]:
        return update_impl(
            codemap_id=codemap_id,
            project=project,
            architecture=architecture,
            confirm_scope=confirm_scope,
            should_stop=stop.is_set,
            on_progress=on_progress,
        )

    payload = await _run_cancellable(work)
    await _notify_maps(ctx)
    return _envelope(payload)


# Renaming a public tool orphans every agent session that already handshook the
# old name: the host validates against its cached list and the call never
# reaches the server. These stay registered (and answer on the current
# contract) while `listed_tool_names` keeps them out of the advertised set, so
# a live session degrades to "deprecated" instead of "tool not found".
@mcp.tool(title="Query CodeMap (compat)", annotations=READ, structured_output=True)
def codemap_explore(
    codemap_id: str = "",
    query: str = "",
    file: str = "",
    line: int = 0,
    line_end: int = 0,
    evidence_id: str = "",
    cursor: str = "",
    limit: int = 20,
    expected_snapshot_id: str = "",
) -> CallToolResult:
    """Deprecated alias for codemap_query. Use codemap_query(operation=search|resolve)."""
    del line_end, evidence_id, expected_snapshot_id
    return _compat_result(
        query_impl(
            operation="resolve",
            codemap_id=codemap_id,
            symbol=query,
            file=file,
            line=line,
            cursor=cursor,
            limit=limit,
        )
    )


@mcp.tool(title="Query CodeMap (compat)", annotations=READ, structured_output=True)
def query_codemap(
    codemap_id: str = "",
    project: str = "",
    architecture: str = "",
    pattern: str = "",
    file: str = "",
    line: int = 0,
    line_end: int = 0,
    limit: int = 20,
    cursor: str = "",
) -> CallToolResult:
    """Deprecated alias for codemap_query. Use codemap_query(operation=search|resolve)."""
    del line_end
    return _compat_result(
        query_impl(
            operation="resolve",
            codemap_id=codemap_id,
            project=project,
            architecture=architecture,
            symbol=pattern,
            file=file,
            line=line,
            cursor=cursor,
            limit=limit,
        )
    )


@mcp.tool(title="Index operator (compat)", annotations=WRITE_ADDITIVE, structured_output=True)
async def index_operator(project: str, architecture: str, ctx: Context) -> Envelope:
    """Compatibility alias for codemap_index."""
    return await codemap_index(project, architecture, ctx)


@mcp.tool(title="Update operator (compat)", annotations=WRITE_ADDITIVE, structured_output=True)
async def update_operator(
    ctx: Context,
    project: str = "",
    architecture: str = "",
    confirm_scope: bool = False,
    codemap_id: str = "",
) -> Envelope:
    """Compatibility alias for codemap_update."""
    return await codemap_update(
        ctx,
        codemap_id=codemap_id,
        project=project,
        architecture=architecture,
        confirm_scope=confirm_scope,
    )


@mcp.resource(
    "codemap://runtime",
    name="codemap-runtime",
    title="CodeMap runtime",
    description="Open query handles and cache bounds for this MCP process.",
    mime_type="application/json",
)
def runtime_resource() -> str:
    return json.dumps(
        {"ok": True, "runtime": runtime.cache_stats(), "version": SERVER_VERSION},
        ensure_ascii=False,
    )


@mcp.resource(
    "codemap://map/{codemap_id}",
    name="codemap-status",
    title="CodeMap snapshot",
    description="Freshness and identity for one operator@architecture CodeMap.",
    mime_type="application/json",
    # `p:<hex>::op@arch` is an opaque id. MCP's default resource policy treats
    # `p:` as a Windows drive path; this id is never joined onto the filesystem.
    security=ResourceSecurity(exempt_params={"codemap_id"}),
)
def map_resource(codemap_id: str) -> str:
    payload = status_impl(codemap_id=codemap_id)
    return json.dumps(payload, ensure_ascii=False, default=str)


@mcp.prompt(
    name="query_operator",
    title="Query an operator CodeMap",
    description="Query a CodeMap for one codemap_id.",
)
def query_operator(codemap_id: str, focus: str = "") -> str:
    extra = f" Focus: {focus}." if str(focus or "").strip() else ""
    return (
        f"Query AscendC CodeMap `{codemap_id}` with MCP tools on server {SERVER_NAME}."
        f"{extra}\n"
        "Unknown name → codemap_search. Known symbol → codemap_trace "
        "(add to_symbol for a path; dim/value for the compiled key space). "
        "file:line → codemap_source. "
        "The returned source is already Read — do not open those files again."
    )


@mcp.prompt(
    name="build_codemap",
    title="Index an operator",
    description="Doctor then cold-index an operator directory into a CodeMap.",
)
def build_codemap(project: str, architecture: str) -> str:
    return (
        f"Build a CodeMap for operator directory `{project}` architecture `{architecture}`.\n"
        "Call codemap_doctor, then codemap_index. Do not index on connect. "
        "Indexing can take minutes and is cancellable between prepare/extract/analyze/commit. "
        "When it finishes, keep the returned codemap.id for queries."
    )


def _complete_ids(prefix: str) -> list[str]:
    needle = str(prefix or "").strip().lower()
    refs = list(runtime.registry.all())
    alias_hits: dict[str, int] = {}
    for ref in refs:
        alias_hits[ref.alias] = alias_hits.get(ref.alias, 0) + 1
    values: list[str] = []
    seen: set[str] = set()
    for ref in refs:
        items = [ref.id]
        if alias_hits.get(ref.alias, 0) == 1:
            items.append(ref.alias)
        for item in items:
            if item in seen:
                continue
            if not needle or needle in item.lower() or needle in ref.op_name.lower():
                seen.add(item)
                values.append(item)
    return values[:20]


def _complete_arches(prefix: str) -> list[str]:
    known = ["arch35", "arch32", "arch22", "default"]
    for ref in runtime.registry.all():
        if ref.architecture and ref.architecture not in known:
            known.append(ref.architecture)
    needle = str(prefix or "").strip().lower()
    return [a for a in known if not needle or a.startswith(needle)][:20]


@mcp.completion()
async def handle_completion(ref, argument, context):
    name = str(getattr(argument, "name", "") or "")
    value = str(getattr(argument, "value", "") or "")
    if isinstance(ref, ResourceTemplateReference) and name == "codemap_id":
        return Completion(values=_complete_ids(value))
    if isinstance(ref, PromptReference):
        if name == "codemap_id":
            return Completion(values=_complete_ids(value))
        if name == "architecture":
            return Completion(values=_complete_arches(value))
    del context
    return Completion(values=[])


def create_server() -> MCPServer:
    return mcp


_orig_list_tools = mcp._tool_manager.list_tools


def _filtered_list_tools():
    tools = _orig_list_tools()
    allow = listed_tool_names()
    if allow is None:
        return tools
    return [tool for tool in tools if getattr(tool, "name", "") in allow]


mcp._tool_manager.list_tools = _filtered_list_tools
