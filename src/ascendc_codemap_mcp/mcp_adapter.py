# -*- coding: utf-8 -*-
"""MCP adapter: official SDK owns transport, schema, cancellation, structured results."""
from __future__ import annotations

import asyncio
import atexit
import json
import threading
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

from mcp.server import MCPServer
from mcp.server.mcpserver.context import Context
from mcp.server.mcpserver.resources.templates import ResourceSecurity
from mcp.types import (
    Completion,
    PromptReference,
    ResourceTemplateReference,
    ToolAnnotations,
)

from ascendc_codemap_mcp.constants import SERVER_NAME, SERVER_VERSION
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
    explore as explore_impl,
    overview as overview_impl,
    query_codemap as query_impl,
    selection as selection_impl,
    symbol as symbol_impl,
)

INSTRUCTIONS = """\
AscendC CodeMap is a semantic compiler graph for one operator + one architecture.
It answers what the code is, not what a previous agent thought.

Workflow:
1. codemap_discover (project= operator directory) → keep codemap.id (p:<workspace>::op@arch). op@arch is an alias; if it matches multiple workspaces, use the canonical id.
2. Read resource codemap://map/{codemap_id} or call codemap_status
3. If not indexed: codemap_doctor then codemap_index (minutes). Do not index on connect. If doctor.ok is false, follow doctor.next_steps (download CANN Toolkit .run, cann-extract, install LLVM 18). Do not run the .run installer.
4. If stale or dirty: codemap_update. If state=needs_confirmation, ask the user before confirm_scope=true.
5. Query with codemap_explore (ident / Dim=V / file:line / phenomenon). evidence_id or cursor continues a prior card.
Typed tools remain as CLI/service primitives.

Rules:
- Never guess architecture. Never pass a natural-language sentence as a symbol.
- Follow evidence[].id (span:...) with evidence[].snapshot_id as expected_snapshot_id. If coverage.truncated and next_cursor is set, pass next_cursor; nested neighbor samples are not a page.
- update ok=true is not "graph refreshed". Read state and updated.
- Do not write patches into .uo. Do not use raw SQL/Cypher.
"""

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


@mcp.tool(title="Explore CodeMap", annotations=READ, structured_output=True)
def codemap_explore(
    codemap_id: str,
    query: str = "",
    file: str = "",
    line: int = 0,
    line_end: int = 0,
    evidence_id: str = "",
    cursor: str = "",
    limit: int = 8,
    expected_snapshot_id: str = "",
) -> Envelope:
    """Unified contract card. Pass one identifier, Dim=V, file+line, a phenomenon phrase, or evidence_id/cursor to continue."""
    return _envelope(
        explore_impl(
            codemap_id=codemap_id,
            query=query,
            file=file,
            line=line,
            line_end=line_end,
            evidence_id=evidence_id,
            cursor=cursor,
            limit=limit,
            expected_snapshot_id=expected_snapshot_id,
        )
    )


@mcp.tool(title="CodeMap overview", annotations=READ, structured_output=True)
def codemap_overview(
    codemap_id: str,
    limit: int = 8,
    cursor: str = "",
) -> Envelope:
    """Index of what this CodeMap can answer: launch phases, dim names, tiling data names."""
    return _envelope(overview_impl(codemap_id=codemap_id, limit=limit, cursor=cursor))


@mcp.tool(title="CodeMap symbol", annotations=READ, structured_output=True)
def codemap_symbol(
    codemap_id: str,
    symbol: str,
    limit: int = 8,
    cursor: str = "",
) -> Envelope:
    """Look up one identifier (e.g. IsPse, deterBandScheduleMode): definition, host writers, kernel readers. Not a sentence."""
    return _envelope(
        symbol_impl(codemap_id=codemap_id, symbol=symbol, limit=limit, cursor=cursor)
    )


@mcp.tool(title="CodeMap selection", annotations=READ, structured_output=True)
def codemap_selection(
    codemap_id: str,
    dim: str,
    value: str = "",
    limit: int = 8,
    cursor: str = "",
) -> Envelope:
    """Template-admissible values for a tiling dim. dim='IsPse' lists the dim; dim='IsPse' and value='1' is Name=Value."""
    return _envelope(
        selection_impl(
            codemap_id=codemap_id, dim=dim, value=value, limit=limit, cursor=cursor
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
    """Resolve a prior evidence handle (span:... / entity id) or a file+line copied from a card. Prefer evidence_id. Pass expected_snapshot_id from evidence[].snapshot_id."""
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


@mcp.tool(title="Query CodeMap (compat)", annotations=READ, structured_output=True)
def query_codemap(
    codemap_id: str = "",
    project: str = "",
    architecture: str = "",
    pattern: str = "",
    file: str = "",
    line: int = 0,
    line_end: int = 0,
    limit: int = 8,
    cursor: str = "",
) -> Envelope:
    """Compatibility facade. Prefer codemap_explore. Requires codemap_id, or project AND architecture together."""
    return _envelope(
        query_impl(
            codemap_id=codemap_id,
            project=project,
            architecture=architecture,
            pattern=pattern,
            file=file,
            line=line,
            line_end=line_end,
            limit=limit,
            cursor=cursor,
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
    description="Explore a CodeMap for one codemap_id.",
)
def query_operator(codemap_id: str, focus: str = "") -> str:
    extra = f" Focus: {focus}." if str(focus or "").strip() else ""
    return (
        f"Query AscendC CodeMap `{codemap_id}` with MCP tools on server {SERVER_NAME}."
        f"{extra}\n"
        "Use codemap_status first (freshness). Then codemap_explore with one "
        "identifier, Dim=V, file+line, or a phenomenon phrase. Continue with "
        "evidence_id or cursor. completeness=COMPLETE is the only finished answer."
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
