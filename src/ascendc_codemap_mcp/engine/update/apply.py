# -*- coding: utf-8 -*-
"""Apply a planned update by re-running uo_init.pilot_engines actions."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from ascendc_codemap_mcp.engine.update.artifacts import (
    load_change_set_if_fresh,
    load_update_plan_if_fresh,
    resolve_uo_root,
    revision_sha,
)
from ascendc_codemap_mcp.engine.update.changes import detect_kb_changes
from ascendc_codemap_mcp.engine.update.diff import export_diff_product
from ascendc_codemap_mcp.engine.update.plan import plan_kb_update
from ascendc_codemap_mcp.engine.yaml_io import read_yaml, write_yaml


def update_operator(
    repo_root: Path,
    op_name: str,
    *,
    architecture: str = "",
    base: str | None = None,
    head: str | None = None,
    confirm_scope: bool = False,
    run_gates: bool = False,
    run_id: str | None = None,
    reuse_artifacts: bool = True,
    cann_root: str | None = None,
    ops_root: str | None = None,
) -> dict[str, Any]:
    del run_gates
    repo_root = Path(repo_root).expanduser().resolve()
    uo_root = resolve_uo_root(repo_root, architecture=architecture)
    from ascendc_codemap_mcp.engine.store.reader import find_uo_product

    product = find_uo_product(repo_root, op_name=op_name, architecture=architecture)
    if product is None or product.suffix != ".uo":
        raise FileNotFoundError(
            f"missing .uo product under {repo_root}; run /uo-init first "
            "(expected .ascendc-codemap/<arch>/<op>.<arch>.uo)"
        )

    change_set: dict[str, Any] | None = None
    plan: dict[str, Any] | None = None
    if reuse_artifacts and base is None and head is None:
        change_set = load_change_set_if_fresh(uo_root, repo_root=repo_root)
        if change_set is not None:
            plan = load_update_plan_if_fresh(uo_root, change_set=change_set)
    if change_set is None:
        change_set = detect_kb_changes(
            repo_root,
            op_name,
            base=base,
            head=head,
            write=True,
            architecture=architecture,
        )
    if plan is None:
        plan = plan_kb_update(
            repo_root,
            op_name,
            change_set=change_set,
            write=True,
            architecture=architecture,
        )

    run_id = str(run_id or "").strip() or _new_run_id()
    update_dir = uo_root / "runs" / run_id / "update"
    update_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(update_dir / "change_set.yaml", change_set)
    write_yaml(update_dir / "update_plan.yaml", plan)

    if (
        plan.get("mode") == "blocked_scope"
        or (
            plan.get("needs_scope_review")
            and not plan.get("scoped_changed_files")
        )
    ) and not confirm_scope:
        export_diff_product(
            repo_root,
            op_name,
            change_set=change_set,
            update_plan=plan,
            status="blocked",
            write=True,
            architecture=architecture,
        )
        receipt = _receipt(
            run_id, change_set, plan, status="blocked", message="needs scope confirmation"
        )
        write_yaml(update_dir / "receipt.yaml", receipt)
        return {
            "status": "blocked",
            "run_id": run_id,
            "plan": plan,
            "change_set": change_set,
            "receipt": receipt,
        }

    if plan.get("needs_scope_review") and not confirm_scope:
        suspicious = [f for f in (change_set.get("files") or []) if f.get("suspicious_out_of_scope")]
        if suspicious:
            export_diff_product(
                repo_root,
                op_name,
                change_set=change_set,
                update_plan=plan,
                status="blocked",
                write=True,
                architecture=architecture,
            )
            receipt = _receipt(
                run_id,
                change_set,
                plan,
                status="blocked",
                message="out-of-scope operator sources detected",
            )
            write_yaml(update_dir / "receipt.yaml", receipt)
            return {
                "status": "blocked",
                "run_id": run_id,
                "plan": plan,
                "change_set": change_set,
                "receipt": receipt,
            }

    if plan.get("mode") == "noop":
        export_diff_product(
            repo_root,
            op_name,
            change_set=change_set,
            update_plan=plan,
            status="ready",
            write=True,
            architecture=architecture,
        )
        _bump_manifest(uo_root, change_set.get("head_revision"), run_id)
        receipt = _receipt(run_id, change_set, plan, status="pass", message="no in-scope changes")
        write_yaml(update_dir / "receipt.yaml", receipt)
        return {
            "status": "pass",
            "run_id": run_id,
            "plan": plan,
            "change_set": change_set,
            "receipt": receipt,
        }

    action_results = _run_rebuild_actions(
        repo_root,
        plan,
        architecture=architecture,
        cann_root=cann_root,
        ops_root=ops_root,
        run_id=run_id,
    )
    write_yaml(update_dir / "rebuild_actions.yaml", {"results": action_results})

    failed = [r for r in action_results if not r.get("ok")]
    if failed:
        export_diff_product(
            repo_root,
            op_name,
            change_set=change_set,
            update_plan=plan,
            status="fail",
            write=True,
            architecture=architecture,
        )
        receipt = _receipt(
            run_id,
            change_set,
            plan,
            status="fail",
            message=f"rebuild failed: {[f.get('action') for f in failed]}",
        )
        write_yaml(update_dir / "receipt.yaml", receipt)
        return {
            "status": "fail",
            "run_id": run_id,
            "plan": plan,
            "change_set": change_set,
            "receipt": receipt,
            "action_results": action_results,
        }

    _bump_manifest(uo_root, change_set.get("head_revision"), run_id)
    export_diff_product(
        repo_root,
        op_name,
        change_set=change_set,
        update_plan=plan,
        status="ready",
        write=True,
        architecture=architecture,
    )
    receipt = _receipt(run_id, change_set, plan, status="pass", message="uo_init rebuild ok")
    write_yaml(update_dir / "receipt.yaml", receipt)
    return {
        "status": "pass",
        "run_id": run_id,
        "plan": plan,
        "change_set": change_set,
        "receipt": receipt,
        "action_results": action_results,
        "publish_deferred": True,
    }


def _run_rebuild_actions(
    repo_root: Path,
    plan: dict[str, Any],
    *,
    architecture: str,
    cann_root: str | None,
    ops_root: str | None,
    run_id: str,
) -> list[dict[str, Any]]:
    from ascendc_codemap_mcp.engine.pilot_engines import ENGINES

    ctx = {
        "op_name": plan.get("op_name") or repo_root.name,
        "arch_dir": architecture,
        "architecture": architecture,
        "run_id": run_id,
        "cann_root": cann_root or "",
        "ops_root": ops_root or str(repo_root.parent),
    }
    results: list[dict[str, Any]] = []
    for action in plan.get("actions") or []:
        fn = ENGINES.get(str(action))
        if fn is None:
            results.append({"action": action, "ok": False, "error": "unknown_action"})
            continue
        try:
            out = fn(repo_root, ctx)
            results.append({"action": action, "ok": bool(out.get("ok")), "result": out})
        except Exception as exc:  # noqa: BLE001
            results.append({"action": action, "ok": False, "error": str(exc)[:300]})
    return results


def _bump_manifest(uo_root: Path, head_revision: Any, run_id: str) -> None:
    man = read_yaml(uo_root / "manifest.yaml") or {}
    source = man.get("source") if isinstance(man.get("source"), dict) else {}
    source = dict(source)
    if head_revision:
        source["revision"] = revision_sha(head_revision) or str(head_revision)
    man["source"] = source
    man["current_run_id"] = run_id
    man["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_yaml(uo_root / "manifest.yaml", man)


def _receipt(
    run_id: str,
    change_set: dict[str, Any],
    plan: dict[str, Any],
    *,
    status: str,
    message: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "status": status,
        "message": message,
        "mode": plan.get("mode"),
        "affected_layers": plan.get("affected_layers"),
        "change_set_fingerprint": change_set.get("change_set_fingerprint"),
        "plan_fingerprint": plan.get("plan_fingerprint"),
        "engine": "uo_init.update",
        "at": datetime.now(timezone.utc).isoformat(),
    }


def _new_run_id() -> str:
    return f"upd-{uuid4().hex[:12]}"
