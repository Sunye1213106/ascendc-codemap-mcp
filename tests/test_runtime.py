# -*- coding: utf-8 -*-
from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tests.conftest import write_uo_fixture
from ascendc_codemap_mcp.engine.store.reader import (
    close_uo_connections,
    open_handle_count,
    shared_uo,
)
from ascendc_codemap_mcp.service import runtime
from ascendc_codemap_mcp.service.control import status, update_operator
from ascendc_codemap_mcp.service.identity import make_id, resolve
from ascendc_codemap_mcp.service.query import (
    encode_cursor,
    evidence,
    paginate,
    query,
    query_fingerprint,
)


def _stale(**_k):
    return {
        "freshness": "stale",
        "source_revision": "fff",
        "indexed_revision": "abc",
        "dirty": False,
        "changed_files": 1,
        "semantic_completeness": 0.9,
    }


def test_sqlite_connections_are_thread_local(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    product = write_uo_fixture(op)
    barrier = threading.Barrier(8)
    conns: list[sqlite3.Connection] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            barrier.wait()
            conn = shared_uo(product)
            conns.append(conn)
            conn.execute("SELECT COUNT(*) FROM entity").fetchone()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert len({id(c) for c in conns}) == 8
    close_uo_connections(product)
    assert open_handle_count(product) == 0
    for conn in conns:
        try:
            conn.execute("SELECT 1")
            raise AssertionError("connection still usable after cross-thread close")
        except sqlite3.ProgrammingError:
            pass
    product.unlink()
    write_uo_fixture(op, symbol="Other", revision="zzz")


def test_mcp_worker_threads_can_query_in_parallel(tmp_path: Path, monkeypatch) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    write_uo_fixture(op)
    status(project=str(op), architecture="arch35")
    monkeypatch.setattr(
        "ascendc_codemap_mcp.service.freshness.compute",
        lambda *a, **k: {
            "freshness": "fresh",
            "source_revision": "abc123",
            "indexed_revision": "abc123",
            "dirty": False,
            "changed_files": 0,
            "semantic_completeness": 1.0,
        },
    )
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            for _ in range(25):
                card = query(codemap_id="toy_op@arch35", symbol="IsPse")
                assert card.get("ok") is True
                ev_id = (card.get("evidence") or [{}])[0].get("id") or "span:e1"
                around = evidence(codemap_id="toy_op@arch35", evidence_id=ev_id)
                assert around.get("ok") is True
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(worker) for _ in range(8)]
        for fut in futs:
            fut.result()
    assert not errors


def test_query_sees_new_snapshot_after_product_replace(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    product = write_uo_fixture(op, symbol="IsPse", revision="abc123")
    first = query(project=str(op), architecture="arch35", symbol="IsPse")
    assert first.get("ok") is True
    sid1 = first["codemap"]["snapshot_id"]
    runtime.cache.drop(product)
    write_uo_fixture(op, symbol="IsFoo", revision="def456", entity_id="e2")
    second = query(project=str(op), architecture="arch35", symbol="IsFoo")
    assert second.get("ok") is True
    assert int(second.get("data", {}).get("count") or 0) >= 1
    assert second["codemap"]["snapshot_id"] != sid1
    old = query(project=str(op), architecture="arch35", symbol="IsPse")
    assert int((old.get("data") or old).get("count") or 0) == 0


def test_drop_releases_sqlite_handle_for_replace(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    product = write_uo_fixture(op)
    query(project=str(op), architecture="arch35", symbol="IsPse")
    assert open_handle_count(product) >= 1
    runtime.cache.drop(product)
    assert open_handle_count(product) == 0
    product.unlink()
    write_uo_fixture(op, symbol="Other", revision="zzz")
    payload = query(project=str(op), architecture="arch35", symbol="Other")
    assert payload.get("ok") is True


def test_shutdown_releases_sqlite_handle_for_unlink(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    product = write_uo_fixture(op)
    query(project=str(op), architecture="arch35", symbol="IsPse")
    assert open_handle_count(product) >= 1
    runtime.shutdown()
    assert open_handle_count(product) == 0
    product.unlink()
    write_uo_fixture(op, symbol="Other", revision="zzz")
    payload = query(project=str(op), architecture="arch35", symbol="Other")
    assert payload.get("ok") is True


def test_idle_close_releases_sqlite_handle_for_unlink(tmp_path: Path, monkeypatch) -> None:
    import time

    monkeypatch.setenv("ASCENDC_CODEMAP_SQLITE_IDLE_SEC", "0.15")
    op = tmp_path / "toy_op"
    op.mkdir()
    product = write_uo_fixture(op)
    query(project=str(op), architecture="arch35", symbol="IsPse")
    assert open_handle_count(product) >= 1
    deadline = time.time() + 2.0
    while time.time() < deadline and open_handle_count(product) > 0:
        time.sleep(0.05)
    assert open_handle_count(product) == 0
    product.unlink()


def test_alias_is_ambiguous_across_workspaces(tmp_path: Path) -> None:
    main = tmp_path / "main" / "flash_attention_score_grad"
    pr = tmp_path / "pr" / "flash_attention_score_grad"
    main.mkdir(parents=True)
    pr.mkdir(parents=True)
    write_uo_fixture(main)
    write_uo_fixture(pr)
    status(project=str(main), architecture="arch35")
    status(project=str(pr), architecture="arch35")
    payload = query(codemap_id="flash_attention_score_grad@arch35", symbol="IsPse")
    assert payload.get("ok") is False
    assert payload.get("error_code") == "AMBIGUOUS_CODEMAP_ID"
    assert len(payload.get("candidates") or []) == 2
    one = resolve(
        codemap_id="flash_attention_score_grad@arch35",
        project=str(main),
        architecture="arch35",
        registry=runtime.registry,
    )
    assert getattr(one, "project", None) == main.resolve()
    typed = query(codemap_id=one.id, symbol="IsPse")
    assert typed.get("ok") is True


def test_cursor_rejects_other_snapshot_and_query(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    write_uo_fixture(op)
    first = query(project=str(op), architecture="arch35", symbol="IsPse")
    sid = first["codemap"]["snapshot_id"]
    fp = query_fingerprint(engine="codemap_query", pattern="IsPse")
    other = encode_cursor(8, snapshot="cm:deadbeefdeadbeef", query=fp)
    miss = query(
        project=str(op), architecture="arch35", symbol="IsPse", cursor=other
    )
    assert miss.get("error_code") == "SNAPSHOT_CHANGED"
    wrong_q = encode_cursor(8, snapshot=sid, query="not-this-query")
    mismatch = query(
        project=str(op), architecture="arch35", symbol="IsPse", cursor=wrong_q
    )
    assert mismatch.get("error_code") == "CURSOR_MISMATCH"


def test_nested_truncation_does_not_mint_cursor() -> None:
    payload = {
        "ok": True,
        "shape": "name",
        "count": 1,
        "cards": [
            {
                "id": "e1",
                "kind": "TILING_KEY",
                "edges": {
                    "WRITES": {
                        "count": 37,
                        "neighbors": [{"id": f"n{i}"} for i in range(8)],
                        "truncated": True,
                    }
                },
            }
        ],
    }
    _, coverage, nxt = paginate(
        payload, limit=8, offset=0, snapshot="cm:abc", query="q1"
    )
    assert coverage["truncated"] is True
    assert coverage["nested_truncated"] is True
    assert nxt is None


def test_evidence_snapshot_epoch(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    write_uo_fixture(op)
    card = query(project=str(op), architecture="arch35", symbol="IsPse")
    ev = card["evidence"][0]
    ok = evidence(
        codemap_id="toy_op@arch35",
        evidence_id=ev["id"],
        expected_snapshot_id=ev["snapshot_id"],
    )
    assert ok.get("ok") is True
    drifted = evidence(
        codemap_id="toy_op@arch35",
        evidence_id=ev["id"],
        expected_snapshot_id="cm:not-this-snapshot",
    )
    assert drifted.get("ok") is False
    assert drifted.get("error_code") == "SNAPSHOT_CHANGED"


def test_cache_does_not_evict_in_use_entries(tmp_path: Path) -> None:
    runtime.cache.max_open = 1
    acquired: list[Path] = []
    try:
        for i in range(3):
            op = tmp_path / f"op{i}"
            op.mkdir()
            product = write_uo_fixture(op)
            runtime.cache.acquire(product)
            acquired.append(product)
        stats = runtime.cache.stats()
        assert stats["inuse"] == 3
        assert stats["cache_size"] == 3
    finally:
        for product in acquired:
            runtime.cache.release(product)
        runtime.cache.max_open = 4
        runtime.cache.close_all()


def test_update_stops_after_detect(monkeypatch, tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    write_uo_fixture(op)
    calls: list[str] = []
    monkeypatch.setattr(
        "ascendc_codemap_mcp.service.freshness.compute",
        lambda *a, **k: _stale(),
    )

    def _detect(*_a, **_k):
        calls.append("detect")
        return {"files": [], "head_revision": "abc", "change_set_fingerprint": "x"}

    monkeypatch.setattr(
        "ascendc_codemap_mcp.engine.update.apply.load_change_set_if_fresh",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "ascendc_codemap_mcp.engine.update.apply.detect_kb_changes",
        _detect,
    )
    payload = update_operator(
        project=str(op),
        architecture="arch35",
        should_stop=lambda: "detect" in calls,
    )
    assert payload.get("error_code") == "CANCELLED"
    assert calls == ["detect"]


def test_update_noop_when_fresh_after_lock(monkeypatch, tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    write_uo_fixture(op)
    called = {"n": 0}

    def _apply(*_a, **_k):
        called["n"] += 1
        return {"status": "pass", "plan": {"mode": "rebuild"}, "change_set": {}}

    monkeypatch.setattr(
        "ascendc_codemap_mcp.service.freshness.compute",
        lambda *a, **k: {
            "freshness": "fresh",
            "source_revision": "abc123",
            "indexed_revision": "abc123",
            "dirty": False,
            "changed_files": 0,
            "semantic_completeness": 0.9,
        },
    )
    monkeypatch.setattr(
        "ascendc_codemap_mcp.engine.update.update_operator",
        _apply,
    )
    payload = update_operator(project=str(op), architecture="arch35")
    assert payload.get("ok") is True
    assert payload.get("updated") is False
    assert payload.get("mode") == "noop"
    assert called["n"] == 0


def test_canonical_ids_differ_per_workspace(tmp_path: Path) -> None:
    a = tmp_path / "wt_a" / "op"
    b = tmp_path / "wt_b" / "op"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    left = make_id("op", "arch35", project=a)
    right = make_id("op", "arch35", project=b)
    assert left != right
    assert "::" in left and "/" not in left
    assert "::" in right and "/" not in right


def test_index_noop_when_product_already_exists(monkeypatch, tmp_path: Path) -> None:
    from ascendc_codemap_mcp.service.control import index_operator

    op = tmp_path / "toy_op"
    op.mkdir()
    write_uo_fixture(op)
    called: list[str] = []

    def _prepare(*_a, **_k):
        called.append("prepare")
        return {"ok": True}

    monkeypatch.setattr("ascendc_codemap_mcp.engine.codemap_engines.prepare", _prepare)
    payload = index_operator(project=str(op), architecture="arch35")
    assert payload.get("ok") is True
    assert payload.get("updated") is False
    assert payload.get("mode") == "noop"
    assert payload.get("error_code") == "ALREADY_INDEXED"
    assert called == []

