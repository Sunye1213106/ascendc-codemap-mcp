# -*- coding: utf-8 -*-
from __future__ import annotations

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
    query_codemap,
    query_fingerprint,
    symbol,
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
    ids: list[int] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            barrier.wait()
            conn = shared_uo(product)
            ids.append(id(conn))
            conn.execute("SELECT COUNT(*) FROM entity").fetchone()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert len(set(ids)) == 8
    close_uo_connections(product)


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
                card = symbol(codemap_id="toy_op@arch35", symbol="IsPse")
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
    first = symbol(project=str(op), architecture="arch35", symbol="IsPse")
    assert first.get("ok") is True
    sid1 = first["codemap"]["snapshot_id"]
    runtime.cache.drop(product)
    write_uo_fixture(op, symbol="IsFoo", revision="def456", entity_id="e2")
    second = symbol(project=str(op), architecture="arch35", symbol="IsFoo")
    assert second.get("ok") is True
    assert int(second.get("data", {}).get("count") or 0) >= 1
    assert second["codemap"]["snapshot_id"] != sid1
    old = symbol(project=str(op), architecture="arch35", symbol="IsPse")
    assert int((old.get("data") or old).get("count") or 0) == 0


def test_drop_releases_sqlite_handle_for_replace(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    product = write_uo_fixture(op)
    query_codemap(project=str(op), architecture="arch35", pattern="IsPse")
    assert open_handle_count(product) >= 1
    runtime.cache.drop(product)
    assert open_handle_count(product) == 0
    product.unlink()
    write_uo_fixture(op, symbol="Other", revision="zzz")
    payload = symbol(project=str(op), architecture="arch35", symbol="Other")
    assert payload.get("ok") is True


def test_alias_is_ambiguous_across_workspaces(tmp_path: Path) -> None:
    main = tmp_path / "main" / "flash_attention_score_grad"
    pr = tmp_path / "pr" / "flash_attention_score_grad"
    main.mkdir(parents=True)
    pr.mkdir(parents=True)
    write_uo_fixture(main)
    write_uo_fixture(pr)
    status(project=str(main), architecture="arch35")
    status(project=str(pr), architecture="arch35")
    payload = symbol(codemap_id="flash_attention_score_grad@arch35", symbol="IsPse")
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
    typed = symbol(codemap_id=one.id, symbol="IsPse")
    assert typed.get("ok") is True


def test_cursor_rejects_other_snapshot_and_query(tmp_path: Path) -> None:
    op = tmp_path / "toy_op"
    op.mkdir()
    write_uo_fixture(op)
    first = symbol(project=str(op), architecture="arch35", symbol="IsPse")
    sid = first["codemap"]["snapshot_id"]
    fp = query_fingerprint(engine="codemap_symbol", pattern="IsPse")
    other = encode_cursor(8, snapshot="cm:deadbeefdeadbeef", query=fp)
    miss = symbol(
        project=str(op), architecture="arch35", symbol="IsPse", cursor=other
    )
    assert miss.get("error_code") == "SNAPSHOT_CHANGED"
    wrong_q = encode_cursor(8, snapshot=sid, query="not-this-query")
    mismatch = symbol(
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
    card = symbol(project=str(op), architecture="arch35", symbol="IsPse")
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
    assert make_id("op", "arch35", project=a) != make_id("op", "arch35", project=b)
