# Benchmarks / regression probes

Q20 and related drivers live here. They are not MCP tools.

- Dialect micro lives in `tests/test_dialect_benchmark.py` (fixture, always on).
- Review-20 lives in `tests/test_review20_benchmark.py` (skips if FAG `.uo` is missing).
- Legacy `_run_q20*.py` probes still run against a rebuilt `.uo`.
- Result JSON/txt is gitignored.
- Metrics: queries_per_question, native_escape, repeat_resolve, invalid_query, server_ms.
  `benchmarks/collect_query_metrics.py` still scripts query steps (it cannot observe grep/read).
  Measure `native_escape` from an exported agent session with `benchmarks/session_metrics.py`.
- Resolve goldens: `resolve(file,1673)` anchors `SetSplitAxis`; `resolve(isBn2MultiBlk)` / `resolve(hasRope)` must show Assignments or Compiled from product facts, not source guesses.
