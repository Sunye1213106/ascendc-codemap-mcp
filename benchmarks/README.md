# Benchmarks / regression probes

These are not MCP tools. Result JSON/txt is gitignored.

- Dialect micro lives in `tests/test_dialect_benchmark.py` (fixture, always on).
- Review-20 lives in `tests/test_review20_benchmark.py` (skips if FAG `.uo` is missing).
- Metrics: queries_per_question, native_escape, repeat_resolve, invalid_query, server_ms.
  `benchmarks/collect_query_metrics.py` scripts query steps (it cannot observe grep/read).
  Measure `native_escape` from an exported agent session with `benchmarks/session_metrics.py`.
