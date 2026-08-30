# Benchmarks / regression probes

Q20 and related drivers live here. They are not MCP tools.

- Dialect micro lives in `tests/test_dialect_benchmark.py` (fixture, always on).
- Review-20 lives in `tests/test_review20_benchmark.py` (skips if FAG `.uo` is missing).
- Legacy `_run_q20*.py` probes still run against a rebuilt `.uo`.
- Result JSON/txt is gitignored.
- Metrics: zero-hit rate, searches-to-first-useful-locator (Q-class ≤2), evidence calls (must stay 0).
- Resolve goldens: `resolve(file,1673)` anchors `SetSplitAxis`; `resolve(isBn2MultiBlk)` / `resolve(hasRope)` must show Assignments or Compiled from product facts, not source guesses.
