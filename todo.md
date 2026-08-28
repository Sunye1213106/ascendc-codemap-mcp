# 任务：MCP runtime correctness

## 目标
不新增功能。修 P0/P1 生命周期：SQLite 线程所有权、旧 snapshot 关闭、workspace identity、request-scoped cancel、pagination/evidence 绑定 snapshot。

## 待办事项
- [x] P0-1 thread-local SQLite connections + 并发测试
- [x] P0-2 QueryCache 真正关闭/失效底层 connection
- [x] P0-3 project_id/op@arch identity + AMBIGUOUS
- [x] P1 request-scoped cancel + update token 下沉
- [x] P1 cursor 绑定 snapshot/query；nested 假 continuation
- [x] P1 evidence snapshot epoch + SNAPSHOT_CHANGED
- [x] P1 write lock 后 re-check freshness；cache 不淘汰 in-use
- [x] 回归测试 + README 收紧

## 进度
8/8
