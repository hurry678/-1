# 问题3正式交付说明

当前正式方案为 `direct`（直接复用问题2滚动插入）。`microbatch` 的 AvgTransferTime 虽较低，但连续净空仅 298 mm，低于 300 mm 硬门，且有 2 项硬违规，因此不可行、不得作为正式方案；`pressure` 与 `full` 也因同类净空硬门失败而淘汰。

- 任务完成：600/600
- AvgTransferTime：854.551420 s
- Makespan：4061.8 s
- 内部连续最小净空：325.000000 mm
- 独立重建最小净空：324.999470 mm
- 独立保守下界：324.977470 mm
- 硬违规：0
- 未来信息泄漏：0，反事实测试通过
- P17：Node70、Link507、距起点303 mm

运行：

```bash
python3 "问题3答案/q3_solver.py" --root .
python3 "问题3答案/verify_q3.py"
```
