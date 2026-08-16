# 问题2正式交付说明

当前正式方案为 `balanced`。`baseline`、`rolling`、`balanced` 在本实例的主指标与 Makespan 完全平局，正式名称由确定性平局规则选出，不宣称 balanced 产生额外性能增益。

- 任务完成：190/190
- AvgTransferTime：142.946305 s
- Makespan：3715.2 s
- 内部连续最小净空：325.000000 mm
- 独立重建最小净空：324.999497 mm
- 独立保守下界：324.977497 mm
- 硬违规：0
- 未来信息泄漏：0，反事实测试通过
- P17：Node70、Link507、距起点303 mm

运行：

```bash
python3 "问题2答案/q2_solver.py" --root .
python3 "问题2答案/verify_q2.py"
```
