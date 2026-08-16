# 问题1正式交付说明

当前正式方案为 `ALNS`。结果真源为 `outputs/q1`，并已与本目录同名结果文件逐项校验 SHA-256。

- 任务完成：32/32
- AvgTransferTime：171.393750 s
- Makespan：332.8 s
- 内部连续最小净空：325.000000 mm
- 独立重建最小净空：324.999513 mm
- 独立保守下界：324.977513 mm
- 硬违规：0
- P17：Node70、Link507、距起点303 mm

运行：

```bash
python3 "问题1答案/q1_solver.py" --root . --iterations 1200
python3 "正式复核结果/独立连续净空复核.py" --links "附件2_轨道连接数据.xlsx" --trace "outputs/q1/问题1_OHT逐步运行记录.csv" --output "正式复核结果/问题1独立连续净空证据.json"
```
