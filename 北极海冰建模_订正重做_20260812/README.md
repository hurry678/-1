# 北极边缘冰区海冰演化与船舶安全航行建模：订正重做

本目录按“先选模型，再选算法，再解模型，最后报告结果”的顺序整理五问。

主要入口：

- `数模全过程.md`：五问的模型、化简、算法、求解、结果和检验；
- `scripts/`：本次订正新增的可复核计算脚本；
- `results/`：结构化数值结果；
- `figures/`：正文引用图。

快速查看数值结论：`results/最终结果摘要.json`。

证据口径：

1. 问题一采用教师评分口径，以科学计数法的指数比较五项力；
2. 问题二优先采用守恒、稳定的分方程数值格式；
3. 问题三至少使用两类优化器，并报告边界最优与可辨识性；
4. 问题四采用约束变分同化表述和低维增量求解；
5. 问题五只规划到距目的地 `E` 最近的 `A < 0.6` 可通行点，不把 `E` 当作可达终点。

根目录中的既有求解器与输出只作为可复用计算证据；本目录中的文档会明确区分本次
复核结果、教师参考值和受算力限制的替代网格结果。

运行环境要求 Python 3.11+（源码使用 `enum.StrEnum`）：

```bash
/opt/homebrew/bin/python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/问题一_量级与闭式解.py
.venv/bin/python scripts/问题四_增量变分同化.py
.venv/bin/python scripts/统一72h预报.py
.venv/bin/python scripts/问题五_严格时空图.py
```
