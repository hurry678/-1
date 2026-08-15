"""问题二 Q2-R4/R5：M1 标准网格正演（100×50、200×100，24 h）。

使用已签署的 P1+P2+P3 优化实现（时间预测 + 受保护 Anderson 深度 3 +
ILU 复用 max=2）。仅用于 M1 正式模型，不改变方程、参数、容限与输出。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from problem2_core.config import ModelMode, Problem2Config
from problem2_core.runner import run_simulation


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "output" / "problem2" / "v2_standard_grid"


def run_config(*, nx: int, ny: int, mode: ModelMode = ModelMode.M1_QUASI_STATIC) -> Problem2Config:
    return Problem2Config(
        nx=nx,
        ny=ny,
        dt=300.0,
        duration_hours=24.0,
        mode=mode,
        temporal_predictor_enabled=True,
        outer_anderson_enabled=True,
        anderson_depth=3,
        anderson_damping=0.8,
        anderson_residual_rise_factor=1.25,
        anderson_condition_limit=1.0e10,
        anderson_step_ratio_limit=10.0,
        ilu_reuse_enabled=True,
        ilu_reuse_max=2,
        picard_iterations_reset_aitken=300,
        picard_iterations_robust=500,
        linear_solver="bicgstab",
        inner_anderson_enabled=True,
        inner_anderson_depth=3,
        inner_anderson_residual_rise_factor=1.2,
        ilu_drop_tol=1.0e-4,
        ilu_fill_factor=16.0,
    )


GRID_OPTIONS = {
    "100x50": (100, 50),
    "200x100": (200, 100),
}


def run_grid(grid: str, mode: str = "M1") -> dict[str, Any]:
    if grid not in GRID_OPTIONS:
        raise ValueError("grid 必须是 100x50 或 200x100")
    if mode not in ("M1", "M0"):
        raise ValueError("mode 必须是 M1 或 M0")
    nx, ny = GRID_OPTIONS[grid]
    config = run_config(
        nx=nx,
        ny=ny,
        mode=ModelMode.M0_FULL if mode == "M0" else ModelMode.M1_QUASI_STATIC,
    )
    output = OUTPUT_ROOT / grid / mode / "run_1"
    if output.exists():
        raise FileExistsError(f"拒绝覆盖既有正式运行目录：{output}")
    return run_simulation(
        config,
        output,
        snapshot_hours=(0.0, 6.0, 12.0, 18.0, 24.0),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", choices=tuple(GRID_OPTIONS), required=True)
    parser.add_argument("--mode", choices=("M1", "M0"), default="M1")
    args = parser.parse_args()
    result = run_grid(args.grid, args.mode)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
