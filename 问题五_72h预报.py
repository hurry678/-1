"""问题五：以问题三参数对与问题四初始场运行 72 h 正演，输出逐小时场。

用法：--cd <C_d> --alpha <α> --coeffs-json <问题四 JSON> --grid 40x20|100x50
输出：output/problem5/forecast_<grid>/ 含 forecast.npz（每小时一层
thickness/ice_u/ice_v/time_hours）与 summary.json。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np

from problem2_core.config import ModelMode, Problem2Config
from problem2_core.solver import Problem2Solver
from problem2_core.state import CoupledState

from 问题四_初始冰厚反演 import build_initial_thickness


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "output" / "problem5"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"无法序列化 {type(value)!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cd", type=float, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--coeffs-json", type=Path, required=True)
    parser.add_argument("--grid", choices=("40x20", "100x50"), default="40x20")
    args = parser.parse_args()
    nx, ny = (40, 20) if args.grid == "40x20" else (100, 50)
    payload = json.loads(args.coeffs_json.read_text(encoding="utf-8"))
    coeffs = np.asarray(payload["selected_coeffs"], dtype=float)
    config = Problem2Config(
        nx=nx,
        ny=ny,
        dt=300.0,
        duration_hours=72.0,
        mode=ModelMode.M1_QUASI_STATIC,
        drag_coefficient=args.cd,
        wind_stress_x=0.12 * args.alpha,
        wind_stress_y=0.04 * args.alpha,
        temporal_predictor_enabled=True,
        outer_anderson_enabled=True,
        anderson_depth=3,
        anderson_damping=0.8,
        anderson_residual_rise_factor=1.25,
        anderson_condition_limit=1.0e10,
        anderson_step_ratio_limit=10.0,
        ilu_reuse_enabled=True,
        ilu_reuse_max=2,
        picard_iterations_reset_aitken=200,
        picard_iterations_robust=300,
    )
    h0 = build_initial_thickness(config, coeffs)
    initial = CoupledState(
        thickness=h0,
        ice_u=np.zeros(config.grid.u_shape),
        ice_v=np.zeros(config.grid.v_shape),
        ocean_u=np.zeros(config.grid.u_shape),
        ocean_v=np.zeros(config.grid.v_shape),
        sea_surface=np.zeros(config.grid.center_shape),
    )
    solver = Problem2Solver(
        config,
        initial,
        failure_directory=OUTPUT_ROOT / f"forecast_{args.grid}" / "failures",
    )
    output = OUTPUT_ROOT / f"forecast_{args.grid}"
    output.mkdir(parents=True, exist_ok=True)
    thickness_rows: list[np.ndarray] = [h0]
    ice_u_rows: list[np.ndarray] = [np.zeros(config.grid.u_shape)]
    ice_v_rows: list[np.ndarray] = [np.zeros(config.grid.v_shape)]
    time_rows: list[float] = [0.0]
    max_residual = 0.0
    max_coupling_residual = 0.0
    completed = 0
    for step in range(1, config.physical_steps + 1):
        advance = solver.advance_one_physical_step(step)
        completed += 1
        max_residual = max(max_residual, advance.diagnostics.ice_residual)
        max_coupling_residual = max(
            max_coupling_residual, advance.diagnostics.coupling_residual
        )
        if step % 12 == 0:
            thickness_rows.append(advance.state.thickness)
            ice_u_rows.append(advance.state.ice_u)
            ice_v_rows.append(advance.state.ice_v)
            time_rows.append(advance.state.time_seconds / 3600.0)
    volume0 = float(np.sum(h0)) * config.grid.dx * config.grid.dy
    volume_final = float(np.sum(thickness_rows[-1])) * config.grid.dx * config.grid.dy
    np.savez(
        output / "forecast.npz",
        thickness=np.asarray(thickness_rows),
        ice_u=np.asarray(ice_u_rows),
        ice_v=np.asarray(ice_v_rows),
        time_hours=np.asarray(time_rows),
    )
    summary = {
        "grid": {"nx": nx, "ny": ny},
        "duration_hours": 72.0,
        "completed_steps": completed,
        "planned_steps": config.physical_steps,
        "max_ice_residual": float(max_residual),
        "max_coupling_residual": float(max_coupling_residual),
        "ice_volume_relative_error": abs(volume_final - volume0) / volume0,
        "minimum_thickness_m": float(np.min(thickness_rows[-1])),
        "maximum_thickness_m": float(np.max(thickness_rows[-1])),
        "cd": args.cd,
        "alpha": args.alpha,
        "generated_at": datetime.now().astimezone().isoformat(),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
