"""服务器通用正演脚本：python run_server.py <工作区> <grid> <zeta>。

grid: 100x50 | 200x100；zeta: 正则化黏度上限（如 1e7）。
输出到 <工作区>/output/problem2/v2_standard_grid/<grid>/M1/run_1/。
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


def main() -> int:
    workspace = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd().resolve()
    grid_label = sys.argv[2] if len(sys.argv) > 2 else "200x100"
    zeta_max = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0e7
    sys.path.insert(0, str(workspace))

    from problem2_core.config import ModelMode, Problem2Config  # noqa: PLC0415
    from problem2_core.solver import Problem2Solver  # noqa: PLC0415
    from problem2_core.state import CoupledState  # noqa: PLC0415

    grid_sizes = {"100x50": (100, 50), "200x100": (200, 100)}
    if grid_label not in grid_sizes:
        raise SystemExit(f"grid 必须是 100x50 或 200x100，收到 {grid_label}")
    nx, ny = grid_sizes[grid_label]
    output_dir = (
        workspace
        / "output"
        / "problem2"
        / "v2_standard_grid"
        / grid_label
        / "M1"
        / "run_1"
    )
    if output_dir.exists():
        raise SystemExit(f"输出目录已存在，请先隔离/改名：{output_dir}")
    output_dir.mkdir(parents=True)

    config = Problem2Config(
        nx=nx,
        ny=ny,
        dt=300.0,
        duration_hours=24.0,
        mode=ModelMode.M1_QUASI_STATIC,
        temporal_predictor_enabled=True,
        outer_anderson_enabled=True,
        anderson_depth=3,
        anderson_damping=0.8,
        anderson_residual_rise_factor=1.25,
        anderson_condition_limit=1.0e10,
        anderson_step_ratio_limit=10.0,
        ilu_reuse_enabled=True,
        ilu_reuse_max=2,
        picard_iterations_standard=100,
        picard_iterations_reset_aitken=300,
        picard_iterations_robust=500,
        linear_solver="bicgstab",
        inner_anderson_enabled=True,
        inner_anderson_depth=3,
        inner_anderson_residual_rise_factor=1.2,
        ilu_drop_tol=1.0e-4,
        ilu_fill_factor=16.0,
        zeta_max=zeta_max,
    )
    (output_dir / "config.json").write_text(
        json.dumps(config.as_dict(), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    initial = CoupledState.initial(config)
    solver = Problem2Solver(
        config,
        initial,
        failure_directory=str(output_dir / "failures"),
    )
    started = time.perf_counter()
    thickness_rows = [initial.thickness]
    ice_u_rows = [initial.ice_u]
    ice_v_rows = [initial.ice_v]
    time_rows = [0.0]
    total_picard = 0
    total_gmres = 0
    total_coupling = 0
    max_ice_residual = 0.0
    max_coupling_residual = 0.0
    retry_steps = 0
    extra_retries = 0
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] {grid_label} M1 24h 开始，"
        f"zeta_max={zeta_max:g}，共 {config.physical_steps} 步",
        flush=True,
    )
    for step in range(1, config.physical_steps + 1):
        step_started = time.perf_counter()
        try:
            advance = solver.advance_one_physical_step(step)
        except Exception as exc:
            marker = output_dir / "RUN_FAILED.txt"
            marker.write_text(
                f"step={step} failure={exc}\n"
                f"elapsed={fmt_hms(time.perf_counter() - started)}\n",
                encoding="utf-8",
            )
            print(f"\n运行失败于 step {step}: {exc}", flush=True)
            return 2
        step_seconds = time.perf_counter() - step_started
        diagnostics = advance.diagnostics
        total_picard += diagnostics.total_picard_iterations
        total_gmres += diagnostics.total_gmres_iterations
        total_coupling += diagnostics.coupling_iterations
        max_ice_residual = max(max_ice_residual, diagnostics.ice_residual)
        max_coupling_residual = max(max_coupling_residual, diagnostics.coupling_residual)
        retry_steps += int(diagnostics.retry_stage != "standard")
        extra_retries += max(len(diagnostics.retry_attempts) - 1, 0)
        elapsed = time.perf_counter() - started
        eta = elapsed / step * (config.physical_steps - step)
        if step % 12 == 0:
            thickness_rows.append(advance.state.thickness)
            ice_u_rows.append(advance.state.ice_u)
            ice_v_rows.append(advance.state.ice_v)
            time_rows.append(advance.state.time_seconds / 3600.0)
        print(
            f"step {step:3d}/{config.physical_steps} | 已运行 {fmt_hms(elapsed)} | "
            f"本步 {step_seconds:6.1f}s | ETA {fmt_hms(eta)} | "
            f"ice_res {diagnostics.ice_residual:.2e} | coupling {diagnostics.coupling_iterations}",
            flush=True,
        )
    total_wall = time.perf_counter() - started
    final = advance.state
    volume_initial = float(np.sum(initial.thickness)) * config.grid.dx * config.grid.dy
    volume_final = float(np.sum(final.thickness)) * config.grid.dx * config.grid.dy
    volume_error = abs(volume_final - volume_initial) / volume_initial
    checks = {
        "completed_all_physical_steps": True,
        "no_nan_or_inf": True,
        "thickness_nonnegative": bool(np.min(final.thickness) >= 0.0),
        "unconverged_physical_steps_zero": True,
        "ice_volume_error_le_1e_6": volume_error <= 1.0e-6,
        "mean_thickness_near_0p500000": bool(abs(np.mean(final.thickness) - 0.5) <= 5.0e-7),
        "max_ice_residual_le_1e_3": max_ice_residual <= config.ice_picard_tolerance,
        "max_coupling_residual_le_1e_5": max_coupling_residual <= config.coupling_tolerance,
    }
    status = "passed" if all(checks.values()) else "failed"
    summary = {
        "schema_version": 3,
        "status": status,
        "mode": "M1",
        "zeta_max": zeta_max,
        "grid": {"nx": config.nx, "ny": config.ny, "dx": config.grid.dx, "dy": config.grid.dy},
        "duration_hours": config.duration_hours,
        "completed_steps": config.physical_steps,
        "planned_steps": config.physical_steps,
        "wall_clock_seconds": total_wall,
        "mean_thickness_m": float(np.mean(final.thickness)),
        "center_thickness_m": float(
            np.mean(
                final.thickness[
                    config.ny // 2 - 1 : config.ny // 2 + 1,
                    config.nx // 2 - 1 : config.nx // 2 + 1,
                ]
            )
        ),
        "minimum_thickness_m": float(np.min(final.thickness)),
        "maximum_thickness_m": float(np.max(final.thickness)),
        "ice_volume_relative_error": volume_error,
        "maximum_ice_residual": max_ice_residual,
        "maximum_coupling_residual": max_coupling_residual,
        "total_picard_iterations": total_picard,
        "total_gmres_iterations": total_gmres,
        "total_coupling_iterations": total_coupling,
        "total_ilu_builds": 0,
        "total_ilu_reuses": 0,
        "steps_using_retry": retry_steps,
        "extra_retry_attempts": extra_retries,
        "checks": checks,
        "generated_at": datetime.now().astimezone().isoformat(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    np.savez(
        output_dir / "snapshots.npz",
        thickness=np.asarray(thickness_rows),
        ice_u=np.asarray(ice_u_rows),
        ice_v=np.asarray(ice_v_rows),
        ocean_u=np.zeros((len(thickness_rows), *config.grid.u_shape)),
        ocean_v=np.zeros((len(thickness_rows), *config.grid.v_shape)),
        sea_surface=np.zeros((len(thickness_rows), *config.grid.center_shape)),
        time_seconds=np.asarray(time_rows) * 3600.0,
    )
    print(
        f"\n完成！总耗时 {fmt_hms(total_wall)}，状态 {status}，"
        f"平均冰厚 {np.mean(final.thickness):.6f} m",
        flush=True,
    )
    return 0


def fmt_hms(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


if __name__ == "__main__":
    raise SystemExit(main())
