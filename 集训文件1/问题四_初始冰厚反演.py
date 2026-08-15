"""问题四：初始冰厚场反演与 48 h 预报。

控制变量为低频谱系数 c（4×3=12 个余弦模式），目标函数为 18 组冰速
观测加权残差 + 背景正则化 + 空间平滑正则化；优化器 L-BFGS-B，梯度为
并行前向差分。固壁观测点按观测误差模型放大方差，不修改观测与边界。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from problem2_core.config import ModelMode, Problem2Config
from problem2_core.solver import Problem2Solver
from problem2_core.state import CoupledState

from 问题三_参数反演 import (
    OBS_SIGMA,
    STATIONS,
    forward_config,
    predicted_station_velocities,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "output" / "problem4"

# 表 2：6 个站点 × 3 个时刻 (6,12,24 h) 的观测。
OBS_TIMES_H = np.array([6.0, 12.0, 24.0])
OBS_U_6 = np.array([0.06, 0.12, 0.07, 0.04, 0.04, 0.02])
OBS_V_6 = np.array([0.04, 0.03, 0.05, 0.10, 0.07, 0.05])
OBS_U_12 = np.array([0.08, 0.15, 0.09, 0.06, 0.05, 0.03])
OBS_V_12 = np.array([0.05, 0.06, 0.07, 0.12, 0.10, 0.07])
OBS_U_24 = np.array([0.10, 0.18, 0.11, 0.07, 0.06, 0.04])
OBS_V_24 = np.array([0.07, 0.08, 0.09, 0.14, 0.12, 0.08])
OBS_U = np.stack([OBS_U_6, OBS_U_12, OBS_U_24])
OBS_V = np.stack([OBS_V_6, OBS_V_12, OBS_V_24])

K_MODES = 4
L_MODES = 3
COEFF_BOUNDS = [(-0.3, 0.3)] * (K_MODES * L_MODES)
COEFF_BOUNDS[0] = (-0.2, 0.2)

INVERSION_GRID = (20, 10)
VERIFICATION_GRID = (40, 20)
# 问题三正式反演结果（网格搜索最优；NM 备选 (0.0076, 1.498) 见敏感性说明）
CD_STAR = 0.015
ALPHA_STAR = 0.5
FD_STEP = 1.0e-3


def background_thickness(config: Problem2Config) -> np.ndarray:
    return CoupledState.initial(config).thickness


def basis(config: Problem2Config) -> np.ndarray:
    """返回 (K*L, ny, nx) 的余弦基函数场。"""
    grid = config.grid
    xs = grid.x0 + (np.arange(grid.nx, dtype=float) + 0.5) * grid.dx
    ys = grid.y0 + (np.arange(grid.ny, dtype=float) + 0.5) * grid.dy
    fields = []
    for k in range(K_MODES):
        for l in range(L_MODES):
            field = np.outer(
                np.cos(l * np.pi * ys / config.length_y),
                np.cos(k * np.pi * xs / config.length_x),
            )
            fields.append(field)
    return np.asarray(fields)


def build_initial_thickness(config: Problem2Config, coeffs: np.ndarray) -> np.ndarray:
    hb = background_thickness(config)
    perturbation = np.tensordot(coeffs, basis(config), axes=(0, 0))
    return np.clip(hb + perturbation, 0.0, 1.5)


def _state_at_times(config: Problem2Config, thickness: np.ndarray) -> dict[int, CoupledState]:
    initial = CoupledState(
        thickness=np.array(thickness, copy=True),
        ice_u=np.zeros(config.grid.u_shape),
        ice_v=np.zeros(config.grid.v_shape),
        ocean_u=np.zeros(config.grid.u_shape),
        ocean_v=np.zeros(config.grid.v_shape),
        sea_surface=np.zeros(config.grid.center_shape),
    )
    solver = Problem2Solver(
        config,
        initial,
        failure_directory=OUTPUT_ROOT / "failures",
    )
    sample_steps = {int(round(h * 3600.0 / config.dt)) for h in OBS_TIMES_H}
    samples: dict[int, CoupledState] = {}
    for step in range(1, config.physical_steps + 1):
        advance = solver.advance_one_physical_step(step)
        if step in sample_steps:
            samples[step] = advance.state
    return samples


def objective_with_coeffs(
    coeffs: np.ndarray,
    *,
    kappa_s: float = 0.5,
    nx: int = INVERSION_GRID[0],
    ny: int = INVERSION_GRID[1],
    cd: float = CD_STAR,
    alpha: float = ALPHA_STAR,
) -> tuple[float, np.ndarray, np.ndarray]:
    config = forward_config(
        cd, alpha, nx=nx, ny=ny, picard_reset=200, picard_robust=300
    )
    h0 = build_initial_thickness(config, np.asarray(coeffs, dtype=float))
    try:
        samples = _state_at_times(config, h0)
        predicted = np.zeros((len(OBS_TIMES_H), len(STATIONS), 2))
        for index, step in enumerate(sorted(samples)):
            predicted[index] = predicted_station_velocities(samples[step])
    except Exception:
        return 1.0e12, np.full((len(OBS_TIMES_H), len(STATIONS), 2), np.nan), h0
    residuals = np.sqrt(
        (OBS_U - predicted[:, :, 0]) ** 2 + (OBS_V - predicted[:, :, 1]) ** 2
    )
    if not np.all(np.isfinite(residuals)):
        return 1.0e12, predicted, h0
    obs_cost = float(np.sum((residuals**2) / (OBS_SIGMA[None, :] ** 2)))
    hb = background_thickness(config)
    grid = config.grid
    dy, dx = grid.dy, grid.dx
    grad_h = np.gradient(h0, dy, dx)
    lambda_b = 1.0 / (2.0 * 0.05**2)
    lambda_s = (
        kappa_s
        * lambda_b
        * float(np.sum(hb**2))
        / (float(np.sum(grad_h[0] ** 2 + grad_h[1] ** 2)) + 1.0e-12)
    )
    reg_b = lambda_b * float(np.sum((h0 - hb) ** 2))
    reg_s = lambda_s * float(np.sum(grad_h[0] ** 2 + grad_h[1] ** 2))
    return obs_cost + reg_b + reg_s, predicted, h0


def _objective_scalar(payload: tuple[np.ndarray, float]) -> float:
    coeffs, kappa_s = payload
    return objective_with_coeffs(coeffs, kappa_s=kappa_s)[0]


def _gradient_batch(
    payload: tuple[np.ndarray, float, float],
) -> tuple[np.ndarray, float]:
    coeffs, kappa_s, step = payload
    f0 = objective_with_coeffs(coeffs, kappa_s=kappa_s)[0]
    grad = np.empty_like(coeffs)
    for index in range(len(coeffs)):
        plus = np.array(coeffs, copy=True)
        plus[index] += step
        fplus = objective_with_coeffs(plus, kappa_s=kappa_s)[0]
        grad[index] = (fplus - f0) / step
    return grad, f0


def parallel_forward_gradient(
    coeffs: np.ndarray,
    *,
    kappa_s: float,
    max_workers: int = 13,
) -> tuple[np.ndarray, float]:
    """并行前向差分梯度：每个坐标的 +h 求值并行执行。"""
    plus_vectors = []
    for index in range(len(coeffs)):
        plus = np.array(coeffs, copy=True)
        plus[index] += FD_STEP
        plus_vectors.append(plus)
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        f0 = objective_with_coeffs(coeffs, kappa_s=kappa_s)[0]
        futures = [
            pool.submit(
                objective_with_coeffs,
                plus,
                kappa_s=kappa_s,
            )
            for plus in plus_vectors
        ]
        fplus = [future.result()[0] for future in futures]
    grad = (np.asarray(fplus) - f0) / FD_STEP
    return grad, f0


def run_inversion(
    *,
    kappa_s: float = 0.5,
    max_iter: int = 20,
    max_workers: int = 13,
) -> dict[str, Any]:
    config = forward_config(CD_STAR, ALPHA_STAR, picard_reset=200, picard_robust=300)
    coeffs0 = np.zeros(K_MODES * L_MODES)
    start = time.perf_counter()
    baseline_cost, baseline_predicted, _ = objective_with_coeffs(coeffs0, kappa_s=kappa_s)

    def callable_gradient(coeffs: np.ndarray) -> tuple[float, np.ndarray]:
        grad, value = parallel_forward_gradient(coeffs, kappa_s=kappa_s, max_workers=max_workers)
        return float(value), grad

    result = minimize(
        callable_gradient,
        coeffs0,
        method="L-BFGS-B",
        jac=True,
        bounds=COEFF_BOUNDS,
        options={"maxiter": max_iter, "ftol": 1.0e-8, "gtol": 1.0e-6},
    )
    elapsed = time.perf_counter() - start
    optimized_cost, optimized_predicted, h0_opt = objective_with_coeffs(
        result.x, kappa_s=kappa_s
    )
    hb = background_thickness(config)
    return {
        "kappa_s": kappa_s,
        "coeffs": result.x.tolist(),
        "optimized_cost": float(optimized_cost),
        "baseline_background_cost": float(baseline_cost),
        "cost_reduction_pct": float(100.0 * (baseline_cost - optimized_cost) / baseline_cost),
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "evaluations": int(result.nfev),
        "wall_clock_seconds": elapsed,
        "max_initial_thickness_m": float(np.max(h0_opt)),
        "min_initial_thickness_m": float(np.min(h0_opt)),
        "mean_abs_background_deviation_m": float(np.mean(np.abs(h0_opt - hb))),
        "max_abs_background_deviation_m": float(np.max(np.abs(h0_opt - hb))),
        "baseline_predicted_velocity_rms_mps": float(np.sqrt(np.mean(baseline_predicted**2))),
        "optimized_predicted_velocity_rms_mps": float(np.sqrt(np.mean(optimized_predicted**2))),
    }


def sensitivity_sweep(*, max_workers: int = 13) -> list[dict[str, Any]]:
    results = []
    for kappa_s in (0.5, 0.0):
        results.append(run_inversion(kappa_s=kappa_s, max_iter=10, max_workers=max_workers))
    return results


def verification_48h(coeffs: np.ndarray, *, kappa_s: float = 0.5) -> dict[str, Any]:
    config = Problem2Config(
        nx=VERIFICATION_GRID[0],
        ny=VERIFICATION_GRID[1],
        dt=300.0,
        duration_hours=48.0,
        mode=ModelMode.M1_QUASI_STATIC,
        drag_coefficient=CD_STAR,
        wind_stress_x=0.12 * ALPHA_STAR,
        wind_stress_y=0.04 * ALPHA_STAR,
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
    h0 = build_initial_thickness(config, np.asarray(coeffs, dtype=float))
    initial = CoupledState(
        thickness=h0,
        ice_u=np.zeros(config.grid.u_shape),
        ice_v=np.zeros(config.grid.v_shape),
        ocean_u=np.zeros(config.grid.u_shape),
        ocean_v=np.zeros(config.grid.v_shape),
        sea_surface=np.zeros(config.grid.center_shape),
    )
    solver = Problem2Solver(config, initial, failure_directory=OUTPUT_ROOT / "failures")
    states: dict[int, CoupledState] = {}
    for step in range(1, config.physical_steps + 1):
        advance = solver.advance_one_physical_step(step)
        if step % 144 == 0:
            states[step] = advance.state
    states[config.physical_steps] = advance.state
    plot_snapshots(states, config, h0, output_dir=OUTPUT_ROOT / "verification_40x20_48h")
    final = advance.state
    ice_uc, ice_vc = config.grid.faces_to_center(final.ice_u, final.ice_v)
    return {
        "final_time_hours": 48.0,
        "mean_thickness_m": float(np.mean(final.thickness)),
        "max_ice_speed_mps": float(np.max(np.hypot(ice_uc, ice_vc))),
        "center_thickness_m": float(np.mean(final.thickness[final.thickness.shape[0] // 2 - 1 : final.thickness.shape[0] // 2 + 1, final.thickness.shape[1] // 2 - 1 : final.thickness.shape[1] // 2 + 1])),
        "ice_volume_relative_error": float(
            abs(
                np.sum(final.thickness) * config.grid.dx * config.grid.dy
                - np.sum(h0) * config.grid.dx * config.grid.dy
            )
            / (np.sum(h0) * config.grid.dx * config.grid.dy)
        ),
        "minimum_thickness_m": float(np.min(final.thickness)),
        "maximum_thickness_m": float(np.max(final.thickness)),
    }


def plot_snapshots(
    states: dict[int, CoupledState],
    config: Problem2Config,
    h0: np.ndarray,
    *,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = config.grid
    xs = grid.x0 + (np.arange(grid.nx, dtype=float) + 0.5) * grid.dx
    ys = grid.y0 + (np.arange(grid.ny, dtype=float) + 0.5) * grid.dy
    all_snapshots = {0: h0, **{step: states[step].thickness for step in states}}
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.5), sharex=True, sharey=True)
    for index, (step, field) in enumerate(sorted(all_snapshots.items())):
        ax = axes.flat[index]
        mesh = ax.pcolormesh(xs, ys, field, shading="auto", cmap="viridis")
        fig.colorbar(mesh, ax=ax)
        ax.set_title(f"t = {step * config.dt / 3600.0:.0f} h" if step else "t = 0 h (initial h0)")
        ax.set_xlabel("x / m")
        ax.set_ylabel("y / m")
    fig.suptitle("Initial thickness (reconstructed) and 48 h forecast snapshots")
    fig.tight_layout()
    fig.savefig(output_dir / "thickness_snapshots_48h.png", dpi=160)
    plt.close(fig)

    final = states[max(states)]
    ice_uc, ice_vc = grid.faces_to_center(final.ice_u, final.ice_v)
    speed = np.hypot(ice_uc, ice_vc)
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    mesh = ax.pcolormesh(xs, ys, speed, shading="auto", cmap="inferno")
    fig.colorbar(mesh, ax=ax, label="ice speed / (m/s)")
    skip = (slice(None, None, 3), slice(None, None, 3))
    ax.quiver(
        xs[::3][None, :] + np.zeros((ys[::3].size, 1)),
        ys[::3][:, None] + np.zeros((1, xs[::3].size)),
        ice_uc[skip],
        ice_vc[skip],
        scale=0.5,
        color="white",
        alpha=0.8,
    )
    ax.set_title("48 h forecast ice velocity field")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    fig.tight_layout()
    fig.savefig(output_dir / "velocity_field_48h.png", dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", action="store_true", help="正则化敏感性扫描")
    parser.add_argument("--verify", action="store_true", help="用最优系数做 40×20、48 h 预报")
    args = parser.parse_args()
    if not args.sweep and not args.verify:
        parser.error("请指定 --sweep 或 --verify")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"generated_at": datetime.now().astimezone().isoformat()}
    if args.sweep:
        result["sensitivity"] = sensitivity_sweep(max_workers=13)
        best = min(result["sensitivity"], key=lambda item: item["optimized_cost"])
        result["selected_kappa_s"] = best["kappa_s"]
        result["selected_coeffs"] = best["coeffs"]
        _write_json(OUTPUT_ROOT / "problem4_inversion.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))
    if args.verify:
        previous = OUTPUT_ROOT / "problem4_inversion.json"
        if previous.exists():
            payload = json.loads(previous.read_text(encoding="utf-8"))
        else:
            payload = {"selected_coeffs": np.zeros(K_MODES * L_MODES).tolist()}
        result["verification_48h"] = verification_48h(
            np.asarray(payload["selected_coeffs"], dtype=float)
        )
        _write_json(OUTPUT_ROOT / "problem4_verify.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))
    return 0


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"无法序列化 {type(value)!r}")


if __name__ == "__main__":
    raise SystemExit(main())
