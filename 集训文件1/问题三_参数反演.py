"""问题三：冰水拖曳系数 C_d 与风应力缩放系数 α 的双参数反演。

使用订正后的 M1 三力平衡正演内核（problem2_core，含时间预测 + 受保护
Anderson + ILU 复用优化）。观测为表 1 的 6 个位置 t=24 h 冰速。
固壁边界观测点（20000, 7500）处海冰无滑移速度为 0，属观测误差，按教师
订正提示给大观测方差并报告排除该点的敏感性，不修改观测值或边界条件。
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
from problem2_core.failure import PhysicalStepFailure
from problem2_core.runner import run_simulation
from problem2_core.solver import Problem2Solver
from problem2_core.state import CoupledState


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "output" / "problem3"

STATIONS = np.array(
    [
        (4000.0, 2500.0),
        (8000.0, 5000.0),
        (12000.0, 2500.0),
        (12000.0, 7500.0),
        (16000.0, 5000.0),
        (20000.0, 7500.0),
    ]
)
OBS_U = np.array([0.10, 0.18, 0.11, 0.07, 0.06, 0.04])
OBS_V = np.array([0.07, 0.08, 0.09, 0.14, 0.12, 0.08])
# 固壁点观测误差方差放大 100 倍（σ 0.1 vs 内部点 0.01 m/s）。
OBS_SIGMA = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.10])

CD_RANGE = (0.002, 0.015)
ALPHA_RANGE = (0.5, 1.5)
INVERSION_GRID = (20, 10)
VERIFICATION_GRID = (40, 20)


def forward_config(
    cd: float,
    alpha: float,
    *,
    nx: int = INVERSION_GRID[0],
    ny: int = INVERSION_GRID[1],
    picard_reset: int = 300,
    picard_robust: int = 500,
) -> Problem2Config:
    return Problem2Config(
        nx=nx,
        ny=ny,
        dt=300.0,
        duration_hours=24.0,
        mode=ModelMode.M1_QUASI_STATIC,
        drag_coefficient=float(cd),
        wind_stress_x=0.12 * float(alpha),
        wind_stress_y=0.04 * float(alpha),
        temporal_predictor_enabled=True,
        outer_anderson_enabled=True,
        anderson_depth=3,
        anderson_damping=0.8,
        anderson_residual_rise_factor=1.25,
        anderson_condition_limit=1.0e10,
        anderson_step_ratio_limit=10.0,
        ilu_reuse_enabled=True,
        ilu_reuse_max=2,
        picard_iterations_reset_aitken=picard_reset,
        picard_iterations_robust=picard_robust,
    )


def run_forward_state(config: Problem2Config) -> CoupledState:
    """直接推进 288 个物理步，返回终态（用于参数搜索，不做完整审计输出）。"""
    initial = CoupledState.initial(config)
    solver = Problem2Solver(
        config,
        initial,
        failure_directory=OUTPUT_ROOT / "failures",
    )
    state = initial
    for step in range(1, config.physical_steps + 1):
        advance = solver.advance_one_physical_step(step)
        state = advance.state
    return state


def _bilinear_center(x: float, y: float, grid: Any, field: np.ndarray) -> float:
    xs = grid.x0 + (np.arange(grid.nx, dtype=float) + 0.5) * grid.dx
    ys = grid.y0 + (np.arange(grid.ny, dtype=float) + 0.5) * grid.dy
    j = min(max(int(np.searchsorted(xs, x) - 1), 0), len(xs) - 2)
    i = min(max(int(np.searchsorted(ys, y) - 1), 0), len(ys) - 2)
    x0, x1 = xs[j], xs[j + 1]
    y0, y1 = ys[i], ys[i + 1]
    tx = 0.0 if x1 <= x0 else (x - x0) / (x1 - x0)
    ty = 0.0 if y1 <= y0 else (y - y0) / (y1 - y0)
    return float(
        (1 - ty) * ((1 - tx) * field[i, j] + tx * field[i, j + 1])
        + ty * ((1 - tx) * field[i + 1, j] + tx * field[i + 1, j + 1])
    )


def predicted_station_velocities(state: CoupledState) -> np.ndarray:
    """返回 (6, 2) 的中心速度预测。固壁点 (20000, 7500) 取无滑移 0。"""
    config = Problem2Config(nx=state.thickness.shape[1], ny=state.thickness.shape[0])
    grid = config.grid
    ice_uc, ice_vc = grid.faces_to_center(state.ice_u, state.ice_v)
    out = np.zeros((len(STATIONS), 2))
    for k, (x, y) in enumerate(STATIONS):
        if x >= 19999.0 or x <= 1.0 or y >= 9999.0 or y <= 1.0:
            out[k] = (0.0, 0.0)
            continue
        out[k, 0] = _bilinear_center(x, y, grid, ice_uc)
        out[k, 1] = _bilinear_center(x, y, grid, ice_vc)
    return out


def objective(
    params: np.ndarray,
    *,
    obs_u: np.ndarray = OBS_U,
    obs_v: np.ndarray = OBS_V,
    sigma: np.ndarray = OBS_SIGMA,
    nx: int = INVERSION_GRID[0],
    ny: int = INVERSION_GRID[1],
) -> tuple[float, np.ndarray]:
    cd, alpha = params
    try:
        state = run_forward_state(forward_config(cd, alpha, nx=nx, ny=ny))
        predicted = predicted_station_velocities(state)
    except Exception:
        predicted = np.full((len(STATIONS), 2), np.nan)
        return 1.0e12, predicted
    residual_u = obs_u - predicted[:, 0]
    residual_v = obs_v - predicted[:, 1]
    if not np.all(np.isfinite(residual_u)) or not np.all(np.isfinite(residual_v)):
        return 1.0e12, predicted
    cost = float(np.sum((residual_u**2 + residual_v**2) / sigma**2))
    return cost, predicted


def objective_only(params: np.ndarray) -> float:
    return objective(np.asarray(params, dtype=float))[0]


def _grid_points(n_per_dim: tuple[int, int]) -> np.ndarray:
    cds = np.linspace(CD_RANGE[0], CD_RANGE[1], n_per_dim[0])
    alphas = np.linspace(ALPHA_RANGE[0], ALPHA_RANGE[1], n_per_dim[1])
    return np.array([[cd, alpha] for cd in cds for alpha in alphas])


def _worker_eval(point: tuple[float, float]) -> tuple[float, float, float, np.ndarray]:
    cd, alpha = point
    start = time.perf_counter()
    cost, predicted = objective(np.array([cd, alpha]))
    return cd, alpha, cost, predicted


def _worker_eval_with_obs(
    payload: tuple[float, float, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[float, float, float, np.ndarray]:
    cd, alpha, obs_u, obs_v, sigma = payload
    cost, predicted = objective(
        np.array([cd, alpha]), obs_u=obs_u, obs_v=obs_v, sigma=sigma
    )
    return cd, alpha, cost, predicted


def grid_search(
    coarse: tuple[int, int] = (7, 6),
    refine: tuple[int, int] = (9, 9),
    *,
    parallel: bool = True,
    max_workers: int = 14,
    obs_u: np.ndarray = OBS_U,
    obs_v: np.ndarray = OBS_V,
    sigma: np.ndarray = OBS_SIGMA,
) -> dict[str, Any]:
    start = time.perf_counter()
    coarse_points = _grid_points(coarse)
    if parallel:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            payloads = [
                (float(p[0]), float(p[1]), obs_u, obs_v, sigma) for p in coarse_points
            ]
            coarse_results = list(pool.map(_worker_eval_with_obs, payloads))
    else:
        coarse_results = [
            _worker_eval_with_obs((float(p[0]), float(p[1]), obs_u, obs_v, sigma))
            for p in coarse_points
        ]
    best = min(coarse_results, key=lambda item: item[2])
    cd0, alpha0 = best[0], best[1]
    span_cd = (CD_RANGE[1] - CD_RANGE[0]) / (coarse[0] - 1)
    span_alpha = (ALPHA_RANGE[1] - ALPHA_RANGE[0]) / (coarse[1] - 1)
    fine_cds = np.linspace(cd0 - span_cd, cd0 + span_cd, refine[0])
    fine_alphas = np.linspace(alpha0 - span_alpha, alpha0 + span_alpha, refine[1])
    fine_points = np.array([[cd, alpha] for cd in fine_cds for alpha in fine_alphas])
    fine_points[:, 0] = np.clip(fine_points[:, 0], *CD_RANGE)
    fine_points[:, 1] = np.clip(fine_points[:, 1], *ALPHA_RANGE)
    if parallel:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            payloads = [
                (float(p[0]), float(p[1]), obs_u, obs_v, sigma) for p in fine_points
            ]
            fine_results = list(pool.map(_worker_eval_with_obs, payloads))
    else:
        fine_results = [
            _worker_eval_with_obs((float(p[0]), float(p[1]), obs_u, obs_v, sigma))
            for p in fine_points
        ]
    best = min(fine_results, key=lambda item: item[2])
    elapsed = time.perf_counter() - start
    return {
        "method": "coarse_to_fine_grid_search",
        "coarse_grid": list(coarse),
        "fine_grid": list(refine),
        "best_parameters": {"cd": float(best[0]), "alpha": float(best[1])},
        "best_cost": float(best[2]),
        "predicted_at_best": np.asarray([item[3][0] for item in fine_results]).tolist(),
        "forward_evaluations": len(coarse_results) + len(fine_results),
        "wall_clock_seconds": elapsed,
        "all_fine_results": [
            {"cd": cd, "alpha": alpha, "cost": cost}
            for cd, alpha, cost, _ in fine_results
        ],
        "fine_grid_cost_triples": [
            (float(cd), float(alpha), float(cost)) for cd, alpha, cost, _ in fine_results
        ],
    }


def nelder_mead(
    starts: list[tuple[float, float]],
    *,
    maxiter: int = 40,
    parallel_starts: bool = True,
    max_workers: int = 3,
    obs_u: np.ndarray = OBS_U,
    obs_v: np.ndarray = OBS_V,
    sigma: np.ndarray = OBS_SIGMA,
) -> dict[str, Any]:
    start = time.perf_counter()
    payloads = [
        (start_point, obs_u, obs_v, sigma, maxiter) for start_point in starts
    ]
    if parallel_starts and len(starts) > 1:
        with ProcessPoolExecutor(max_workers=min(max_workers, len(starts))) as pool:
            records = list(pool.map(_nm_worker, payloads))
    else:
        records = [_nm_worker(payload) for payload in payloads]
    elapsed = time.perf_counter() - start
    best = min(records, key=lambda item: item["cost"])
    return {
        "method": "nelder_mead_multi_start",
        "parallel_starts": parallel_starts,
        "best_parameters": {"cd": float(best["estimate"][0]), "alpha": float(best["estimate"][1])},
        "best_cost": float(best["cost"]),
        "records": records,
        "forward_evaluations": sum(item["evaluations"] for item in records),
        "wall_clock_seconds": elapsed,
    }


def _nm_worker(
    payload: tuple[tuple[float, float], np.ndarray, np.ndarray, np.ndarray, int],
) -> dict[str, Any]:
    start_point, obs_u_local, obs_v_local, sigma_local, maxiter = payload
    t0 = time.perf_counter()
    result = minimize(
        lambda p: objective(
            np.asarray(p, dtype=float),
            obs_u=obs_u_local,
            obs_v=obs_v_local,
            sigma=sigma_local,
        )[0],
        np.asarray(start_point, dtype=float),
        method="Nelder-Mead",
        options={"xatol": 1.0e-6, "fatol": 1.0e-9, "maxiter": maxiter, "maxfev": 40},
        bounds=[CD_RANGE, ALPHA_RANGE],
    )
    return {
        "start": list(start_point),
        "estimate": [float(result.x[0]), float(result.x[1])],
        "cost": float(result.fun),
        "iterations": int(result.nit),
        "evaluations": int(result.nfev),
        "success": bool(result.success),
        "message": str(result.message),
        "wall_clock_seconds": time.perf_counter() - t0,
    }


def twin_experiment(
    truth: tuple[float, float],
    *,
    noise_sigma: float = 0.01,
    seed: int = 20260807,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    state = run_forward_state(forward_config(truth[0], truth[1]))
    predicted = predicted_station_velocities(state)
    noise = rng.normal(0.0, noise_sigma, size=predicted.shape)
    obs_u = predicted[:, 0] + noise[:, 0]
    obs_v = predicted[:, 1] + noise[:, 1]
    sigma = np.full(len(STATIONS), noise_sigma)
    sigma[-1] = 0.10
    obs_u[-1] = 0.04
    obs_v[-1] = 0.08
    start = time.perf_counter()
    grid_result = grid_search(
        coarse=(6, 5),
        refine=(7, 7),
        parallel=True,
        max_workers=12,
        obs_u=obs_u,
        obs_v=obs_v,
        sigma=sigma,
    )
    nm_result = nelder_mead(
        [(0.005, 1.0), (0.010, 0.8), (0.004, 1.3)],
        parallel_starts=True,
        max_workers=3,
        maxiter=20,
        obs_u=obs_u,
        obs_v=obs_v,
        sigma=sigma,
    )
    return {
        "truth": {"cd": truth[0], "alpha": truth[1]},
        "noise_sigma_mps": noise_sigma,
        "grid_search": {
            key: value
            for key, value in grid_result.items()
            if key != "all_fine_results"
        },
        "nelder_mead": {
            key: value for key, value in nm_result.items() if key != "records"
        },
        "recovery_error_grid": {
            "cd": float(abs(grid_result["best_parameters"]["cd"] - truth[0])),
            "alpha": float(abs(grid_result["best_parameters"]["alpha"] - truth[1])),
        },
        "recovery_error_nm": {
            "cd": float(abs(nm_result["best_parameters"]["cd"] - truth[0])),
            "alpha": float(abs(nm_result["best_parameters"]["alpha"] - truth[1])),
        },
        "wall_clock_seconds": time.perf_counter() - start,
    }


def contour_figure(
    cd_alpha_costs: list[tuple[float, float, float]],
    *,
    best: tuple[float, float],
) -> Path:
    points = np.asarray(cd_alpha_costs)
    cds = np.unique(points[:, 0])
    alphas = np.unique(points[:, 1])
    grid = np.full((len(cds), len(alphas)), np.nan)
    for cd, alpha, cost in points:
        grid[int(np.searchsorted(cds, cd)), int(np.searchsorted(alphas, alpha))] = cost
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    cs = ax.contourf(cds, alphas, grid.T, levels=30, cmap="viridis")
    ax.contour(cds, alphas, grid.T, levels=12, colors="white", linewidths=0.6, alpha=0.6)
    ax.plot(best[0], best[1], "r*", ms=16, label="estimated optimum")
    ax.set_xlabel(r"$C_d$")
    ax.set_ylabel(r"$\alpha$")
    ax.set_title("Objective J on refined grid (twin data)")
    fig.colorbar(cs, label="J")
    ax.legend()
    fig.tight_layout()
    path = OUTPUT_ROOT / "objective_contour.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def verification_run(cd: float, alpha: float) -> dict[str, Any]:
    config = forward_config(cd, alpha, nx=VERIFICATION_GRID[0], ny=VERIFICATION_GRID[1])
    output = OUTPUT_ROOT / "verification_40x20_24h"
    return run_simulation(
        config,
        output,
        snapshot_hours=(0.0, 6.0, 12.0, 18.0, 24.0),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-data", action="store_true", help="对表 1 真实观测反演")
    parser.add_argument("--twin", action="store_true", help="运行孪生试验")
    parser.add_argument("--verify", type=float, nargs=2, metavar=("CD", "ALPHA"))
    parser.add_argument("--sequential", action="store_true", help="网格搜索串行")
    parser.add_argument("--smoke", action="store_true", help="单次目标函数冒烟")
    args = parser.parse_args()
    if not any((args.real_data, args.twin, args.verify, args.smoke)):
        parser.error("请指定 --real-data、--twin、--verify 或 --smoke")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"generated_at": datetime.now().astimezone().isoformat()}
    if args.smoke:
        started = time.perf_counter()
        cost, predicted = objective(np.array([0.005, 1.0]))
        result["smoke"] = {
            "cost": cost,
            "predicted_station_velocities": predicted.tolist(),
            "elapsed_seconds": time.perf_counter() - started,
        }
    if args.twin:
        result["twin"] = twin_experiment((0.008, 1.2), seed=20260807)
    if args.real_data:
        grid_result = grid_search(parallel=not args.sequential, max_workers=14)
        partial = {
            "generated_at": datetime.now().astimezone().isoformat(),
            "phase": "grid_search_done",
            "grid_search": {
                key: value
                for key, value in grid_result.items()
                if key not in ("all_fine_results", "fine_grid_cost_triples")
            },
        }
        partial_path = OUTPUT_ROOT / "problem3_grid_partial.json"
        partial_path.write_text(
            json.dumps(partial, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        nm_result = nelder_mead(
            [
                (0.005, 1.0),
                (0.002, 0.5),
                (0.012, 1.2),
            ],
            parallel_starts=True,
            max_workers=3,
        )
        nm_partial = {
            "generated_at": datetime.now().astimezone().isoformat(),
            "phase": "nelder_mead_done",
            "nelder_mead": {
                key: value for key, value in nm_result.items() if key != "records"
            },
            "nelder_mead_records": nm_result["records"],
        }
        (OUTPUT_ROOT / "problem3_nm_partial.json").write_text(
            json.dumps(nm_partial, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        result["grid_search"] = {
            key: value for key, value in grid_result.items() if key != "all_fine_results"
        }
        result["nelder_mead"] = {
            key: value for key, value in nm_result.items() if key != "records"
        }
        result["nelder_mead_records"] = nm_result["records"]
        best = grid_result["best_parameters"]
        state = run_forward_state(
            forward_config(best["cd"], best["alpha"])
        )
        result["predicted_station_velocities_at_best"] = (
            predicted_station_velocities(state).tolist()
        )
        result["residuals_at_best"] = {
            "u": (OBS_U - predicted_station_velocities(state)[:, 0]).tolist(),
            "v": (OBS_V - predicted_station_velocities(state)[:, 1]).tolist(),
        }
        result["station_6_included"] = True
        result["station_6_note"] = (
            "固壁点 (20000,7500) 海冰无滑移模型预测为 0，观测方差放大 100 倍；"
            "反演以内部 5 点为主导。"
        )
        if "fine_grid_cost_triples" in grid_result:
            contour_figure(
                grid_result["fine_grid_cost_triples"],
                best=(
                    grid_result["best_parameters"]["cd"],
                    grid_result["best_parameters"]["alpha"],
                ),
            )
        result["grid_search"].pop("fine_grid_cost_triples", None)
    if args.verify is not None:
        result["verification_40x20"] = verification_run(args.verify[0], args.verify[1])
    output_name = (
        "problem3_twin_summary.json"
        if args.twin
        else "problem3_verify_summary.json"
        if args.verify is not None
        else "problem3_summary.json"
    )
    path = OUTPUT_ROOT / output_name
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))
    return 0


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
