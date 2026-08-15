"""问题四：低频控制空间中的增量变分同化。

在背景场处用有限差分构造观测算子切线矩阵 G，将一次增量 4D-Var
化为 12 x 12 的正则化法方程，再用非线性正演在线搜索上验收。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


DELIVERY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DELIVERY_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from 问题四_初始冰厚反演 import (  # noqa: E402
    K_MODES,
    L_MODES,
    OBS_U,
    OBS_V,
    build_initial_thickness,
    objective_with_coeffs,
)
from 问题三_参数反演 import OBS_SIGMA, forward_config  # noqa: E402


FD_STEP = 1.0e-3
MODE_SIGMA0_M = 0.05


def _simulate(coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, predicted, h0 = objective_with_coeffs(np.asarray(coeffs), kappa_s=0.0)
    if not np.all(np.isfinite(predicted)):
        raise RuntimeError("正演返回非有限观测")
    return predicted.reshape(-1), h0


def _simulate_worker(coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return _simulate(coeffs)


def _mode_sigmas() -> np.ndarray:
    sigmas = []
    for k in range(K_MODES):
        for l in range(L_MODES):
            # 高频模态先验方差更小，相当于背景项与 Sobolev 平滑项的合并。
            sigmas.append(MODE_SIGMA0_M / np.sqrt(1.0 + k * k + l * l))
    return np.asarray(sigmas)


def _observation_vector() -> np.ndarray:
    return np.stack([OBS_U, OBS_V], axis=-1).reshape(-1)


def _inverse_variances() -> np.ndarray:
    by_time_station = np.repeat(OBS_SIGMA[None, :] ** -2, 3, axis=0)
    return np.repeat(by_time_station[:, :, None], 2, axis=2).reshape(-1)


def _cost(coeffs: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = predicted - _observation_vector()
    obs = float(np.sum(_inverse_variances() * residual**2))
    prior = float(np.sum((coeffs / _mode_sigmas()) ** 2))
    return {"total": 0.5 * (obs + prior), "observation": 0.5 * obs, "prior": 0.5 * prior}


def run(max_workers: int) -> dict[str, Any]:
    started = time.perf_counter()
    n_control = K_MODES * L_MODES
    zero = np.zeros(n_control)
    perturbations = []
    for index in range(n_control):
        c = zero.copy()
        c[index] = FD_STEP
        perturbations.append(c)

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        simulations = list(pool.map(_simulate_worker, [zero, *perturbations]))

    y_background, h_background = simulations[0]
    jacobian = np.column_stack(
        [(simulations[i + 1][0] - y_background) / FD_STEP for i in range(n_control)]
    )
    innovations = _observation_vector() - y_background
    rinv = _inverse_variances()
    binv = np.diag(_mode_sigmas() ** -2)
    hessian = jacobian.T @ (rinv[:, None] * jacobian) + binv
    rhs = jacobian.T @ (rinv * innovations)
    raw_increment = np.linalg.solve(hessian, rhs)

    # 非线性前向模型上做离散线搜索，防止切线近似步长过大。
    line_factors = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    candidates = [factor * raw_increment for factor in line_factors]
    with ProcessPoolExecutor(max_workers=min(max_workers, len(candidates))) as pool:
        candidate_runs = list(pool.map(_simulate_worker, candidates))
    records = []
    for factor, coeffs, (predicted, h0) in zip(line_factors, candidates, candidate_runs):
        records.append(
            {
                "line_factor": float(factor),
                "coeffs": coeffs,
                "predicted": predicted,
                "h0": h0,
                "cost": _cost(coeffs, predicted),
            }
        )
    selected = min(records, key=lambda item: item["cost"]["total"])
    config = forward_config(0.015, 0.5)
    h_selected = build_initial_thickness(config, selected["coeffs"])

    singular_values = np.linalg.svd(
        np.sqrt(rinv)[:, None] * jacobian, compute_uv=False
    )
    result = {
        "method": "incremental_variational_assimilation",
        "control_dimension": n_control,
        "observation_dimension": int(_observation_vector().size),
        "fd_step_m": FD_STEP,
        "selected_line_factor": selected["line_factor"],
        "selected_coeffs": selected["coeffs"].tolist(),
        "background_cost": _cost(zero, y_background),
        "selected_cost": selected["cost"],
        "cost_reduction_pct": float(
            100.0
            * (_cost(zero, y_background)["total"] - selected["cost"]["total"])
            / _cost(zero, y_background)["total"]
        ),
        "mean_abs_background_deviation_m": float(
            np.mean(np.abs(h_selected - h_background))
        ),
        "max_abs_background_deviation_m": float(
            np.max(np.abs(h_selected - h_background))
        ),
        "minimum_initial_thickness_m": float(np.min(h_selected)),
        "maximum_initial_thickness_m": float(np.max(h_selected)),
        "weighted_jacobian_singular_values": singular_values.tolist(),
        "weighted_jacobian_condition_number": float(
            singular_values[0] / max(singular_values[-1], 1.0e-15)
        ),
        "line_search": [
            {
                "factor": item["line_factor"],
                "total_cost": item["cost"]["total"],
                "observation_cost": item["cost"]["observation"],
                "prior_cost": item["cost"]["prior"],
            }
            for item in records
        ],
        "wall_clock_seconds": time.perf_counter() - started,
    }
    return result


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=13)
    args = parser.parse_args()
    result = run(max_workers=args.workers)
    output = DELIVERY_ROOT / "results" / "问题四_增量变分.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
