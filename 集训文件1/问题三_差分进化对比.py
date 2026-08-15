"""问题三对比实验：差分进化（全局优化）与网格搜索/Nelder-Mead 对照。"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

from 问题三_参数反演 import (
    ALPHA_RANGE,
    CD_RANGE,
    OBS_SIGMA,
    OBS_U,
    OBS_V,
    forward_config,
    predicted_station_velocities,
    run_forward_state,
)


ROOT = Path(__file__).resolve().parent


def _objective_de(params: list[float]) -> float:
    try:
        cd, alpha = params
        state = run_forward_state(
            forward_config(cd, alpha, picard_reset=200, picard_robust=300)
        )
        predicted = predicted_station_velocities(state)
        residual_u = OBS_U - predicted[:, 0]
        residual_v = OBS_V - predicted[:, 1]
        if not (
            np.all(np.isfinite(residual_u)) and np.all(np.isfinite(residual_v))
        ):
            return 1.0e12
        return float(np.sum((residual_u**2 + residual_v**2) / OBS_SIGMA**2))
    except Exception:
        return 1.0e12


def main() -> int:
    started = time.perf_counter()
    result = differential_evolution(
        _objective_de,
        bounds=[CD_RANGE, ALPHA_RANGE],
        seed=20260808,
        popsize=6,
        maxiter=8,
        tol=1.0e-9,
        mutation=(0.5, 1.0),
        recombination=0.7,
        workers=14,
        updating="deferred",
    )
    payload = {
        "method": "differential_evolution",
        "best_parameters": {"cd": float(result.x[0]), "alpha": float(result.x[1])},
        "best_cost": float(result.fun),
        "evaluations": int(result.nfev),
        "iterations": int(result.nit),
        "success": bool(result.success),
        "message": str(result.message),
        "wall_clock_seconds": time.perf_counter() - started,
        "generated_at": datetime.now().astimezone().isoformat(),
    }
    (ROOT / "output" / "problem3" / "problem3_de_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
