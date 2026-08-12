"""用问题三参数和问题四增量变分初值统一运行 72 h 正演。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DELIVERY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DELIVERY_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from problem2_core.config import ModelMode, Problem2Config  # noqa: E402
from problem2_core.solver import Problem2Solver  # noqa: E402
from problem2_core.state import CoupledState  # noqa: E402
from 问题四_初始冰厚反演 import build_initial_thickness  # noqa: E402


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _plot(
    config: Problem2Config,
    time_hours: np.ndarray,
    thickness: np.ndarray,
    ice_u: np.ndarray,
    ice_v: np.ndarray,
) -> None:
    xs = (np.arange(config.nx) + 0.5) * config.grid.dx
    ys = (np.arange(config.ny) + 0.5) * config.grid.dy

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True, sharey=True)
    for ax, target_h in zip(axes, (24.0, 48.0, 72.0)):
        index = int(np.argmin(np.abs(time_hours - target_h)))
        field = np.clip(thickness[index], 0.0, 1.0)
        mesh = ax.pcolormesh(xs, ys, field, shading="auto", cmap="Blues", vmin=0, vmax=1)
        ax.contour(xs, ys, field, levels=[0.6], colors="red", linewidths=1.2)
        ax.set_title(f"t={target_h:.0f} h")
        ax.set_xlabel("x / m")
        ax.set_ylabel("y / m")
        fig.colorbar(mesh, ax=ax, label="A")
    fig.tight_layout()
    fig.savefig(DELIVERY_ROOT / "figures" / "问题五_24_48_72h冰密集度.png", dpi=180)
    plt.close(fig)

    index48 = int(np.argmin(np.abs(time_hours - 48.0)))
    uc, vc = config.grid.faces_to_center(ice_u[index48], ice_v[index48])
    speed = np.hypot(uc, vc)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    mesh0 = axes[0].pcolormesh(xs, ys, thickness[index48], shading="auto", cmap="viridis")
    fig.colorbar(mesh0, ax=axes[0], label="h_i / m")
    axes[0].set_title("48 h ice thickness")
    mesh1 = axes[1].pcolormesh(xs, ys, speed, shading="auto", cmap="inferno")
    fig.colorbar(mesh1, ax=axes[1], label="speed / (m/s)")
    skip = (slice(None, None, 3), slice(None, None, 3))
    axes[1].quiver(
        xs[::3][None, :] + np.zeros((ys[::3].size, 1)),
        ys[::3][:, None] + np.zeros((1, xs[::3].size)),
        uc[skip],
        vc[skip],
        color="white",
        scale=0.5,
    )
    axes[1].set_title("48 h ice velocity")
    for ax in axes:
        ax.set_xlabel("x / m")
        ax.set_ylabel("y / m")
    fig.tight_layout()
    fig.savefig(DELIVERY_ROOT / "figures" / "问题四_增量变分48h.png", dpi=180)
    plt.close(fig)


def run(coefficients_path: Path) -> dict[str, Any]:
    payload = json.loads(coefficients_path.read_text(encoding="utf-8"))
    coeffs = np.asarray(payload["selected_coeffs"], dtype=float)
    config = Problem2Config(
        nx=40,
        ny=20,
        dt=300.0,
        duration_hours=72.0,
        mode=ModelMode.M1_QUASI_STATIC,
        drag_coefficient=0.015,
        wind_stress_x=0.12 * 0.5,
        wind_stress_y=0.04 * 0.5,
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
        failure_directory=DELIVERY_ROOT / "results" / "forecast_failures",
    )
    thickness_rows = [h0.copy()]
    ice_u_rows = [initial.ice_u.copy()]
    ice_v_rows = [initial.ice_v.copy()]
    time_rows = [0.0]
    max_ice_residual = 0.0
    max_coupling_residual = 0.0
    max_cfl = 0.0
    started = time.perf_counter()
    for step in range(1, config.physical_steps + 1):
        advance = solver.advance_one_physical_step(step)
        max_ice_residual = max(max_ice_residual, advance.diagnostics.ice_residual)
        max_coupling_residual = max(
            max_coupling_residual, advance.diagnostics.coupling_residual
        )
        max_cfl = max(max_cfl, advance.diagnostics.thickness_cfl_macro)
        if step % 12 == 0:
            thickness_rows.append(advance.state.thickness.copy())
            ice_u_rows.append(advance.state.ice_u.copy())
            ice_v_rows.append(advance.state.ice_v.copy())
            time_rows.append(advance.state.time_seconds / 3600.0)

    thickness = np.asarray(thickness_rows)
    ice_u = np.asarray(ice_u_rows)
    ice_v = np.asarray(ice_v_rows)
    times = np.asarray(time_rows)
    output_npz = DELIVERY_ROOT / "results" / "统一72h预报.npz"
    np.savez(
        output_npz,
        thickness=thickness,
        ice_u=ice_u,
        ice_v=ice_v,
        time_hours=times,
    )
    index48 = int(np.argmin(np.abs(times - 48.0)))
    uc48, vc48 = config.grid.faces_to_center(ice_u[index48], ice_v[index48])
    volume0 = float(np.sum(h0) * config.grid.dx * config.grid.dy)
    volume72 = float(np.sum(thickness[-1]) * config.grid.dx * config.grid.dy)
    summary = {
        "grid": {"nx": config.nx, "ny": config.ny},
        "completed_steps": config.physical_steps,
        "planned_steps": config.physical_steps,
        "cd": 0.015,
        "alpha": 0.5,
        "wall_clock_seconds": time.perf_counter() - started,
        "maximum_ice_residual": max_ice_residual,
        "maximum_coupling_residual": max_coupling_residual,
        "maximum_thickness_cfl": max_cfl,
        "ice_volume_relative_error_72h": abs(volume72 - volume0) / volume0,
        "initial_mean_thickness_m": float(np.mean(h0)),
        "forecast_48h": {
            "mean_thickness_m": float(np.mean(thickness[index48])),
            "maximum_ice_speed_mps": float(np.max(np.hypot(uc48, vc48))),
            "minimum_thickness_m": float(np.min(thickness[index48])),
            "maximum_thickness_m": float(np.max(thickness[index48])),
            "center_thickness_m": float(
                np.mean(
                    thickness[index48][
                        config.ny // 2 - 1 : config.ny // 2 + 1,
                        config.nx // 2 - 1 : config.nx // 2 + 1,
                    ]
                )
            ),
        },
        "forecast_72h": {
            "mean_thickness_m": float(np.mean(thickness[-1])),
            "minimum_thickness_m": float(np.min(thickness[-1])),
            "maximum_thickness_m": float(np.max(thickness[-1])),
        },
    }
    (DELIVERY_ROOT / "results" / "统一72h预报摘要.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    _plot(config, times, thickness, ice_u, ice_v)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coefficients",
        type=Path,
        default=DELIVERY_ROOT / "results" / "问题四_增量变分.json",
    )
    args = parser.parse_args()
    summary = run(args.coefficients)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
