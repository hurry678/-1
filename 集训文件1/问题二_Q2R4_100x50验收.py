"""问题二 Q2-R4：100×50 中间网格 M1 验收与分析。

只读分析已完成的 100×50 运行；不启动新正演。100×50 与 40×20 为 2.5:1
非嵌套，仅作趋势对照；正式 2:1 嵌套守恒限制比较在 200×100 完成后进行。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
GRID_RUN = ROOT / "output" / "problem2" / "v2_standard_grid" / "100x50" / "M1" / "run_1"
REF40_RUN = ROOT / "output" / "problem2" / "v2_24h_verification" / "formal_40x20_24h" / "optimized" / "run_1"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_snapshots(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def main() -> int:
    if not (GRID_RUN / "summary.json").exists():
        raise FileNotFoundError(f"100×50 运行尚未完成：{GRID_RUN}")
    summary = _load_json(GRID_RUN / "summary.json")
    snapshots = _load_snapshots(GRID_RUN / "snapshots.npz")
    checks = dict(summary["checks"])
    all_checks_pass = bool(summary["status"] == "passed" and all(checks.values()))
    thickness = snapshots["thickness"]
    ice_u = snapshots["ice_u"]
    ice_v = snapshots["ice_v"]
    final_index = len(thickness) - 1
    ny, nx = thickness.shape[1:]
    dx = dy = 20000.0 / nx
    center_u = 0.5 * (ice_u[final_index, :, :-1] + ice_u[final_index, :, 1:])
    center_v = 0.5 * (ice_v[final_index, :-1, :] + ice_v[final_index, 1:, :])
    speed = np.hypot(center_u, center_v)
    h_final = thickness[final_index]
    center_thickness = float(np.mean(h_final[ny // 2 - 1 : ny // 2 + 1, nx // 2 - 1 : nx // 2 + 1]))
    max_speed_index = np.unravel_index(np.argmax(speed), speed.shape)
    stats = {
        "grid": {"nx": nx, "ny": ny, "dx_m": dx, "dy_m": dy},
        "completed_steps": summary["completed_steps"],
        "planned_steps": summary["planned_steps"],
        "status": summary["status"],
        "all_frozen_checks_passed": all_checks_pass,
        "failed_checks": [key for key, value in checks.items() if not value],
        "wall_clock_seconds": summary["wall_clock_seconds"],
        "total_picard_iterations": summary["total_picard_iterations"],
        "total_gmres_iterations": summary["total_gmres_iterations"],
        "total_coupling_iterations": summary["total_coupling_iterations"],
        "ilu_builds": summary["total_ilu_builds"],
        "ilu_reuses": summary["total_ilu_reuses"],
        "retry_steps": summary["steps_using_retry"],
        "mean_thickness_m": float(np.mean(h_final)),
        "max_thickness_m": float(np.max(h_final)),
        "min_thickness_m": float(np.min(h_final)),
        "center_thickness_m": center_thickness,
        "max_ice_speed_mps": float(np.max(speed)),
        "max_speed_center_x_m": float((max_speed_index[1] + 0.5) * dx),
        "max_speed_center_y_m": float((max_speed_index[0] + 0.5) * dy),
        "ice_volume_relative_error": summary["ice_volume_relative_error"],
        "maximum_ice_residual": summary["maximum_ice_residual"],
        "maximum_coupling_residual": summary["maximum_coupling_residual"],
        "source_hash_unchanged": checks.get("source_unchanged_during_run"),
    }

    ref_payload: dict[str, Any] = {}
    if (REF40_RUN / "summary.json").exists():
        ref_summary = _load_json(REF40_RUN / "summary.json")
        ref_snapshots = _load_snapshots(REF40_RUN / "snapshots.npz")
        rny, rnx = ref_snapshots["thickness"].shape[1:]
        rcenter_u = 0.5 * (ref_snapshots["ice_u"][-1, :, :-1] + ref_snapshots["ice_u"][-1, :, 1:])
        rcenter_v = 0.5 * (ref_snapshots["ice_v"][-1, :-1, :] + ref_snapshots["ice_v"][-1, 1:, :])
        rspeed = np.hypot(rcenter_u, rcenter_v)
        rh = ref_snapshots["thickness"][-1]
        ref_payload = {
            "40x20_mean_thickness_m": float(np.mean(rh)),
            "40x20_max_speed_mps": float(np.max(rspeed)),
            "40x20_center_thickness_m": float(
                np.mean(rh[rny // 2 - 1 : rny // 2 + 1, rnx // 2 - 1 : rnx // 2 + 1])
            ),
            "40x20_wall_clock_seconds": ref_summary["wall_clock_seconds"],
            "mean_thickness_ratio_100x50_over_40x20": float(np.mean(h_final) / np.mean(rh)),
            "max_speed_ratio_100x50_over_40x20": float(np.max(speed) / np.max(rspeed)),
        }
        stats.update(ref_payload)

    output_dir = GRID_RUN.parent
    (output_dir / "q2r4_acceptance.json").write_text(
        json.dumps(
            {"generated_at": datetime.now().astimezone().isoformat(), "stats": stats},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    xs = (np.arange(nx) + 0.5) * dx
    ys = (np.arange(ny) + 0.5) * dy
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    mesh = axes[0].pcolormesh(xs, ys, h_final, shading="auto", cmap="viridis")
    fig.colorbar(mesh, ax=axes[0], label="h_i / m")
    axes[0].set_title("100×50 M1 24 h ice thickness")
    axes[0].set_xlabel("x / m")
    axes[0].set_ylabel("y / m")
    mesh = axes[1].pcolormesh(xs, ys, speed, shading="auto", cmap="inferno")
    fig.colorbar(mesh, ax=axes[1], label="|U| / (m/s)")
    axes[1].set_title("100×50 M1 24 h ice speed")
    axes[1].set_xlabel("x / m")
    axes[1].set_ylabel("y / m")
    fig.tight_layout()
    fig.savefig(output_dir / "q2r4_100x50_fields.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"stats": stats}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
