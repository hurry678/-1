"""问题二正式出图：从任一冻结运行目录生成题面要求的图与统计。

用法：python 问题二_正式出图.py --run-dir <目录> --out-dir <目录>
要求目录内含 snapshots.npz（0/6/12/18/24 h）与 summary.json。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_snapshots(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def _center_velocity(ice_u: np.ndarray, ice_v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        0.5 * (ice_u[:, :-1] + ice_u[:, 1:]),
        0.5 * (ice_v[:-1, :] + ice_v[1:, :]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshots = _load_snapshots(run_dir / "snapshots.npz")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    times = snapshots["time_seconds"]
    ny, nx = snapshots["thickness"].shape[1:]
    dx = dy = 20000.0 / nx
    xs = (np.arange(nx) + 0.5) * dx
    ys = (np.arange(ny) + 0.5) * dy

    thickness = snapshots["thickness"]
    ice_u = snapshots["ice_u"]
    ice_v = snapshots["ice_v"]
    center_u, center_v = _center_velocity(ice_u[-1], ice_v[-1])
    speed = np.hypot(center_u, center_v)
    h_final = thickness[-1]

    # 图 1：t=0,6,12,18,24 h 冰厚分布（统一色标）
    target_hours = (0.0, 6.0, 12.0, 18.0, 24.0)
    indices = [int(np.argmin(np.abs(times - h * 3600.0))) for h in target_hours]
    vmin, vmax = float(np.min(thickness)), float(np.max(thickness))
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.6), sharex=True, sharey=True)
    for pos, (hour, index) in enumerate(zip(target_hours, indices)):
        ax = axes.flat[pos]
        mesh = ax.pcolormesh(xs, ys, thickness[index], shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"t = {hour:g} h")
        ax.set_xlabel("x / m")
        ax.set_ylabel("y / m")
    fig.delaxes(axes.flat[-1])
    fig.colorbar(mesh, ax=axes.flat[-1], label="h_i / m", shrink=0.9)
    fig.suptitle(f"M1 formal grid {nx}x{ny}: ice thickness evolution")
    fig.tight_layout()
    fig.savefig(out_dir / "thickness_evolution.png", dpi=160)
    plt.close(fig)

    # 图 2：24 h 冰厚等值线 + 冰速矢量
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    levels = np.linspace(vmin, vmax, 13)
    cs = ax.contourf(xs, ys, h_final, levels=levels, cmap="viridis")
    ax.contour(xs, ys, h_final, levels=levels[::2], colors="white", linewidths=0.5, alpha=0.6)
    fig.colorbar(cs, ax=ax, label="h_i / m")
    skip = (slice(None, None, max(1, nx // 20)), slice(None, None, max(1, ny // 10)))
    ax.quiver(
        xs[skip[1]][None, :] + np.zeros((ys[skip[0]].size, 1)),
        ys[skip[0]][:, None] + np.zeros((1, xs[skip[1]].size)),
        center_u[skip],
        center_v[skip],
        scale=0.5,
        color="white",
        alpha=0.9,
    )
    ax.set_title(f"24 h ice thickness and velocity ({nx}×{ny})")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    fig.tight_layout()
    fig.savefig(out_dir / "thickness_contour_velocity_24h.png", dpi=160)
    plt.close(fig)

    # 图 3：中心点冰厚时间序列
    center_h = [
        float(
            np.mean(
                thickness[index, ny // 2 - 1 : ny // 2 + 1, nx // 2 - 1 : nx // 2 + 1]
            )
        )
        for index in range(len(times))
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(times / 3600.0, center_h, "o-", ms=4)
    ax.set_xlabel("t / h")
    ax.set_ylabel("h_i at (10000, 5000) / m")
    ax.set_title("Center-point thickness time series")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "center_thickness_timeseries.png", dpi=160)
    plt.close(fig)

    stats = {
        "mean_thickness_m": float(np.mean(h_final)),
        "max_thickness_m": float(np.max(h_final)),
        "min_thickness_m": float(np.min(h_final)),
        "max_ice_speed_mps": float(np.max(speed)),
        "center_thickness_m": center_h[-1],
        "grid": {"nx": nx, "ny": ny, "dx_m": dx, "dy_m": dy},
        "ice_volume_relative_error": summary["ice_volume_relative_error"],
    }
    (out_dir / "formal_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
