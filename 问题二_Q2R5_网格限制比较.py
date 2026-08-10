"""问题二 Q2-R5：100×50 与 200×100 的 2:1 守恒限制比较。

将 200×100 终态与逐步场用守恒限制算子（面积平均/长度平均）限制到
100×50，与 100×50 直接运行结果比较，报告四类相对误差与守恒检查。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from problem2_core.grid import CGrid
from problem2_core.restriction import (
    restrict_center_area_average,
    restrict_u_face_length_average,
    restrict_v_face_length_average,
)


ROOT = Path(__file__).resolve().parent
GRID100 = ROOT / "output" / "problem2" / "v2_standard_grid" / "100x50" / "M1" / "run_1"
GRID200 = ROOT / "output" / "problem2" / "v2_standard_grid" / "200x100" / "M1" / "run_1"


def _load_snapshots(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def _center_velocity(ice_u: np.ndarray, ice_v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        0.5 * (ice_u[:, :-1] + ice_u[:, 1:]),
        0.5 * (ice_v[:-1, :] + ice_v[1:, :]),
    )


def main() -> int:
    for path in (GRID100, GRID200):
        if not (path / "summary.json").exists():
            raise FileNotFoundError(f"缺少运行目录：{path}")
    fine = _load_snapshots(GRID200)
    coarse = _load_snapshots(GRID100)
    fine_grid = CGrid(nx=200, ny=100, dx=100.0, dy=100.0)
    coarse_grid = CGrid(nx=100, ny=50, dx=200.0, dy=200.0)

    def compare_time_index(index: int) -> dict[str, float]:
        restricted_h = restrict_center_area_average(
            fine_grid, coarse_grid, fine["thickness"][index]
        )
        restricted_u = restrict_u_face_length_average(
            fine_grid, coarse_grid, fine["ice_u"][index]
        )
        restricted_v = restrict_v_face_length_average(
            fine_grid, coarse_grid, fine["ice_v"][index]
        )
        fine_uc, fine_vc = _center_velocity(fine["ice_u"][index], fine["ice_v"][index])
        coarse_uc, coarse_vc = _center_velocity(coarse["ice_u"][index], coarse["ice_v"][index])
        r_uc, r_vc = _center_velocity(restricted_u, restricted_v)
        active = coarse["thickness"][index] >= 1.0e-6
        velocity_error = float(
            np.sqrt(np.sum((r_uc[active] - coarse_uc[active]) ** 2 + (r_vc[active] - coarse_vc[active]) ** 2))
            / (np.sqrt(np.sum(coarse_uc[active] ** 2 + coarse_vc[active] ** 2)) + 1.0e-12)
        )
        thickness_error = float(
            np.sqrt(np.sum((restricted_h - coarse["thickness"][index]) ** 2))
            / (np.sqrt(np.sum(coarse["thickness"][index] ** 2)) + 1.0e-12)
        )
        volume_fine = float(np.sum(fine["thickness"][index])) * fine_grid.dx * fine_grid.dy
        volume_restricted = float(np.sum(restricted_h)) * coarse_grid.dx * coarse_grid.dy
        volume_coarse = float(np.sum(coarse["thickness"][index])) * coarse_grid.dx * coarse_grid.dy
        return {
            "velocity_terminal_relative_error": velocity_error,
            "thickness_terminal_relative_error": thickness_error,
            "volume_restriction_relative_error": abs(volume_restricted - volume_fine) / volume_fine,
            "volume_coarse_vs_fine_relative_error": abs(volume_coarse - volume_fine) / volume_fine,
            "mean_thickness_coarse_m": float(np.mean(coarse["thickness"][index])),
            "mean_thickness_restricted_m": float(np.mean(restricted_h)),
            "max_speed_coarse_mps": float(np.max(np.hypot(coarse_uc, coarse_vc))),
            "max_speed_restricted_mps": float(np.max(np.hypot(r_uc, r_vc))),
        }

    final_index = len(fine["thickness"]) - 1
    final_compare = compare_time_index(final_index)
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "scope": "100x50_direct_vs_restricted_200x100_2to1_nested",
        "final_time_comparison": final_compare,
        "note": "速度/冰厚相对误差按 100×50 直接运行场为参照；体积误差衡量保守限制的守恒性。",
    }
    out = ROOT / "output" / "problem2" / "v2_standard_grid"
    (out / "grid_restriction_comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
