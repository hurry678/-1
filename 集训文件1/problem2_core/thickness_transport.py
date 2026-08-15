"""冰厚守恒输运的一阶迎风验证基线。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from .grid import CGrid

Array = NDArray[np.float64]


@dataclass(frozen=True)
class ThicknessTransportDiagnostics:
    substeps: int
    cfl_macro: float
    minimum_thickness: float
    boundary_flux: float
    mass_correction: float = 0.0


def total_ice_volume(grid: CGrid, thickness: Array) -> float:
    grid.require_center(thickness)
    return float(np.sum(thickness, dtype=np.float64) * grid.cell_area)


def _upwind_fluxes(
    grid: CGrid, thickness: Array, u_face: Array, v_face: Array
) -> tuple[Array, Array]:
    flux_u = np.zeros(grid.u_shape, dtype=float)
    flux_v = np.zeros(grid.v_shape, dtype=float)

    interior_u = u_face[:, 1:-1]
    flux_u[:, 1:-1] = interior_u * np.where(
        interior_u >= 0.0, thickness[:, :-1], thickness[:, 1:]
    )
    interior_v = v_face[1:-1, :]
    flux_v[1:-1, :] = interior_v * np.where(
        interior_v >= 0.0, thickness[:-1, :], thickness[1:, :]
    )
    # 物理边界面严格零法向通量；不从速度掩膜间接推断。
    flux_u[:, (0, -1)] = 0.0
    flux_v[(0, -1), :] = 0.0
    return flux_u, flux_v


def advect_thickness_upwind(
    grid: CGrid,
    thickness: Array,
    u_face: Array,
    v_face: Array,
    *,
    dt: float,
    cfl_limit: float = 0.8,
) -> tuple[Array, ThicknessTransportDiagnostics]:
    """一阶守恒迎风输运，并按冻结 CFL 定义自动划分子步。"""

    grid.require_center(thickness)
    grid.require_u(u_face)
    grid.require_v(v_face)
    if dt <= 0.0 or not (0.0 < cfl_limit <= 1.0):
        raise ValueError("dt 必须为正，cfl_limit 必须位于 (0, 1]")
    if np.min(thickness) < 0.0:
        raise ValueError("输入冰厚必须非负")

    cfl_macro = float(
        np.max(np.abs(u_face)) * dt / grid.dx
        + np.max(np.abs(v_face)) * dt / grid.dy
    )
    substeps = max(1, int(math.ceil(cfl_macro / cfl_limit)))
    dt_sub = dt / substeps
    updated = np.array(thickness, dtype=float, copy=True)

    boundary_flux = 0.0
    for _ in range(substeps):
        flux_u, flux_v = _upwind_fluxes(grid, updated, u_face, v_face)
        boundary_flux += dt_sub * (
            grid.dy
            * float(np.sum(np.abs(flux_u[:, 0])) + np.sum(np.abs(flux_u[:, -1])))
            + grid.dx
            * float(np.sum(np.abs(flux_v[0, :])) + np.sum(np.abs(flux_v[-1, :])))
        )
        divergence = (
            (flux_u[:, 1:] - flux_u[:, :-1]) / grid.dx
            + (flux_v[1:, :] - flux_v[:-1, :]) / grid.dy
        )
        updated = updated - dt_sub * divergence
        if np.min(updated) < -1.0e-12:
            raise FloatingPointError("迎风基线产生负冰厚，CFL 或输入状态不合法")
        updated[updated < 0.0] = 0.0

    diagnostics = ThicknessTransportDiagnostics(
        substeps=substeps,
        cfl_macro=cfl_macro,
        minimum_thickness=float(np.min(updated)),
        boundary_flux=float(boundary_flux),
    )
    return updated, diagnostics
