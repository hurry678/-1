"""问题二冻结的海冰、海水、冰厚与 Helmholtz 边界模板。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .grid import CGrid

Array = NDArray[np.float64]


@dataclass(frozen=True)
class TangentialGhosts:
    u_bottom: Array
    u_top: Array
    v_left: Array
    v_right: Array


def zero_normal_face_fluxes(
    grid: CGrid, u_face: Array, v_face: Array
) -> tuple[Array, Array]:
    """将四周法向面量严格置零，不改变内部面。"""

    grid.require_u(u_face)
    grid.require_v(v_face)
    bounded_u = np.array(u_face, dtype=float, copy=True)
    bounded_v = np.array(v_face, dtype=float, copy=True)
    bounded_u[:, 0] = 0.0
    bounded_u[:, -1] = 0.0
    bounded_v[0, :] = 0.0
    bounded_v[-1, :] = 0.0
    return bounded_u, bounded_v


def apply_ice_no_slip(
    grid: CGrid, u_face: Array, v_face: Array
) -> tuple[Array, Array]:
    """海冰无滑移：法向面为零，切向条件由奇延拓幽灵值表达。"""

    return zero_normal_face_fluxes(grid, u_face, v_face)


def ice_tangential_ghosts(
    grid: CGrid, u_face: Array, v_face: Array
) -> TangentialGhosts:
    grid.require_u(u_face)
    grid.require_v(v_face)
    return TangentialGhosts(
        u_bottom=-np.array(u_face[0, :], copy=True),
        u_top=-np.array(u_face[-1, :], copy=True),
        v_left=-np.array(v_face[:, 0], copy=True),
        v_right=-np.array(v_face[:, -1], copy=True),
    )


def apply_water_free_slip(
    grid: CGrid, u_face: Array, v_face: Array
) -> tuple[Array, Array]:
    """海水自由滑移：法向面为零，切向量采用偶延拓。"""

    return zero_normal_face_fluxes(grid, u_face, v_face)


def water_tangential_ghosts(
    grid: CGrid, u_face: Array, v_face: Array
) -> TangentialGhosts:
    grid.require_u(u_face)
    grid.require_v(v_face)
    return TangentialGhosts(
        u_bottom=np.array(u_face[0, :], copy=True),
        u_top=np.array(u_face[-1, :], copy=True),
        v_left=np.array(v_face[:, 0], copy=True),
        v_right=np.array(v_face[:, -1], copy=True),
    )


def thickness_even_ghost(grid: CGrid, thickness: Array) -> Array:
    """冰厚偶延拓，返回带一层中心幽灵单元的数组。"""

    grid.require_center(thickness)
    return np.pad(thickness, ((1, 1), (1, 1)), mode="edge")


def homogeneous_neumann_ghost(grid: CGrid, center_field: Array) -> Array:
    """中心量齐次 Neumann 边界，对应一层偶延拓幽灵值。"""

    grid.require_center(center_field)
    return np.pad(center_field, ((1, 1), (1, 1)), mode="edge")


def ice_active_face_masks(
    grid: CGrid, thickness: Array, *, h_min: float
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]:
    """按冻结的逻辑“或”规则从中心冰区掩膜构造速度面掩膜。"""

    grid.require_center(thickness)
    if h_min < 0.0:
        raise ValueError("h_min 不得为负")
    center = thickness >= h_min
    u_face = np.zeros(grid.u_shape, dtype=bool)
    v_face = np.zeros(grid.v_shape, dtype=bool)
    u_face[:, 1:-1] = center[:, :-1] | center[:, 1:]
    u_face[:, 0] = center[:, 0]
    u_face[:, -1] = center[:, -1]
    v_face[1:-1, :] = center[:-1, :] | center[1:, :]
    v_face[0, :] = center[0, :]
    v_face[-1, :] = center[-1, :]
    return center, u_face, v_face
