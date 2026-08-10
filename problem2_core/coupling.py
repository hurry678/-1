"""冰水界面拖曳的共享应力与离散冲量闭合。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .grid import CGrid
from .boundary import ice_active_face_masks

Array = NDArray[np.float64]


@dataclass(frozen=True)
class InterfaceDrag:
    tau_u: Array
    tau_v: Array
    coefficient_u: Array | None = None
    coefficient_v: Array | None = None
    active_u: NDArray[np.bool_] | None = None
    active_v: NDArray[np.bool_] | None = None


@dataclass(frozen=True)
class DragImpulseBalance:
    ice_u: Array
    ice_v: Array
    water_u: Array
    water_v: Array
    absolute_closure_error: float
    relative_closure_error: float
    face_l1_absolute_closure_error: float
    face_l1_relative_closure_error: float


def compute_interface_drag(
    grid: CGrid,
    *,
    ice_u: Array,
    ice_v: Array,
    water_u: Array,
    water_v: Array,
    rho_water: float,
    drag_coefficient: float,
    thickness: Array,
    h_min: float,
) -> InterfaceDrag:
    """在两个方向的 C 网格面上计算同一轮次的二次冰水拖曳应力。"""

    grid.require_u(ice_u)
    grid.require_v(ice_v)
    grid.require_u(water_u)
    grid.require_v(water_v)
    grid.require_center(thickness)
    if rho_water <= 0.0 or drag_coefficient < 0.0:
        raise ValueError("rho_water 必须为正，drag_coefficient 不得为负")
    if h_min < 0.0:
        raise ValueError("h_min 不得为负")
    _, active_u, active_v = ice_active_face_masks(
        grid, thickness, h_min=h_min
    )
    active_u = np.array(active_u, copy=True)
    active_v = np.array(active_v, copy=True)
    active_u[:, (0, -1)] = False
    active_v[(0, -1), :] = False

    relative_u = water_u - ice_u
    relative_v = water_v - ice_v
    relative_v_at_u = grid.v_to_u(relative_v)
    relative_u_at_v = grid.u_to_v(relative_u)

    speed_at_u = np.hypot(relative_u, relative_v_at_u)
    speed_at_v = np.hypot(relative_u_at_v, relative_v)
    coefficient_u = np.where(
        active_u, rho_water * drag_coefficient * speed_at_u, 0.0
    )
    coefficient_v = np.where(
        active_v, rho_water * drag_coefficient * speed_at_v, 0.0
    )
    tau_u = coefficient_u * relative_u
    tau_v = coefficient_v * relative_v
    return InterfaceDrag(
        tau_u=tau_u,
        tau_v=tau_v,
        coefficient_u=coefficient_u,
        coefficient_v=coefficient_v,
        active_u=active_u,
        active_v=active_v,
    )


def diagnose_applied_drag_closure(
    grid: CGrid,
    *,
    ice_force_u: Array,
    ice_force_v: Array,
    water_force_u: Array,
    water_force_v: Array,
    dt: float,
) -> DragImpulseBalance:
    """比较冰、水方程实际施加的四个力数组，不人为构造反作用。"""

    grid.require_u(ice_force_u)
    grid.require_v(ice_force_v)
    grid.require_u(water_force_u)
    grid.require_v(water_force_v)
    if dt <= 0.0:
        raise ValueError("dt 必须为正")
    weights_u = grid.u_dual_area_weights() * dt
    weights_v = grid.v_dual_area_weights() * dt
    ice_u = ice_force_u * weights_u
    ice_v = ice_force_v * weights_v
    water_u = water_force_u * weights_u
    water_v = water_force_v * weights_v
    closure_u = ice_u + water_u
    closure_v = ice_v + water_v
    ice_resultant = np.array(
        [np.sum(ice_u, dtype=np.float64), np.sum(ice_v, dtype=np.float64)]
    )
    water_resultant = np.array(
        [np.sum(water_u, dtype=np.float64), np.sum(water_v, dtype=np.float64)]
    )
    absolute_error = float(np.linalg.norm(ice_resultant + water_resultant))
    reference = float(np.linalg.norm(ice_resultant))
    relative_error = absolute_error / max(reference, np.finfo(float).tiny)
    face_l1_absolute_error = float(
        np.sum(np.abs(closure_u), dtype=np.float64)
        + np.sum(np.abs(closure_v), dtype=np.float64)
    )
    face_l1_reference = float(
        np.sum(np.abs(ice_u), dtype=np.float64)
        + np.sum(np.abs(ice_v), dtype=np.float64)
    )
    face_l1_relative_error = face_l1_absolute_error / max(
        face_l1_reference, np.finfo(float).tiny
    )
    return DragImpulseBalance(
        ice_u=ice_u,
        ice_v=ice_v,
        water_u=water_u,
        water_v=water_v,
        absolute_closure_error=absolute_error,
        relative_closure_error=relative_error,
        face_l1_absolute_closure_error=face_l1_absolute_error,
        face_l1_relative_closure_error=face_l1_relative_error,
    )
