"""冻结规格 v2 的 VP 应变率、应力与厚度加权应力通量散度。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .grid import CGrid

Array = NDArray[np.float64]


@dataclass(frozen=True)
class StrainRates:
    exx: Array
    eyy: Array
    exy_corner: Array

    @property
    def exy_square_center(self) -> Array:
        squared = self.exy_corner**2
        return 0.25 * (
            squared[:-1, :-1]
            + squared[:-1, 1:]
            + squared[1:, :-1]
            + squared[1:, 1:]
        )


@dataclass(frozen=True)
class VPStress:
    sigma_xx: Array
    sigma_yy: Array
    sigma_xy_corner: Array
    pressure: Array
    zeta: Array
    eta: Array
    delta_epsilon: Array


def strain_rates(
    grid: CGrid,
    u_face: Array,
    v_face: Array,
    u_y_ghost: tuple[Array, Array] | None = None,
    v_x_ghost: tuple[Array, Array] | None = None,
) -> StrainRates:
    """由 C 网格面速度计算中心法向应变率和角点剪切应变率。

    未显式给出幽灵值时，采用海冰无滑移所需的切向奇延拓。
    解析制造场测试可传入位于相邻幽灵面中心的精确值。
    """

    grid.require_u(u_face)
    grid.require_v(v_face)
    exx = (u_face[:, 1:] - u_face[:, :-1]) / grid.dx
    eyy = (v_face[1:, :] - v_face[:-1, :]) / grid.dy

    if u_y_ghost is None:
        u_bottom, u_top = -u_face[0, :], -u_face[-1, :]
    else:
        u_bottom, u_top = u_y_ghost
    if v_x_ghost is None:
        v_left, v_right = -v_face[:, 0], -v_face[:, -1]
    else:
        v_left, v_right = v_x_ghost

    u_bottom = np.asarray(u_bottom, dtype=float)
    u_top = np.asarray(u_top, dtype=float)
    v_left = np.asarray(v_left, dtype=float)
    v_right = np.asarray(v_right, dtype=float)
    if u_bottom.shape != (grid.nx + 1,) or u_top.shape != (grid.nx + 1,):
        raise ValueError("U 的上下幽灵值长度必须为 nx + 1")
    if v_left.shape != (grid.ny + 1,) or v_right.shape != (grid.ny + 1,):
        raise ValueError("V 的左右幽灵值长度必须为 ny + 1")

    u_extended = np.vstack((u_bottom[None, :], u_face, u_top[None, :]))
    v_extended = np.column_stack((v_left, v_face, v_right))
    du_dy_corner = (u_extended[1:, :] - u_extended[:-1, :]) / grid.dy
    dv_dx_corner = (v_extended[:, 1:] - v_extended[:, :-1]) / grid.dx
    exy_corner = 0.5 * (du_dy_corner + dv_dx_corner)
    return StrainRates(exx=exx, eyy=eyy, exy_corner=exy_corner)


def strain_invariant_squared(
    strain: StrainRates,
    ellipse_ratio: float = 2.0,
) -> Array:
    """计算未加入 ``delta_min`` 的中心 VP 应变率不变量平方。"""

    if ellipse_ratio <= 0.0:
        raise ValueError("ellipse_ratio 必须为正数")
    inverse_e2 = ellipse_ratio**-2
    return (
        (strain.exx**2 + strain.eyy**2) * (1.0 + inverse_e2)
        + 4.0 * inverse_e2 * strain.exy_square_center
        + 2.0 * strain.exx * strain.eyy * (1.0 - inverse_e2)
    )


def compute_vp_stress(
    grid: CGrid,
    thickness: Array,
    strain: StrainRates,
    *,
    h_i0: float = 1.0,
    p_star: float = 5000.0,
    concentration_decay: float = 20.0,
    ellipse_ratio: float = 2.0,
    delta_min: float = 1.0e-9,
    zeta_max: float = 1.0e8,
) -> VPStress:
    """按冻结定义计算正则化 VP 应力，不改变任何物理公式。"""

    grid.require_center(thickness)
    if h_i0 <= 0.0 or p_star < 0.0 or delta_min <= 0.0 or zeta_max <= 0.0:
        raise ValueError("VP 参数超出允许范围")

    concentration = np.clip(thickness / h_i0, 0.0, 1.0)
    pressure = p_star * thickness * np.exp(
        -concentration_decay * (1.0 - concentration)
    )
    invariant_sq = strain_invariant_squared(strain, ellipse_ratio)
    delta_epsilon = np.sqrt(invariant_sq + delta_min**2)
    zeta = np.minimum(pressure / (2.0 * delta_epsilon), zeta_max)
    eta = zeta / ellipse_ratio**2
    divergence = strain.exx + strain.eyy
    sigma_xx = 2.0 * eta * strain.exx + (zeta - eta) * divergence - 0.5 * pressure
    sigma_yy = 2.0 * eta * strain.eyy + (zeta - eta) * divergence - 0.5 * pressure
    eta_corner = grid.center_to_corner(eta)
    sigma_xy_corner = 2.0 * eta_corner * strain.exy_corner
    return VPStress(
        sigma_xx=sigma_xx,
        sigma_yy=sigma_yy,
        sigma_xy_corner=sigma_xy_corner,
        pressure=pressure,
        zeta=zeta,
        eta=eta,
        delta_epsilon=delta_epsilon,
    )


def stress_divergence(
    grid: CGrid,
    thickness: Array,
    sigma_xx: Array,
    sigma_yy: Array,
    sigma_xy_corner: Array,
) -> tuple[Array, Array]:
    """将 ``div(h sigma)`` 从中心/角点通量离散回 U、V 面。

    中心厚度与法向应力先形成 ``Q_xx,Q_yy``；冰厚以偶延拓的
    四点/单边平均插值到角点后形成 ``Q_xy``。物理边界上的法向
    通量采用相邻中心值的单边延拓。
    """

    grid.require_center(thickness)
    grid.require_center(sigma_xx)
    grid.require_center(sigma_yy)
    grid.require_corner(sigma_xy_corner)

    q_xx = thickness * sigma_xx
    q_yy = thickness * sigma_yy
    q_xy_corner = grid.center_to_corner(thickness) * sigma_xy_corner

    q_xx_padded = np.pad(q_xx, ((0, 0), (1, 1)), mode="edge")
    dq_xx_dx = (
        q_xx_padded[:, 1:] - q_xx_padded[:, :-1]
    ) / grid.dx
    dq_xy_dy = (
        q_xy_corner[1:, :] - q_xy_corner[:-1, :]
    ) / grid.dy
    divergence_u = dq_xx_dx + dq_xy_dy

    q_yy_padded = np.pad(q_yy, ((1, 1), (0, 0)), mode="edge")
    dq_yy_dy = (
        q_yy_padded[1:, :] - q_yy_padded[:-1, :]
    ) / grid.dy
    dq_xy_dx = (
        q_xy_corner[:, 1:] - q_xy_corner[:, :-1]
    ) / grid.dx
    divergence_v = dq_xy_dx + dq_yy_dy
    return divergence_u, divergence_v
