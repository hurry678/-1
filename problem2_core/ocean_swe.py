"""C 网格海水浅水方程的 IMEX/Helmholtz 求解。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import splu

from .boundary import zero_normal_face_fluxes
from .config import Problem2Config
from .coupling import InterfaceDrag
from .discrete_operators import SparseCGridOperators
from .grid import CGrid

Array = NDArray[np.float64]


@dataclass(frozen=True)
class OceanState:
    xi: Array
    u: Array
    v: Array


@dataclass(frozen=True)
class OceanStepResult:
    state: OceanState
    converged: bool
    helmholtz_residual: float
    equation_residual: float
    residual_xi: float
    residual_u: float
    residual_v: float
    coriolis_iterations: int
    adv_u: Array
    adv_v: Array
    applied_drag: InterfaceDrag
    water_drag_force_u: Array
    water_drag_force_v: Array


def _explicit_upwind_advection(
    grid: CGrid, u_face: Array, v_face: Array
) -> tuple[Array, Array]:
    """以旧时刻速度显式计算海水平流；切向采用自由滑移偶延拓。"""

    grid.require_u(u_face)
    grid.require_v(v_face)
    transverse_v = grid.v_to_u(v_face)
    transverse_u = grid.u_to_v(u_face)
    adv_u = np.zeros(grid.u_shape, dtype=float)
    adv_v = np.zeros(grid.v_shape, dtype=float)

    for j in range(grid.ny):
        for i in range(1, grid.nx):
            ax = float(u_face[j, i])
            ay = float(transverse_v[j, i])
            dx_term = (
                max(ax, 0.0) * (u_face[j, i] - u_face[j, i - 1]) / grid.dx
                + min(ax, 0.0) * (u_face[j, i + 1] - u_face[j, i]) / grid.dx
            )
            lower = u_face[j - 1, i] if j > 0 else u_face[j, i]
            upper = u_face[j + 1, i] if j < grid.ny - 1 else u_face[j, i]
            dy_term = (
                max(ay, 0.0) * (u_face[j, i] - lower) / grid.dy
                + min(ay, 0.0) * (upper - u_face[j, i]) / grid.dy
            )
            adv_u[j, i] = dx_term + dy_term

    for j in range(1, grid.ny):
        for i in range(grid.nx):
            ax = float(transverse_u[j, i])
            ay = float(v_face[j, i])
            left = v_face[j, i - 1] if i > 0 else v_face[j, i]
            right = v_face[j, i + 1] if i < grid.nx - 1 else v_face[j, i]
            dx_term = (
                max(ax, 0.0) * (v_face[j, i] - left) / grid.dx
                + min(ax, 0.0) * (right - v_face[j, i]) / grid.dx
            )
            dy_term = (
                max(ay, 0.0) * (v_face[j, i] - v_face[j - 1, i]) / grid.dy
                + min(ay, 0.0) * (v_face[j + 1, i] - v_face[j, i]) / grid.dy
            )
            adv_v[j, i] = dx_term + dy_term
    return adv_u, adv_v


class OceanIMEXSolver:
    """显式慢项、隐式重力波的标准 Helmholtz 压力校正器。"""

    def __init__(self, config: Problem2Config):
        self.config = config
        self.grid = config.grid
        self.ops = SparseCGridOperators.build(self.grid)
        laplacian = (
            self.ops.divergence_u @ self.ops.center_x_to_u
            + self.ops.divergence_v @ self.ops.center_y_to_v
        ).tocsr()
        helmholtz = sparse.eye(self.ops.n_center, format="csr") - (
            config.gravity
            * config.water_depth
            * config.dt**2
            * laplacian
        )
        self._helmholtz = helmholtz.tocsr()
        self._factor = splu(self._helmholtz.tocsc())

    def advance(self, old: OceanState, drag: InterfaceDrag) -> OceanStepResult:
        cfg, grid, ops = self.config, self.grid, self.ops
        grid.require_center(old.xi)
        grid.require_u(old.u)
        grid.require_v(old.v)
        grid.require_u(drag.tau_u)
        grid.require_v(drag.tau_v)
        adv_u, adv_v = _explicit_upwind_advection(grid, old.u, old.v)
        water_mass = cfg.rho_water * cfg.water_depth
        iterate_u = np.array(old.u, copy=True)
        iterate_v = np.array(old.v, copy=True)
        xi = np.array(old.xi, copy=True)
        new_u = np.array(old.u, copy=True)
        new_v = np.array(old.v, copy=True)
        grad_x = np.zeros(grid.u_shape, dtype=float)
        grad_y = np.zeros(grid.v_shape, dtype=float)
        helmholtz_residual = float("inf")
        residual_xi = residual_u = residual_v = float("inf")
        coriolis_iterations = 0
        for coriolis_iterations in range(1, cfg.ocean_coriolis_max_iterations + 1):
            predicted_u = (
                old.u
                - cfg.dt * adv_u
                + cfg.dt * cfg.coriolis * grid.v_to_u(iterate_v)
                - cfg.dt * drag.tau_u / water_mass
            )
            predicted_v = (
                old.v
                - cfg.dt * adv_v
                - cfg.dt * cfg.coriolis * grid.u_to_v(iterate_u)
                - cfg.dt * drag.tau_v / water_mass
            )
            predicted_u, predicted_v = zero_normal_face_fluxes(
                grid, predicted_u, predicted_v
            )
            divergence_predicted = (
                ops.divergence_u @ predicted_u.ravel()
                + ops.divergence_v @ predicted_v.ravel()
            )
            rhs = old.xi.ravel() - cfg.water_depth * cfg.dt * divergence_predicted
            xi_vector = self._factor.solve(rhs)
            xi = xi_vector.reshape(grid.center_shape)
            grad_x = (ops.center_x_to_u @ xi_vector).reshape(grid.u_shape)
            grad_y = (ops.center_y_to_v @ xi_vector).reshape(grid.v_shape)
            new_u = predicted_u - cfg.gravity * cfg.dt * grad_x
            new_v = predicted_v - cfg.gravity * cfg.dt * grad_y
            new_u, new_v = zero_normal_face_fluxes(grid, new_u, new_v)
            helmholtz_residual = float(
                np.linalg.norm(self._helmholtz @ xi_vector - rhs)
                / (np.linalg.norm(rhs) + cfg.norm_epsilon)
            )

            divergence_new = (
                ops.divergence_u @ new_u.ravel()
                + ops.divergence_v @ new_v.ravel()
            ).reshape(grid.center_shape)
            time_xi = (xi - old.xi) / cfg.dt
            continuity = time_xi + cfg.water_depth * divergence_new
            time_u = (new_u - old.u) / cfg.dt
            time_v = (new_v - old.v) / cfg.dt
            coriolis_u = cfg.coriolis * grid.v_to_u(new_v)
            coriolis_v = cfg.coriolis * grid.u_to_v(new_u)
            drag_u = drag.tau_u / water_mass
            drag_v = drag.tau_v / water_mass
            momentum_u = time_u + adv_u + cfg.gravity * grad_x - coriolis_u + drag_u
            momentum_v = time_v + adv_v + cfg.gravity * grad_y + coriolis_v + drag_v

            residual_xi = float(
                np.linalg.norm(continuity)
                / (
                    np.linalg.norm(time_xi)
                    + cfg.water_depth * np.linalg.norm(divergence_new)
                    + cfg.norm_epsilon
                )
            )
            residual_u = float(
                np.linalg.norm(momentum_u[:, 1:-1])
                / (
                    np.linalg.norm(time_u[:, 1:-1])
                    + np.linalg.norm(adv_u[:, 1:-1])
                    + cfg.gravity * np.linalg.norm(grad_x[:, 1:-1])
                    + np.linalg.norm(coriolis_u[:, 1:-1])
                    + np.linalg.norm(drag_u[:, 1:-1])
                    + cfg.norm_epsilon
                )
            )
            residual_v = float(
                np.linalg.norm(momentum_v[1:-1, :])
                / (
                    np.linalg.norm(time_v[1:-1, :])
                    + np.linalg.norm(adv_v[1:-1, :])
                    + cfg.gravity * np.linalg.norm(grad_y[1:-1, :])
                    + np.linalg.norm(coriolis_v[1:-1, :])
                    + np.linalg.norm(drag_v[1:-1, :])
                    + cfg.norm_epsilon
                )
            )
            equation_residual = max(residual_xi, residual_u, residual_v)
            if (
                helmholtz_residual <= cfg.water_residual_tolerance
                and equation_residual <= cfg.water_residual_tolerance
            ):
                break
            iterate_u, iterate_v = new_u, new_v

        equation_residual = max(residual_xi, residual_u, residual_v)
        converged = bool(
            np.all(np.isfinite(xi))
            and np.all(np.isfinite(new_u))
            and np.all(np.isfinite(new_v))
            and helmholtz_residual <= cfg.water_residual_tolerance
            and equation_residual <= cfg.water_residual_tolerance
        )
        return OceanStepResult(
            state=OceanState(xi=xi, u=new_u, v=new_v),
            converged=converged,
            helmholtz_residual=helmholtz_residual,
            equation_residual=equation_residual,
            residual_xi=residual_xi,
            residual_u=residual_u,
            residual_v=residual_v,
            coriolis_iterations=coriolis_iterations,
            adv_u=adv_u,
            adv_v=adv_v,
            applied_drag=drag,
            water_drag_force_u=-drag.tau_u,
            water_drag_force_v=-drag.tau_v,
        )


def lake_at_rest_residual(
    grid: CGrid,
    state: OceanState,
    *,
    water_depth: float,
    gravity: float,
    coriolis: float = 1.4e-4,
) -> OceanState:
    """计算零外力浅水算子的离散残差，用于静水保持验收。

    该函数没有替代正式 IMEX 时间推进器；它只验证 C 网格散度、压力梯度、
    科里奥利插值和固壁法向模板在静水状态下不会产生伪运动。
    """

    grid.require_center(state.xi)
    grid.require_u(state.u)
    grid.require_v(state.v)
    if water_depth <= 0.0 or gravity <= 0.0:
        raise ValueError("water_depth 和 gravity 必须为正数")

    divergence = (
        (state.u[:, 1:] - state.u[:, :-1]) / grid.dx
        + (state.v[1:, :] - state.v[:-1, :]) / grid.dy
    )
    xi_tendency = -water_depth * divergence

    xi_x_padded = np.pad(state.xi, ((0, 0), (1, 1)), mode="edge")
    grad_x = (xi_x_padded[:, 1:] - xi_x_padded[:, :-1]) / grid.dx
    xi_y_padded = np.pad(state.xi, ((1, 1), (0, 0)), mode="edge")
    grad_y = (xi_y_padded[1:, :] - xi_y_padded[:-1, :]) / grid.dy

    u_tendency = -gravity * grad_x + coriolis * grid.v_to_u(state.v)
    v_tendency = -gravity * grad_y - coriolis * grid.u_to_v(state.u)
    u_tendency, v_tendency = zero_normal_face_fluxes(
        grid, u_tendency, v_tendency
    )
    return OceanState(xi=xi_tendency, u=u_tendency, v=v_tendency)
