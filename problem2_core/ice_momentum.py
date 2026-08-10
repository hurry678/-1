"""规格 v2 的 M0/M1 后向欧拉—Picard—GMRES 海冰动量求解器。"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, bicgstab, gmres, spilu, splu

from .boundary import apply_ice_no_slip, ice_active_face_masks
from .config import Problem2Config
from .coupling import InterfaceDrag, compute_interface_drag
from .discrete_operators import SparseCGridOperators
from .rheology_vp import compute_vp_stress, strain_rates, stress_divergence

Array = NDArray[np.float64]


@dataclass(frozen=True)
class IcePicardRecord:
    iteration: int
    nonlinear_residual: float
    gmres_residual: float
    gmres_iterations: int
    relaxation_factor: float
    preconditioner: str
    matrix_assembly_seconds: float = 0.0
    preconditioner_build_seconds: float = 0.0
    gmres_seconds: float = 0.0
    residual_evaluation_seconds: float = 0.0
    ilu_builds: int = 0
    ilu_reuses: int = 0
    ilu_rebuilds: int = 0
    cache_invalidation_reason: str = ""
    preconditioner_fallbacks: int = 0


@dataclass(frozen=True)
class IceMomentumResult:
    u: Array
    v: Array
    converged: bool
    nonlinear_residual: float
    picard_iterations: int
    gmres_iterations: int
    records: tuple[IcePicardRecord, ...]
    applied_drag: InterfaceDrag
    ice_drag_force_u: Array
    ice_drag_force_v: Array
    ice_solve_seconds: float = 0.0
    matrix_assembly_seconds: float = 0.0
    preconditioner_build_seconds: float = 0.0
    gmres_seconds: float = 0.0
    residual_evaluation_seconds: float = 0.0
    ilu_builds: int = 0
    ilu_reuses: int = 0
    ilu_rebuilds: int = 0
    cache_invalidation_reasons: tuple[str, ...] = ()
    preconditioner_fallbacks: int = 0


@dataclass(frozen=True)
class ForceBudget:
    """已接受海冰动量离散中的实际逐面力与中心诊断。"""

    face_arrays: Mapping[str, Array]
    center_arrays: Mapping[str, Array]
    summary: Mapping[str, Mapping[str, float]]


class IceMomentumSolver:
    """在统一 C 网格路径上求解独立海冰惯性/科氏权重的动量方程。"""

    def __init__(
        self,
        config: Problem2Config,
        *,
        ice_coriolis_weight: float | None = None,
    ):
        self.config = config
        self.grid = config.grid
        self.ops = SparseCGridOperators.build(self.grid)
        self.ice_coriolis_weight = (
            config.ice_coriolis_weight
            if ice_coriolis_weight is None
            else float(ice_coriolis_weight)
        )
        if self.ice_coriolis_weight not in (0.0, 1.0):
            raise ValueError("海冰科里奥利权重仅允许 0 或 1")

    def _active_vector(self, thickness: Array) -> Array:
        _, active_u, active_v = ice_active_face_masks(
            self.grid, thickness, h_min=self.config.h_min
        )
        active_u = np.array(active_u, copy=True)
        active_v = np.array(active_v, copy=True)
        active_u[:, (0, -1)] = False
        active_v[(0, -1), :] = False
        return np.concatenate((active_u.ravel(), active_v.ravel()))

    def _stress_blocks(
        self, thickness: Array, zeta: Array, eta: Array
    ) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
        self.grid.require_center(thickness)
        h_corner = self.grid.center_to_corner(thickness)
        eta_corner = self.grid.center_to_corner(eta)
        normal_plus = sparse.diags((thickness * (zeta + eta)).ravel())
        normal_minus = sparse.diags((thickness * (zeta - eta)).ravel())
        shear = sparse.diags((2.0 * h_corner * eta_corner).ravel())

        b_uu = (
            self.ops.center_x_to_u @ normal_plus @ self.ops.exx_from_u
            + self.ops.corner_y_to_u @ shear @ self.ops.exy_from_u
        ).tocsr()
        b_uv = (
            self.ops.center_x_to_u @ normal_minus @ self.ops.eyy_from_v
            + self.ops.corner_y_to_u @ shear @ self.ops.exy_from_v
        ).tocsr()
        b_vu = (
            self.ops.corner_x_to_v @ shear @ self.ops.exy_from_u
            + self.ops.center_y_to_v @ normal_minus @ self.ops.exx_from_u
        ).tocsr()
        b_vv = (
            self.ops.corner_x_to_v @ shear @ self.ops.exy_from_v
            + self.ops.center_y_to_v @ normal_plus @ self.ops.eyy_from_v
        ).tocsr()
        return b_uu, b_uv, b_vu, b_vv

    def assemble_linear_system(
        self,
        *,
        thickness: Array,
        old_u: Array,
        old_v: Array,
        water_u: Array,
        water_v: Array,
        iterate_u: Array,
        iterate_v: Array,
        inertia_weight: float,
    ) -> tuple[sparse.csr_matrix, Array]:
        cfg, grid, ops = self.config, self.grid, self.ops
        strain = strain_rates(grid, iterate_u, iterate_v)
        stress = compute_vp_stress(
            grid,
            thickness,
            strain,
            h_i0=cfg.h_i0,
            p_star=cfg.p_star,
            concentration_decay=cfg.concentration_decay,
            ellipse_ratio=cfg.ellipse_ratio,
            delta_min=cfg.delta_min,
            zeta_max=cfg.zeta_max,
        )
        b_uu, b_uv, b_vu, b_vv = self._stress_blocks(
            thickness, stress.zeta, stress.eta
        )

        linearized_drag = compute_interface_drag(
            grid,
            thickness=thickness,
            h_min=cfg.h_min,
            ice_u=iterate_u,
            ice_v=iterate_v,
            water_u=water_u,
            water_v=water_v,
            rho_water=cfg.rho_water,
            drag_coefficient=cfg.drag_coefficient,
        )
        if linearized_drag.coefficient_u is None or linearized_drag.coefficient_v is None:
            raise RuntimeError("正式拖曳接口必须返回 Picard 冻结系数")
        drag_u = linearized_drag.coefficient_u.ravel()
        drag_v = linearized_drag.coefficient_v.ravel()

        if inertia_weight:
            mass_u = cfg.rho_ice * grid.center_to_u(thickness).ravel()
            mass_v = cfg.rho_ice * grid.center_to_v(thickness).ravel()
            transient_u = mass_u / cfg.dt
            transient_v = mass_v / cfg.dt
            adv_u = ops.implicit_upwind_u(iterate_u, grid.v_to_u(iterate_v))
            adv_v = ops.implicit_upwind_v(grid.u_to_v(iterate_u), iterate_v)
            a_uu = (
                sparse.diags(transient_u + drag_u)
                + sparse.diags(mass_u) @ adv_u
                - b_uu
            )
            a_vv = (
                sparse.diags(transient_v + drag_v)
                + sparse.diags(mass_v) @ adv_v
                - b_vv
            )
        else:
            transient_u = np.zeros_like(drag_u)
            transient_v = np.zeros_like(drag_v)
            a_uu = sparse.diags(drag_u) - b_uu
            a_vv = sparse.diags(drag_v) - b_vv

        if self.ice_coriolis_weight:
            if not inertia_weight:
                mass_u = cfg.rho_ice * grid.center_to_u(thickness).ravel()
                mass_v = cfg.rho_ice * grid.center_to_v(thickness).ravel()
            a_uv = -sparse.diags(mass_u * cfg.coriolis) @ ops.v_to_u - b_uv
            a_vu = sparse.diags(mass_v * cfg.coriolis) @ ops.u_to_v - b_vu
        else:
            a_uv = -b_uv
            a_vu = -b_vu
        matrix = sparse.bmat(((a_uu, a_uv), (a_vu, a_vv)), format="csr")

        pressure_vector = (-0.5 * thickness * stress.pressure).ravel()
        pressure_u = ops.center_x_to_u @ pressure_vector
        pressure_v = ops.center_y_to_v @ pressure_vector
        rhs_u = (
            transient_u * old_u.ravel()
            + pressure_u
            + drag_u * water_u.ravel()
            + cfg.wind_stress_x
        )
        rhs_v = (
            transient_v * old_v.ravel()
            + pressure_v
            + drag_v * water_v.ravel()
            + cfg.wind_stress_y
        )
        rhs = np.concatenate((rhs_u, rhs_v))

        active = self._active_vector(thickness)
        active_diag = sparse.diags(active.astype(float))
        fixed_diag = sparse.diags((~active).astype(float))
        matrix = (active_diag @ matrix @ active_diag + fixed_diag).tocsr()
        rhs = np.where(active, rhs, 0.0)
        return matrix, rhs

    def _preconditioner(
        self, matrix: sparse.csr_matrix, kind: str
    ) -> tuple[LinearOperator, str]:
        if kind == "jacobi":
            diagonal = matrix.diagonal()
            safe = np.where(np.abs(diagonal) > 1.0e-14, diagonal, 1.0)
            return LinearOperator(matrix.shape, matvec=lambda x: x / safe), "jacobi"
        try:
            if kind == "robust_ilu":
                ilu = spilu(
                    matrix.tocsc(),
                    drop_tol=self.config.robust_ilu_drop_tol,
                    fill_factor=self.config.robust_ilu_fill_factor,
                )
            else:
                ilu = spilu(
                    matrix.tocsc(),
                    drop_tol=self.config.ilu_drop_tol,
                    fill_factor=self.config.ilu_fill_factor,
                )
            return LinearOperator(matrix.shape, matvec=ilu.solve), kind
        except Exception:
            diagonal = matrix.diagonal()
            safe = np.where(np.abs(diagonal) > 1.0e-14, diagonal, 1.0)
            return LinearOperator(matrix.shape, matvec=lambda x: x / safe), "jacobi_fallback"

    def _ilu_reuse_decision(
        self,
        matrix: sparse.csr_matrix,
        kind: str,
        cache: dict[str, Any],
        previous_gmres_iterations: int | None,
    ) -> tuple[LinearOperator | None, str]:
        """P3：受保护 ILU 复用决策。返回 (缓存预条件器, 决策原因)。

        仅在同一物理步、同一外层耦合轮次内的相邻 Picard 迭代间复用；跨调用
        （新物理步/新耦合轮次）时 cache 为空，必然重建。任何失效条件都返回
        ``None`` 并给出唯一失效原因，由调用方重建并记录。
        """
        cfg = self.config
        if not cfg.ilu_reuse_enabled:
            return None, "cache_disabled"
        if not cache:
            return None, "new_picard_round"
        reference = cache["ref_matrix"]
        if kind != cache["ref_kind"]:
            return None, "kind_changed"
        if matrix.shape != reference.shape:
            return None, "shape_changed"
        if cache["consecutive_reuse"] >= cfg.ilu_reuse_max:
            return None, "max_reuse_reached"
        denominator = float(sparse.linalg.norm(reference, ord=np.inf))
        if denominator <= 0.0:
            denominator = 1.0
        drift = float(sparse.linalg.norm(abs(matrix - reference), ord=np.inf)) / denominator
        if drift > 0.05:
            return None, "matrix_drift_exceeded"
        reference_gmres = int(cache["ref_gmres"])
        if (
            previous_gmres_iterations is not None
            and reference_gmres > 0
            and previous_gmres_iterations > 1.5 * reference_gmres
        ):
            return None, "gmres_degraded"
        return cache["ref_op"], "reuse"

    def _inner_anderson_combine(
        self,
        y_history: list[np.ndarray],
        f_history: list[np.ndarray],
        depth: int,
        active: np.ndarray,
    ) -> np.ndarray:
        """P4：内层 Picard 固定点的受保护 Anderson 组合。

        对最近 ``depth`` 个迭代 (y_j, f_j) 求解
        min_{γ: Σγ=1} ||Σ γ_j f_j||_2 的最小二乘问题，返回组合候选。
        秩亏、非有限或权重过大时回退到最近迭代。
        """
        ys = y_history[-depth:]
        fs = f_history[-depth:]
        if len(ys) < 2:
            return ys[-1]
        f_last = fs[-1]
        delta_f = np.column_stack([fs[j] - f_last for j in range(len(ys) - 1)])
        try:
            gamma_prime, *_ = np.linalg.lstsq(delta_f, -f_last, rcond=None)
        except np.linalg.LinAlgError:
            return ys[-1]
        gamma = np.concatenate([gamma_prime, [1.0 - float(np.sum(gamma_prime))]])
        if not np.all(np.isfinite(gamma)) or float(np.max(np.abs(gamma))) > 10.0:
            return ys[-1]
        candidate = np.zeros_like(ys[-1])
        for weight, y in zip(gamma, ys):
            candidate += weight * y
        candidate[~active] = 0.0
        return candidate

    def physical_residual_components(
        self,
        *,
        thickness: Array,
        old_u: Array,
        old_v: Array,
        water_u: Array,
        water_v: Array,
        candidate_u: Array,
        candidate_v: Array,
        applied_drag: InterfaceDrag,
        inertia_weight: float,
    ) -> tuple[Array, Array, Array, Array]:
        """返回规格 v2 的海冰物理残差与逐点完整项尺度。"""

        cfg = self.config
        terms = self._force_face_arrays(
            thickness=thickness,
            old_u=old_u,
            old_v=old_v,
            water_u=water_u,
            water_v=water_v,
            candidate_u=candidate_u,
            candidate_v=candidate_v,
            applied_drag=applied_drag,
            inertia_weight=inertia_weight,
        )
        inertia_u, inertia_v = terms["F_I_u"], terms["F_I_v"]
        coriolis_u, coriolis_v = terms["F_C_u"], terms["F_C_v"]
        stress_u, stress_v = terms["F_sigma_u"], terms["F_sigma_v"]
        drag_u, drag_v = terms["F_D_u"], terms["F_D_v"]
        wind_u, wind_v = terms["F_W_u"], terms["F_W_v"]
        residual_u = inertia_u - (coriolis_u + stress_u + drag_u + wind_u)
        residual_v = inertia_v - (coriolis_v + stress_v + drag_v + wind_v)
        scale_u = (
            np.abs(inertia_u)
            + np.abs(coriolis_u)
            + np.abs(stress_u)
            + np.abs(drag_u)
            + np.abs(wind_u)
        )
        scale_v = (
            np.abs(inertia_v)
            + np.abs(coriolis_v)
            + np.abs(stress_v)
            + np.abs(drag_v)
            + np.abs(wind_v)
        )
        return residual_u, residual_v, scale_u, scale_v

    def _force_face_arrays(
        self,
        *,
        thickness: Array,
        old_u: Array,
        old_v: Array,
        water_u: Array,
        water_v: Array,
        candidate_u: Array,
        candidate_v: Array,
        applied_drag: InterfaceDrag,
        inertia_weight: float,
    ) -> dict[str, Array]:
        """返回与正式物理残差完全相同的逐面离散力。"""

        cfg, grid, ops = self.config, self.grid, self.ops
        grid.require_u(water_u)
        grid.require_v(water_v)
        strain = strain_rates(grid, candidate_u, candidate_v)
        stress = compute_vp_stress(
            grid,
            thickness,
            strain,
            h_i0=cfg.h_i0,
            p_star=cfg.p_star,
            concentration_decay=cfg.concentration_decay,
            ellipse_ratio=cfg.ellipse_ratio,
            delta_min=cfg.delta_min,
            zeta_max=cfg.zeta_max,
        )
        stress_u, stress_v = stress_divergence(
            grid,
            thickness,
            stress.sigma_xx,
            stress.sigma_yy,
            stress.sigma_xy_corner,
        )
        grid.require_u(applied_drag.tau_u)
        grid.require_v(applied_drag.tau_v)
        mass_u = None
        mass_v = None
        if inertia_weight:
            mass_u = cfg.rho_ice * grid.center_to_u(thickness)
            mass_v = cfg.rho_ice * grid.center_to_v(thickness)
            adv_u = (
                ops.implicit_upwind_u(candidate_u, grid.v_to_u(candidate_v))
                @ candidate_u.ravel()
            ).reshape(grid.u_shape)
            adv_v = (
                ops.implicit_upwind_v(grid.u_to_v(candidate_u), candidate_v)
                @ candidate_v.ravel()
            ).reshape(grid.v_shape)
            inertia_u = mass_u * ((candidate_u - old_u) / cfg.dt + adv_u)
            inertia_v = mass_v * ((candidate_v - old_v) / cfg.dt + adv_v)
        else:
            inertia_u = np.zeros_like(candidate_u)
            inertia_v = np.zeros_like(candidate_v)

        if self.ice_coriolis_weight:
            if mass_u is None or mass_v is None:
                mass_u = cfg.rho_ice * grid.center_to_u(thickness)
                mass_v = cfg.rho_ice * grid.center_to_v(thickness)
            coriolis_u = mass_u * cfg.coriolis * grid.v_to_u(candidate_v)
            coriolis_v = -mass_v * cfg.coriolis * grid.u_to_v(candidate_u)
        else:
            coriolis_u = np.zeros_like(candidate_u)
            coriolis_v = np.zeros_like(candidate_v)
        _, active_u, active_v = ice_active_face_masks(grid, thickness, h_min=cfg.h_min)
        active_u = np.array(active_u, copy=True)
        active_v = np.array(active_v, copy=True)
        active_u[:, (0, -1)] = False
        active_v[(0, -1), :] = False

        def masked(values: Array, active: Array) -> Array:
            return np.where(active, values, 0.0)

        return {
            "F_I_u": masked(inertia_u, active_u),
            "F_I_v": masked(inertia_v, active_v),
            "F_C_u": masked(coriolis_u, active_u),
            "F_C_v": masked(coriolis_v, active_v),
            "F_sigma_u": masked(stress_u, active_u),
            "F_sigma_v": masked(stress_v, active_v),
            "F_D_u": masked(applied_drag.tau_u, active_u),
            "F_D_v": masked(applied_drag.tau_v, active_v),
            "F_W_u": np.where(active_u, cfg.wind_stress_x, 0.0),
            "F_W_v": np.where(active_v, cfg.wind_stress_y, 0.0),
        }

    def force_budget(
        self,
        *,
        thickness: Array,
        old_u: Array,
        old_v: Array,
        water_u: Array,
        water_v: Array,
        candidate_u: Array,
        candidate_v: Array,
        applied_drag: InterfaceDrag,
        inertia_weight: float,
    ) -> ForceBudget:
        """从已接受离散数组计算删项、保留项及近抵消比值。"""

        cfg, grid = self.config, self.grid
        faces = self._force_face_arrays(
            thickness=thickness,
            old_u=old_u,
            old_v=old_v,
            water_u=water_u,
            water_v=water_v,
            candidate_u=candidate_u,
            candidate_v=candidate_v,
            applied_drag=applied_drag,
            inertia_weight=inertia_weight,
        )

        centers: dict[str, Array] = {}
        center_vectors: dict[str, tuple[Array, Array]] = {}
        for prefix in ("F_I", "F_C", "F_sigma", "F_D", "F_W"):
            u_center, v_center = grid.faces_to_center(
                faces[f"{prefix}_u"], faces[f"{prefix}_v"]
            )
            center_vectors[prefix] = (u_center, v_center)
            centers[f"{prefix}_center_u"] = u_center
            centers[f"{prefix}_center_v"] = v_center
            centers[f"{prefix}_abs"] = np.hypot(u_center, v_center)

        inertia_u, inertia_v = center_vectors["F_I"]
        coriolis_u, coriolis_v = center_vectors["F_C"]
        stress_u, stress_v = center_vectors["F_sigma"]
        drag_u, drag_v = center_vectors["F_D"]
        wind_u, wind_v = center_vectors["F_W"]
        combined_missing_u = inertia_u - coriolis_u
        combined_missing_v = inertia_v - coriolis_v
        deprecated_fi_plus_fc_u = inertia_u + coriolis_u
        deprecated_fi_plus_fc_v = inertia_v + coriolis_v
        retained_u = stress_u + drag_u + wind_u
        retained_v = stress_v + drag_v + wind_v
        combined_missing_abs = np.hypot(combined_missing_u, combined_missing_v)
        deprecated_fi_plus_fc_abs = np.hypot(
            deprecated_fi_plus_fc_u, deprecated_fi_plus_fc_v
        )
        retained_net_abs = np.hypot(retained_u, retained_v)
        undirected_ic_abs_sum = centers["F_I_abs"] + centers["F_C_abs"]
        retained_abs_sum = (
            centers["F_sigma_abs"] + centers["F_D_abs"] + centers["F_W_abs"]
        )
        active_center = thickness >= cfg.h_min
        centers.update(
            {
                "combined_missing_u": combined_missing_u,
                "combined_missing_v": combined_missing_v,
                "combined_missing_abs": combined_missing_abs,
                "undirected_IC_abs_sum": undirected_ic_abs_sum,
                "retained_abs_sum": retained_abs_sum,
                "retained_net_u": retained_u,
                "retained_net_v": retained_v,
                "retained_net_abs": retained_net_abs,
                "r_IC_corrected": combined_missing_abs
                / (retained_abs_sum + cfg.force_epsilon),
                "r_Sigma": retained_net_abs / (retained_abs_sum + cfg.force_epsilon),
                "r_net_corrected": combined_missing_abs
                / (retained_net_abs + cfg.force_epsilon),
                "deprecated_FI_plus_FC_abs": deprecated_fi_plus_fc_abs,
                "r_IC_deprecated_FI_plus_FC": deprecated_fi_plus_fc_abs
                / (retained_abs_sum + cfg.force_epsilon),
                "r_net_deprecated_FI_plus_FC": deprecated_fi_plus_fc_abs
                / (retained_net_abs + cfg.force_epsilon),
                "active_center": active_center,
            }
        )
        for name, values in tuple(centers.items()):
            if name == "active_center":
                continue
            centers[name] = np.where(active_center, values, 0.0)

        def statistics(values: Array) -> dict[str, float]:
            selected = np.asarray(values, dtype=float)[active_center]
            if selected.size == 0:
                return {"median": 0.0, "p95": 0.0, "maximum": 0.0}
            return {
                "median": float(np.median(selected)),
                "p95": float(np.percentile(selected, 95.0)),
                "maximum": float(np.max(selected)),
            }

        reported = (
            "F_I_abs",
            "F_C_abs",
            "F_sigma_abs",
            "F_D_abs",
            "F_W_abs",
            "combined_missing_abs",
            "undirected_IC_abs_sum",
            "retained_abs_sum",
            "retained_net_abs",
            "r_IC_corrected",
            "r_Sigma",
            "r_net_corrected",
            "deprecated_FI_plus_FC_abs",
            "r_IC_deprecated_FI_plus_FC",
            "r_net_deprecated_FI_plus_FC",
        )
        summary = {name: statistics(centers[name]) for name in reported}
        summary["active_center_count"] = {
            "median": float(np.count_nonzero(active_center)),
            "p95": float(np.count_nonzero(active_center)),
            "maximum": float(np.count_nonzero(active_center)),
        }
        return ForceBudget(face_arrays=faces, center_arrays=centers, summary=summary)

    def nonlinear_residual(
        self,
        *,
        thickness: Array,
        old_u: Array,
        old_v: Array,
        water_u: Array,
        water_v: Array,
        candidate_u: Array,
        candidate_v: Array,
        applied_drag: InterfaceDrag,
        inertia_weight: float,
    ) -> float:
        """按活动内部冰面聚合规格 v2 的归一化物理残差。"""

        residual_u, residual_v, scale_u, scale_v = self.physical_residual_components(
            thickness=thickness,
            old_u=old_u,
            old_v=old_v,
            water_u=water_u,
            water_v=water_v,
            candidate_u=candidate_u,
            candidate_v=candidate_v,
            applied_drag=applied_drag,
            inertia_weight=inertia_weight,
        )
        cfg = self.config
        active = self._active_vector(thickness)
        residual = np.concatenate((residual_u.ravel(), residual_v.ravel()))[active]
        scale = np.concatenate((scale_u.ravel(), scale_v.ravel()))[active]
        return float(
            np.linalg.norm(residual)
            / (np.linalg.norm(scale) + cfg.force_epsilon)
        )

    def solve(
        self,
        *,
        thickness: Array,
        old_u: Array,
        old_v: Array,
        water_u: Array,
        water_v: Array,
        inertia_weight: float,
        max_picard_iterations: int,
        preconditioner: str,
        initial_u: Array | None = None,
        initial_v: Array | None = None,
    ) -> IceMomentumResult:
        solve_started = time.perf_counter()
        cfg, grid, ops = self.config, self.grid, self.ops
        grid.require_center(thickness)
        grid.require_u(old_u)
        grid.require_v(old_v)
        grid.require_u(water_u)
        grid.require_v(water_v)
        if inertia_weight not in (0.0, 1.0):
            raise ValueError("M0/M1 仅允许海冰惯性权重 0 或 1")
        if max_picard_iterations <= 0:
            raise ValueError("Picard 迭代上限必须为正")

        if initial_u is not None:
            grid.require_u(initial_u)
        if initial_v is not None:
            grid.require_v(initial_v)
        current_u, current_v = apply_ice_no_slip(
            grid,
            np.array(old_u if initial_u is None else initial_u, copy=True),
            np.array(old_v if initial_v is None else initial_v, copy=True),
        )
        current = ops.ice_velocity_vector(current_u, current_v)
        active = self._active_vector(thickness)
        current[~active] = 0.0
        records: list[IcePicardRecord] = []
        total_gmres = 0
        last_residual = float("inf")
        ilu_cache: dict[str, Any] = {}
        previous_gmres_iterations: int | None = None
        step_ilu_reasons: list[str] = []
        inner_y_history: list[np.ndarray] = []
        inner_f_history: list[np.ndarray] = []
        last_drag = compute_interface_drag(
            grid,
            thickness=thickness,
            h_min=cfg.h_min,
            ice_u=current_u,
            ice_v=current_v,
            water_u=water_u,
            water_v=water_v,
            rho_water=cfg.rho_water,
            drag_coefficient=cfg.drag_coefficient,
        )

        for iteration in range(1, max_picard_iterations + 1):
            current_u, current_v = ops.split_ice_velocity(current)
            assembly_started = time.perf_counter()
            matrix, rhs = self.assemble_linear_system(
                thickness=thickness,
                old_u=old_u,
                old_v=old_v,
                water_u=water_u,
                water_v=water_v,
                iterate_u=current_u,
                iterate_v=current_v,
                inertia_weight=inertia_weight,
            )
            assembly_seconds = time.perf_counter() - assembly_started
            linear_history: list[float] = []
            if cfg.linear_solver == "splu":
                # M1 的冻结系数矩阵是稀疏对称系统。直接 LU 仅替换线性
                # 子求解器；后面的真实残差门仍使用统一的 1e-8 标准。
                ilu_cache.clear()
                cache_reason = "direct_sparse_lu"
                used_preconditioner = "splu_direct"
                ilu_builds_now = 0
                ilu_reuses_now = 0
                ilu_rebuilds_now = 0
                preconditioner_started = time.perf_counter()
                try:
                    direct_factor = splu(
                        matrix.tocsc(), permc_spec="MMD_AT_PLUS_A"
                    )
                    factor_ok = True
                except RuntimeError:
                    direct_factor = None
                    factor_ok = False
                preconditioner_seconds = time.perf_counter() - preconditioner_started
                gmres_started = time.perf_counter()
                if factor_ok and direct_factor is not None:
                    linear_solution = direct_factor.solve(rhs)
                    info = 0
                    linear_history.append(0.0)
                else:
                    linear_solution = current.copy()
                    info = 1
                gmres_seconds = time.perf_counter() - gmres_started
            else:
                preconditioner_started = time.perf_counter()
                cached_op, cache_reason = self._ilu_reuse_decision(
                    matrix, preconditioner, ilu_cache, previous_gmres_iterations
                )
                if cached_op is not None:
                    preconditioner_op = cached_op
                    used_preconditioner = str(ilu_cache["ref_kind"])
                    ilu_builds_now = 0
                    ilu_reuses_now = 1
                    ilu_rebuilds_now = 0
                else:
                    had_cache = bool(ilu_cache)
                    preconditioner_op, used_preconditioner = self._preconditioner(
                        matrix, preconditioner
                    )
                    ilu_cache.clear()
                    ilu_builds_now = 1
                    ilu_reuses_now = 0
                    ilu_rebuilds_now = 1 if had_cache else 0
                    if used_preconditioner == "jacobi_fallback":
                        cache_reason = "jacobi_fallback"
                    elif not cfg.ilu_reuse_enabled:
                        cache_reason = "cache_disabled"
                    elif not had_cache:
                        cache_reason = "new_picard_round"
                    if cfg.ilu_reuse_enabled and used_preconditioner != "jacobi_fallback":
                        ilu_cache["ref_matrix"] = matrix
                        ilu_cache["ref_kind"] = used_preconditioner
                        ilu_cache["ref_op"] = preconditioner_op
                        ilu_cache["ref_gmres"] = 0
                        ilu_cache["consecutive_reuse"] = 0
                preconditioner_seconds = time.perf_counter() - preconditioner_started
                gmres_started = time.perf_counter()
                if cfg.linear_solver == "bicgstab":
                    linear_solution, info = bicgstab(
                        matrix,
                        rhs,
                        x0=current,
                        rtol=cfg.gmres_relative_tolerance,
                        atol=0.0,
                        maxiter=cfg.gmres_max_iterations,
                        M=preconditioner_op,
                        callback=lambda xk: linear_history.append(0.0),
                    )
                else:
                    linear_solution, info = gmres(
                        matrix,
                        rhs,
                        x0=current,
                        rtol=cfg.gmres_relative_tolerance,
                        atol=0.0,
                        restart=cfg.gmres_restart,
                        maxiter=cfg.gmres_max_iterations,
                        M=preconditioner_op,
                        callback=lambda value: linear_history.append(float(value)),
                        callback_type="pr_norm",
                    )
                gmres_seconds = time.perf_counter() - gmres_started
            gmres_iterations = len(linear_history)
            total_gmres += gmres_iterations
            previous_gmres_iterations = gmres_iterations
            step_ilu_reasons.append(cache_reason)
            linear_residual = float(
                np.linalg.norm(matrix @ linear_solution - rhs)
                / (np.linalg.norm(rhs) + cfg.norm_epsilon)
            )
            if info != 0 or not np.isfinite(linear_residual):
                ilu_cache.clear()
                records.append(
                    IcePicardRecord(
                        iteration=iteration,
                        nonlinear_residual=float("inf"),
                        gmres_residual=linear_residual,
                        gmres_iterations=gmres_iterations,
                        relaxation_factor=cfg.inner_picard_relaxation,
                        preconditioner=used_preconditioner,
                        matrix_assembly_seconds=assembly_seconds,
                        preconditioner_build_seconds=preconditioner_seconds,
                        gmres_seconds=gmres_seconds,
                        residual_evaluation_seconds=0.0,
                        ilu_builds=ilu_builds_now,
                        ilu_reuses=ilu_reuses_now,
                        ilu_rebuilds=ilu_rebuilds_now,
                        cache_invalidation_reason=cache_reason,
                        preconditioner_fallbacks=int(used_preconditioner == "jacobi_fallback"),
                    )
                )
                break

            if ilu_cache:
                if ilu_builds_now == 1:
                    ilu_cache["ref_gmres"] = gmres_iterations
                    ilu_cache["consecutive_reuse"] = 0
                else:
                    ilu_cache["consecutive_reuse"] = int(ilu_cache["consecutive_reuse"]) + 1

            fixed_point_delta = linear_solution - current
            anderson_accepted = False
            residual_seconds = 0.0
            if cfg.inner_anderson_enabled and np.isfinite(linear_residual):
                inner_y_history.append(current.copy())
                inner_f_history.append(fixed_point_delta.copy())
                if len(inner_y_history) >= 3:
                    anderson_started = time.perf_counter()
                    candidate = self._inner_anderson_combine(
                        inner_y_history,
                        inner_f_history,
                        cfg.inner_anderson_depth,
                        active,
                    )
                    candidate_u, candidate_v = ops.split_ice_velocity(candidate)
                    candidate_u, candidate_v = apply_ice_no_slip(
                        grid, candidate_u, candidate_v
                    )
                    candidate = ops.ice_velocity_vector(candidate_u, candidate_v)
                    candidate_drag = compute_interface_drag(
                        grid,
                        thickness=thickness,
                        h_min=cfg.h_min,
                        ice_u=candidate_u,
                        ice_v=candidate_v,
                        water_u=water_u,
                        water_v=water_v,
                        rho_water=cfg.rho_water,
                        drag_coefficient=cfg.drag_coefficient,
                    )
                    candidate_residual = self.nonlinear_residual(
                        thickness=thickness,
                        old_u=old_u,
                        old_v=old_v,
                        water_u=water_u,
                        water_v=water_v,
                        candidate_u=candidate_u,
                        candidate_v=candidate_v,
                        applied_drag=candidate_drag,
                        inertia_weight=inertia_weight,
                    )
                    residual_seconds = time.perf_counter() - anderson_started
                    if (
                        np.isfinite(candidate_residual)
                        and candidate_residual <= 0.99 * last_residual
                    ):
                        current = candidate
                        last_drag = candidate_drag
                        last_residual = candidate_residual
                        anderson_accepted = True
                    else:
                        inner_y_history.clear()
                        inner_f_history.clear()
            if not anderson_accepted:
                if cfg.inner_backtracking_enabled:
                    omegas = []
                    omega = float(
                        np.clip(
                            cfg.inner_backtracking_initial,
                            cfg.inner_backtracking_min,
                            cfg.inner_backtracking_max,
                        )
                    )
                    while omega >= cfg.inner_backtracking_min - 1.0e-15:
                        omegas.append(omega)
                        omega *= cfg.inner_backtracking_factor
                else:
                    omegas = (
                        (0.5, 1.0)
                        if cfg.inner_anderson_enabled
                        else (cfg.inner_picard_relaxation,)
                    )
                best_candidate = None
                best_drag = None
                best_cu = None
                best_cv = None
                best_residual = float("inf")
                accepted_candidate = None
                accepted_drag = None
                accepted_cu = None
                accepted_cv = None
                accepted_residual = None
                residual_started = time.perf_counter()
                for omega in omegas:
                    candidate = current + omega * fixed_point_delta
                    candidate[~active] = 0.0
                    cu, cv = ops.split_ice_velocity(candidate)
                    cu, cv = apply_ice_no_slip(grid, cu, cv)
                    candidate = ops.ice_velocity_vector(cu, cv)
                    drag = compute_interface_drag(
                        grid,
                        thickness=thickness,
                        h_min=cfg.h_min,
                        ice_u=cu,
                        ice_v=cv,
                        water_u=water_u,
                        water_v=water_v,
                        rho_water=cfg.rho_water,
                        drag_coefficient=cfg.drag_coefficient,
                    )
                    residual = self.nonlinear_residual(
                        thickness=thickness,
                        old_u=old_u,
                        old_v=old_v,
                        water_u=water_u,
                        water_v=water_v,
                        candidate_u=cu,
                        candidate_v=cv,
                        applied_drag=drag,
                        inertia_weight=inertia_weight,
                    )
                    if np.isfinite(residual) and residual < best_residual:
                        best_residual = residual
                        best_candidate = candidate
                        best_drag = drag
                        best_cu = cu
                        best_cv = cv
                    if (
                        cfg.inner_backtracking_enabled
                        and accepted_candidate is None
                        and np.isfinite(residual)
                        and residual
                        <= cfg.inner_backtracking_growth_limit * last_residual
                    ):
                        accepted_candidate = candidate
                        accepted_drag = drag
                        accepted_cu = cu
                        accepted_cv = cv
                        accepted_residual = residual
                residual_seconds = time.perf_counter() - residual_started
                if cfg.inner_backtracking_enabled and accepted_candidate is not None:
                    current = accepted_candidate
                    candidate_u = accepted_cu
                    candidate_v = accepted_cv
                    last_drag = accepted_drag
                    last_residual = accepted_residual
                else:
                    current = best_candidate
                    candidate_u = best_cu
                    candidate_v = best_cv
                    last_drag = best_drag
                    last_residual = best_residual
            records.append(
                IcePicardRecord(
                    iteration=iteration,
                    nonlinear_residual=last_residual,
                    gmres_residual=linear_residual,
                    gmres_iterations=gmres_iterations,
                    relaxation_factor=cfg.inner_picard_relaxation,
                    preconditioner=used_preconditioner,
                    matrix_assembly_seconds=assembly_seconds,
                    preconditioner_build_seconds=preconditioner_seconds,
                    gmres_seconds=gmres_seconds,
                    residual_evaluation_seconds=residual_seconds,
                    ilu_builds=ilu_builds_now,
                    ilu_reuses=ilu_reuses_now,
                    ilu_rebuilds=ilu_rebuilds_now,
                    cache_invalidation_reason=cache_reason,
                    preconditioner_fallbacks=int(used_preconditioner == "jacobi_fallback"),
                )
            )
            if (
                last_residual <= cfg.ice_picard_tolerance
                and linear_residual <= cfg.gmres_relative_tolerance
            ):
                return IceMomentumResult(
                    u=candidate_u,
                    v=candidate_v,
                    converged=True,
                    nonlinear_residual=last_residual,
                    picard_iterations=iteration,
                    gmres_iterations=total_gmres,
                    records=tuple(records),
                    applied_drag=last_drag,
                    ice_drag_force_u=last_drag.tau_u,
                    ice_drag_force_v=last_drag.tau_v,
                    ice_solve_seconds=time.perf_counter() - solve_started,
                    matrix_assembly_seconds=sum(item.matrix_assembly_seconds for item in records),
                    preconditioner_build_seconds=sum(
                        item.preconditioner_build_seconds for item in records
                    ),
                    gmres_seconds=sum(item.gmres_seconds for item in records),
                    residual_evaluation_seconds=sum(
                        item.residual_evaluation_seconds for item in records
                    ),
                    ilu_builds=sum(item.ilu_builds for item in records),
                    ilu_reuses=sum(item.ilu_reuses for item in records),
                    ilu_rebuilds=sum(item.ilu_rebuilds for item in records),
                    cache_invalidation_reasons=tuple(
                        item.cache_invalidation_reason for item in records
                    ),
                    preconditioner_fallbacks=sum(
                        item.preconditioner_fallbacks for item in records
                    ),
                )

        final_u, final_v = ops.split_ice_velocity(current)
        final_u, final_v = apply_ice_no_slip(grid, final_u, final_v)
        return IceMomentumResult(
            u=final_u,
            v=final_v,
            converged=False,
            nonlinear_residual=last_residual,
            picard_iterations=len(records),
            gmres_iterations=total_gmres,
            records=tuple(records),
            applied_drag=last_drag,
            ice_drag_force_u=last_drag.tau_u,
            ice_drag_force_v=last_drag.tau_v,
            ice_solve_seconds=time.perf_counter() - solve_started,
            matrix_assembly_seconds=sum(item.matrix_assembly_seconds for item in records),
            preconditioner_build_seconds=sum(
                item.preconditioner_build_seconds for item in records
            ),
            gmres_seconds=sum(item.gmres_seconds for item in records),
            residual_evaluation_seconds=sum(
                item.residual_evaluation_seconds for item in records
            ),
            ilu_builds=sum(item.ilu_builds for item in records),
            ilu_reuses=sum(item.ilu_reuses for item in records),
            ilu_rebuilds=sum(item.ilu_rebuilds for item in records),
            cache_invalidation_reasons=tuple(
                item.cache_invalidation_reason for item in records
            ),
            preconditioner_fallbacks=sum(
                item.preconditioner_fallbacks for item in records
            ),
        )
