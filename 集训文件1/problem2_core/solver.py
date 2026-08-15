"""规格 v2 的 M0/M1 海冰—海水物理时间步与固定重试流程。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
import time
from typing import Any

import numpy as np

from .acceleration import ProtectedAnderson, build_temporal_initial_guess
from .boundary import apply_ice_no_slip, apply_water_free_slip, ice_active_face_masks
from .config import ModelMode, Problem2Config
from .coupling import compute_interface_drag, diagnose_applied_drag_closure
from .diagnostic_modes import DiagnosticRunSpec
from .failure import (
    PhysicalStepResult,
    RetryPolicy,
    RetryStage,
    StepAttempt,
    execute_physical_step,
)
from .ice_momentum import IceMomentumSolver
from .ocean_swe import OceanIMEXSolver, OceanState
from .state import CoupledState
from .thickness_transport import advect_thickness_upwind, total_ice_volume


@dataclass(frozen=True)
class CouplingRecord:
    iteration: int
    coupling_residual: float
    ice_residual: float
    water_fixed_point_residual: float
    water_equation_residual: float
    water_residual_xi: float
    water_residual_u: float
    water_residual_v: float
    helmholtz_residual: float
    ocean_coriolis_iterations: int
    picard_iterations: int
    gmres_iterations: int
    gmres_residual: float
    outer_aitken_factor_applied: float | None
    drag_object_reused: bool
    outer_acceleration_method: str = "aitken"
    acceleration_fallback_reason: str = ""
    acceleration_history_cleared: bool = False
    acceleration_history_depth: int = 0
    ice_solve_seconds: float = 0.0
    ocean_solve_seconds: float = 0.0
    coupling_iteration_seconds: float = 0.0
    coupling_overhead_seconds: float = 0.0
    matrix_assembly_seconds: float = 0.0
    preconditioner_build_seconds: float = 0.0
    gmres_seconds: float = 0.0
    ice_residual_evaluation_seconds: float = 0.0
    ilu_builds: int = 0
    ilu_reuses: int = 0
    ilu_rebuilds: int = 0
    cache_invalidation_reason: str = ""
    preconditioner_fallbacks: int = 0


@dataclass(frozen=True)
class PhysicalStepDiagnostics:
    step_index: int
    time_start_seconds: float
    time_end_seconds: float
    chi: float
    chi_coriolis: float
    retry_stage: str
    coupling_iterations: int
    total_picard_iterations: int
    total_gmres_iterations: int
    ice_residual: float
    coupling_residual: float
    water_residual: float
    water_residual_xi: float
    water_residual_u: float
    water_residual_v: float
    helmholtz_residual: float
    ocean_coriolis_iterations: int
    gmres_residual: float
    drag_closure_error: float
    drag_face_l1_closure_error: float
    drag_object_reused: bool
    maximum_inactive_drag: float
    mean_thickness: float
    ice_volume_m3: float
    ice_volume_relative_error: float
    minimum_thickness: float
    maximum_thickness: float
    maximum_ice_speed: float
    maximum_ocean_speed: float
    mean_sea_surface: float
    thickness_cfl_macro: float
    thickness_substeps: int
    coupling_records: tuple[CouplingRecord, ...] = field(default_factory=tuple)
    residual_records: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    retry_attempts: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    timing_seconds: dict[str, float] = field(default_factory=dict)
    ilu_builds: int = 0
    ilu_reuses: int = 0
    ilu_rebuilds: int = 0
    ilu_invalidation_reasons: tuple[str, ...] = field(default_factory=tuple)
    preconditioner_fallbacks: int = 0
    initial_guess_strategy: str = "previous_accepted_state"
    initial_guess_fallback_reason: str = ""
    outer_acceleration: str = "aitken"
    acceleration_fallbacks: int = 0
    acceleration_history_clears: int = 0
    acceleration_clear_reasons: tuple[str, ...] = field(default_factory=tuple)
    cache_invalidation_reason: str = "no_cache_phase1"
    failure_reason: str = ""
    active_center_count_before: int = 0
    active_center_count_after: int = 0
    active_center_mask_changes: int = 0
    active_u_mask_changes: int = 0
    active_v_mask_changes: int = 0
    force_budget_summary: dict[str, Any] = field(default_factory=dict)
    force_budget_face_arrays: dict[str, np.ndarray] = field(default_factory=dict)
    force_budget_center_arrays: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class PhysicalStepAdvance:
    state: CoupledState
    diagnostics: PhysicalStepDiagnostics


def _relative_change(new: np.ndarray, old: np.ndarray, epsilon: float) -> float:
    return float(np.linalg.norm(new - old) / (np.linalg.norm(new) + epsilon))


def _pair_relative_change(
    new_u: np.ndarray,
    new_v: np.ndarray,
    old_u: np.ndarray,
    old_v: np.ndarray,
    epsilon: float,
    *,
    mask_u: np.ndarray | None = None,
    mask_v: np.ndarray | None = None,
) -> float:
    if mask_u is None:
        mask_u = np.ones(new_u.shape, dtype=bool)
    if mask_v is None:
        mask_v = np.ones(new_v.shape, dtype=bool)
    difference = np.concatenate(((new_u - old_u)[mask_u], (new_v - old_v)[mask_v]))
    reference = np.concatenate((new_u[mask_u], new_v[mask_v]))
    return float(np.linalg.norm(difference) / (np.linalg.norm(reference) + epsilon))


class Problem2Solver:
    """完整时间步求解器；海冰惯性与科里奥利使用独立冻结权重。"""

    def __init__(
        self,
        config: Problem2Config,
        state: CoupledState | None = None,
        *,
        previous_accepted_state: CoupledState | None = None,
        failure_directory: str | Path = "output/problem2/round2/failures",
        diagnostic_spec: DiagnosticRunSpec | None = None,
    ):
        if config.mode not in (ModelMode.M0_FULL, ModelMode.M1_QUASI_STATIC):
            raise ValueError("数值规格 v2 正式求解器只接受 M0 或 M1；旧 M2 已退出正式路线")
        self.config = config
        self.diagnostic_spec = diagnostic_spec
        self.ice_inertia_weight = (
            config.ice_inertia_weight
            if diagnostic_spec is None
            else diagnostic_spec.inertia_weight
        )
        self.ice_coriolis_weight = (
            config.ice_coriolis_weight
            if diagnostic_spec is None
            else diagnostic_spec.ice_coriolis_weight
        )
        self.state = state.copy() if state is not None else CoupledState.initial(config)
        self.state.validate(config)
        if previous_accepted_state is not None:
            previous_accepted_state.validate(config)
            if not np.isclose(
                self.state.time_seconds - previous_accepted_state.time_seconds,
                config.dt,
            ):
                raise ValueError("重启预测历史必须是当前态之前一个物理时间步")
        self.ice_solver = IceMomentumSolver(
            config, ice_coriolis_weight=self.ice_coriolis_weight
        )
        self.ocean_solver = OceanIMEXSolver(config)
        self.failure_directory = Path(failure_directory)
        self.initial_volume = total_ice_volume(config.grid, self.state.thickness)
        self._attempt_diagnostics: dict[str, PhysicalStepDiagnostics] = {}
        self._previous_accepted_state = (
            None
            if previous_accepted_state is None
            else previous_accepted_state.copy()
        )

    def _attempt(
        self,
        old: CoupledState,
        *,
        step_index: int,
        stage: RetryStage,
        target_residual: float,
    ) -> tuple[CoupledState, PhysicalStepDiagnostics, bool]:
        attempt_started = time.perf_counter()
        cfg, grid = self.config, self.config.grid
        chi = self.ice_inertia_weight
        chi_coriolis = self.ice_coriolis_weight
        _, active_u, active_v = ice_active_face_masks(
            grid, old.thickness, h_min=cfg.h_min
        )
        active_u = np.array(active_u, copy=True)
        active_v = np.array(active_v, copy=True)
        active_u[:, (0, -1)] = False
        active_v[(0, -1), :] = False
        active_center_before = old.thickness >= cfg.h_min
        initial_guess = build_temporal_initial_guess(
            cfg,
            current=old,
            previous=self._previous_accepted_state,
        )
        guess_ice_u = initial_guess.ice_u
        guess_ice_v = initial_guess.ice_v
        guess_ocean_u = initial_guess.ocean_u
        guess_ocean_v = initial_guess.ocean_v
        guess_xi = initial_guess.sea_surface
        previous_drag = compute_interface_drag(
            grid,
            thickness=old.thickness,
            h_min=cfg.h_min,
            ice_u=guess_ice_u,
            ice_v=guess_ice_v,
            water_u=guess_ocean_u,
            water_v=guess_ocean_v,
            rho_water=cfg.rho_water,
            drag_coefficient=cfg.drag_coefficient,
        )
        current_vector = np.concatenate(
            (
                guess_ice_u.ravel(),
                guess_ice_v.ravel(),
                guess_ocean_u.ravel(),
                guess_ocean_v.ravel(),
                guess_xi.ravel(),
            )
        )
        previous_delta: np.ndarray | None = None
        omega = float(
            np.clip(stage.outer_aitken_initial, cfg.aitken_min, cfg.aitken_max)
        )
        anderson = (
            ProtectedAnderson(
                depth=cfg.anderson_depth,
                damping=cfg.anderson_damping,
                residual_rise_factor=cfg.anderson_residual_rise_factor,
                condition_limit=cfg.anderson_condition_limit,
                step_ratio_limit=cfg.anderson_step_ratio_limit,
            )
            if cfg.outer_anderson_enabled
            else None
        )
        acceleration_fallbacks = 0
        coupling_limit = stage.max_coupling_iterations
        coupling_records: list[CouplingRecord] = []
        residual_records: list[dict[str, Any]] = []
        total_picard = 0
        total_gmres = 0
        total_ice_solve_seconds = 0.0
        total_matrix_assembly_seconds = 0.0
        total_preconditioner_build_seconds = 0.0
        total_gmres_seconds = 0.0
        total_ice_residual_evaluation_seconds = 0.0
        total_ocean_solve_seconds = 0.0
        total_ilu_builds = 0
        total_ilu_reuses = 0
        total_ilu_rebuilds = 0
        total_preconditioner_fallbacks = 0
        step_ilu_reasons: list[str] = []
        last_ice_residual = float("inf")
        last_coupling_residual = float("inf")
        last_water_residual = float("inf")
        last_water_residual_xi = float("inf")
        last_water_residual_u = float("inf")
        last_water_residual_v = float("inf")
        last_helmholtz_residual = float("inf")
        last_ocean_coriolis_iterations = 0
        last_gmres_residual = float("inf")
        converged = False
        failure_reason = "coupling_iteration_limit"
        accepted_ice = None
        accepted_ocean = None

        old_ocean = OceanState(
            xi=old.sea_surface, u=old.ocean_u, v=old.ocean_v
        )
        coupling_started = time.perf_counter()
        for coupling_iteration in range(1, coupling_limit + 1):
            coupling_iteration_started = time.perf_counter()
            ice = self.ice_solver.solve(
                thickness=old.thickness,
                old_u=old.ice_u,
                old_v=old.ice_v,
                water_u=guess_ocean_u,
                water_v=guess_ocean_v,
                inertia_weight=chi,
                max_picard_iterations=stage.max_picard_iterations,
                preconditioner=stage.preconditioner,
                initial_u=guess_ice_u,
                initial_v=guess_ice_v,
            )
            total_picard += ice.picard_iterations
            total_gmres += ice.gmres_iterations
            total_ice_solve_seconds += ice.ice_solve_seconds
            total_matrix_assembly_seconds += ice.matrix_assembly_seconds
            total_preconditioner_build_seconds += ice.preconditioner_build_seconds
            total_gmres_seconds += ice.gmres_seconds
            total_ice_residual_evaluation_seconds += ice.residual_evaluation_seconds
            total_ilu_builds += ice.ilu_builds
            total_ilu_reuses += ice.ilu_reuses
            total_ilu_rebuilds += ice.ilu_rebuilds
            step_ilu_reasons.extend(ice.cache_invalidation_reasons)
            total_preconditioner_fallbacks += ice.preconditioner_fallbacks
            for record in ice.records:
                residual_records.append(
                    {
                        "step": step_index,
                        "stage": stage.name,
                        "coupling_iteration": coupling_iteration,
                        "picard_iteration": record.iteration,
                        "ice_residual": record.nonlinear_residual,
                        "gmres_residual": record.gmres_residual,
                        "gmres_iterations": record.gmres_iterations,
                        "inner_relaxation": record.relaxation_factor,
                        "preconditioner": record.preconditioner,
                        "matrix_assembly_seconds": record.matrix_assembly_seconds,
                        "preconditioner_build_seconds": record.preconditioner_build_seconds,
                        "gmres_seconds": record.gmres_seconds,
                        "ice_residual_evaluation_seconds": record.residual_evaluation_seconds,
                        "ilu_builds": record.ilu_builds,
                        "ilu_reuses": record.ilu_reuses,
                        "ilu_rebuilds": record.ilu_rebuilds,
                        "cache_invalidation_reason": record.cache_invalidation_reason,
                        "preconditioner_fallbacks": record.preconditioner_fallbacks,
                    }
                )
            if not ice.converged:
                failure_reason = "ice_picard_or_gmres_not_converged"
                last_ice_residual = ice.nonlinear_residual
                last_gmres_residual = max(
                    (record.gmres_residual for record in ice.records),
                    default=float("inf"),
                )
                break

            ocean_started = time.perf_counter()
            ocean = self.ocean_solver.advance(old_ocean, ice.applied_drag)
            ocean_seconds = time.perf_counter() - ocean_started
            total_ocean_solve_seconds += ocean_seconds
            if not ocean.converged:
                failure_reason = "ocean_solver_not_converged"
                last_ice_residual = ice.nonlinear_residual
                last_water_residual = ocean.equation_residual
                last_water_residual_xi = ocean.residual_xi
                last_water_residual_u = ocean.residual_u
                last_water_residual_v = ocean.residual_v
                last_helmholtz_residual = ocean.helmholtz_residual
                last_ocean_coriolis_iterations = ocean.coriolis_iterations
                break
            raw_vector = np.concatenate(
                (
                    ice.u.ravel(),
                    ice.v.ravel(),
                    ocean.state.u.ravel(),
                    ocean.state.v.ravel(),
                    ocean.state.xi.ravel(),
                )
            )
            delta = raw_vector - current_vector
            ice_change = _pair_relative_change(
                ice.u,
                ice.v,
                guess_ice_u,
                guess_ice_v,
                cfg.norm_epsilon,
                mask_u=active_u,
                mask_v=active_v,
            )
            water_change = _pair_relative_change(
                ocean.state.u,
                ocean.state.v,
                guess_ocean_u,
                guess_ocean_v,
                cfg.norm_epsilon,
            )
            surface_change = _relative_change(
                ocean.state.xi, guess_xi, cfg.norm_epsilon
            )
            drag_change = _pair_relative_change(
                ice.applied_drag.tau_u,
                ice.applied_drag.tau_v,
                previous_drag.tau_u,
                previous_drag.tau_v,
                cfg.norm_epsilon,
                mask_u=active_u,
                mask_v=active_v,
            )
            water_fixed_point = max(water_change, surface_change)
            coupling_residual = max(
                ice_change, water_change, surface_change, drag_change
            )
            gmres_residual = max(
                (record.gmres_residual for record in ice.records), default=float("inf")
            )
            drag_object_reused = ocean.applied_drag is ice.applied_drag
            meets_convergence = bool(
                coupling_residual <= target_residual
                and ice.nonlinear_residual <= cfg.ice_picard_tolerance
                and gmres_residual <= cfg.gmres_relative_tolerance
                and ocean.converged
                and drag_object_reused
            )
            applied_outer_omega: float | None = None
            acceleration_method = "converged_no_update"
            acceleration_fallback_reason = ""
            acceleration_history_cleared = False
            acceleration_history_depth = 0
            updated_vector: np.ndarray | None = None
            if not meets_convergence:
                next_omega = omega
                if previous_delta is not None:
                    difference = delta - previous_delta
                    denominator = float(np.dot(difference, difference))
                    if denominator > np.finfo(float).tiny:
                        next_omega = float(
                            np.clip(
                                -omega
                                * float(np.dot(previous_delta, difference))
                                / denominator,
                                cfg.aitken_min,
                                cfg.aitken_max,
                            )
                        )
                applied_outer_omega = next_omega
                aitken_vector = current_vector + applied_outer_omega * delta
                if anderson is None:
                    acceleration_method = "aitken"
                    updated_vector = aitken_vector
                else:
                    mask_signature = np.concatenate(
                        (
                            active_u.ravel(),
                            active_v.ravel(),
                            active_center_before.ravel(),
                        )
                    )
                    decision = anderson.propose(
                        current=current_vector,
                        raw=raw_vector,
                        residual=coupling_residual,
                        fallback=aitken_vector,
                        mask_signature=mask_signature,
                    )
                    updated_vector = decision.vector
                    acceleration_method = decision.method
                    acceleration_fallback_reason = decision.fallback_reason
                    acceleration_history_cleared = decision.history_cleared
                    acceleration_history_depth = decision.history_depth
                    if decision.method == "aitken_fallback":
                        acceleration_fallbacks += 1
            coupling_records.append(
                CouplingRecord(
                    iteration=coupling_iteration,
                    coupling_residual=coupling_residual,
                    ice_residual=ice.nonlinear_residual,
                    water_fixed_point_residual=water_fixed_point,
                    water_equation_residual=ocean.equation_residual,
                    water_residual_xi=ocean.residual_xi,
                    water_residual_u=ocean.residual_u,
                    water_residual_v=ocean.residual_v,
                    helmholtz_residual=ocean.helmholtz_residual,
                    ocean_coriolis_iterations=ocean.coriolis_iterations,
                    picard_iterations=ice.picard_iterations,
                    gmres_iterations=ice.gmres_iterations,
                    gmres_residual=gmres_residual,
                    outer_aitken_factor_applied=applied_outer_omega,
                    drag_object_reused=drag_object_reused,
                    outer_acceleration_method=acceleration_method,
                    acceleration_fallback_reason=acceleration_fallback_reason,
                    acceleration_history_cleared=acceleration_history_cleared,
                    acceleration_history_depth=acceleration_history_depth,
                )
            )
            last_ice_residual = ice.nonlinear_residual
            last_coupling_residual = coupling_residual
            last_water_residual = ocean.equation_residual
            last_water_residual_xi = ocean.residual_xi
            last_water_residual_u = ocean.residual_u
            last_water_residual_v = ocean.residual_v
            last_helmholtz_residual = ocean.helmholtz_residual
            last_ocean_coriolis_iterations = ocean.coriolis_iterations
            last_gmres_residual = gmres_residual

            if meets_convergence:
                accepted_ice = ice
                accepted_ocean = ocean
                converged = True
                failure_reason = ""
                coupling_iteration_seconds = time.perf_counter() - coupling_iteration_started
                coupling_records[-1] = replace(
                    coupling_records[-1],
                    ice_solve_seconds=ice.ice_solve_seconds,
                    ocean_solve_seconds=ocean_seconds,
                    coupling_iteration_seconds=coupling_iteration_seconds,
                    coupling_overhead_seconds=max(
                        coupling_iteration_seconds - ice.ice_solve_seconds - ocean_seconds,
                        0.0,
                    ),
                    matrix_assembly_seconds=ice.matrix_assembly_seconds,
                    preconditioner_build_seconds=ice.preconditioner_build_seconds,
                    gmres_seconds=ice.gmres_seconds,
                    ice_residual_evaluation_seconds=ice.residual_evaluation_seconds,
                    ilu_builds=ice.ilu_builds,
                    ilu_reuses=ice.ilu_reuses,
                    ilu_rebuilds=ice.ilu_rebuilds,
                    cache_invalidation_reason=(
                        ice.cache_invalidation_reasons[-1]
                        if ice.cache_invalidation_reasons
                        else ""
                    ),
                    preconditioner_fallbacks=ice.preconditioner_fallbacks,
                )
                break

            if applied_outer_omega is None:
                raise RuntimeError("未收敛外层迭代缺少实际 Aitken 松弛因子")
            if updated_vector is None:
                raise RuntimeError("未收敛外层迭代缺少受保护更新向量")
            omega = applied_outer_omega
            previous_delta = delta
            n_u = grid.ny * (grid.nx + 1)
            n_v = (grid.ny + 1) * grid.nx
            offset = 0
            guess_ice_u = updated_vector[offset : offset + n_u].reshape(grid.u_shape)
            offset += n_u
            guess_ice_v = updated_vector[offset : offset + n_v].reshape(grid.v_shape)
            offset += n_v
            guess_ocean_u = updated_vector[offset : offset + n_u].reshape(grid.u_shape)
            offset += n_u
            guess_ocean_v = updated_vector[offset : offset + n_v].reshape(grid.v_shape)
            offset += n_v
            guess_xi = updated_vector[offset:].reshape(grid.center_shape)
            guess_ice_u, guess_ice_v = apply_ice_no_slip(
                grid, guess_ice_u, guess_ice_v
            )
            guess_ice_u[~active_u] = 0.0
            guess_ice_v[~active_v] = 0.0
            guess_ocean_u, guess_ocean_v = apply_water_free_slip(
                grid, guess_ocean_u, guess_ocean_v
            )
            current_vector = np.concatenate(
                (
                    guess_ice_u.ravel(),
                    guess_ice_v.ravel(),
                    guess_ocean_u.ravel(),
                    guess_ocean_v.ravel(),
                    guess_xi.ravel(),
                )
            )
            previous_drag = ice.applied_drag
            coupling_iteration_seconds = time.perf_counter() - coupling_iteration_started
            coupling_records[-1] = replace(
                coupling_records[-1],
                ice_solve_seconds=ice.ice_solve_seconds,
                ocean_solve_seconds=ocean_seconds,
                coupling_iteration_seconds=coupling_iteration_seconds,
                coupling_overhead_seconds=max(
                    coupling_iteration_seconds - ice.ice_solve_seconds - ocean_seconds,
                    0.0,
                ),
                matrix_assembly_seconds=ice.matrix_assembly_seconds,
                preconditioner_build_seconds=ice.preconditioner_build_seconds,
                gmres_seconds=ice.gmres_seconds,
                ice_residual_evaluation_seconds=ice.residual_evaluation_seconds,
                ilu_builds=ice.ilu_builds,
                ilu_reuses=ice.ilu_reuses,
                ilu_rebuilds=ice.ilu_rebuilds,
                cache_invalidation_reason=(
                    ice.cache_invalidation_reasons[-1]
                    if ice.cache_invalidation_reasons
                    else ""
                ),
                preconditioner_fallbacks=ice.preconditioner_fallbacks,
            )

        if not converged and anderson is not None:
            anderson.clear("failed_attempt_rollback")
        coupling_seconds = time.perf_counter() - coupling_started

        if converged:
            if accepted_ice is None or accepted_ocean is None:
                raise RuntimeError("收敛标志与同步冰水解不一致")
            final_ice_u = accepted_ice.u
            final_ice_v = accepted_ice.v
            final_ocean_u = accepted_ocean.state.u
            final_ocean_v = accepted_ocean.state.v
            final_xi = accepted_ocean.state.xi
            thickness_started = time.perf_counter()
            new_thickness, thickness_diagnostics = advect_thickness_upwind(
                grid,
                old.thickness,
                final_ice_u,
                final_ice_v,
                dt=cfg.dt,
                cfl_limit=cfg.thickness_cfl_limit,
            )
            thickness_seconds = time.perf_counter() - thickness_started
            new_state = CoupledState(
                thickness=new_thickness,
                ice_u=final_ice_u,
                ice_v=final_ice_v,
                ocean_u=final_ocean_u,
                ocean_v=final_ocean_v,
                sea_surface=final_xi,
                time_seconds=old.time_seconds + cfg.dt,
            )
            new_state.validate(cfg)
            drag_balance = diagnose_applied_drag_closure(
                grid,
                ice_force_u=accepted_ice.ice_drag_force_u,
                ice_force_v=accepted_ice.ice_drag_force_v,
                water_force_u=accepted_ocean.water_drag_force_u,
                water_force_v=accepted_ocean.water_drag_force_v,
                dt=cfg.dt,
            )
            drag_object_reused = accepted_ocean.applied_drag is accepted_ice.applied_drag
            inactive_u = ~accepted_ice.applied_drag.active_u
            inactive_v = ~accepted_ice.applied_drag.active_v
            maximum_inactive_drag = max(
                float(np.max(np.abs(accepted_ice.ice_drag_force_u[inactive_u]))),
                float(np.max(np.abs(accepted_ice.ice_drag_force_v[inactive_v]))),
            )
            force_budget_started = time.perf_counter()
            force_budget = self.ice_solver.force_budget(
                thickness=old.thickness,
                old_u=old.ice_u,
                old_v=old.ice_v,
                water_u=final_ocean_u,
                water_v=final_ocean_v,
                candidate_u=final_ice_u,
                candidate_v=final_ice_v,
                applied_drag=accepted_ice.applied_drag,
                inertia_weight=chi,
            )
            force_budget_seconds = time.perf_counter() - force_budget_started
        else:
            new_state = old.copy()
            final_ice_u = guess_ice_u
            final_ice_v = guess_ice_v
            final_ocean_u = guess_ocean_u
            final_ocean_v = guess_ocean_v
            thickness_diagnostics = type(
                "FailedThicknessDiagnostics",
                (),
                {"cfl_macro": float("nan"), "substeps": 0},
            )()
            drag_balance = None
            drag_object_reused = False
            maximum_inactive_drag = float("inf")
            thickness_seconds = 0.0
            force_budget_seconds = 0.0
            force_budget = None
        ice_uc, ice_vc = grid.faces_to_center(final_ice_u, final_ice_v)
        water_uc, water_vc = grid.faces_to_center(final_ocean_u, final_ocean_v)
        volume = total_ice_volume(grid, new_state.thickness)
        active_center_after, active_u_after, active_v_after = ice_active_face_masks(
            grid, new_state.thickness, h_min=cfg.h_min
        )
        active_u_after = np.array(active_u_after, copy=True)
        active_v_after = np.array(active_v_after, copy=True)
        active_u_after[:, (0, -1)] = False
        active_v_after[(0, -1), :] = False
        postprocess_started = time.perf_counter()
        timing_seconds = {
            "attempt_total": 0.0,
            "outer_coupling": coupling_seconds,
            "coupling_overhead": max(
                coupling_seconds - total_ice_solve_seconds - total_ocean_solve_seconds,
                0.0,
            ),
            "ice_solve": total_ice_solve_seconds,
            "matrix_assembly": total_matrix_assembly_seconds,
            "preconditioner_build": total_preconditioner_build_seconds,
            "gmres": total_gmres_seconds,
            "ice_residual_evaluation": total_ice_residual_evaluation_seconds,
            "ocean_solve": total_ocean_solve_seconds,
            "thickness_transport": thickness_seconds,
            "force_budget": force_budget_seconds,
            "postprocess": 0.0,
        }
        diagnostics = PhysicalStepDiagnostics(
            step_index=step_index,
            time_start_seconds=old.time_seconds,
            time_end_seconds=new_state.time_seconds,
            chi=chi,
            chi_coriolis=chi_coriolis,
            retry_stage=stage.name,
            coupling_iterations=len(coupling_records),
            total_picard_iterations=total_picard,
            total_gmres_iterations=total_gmres,
            ice_residual=last_ice_residual,
            coupling_residual=last_coupling_residual,
            water_residual=last_water_residual,
            water_residual_xi=last_water_residual_xi,
            water_residual_u=last_water_residual_u,
            water_residual_v=last_water_residual_v,
            helmholtz_residual=last_helmholtz_residual,
            ocean_coriolis_iterations=last_ocean_coriolis_iterations,
            gmres_residual=last_gmres_residual,
            drag_closure_error=(
                float("inf") if drag_balance is None else drag_balance.relative_closure_error
            ),
            drag_face_l1_closure_error=(
                float("inf")
                if drag_balance is None
                else drag_balance.face_l1_relative_closure_error
            ),
            drag_object_reused=drag_object_reused,
            maximum_inactive_drag=maximum_inactive_drag,
            mean_thickness=float(np.mean(new_state.thickness)),
            ice_volume_m3=volume,
            ice_volume_relative_error=abs(volume - self.initial_volume) / self.initial_volume,
            minimum_thickness=float(np.min(new_state.thickness)),
            maximum_thickness=float(np.max(new_state.thickness)),
            maximum_ice_speed=float(np.max(np.hypot(ice_uc, ice_vc))),
            maximum_ocean_speed=float(np.max(np.hypot(water_uc, water_vc))),
            mean_sea_surface=float(np.mean(new_state.sea_surface)),
            thickness_cfl_macro=float(thickness_diagnostics.cfl_macro),
            thickness_substeps=int(thickness_diagnostics.substeps),
            coupling_records=tuple(coupling_records),
            residual_records=tuple(residual_records),
            timing_seconds=timing_seconds,
            ilu_builds=total_ilu_builds,
            ilu_reuses=total_ilu_reuses,
            ilu_rebuilds=total_ilu_rebuilds,
            ilu_invalidation_reasons=tuple(step_ilu_reasons),
            preconditioner_fallbacks=total_preconditioner_fallbacks,
            initial_guess_strategy=initial_guess.strategy,
            initial_guess_fallback_reason=initial_guess.fallback_reason,
            outer_acceleration=(
                "protected_anderson" if anderson is not None else "aitken"
            ),
            acceleration_fallbacks=acceleration_fallbacks,
            acceleration_history_clears=(
                0 if anderson is None else anderson.history_clears
            ),
            acceleration_clear_reasons=(
                () if anderson is None else tuple(anderson.clear_reasons)
            ),
            failure_reason=failure_reason,
            active_center_count_before=int(np.count_nonzero(active_center_before)),
            active_center_count_after=int(np.count_nonzero(active_center_after)),
            active_center_mask_changes=int(
                np.count_nonzero(active_center_before != active_center_after)
            ),
            active_u_mask_changes=int(np.count_nonzero(active_u != active_u_after)),
            active_v_mask_changes=int(np.count_nonzero(active_v != active_v_after)),
            force_budget_summary=(
                {} if force_budget is None else dict(force_budget.summary)
            ),
            force_budget_face_arrays=(
                {} if force_budget is None else dict(force_budget.face_arrays)
            ),
            force_budget_center_arrays=(
                {} if force_budget is None else dict(force_budget.center_arrays)
            ),
        )
        postprocess_seconds = time.perf_counter() - postprocess_started
        timing_seconds["postprocess"] = postprocess_seconds
        timing_seconds["attempt_total"] = time.perf_counter() - attempt_started
        finite_and_bounded = bool(
            np.isfinite(diagnostics.maximum_ice_speed)
            and np.isfinite(diagnostics.maximum_ocean_speed)
            and diagnostics.maximum_ice_speed <= cfg.maximum_speed
            and diagnostics.maximum_ocean_speed <= cfg.maximum_speed
        )
        accepted = bool(
            converged
            and finite_and_bounded
            and diagnostics.minimum_thickness >= 0.0
            and diagnostics.ice_volume_relative_error <= 1.0e-6
            and diagnostics.drag_closure_error <= 1.0e-12
            and diagnostics.drag_object_reused
        )
        return new_state, diagnostics, accepted

    def advance_one_physical_step(self, step_index: int) -> PhysicalStepAdvance:
        cfg = self.config
        policy = RetryPolicy.default(
            target_residual=cfg.coupling_tolerance,
            picard_iterations=(
                cfg.picard_iterations_standard,
                cfg.picard_iterations_reset_aitken,
                cfg.picard_iterations_robust,
            ),
            coupling_iterations=(
                cfg.coupling_iterations_standard,
                cfg.coupling_iterations_reset_aitken,
                cfg.coupling_iterations_robust,
            ),
            outer_aitken_initials=(
                cfg.outer_aitken_standard,
                cfg.outer_aitken_reset,
                cfg.outer_aitken_robust,
            ),
            preconditioners=(
                cfg.preconditioner_standard,
                cfg.preconditioner_reset_aitken,
                cfg.preconditioner_robust,
            ),
        )
        self._attempt_diagnostics = {}

        def callback(values, *, stage, target_residual):
            old = CoupledState.from_mapping(values)
            state, diagnostics, accepted = self._attempt(
                old,
                step_index=step_index,
                stage=stage,
                target_residual=target_residual,
            )
            self._attempt_diagnostics[stage.name] = diagnostics
            attempt_diagnostics = {
                "coupling_iterations": diagnostics.coupling_iterations,
                "picard_iterations": diagnostics.total_picard_iterations,
                "gmres_iterations": diagnostics.total_gmres_iterations,
                "ice_residual": diagnostics.ice_residual,
                "gmres_residual": diagnostics.gmres_residual,
                "coupling_residual": diagnostics.coupling_residual,
                "water_residual": diagnostics.water_residual,
                "helmholtz_residual": diagnostics.helmholtz_residual,
                "timing_seconds": dict(diagnostics.timing_seconds),
                "ilu_builds": diagnostics.ilu_builds,
                "ilu_reuses": diagnostics.ilu_reuses,
                "ilu_rebuilds": diagnostics.ilu_rebuilds,
                "preconditioner_fallbacks": diagnostics.preconditioner_fallbacks,
                "initial_guess_strategy": diagnostics.initial_guess_strategy,
                "initial_guess_fallback_reason": diagnostics.initial_guess_fallback_reason,
                "outer_acceleration": diagnostics.outer_acceleration,
                "acceleration_fallbacks": diagnostics.acceleration_fallbacks,
                "acceleration_history_clears": diagnostics.acceleration_history_clears,
                "acceleration_clear_reasons": diagnostics.acceleration_clear_reasons,
                "cache_invalidation_reason": diagnostics.cache_invalidation_reason,
                "failure_reason": diagnostics.failure_reason,
                "active_center_count_before": diagnostics.active_center_count_before,
                "active_center_count_after": diagnostics.active_center_count_after,
                "active_center_mask_changes": diagnostics.active_center_mask_changes,
                "active_u_mask_changes": diagnostics.active_u_mask_changes,
                "active_v_mask_changes": diagnostics.active_v_mask_changes,
            }
            return StepAttempt(
                converged=accepted,
                state=state.as_mapping(),
                residual_history=[
                    record.coupling_residual for record in diagnostics.coupling_records
                ],
                iterations=diagnostics.coupling_iterations,
                message=("converged" if accepted else "frozen residual or hard check failed"),
                diagnostics=attempt_diagnostics,
            )

        result: PhysicalStepResult = execute_physical_step(
            self.state.as_mapping(),
            callback,
            policy=policy,
            step_index=step_index,
            failure_directory=self.failure_directory,
            config=self.config.as_dict(),
        )
        accepted_stage = result.attempts[-1].stage
        diagnostics = self._attempt_diagnostics[accepted_stage]
        diagnostics = replace(
            diagnostics,
            retry_attempts=tuple(asdict(item) for item in result.attempts),
        )
        self._previous_accepted_state = self.state.copy()
        self.state = CoupledState.from_mapping(result.state)
        self.state.validate(self.config)
        return PhysicalStepAdvance(state=self.state.copy(), diagnostics=diagnostics)
