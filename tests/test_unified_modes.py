import numpy as np
import pytest

from problem2_core.boundary import ice_active_face_masks
from problem2_core.config import ModelMode, Problem2Config
from problem2_core.coupling import compute_interface_drag
from problem2_core.ice_momentum import IceMomentumSolver
from problem2_core.solver import Problem2Solver
from problem2_core.state import CoupledState


def test_v2_modes_switch_both_ice_terms_but_keep_ocean_coriolis():
    common = {"nx": 20, "ny": 10}
    m0 = Problem2Config(**common, mode=ModelMode.M0_FULL)
    m1 = Problem2Config(**common, mode=ModelMode.M1_QUASI_STATIC)

    assert m0.ice_inertia_weight == 1.0
    assert m0.ice_coriolis_weight == 1.0
    assert m1.ice_inertia_weight == 0.0
    assert m1.ice_coriolis_weight == 0.0
    assert m0.ocean_coriolis_enabled
    assert m1.ocean_coriolis_enabled
    assert m0.coriolis == m1.coriolis == 1.4e-4

    assert m0.as_dict()["ice_inertia_weight"] == 1.0
    assert m0.as_dict()["ice_coriolis_weight"] == 1.0
    assert m1.as_dict()["ice_inertia_weight"] == 0.0
    assert m1.as_dict()["ice_coriolis_weight"] == 0.0


def test_m1_deleted_inertia_does_not_evaluate_upwind_operators(monkeypatch):
    """M1 已删除惯性项，装配与物理残差都不应再构造平流算子。"""

    config = Problem2Config(
        nx=6,
        ny=4,
        duration_hours=0.5,
        mode=ModelMode.M1_QUASI_STATIC,
    )
    state = CoupledState.initial(config)
    solver = IceMomentumSolver(config)

    def deleted_term_called(*_args, **_kwargs):
        raise AssertionError("M1 不应计算已删除惯性项的迎风算子")

    monkeypatch.setattr(type(solver.ops), "implicit_upwind_u", deleted_term_called)
    monkeypatch.setattr(type(solver.ops), "implicit_upwind_v", deleted_term_called)

    matrix, rhs = solver.assemble_linear_system(
        thickness=state.thickness,
        old_u=state.ice_u,
        old_v=state.ice_v,
        water_u=state.ocean_u,
        water_v=state.ocean_v,
        iterate_u=state.ice_u,
        iterate_v=state.ice_v,
        inertia_weight=0.0,
    )
    drag = compute_interface_drag(
        config.grid,
        thickness=state.thickness,
        h_min=config.h_min,
        ice_u=state.ice_u,
        ice_v=state.ice_v,
        water_u=state.ocean_u,
        water_v=state.ocean_v,
        rho_water=config.rho_water,
        drag_coefficient=config.drag_coefficient,
    )
    residual = solver.nonlinear_residual(
        thickness=state.thickness,
        old_u=state.ice_u,
        old_v=state.ice_v,
        water_u=state.ocean_u,
        water_v=state.ocean_v,
        candidate_u=state.ice_u,
        candidate_v=state.ice_v,
        applied_drag=drag,
        inertia_weight=0.0,
    )

    assert matrix.shape[0] == rhs.size
    assert np.isfinite(residual)


def test_v2_formal_solver_rejects_the_retired_physical_time_m2_mode():
    config = Problem2Config(
        nx=4,
        ny=2,
        duration_hours=0.5,
        mode=ModelMode.M2_STARTUP_WINDOW,
    )

    with pytest.raises(ValueError, match="旧 M2 已退出正式路线"):
        Problem2Solver(config)


def test_picard_pressure_rhs_is_divergence_of_minus_half_h_times_pressure():
    config = Problem2Config(
        nx=6,
        ny=4,
        duration_hours=0.5,
        mode=ModelMode.M1_QUASI_STATIC,
        wind_stress_x=0.0,
        wind_stress_y=0.0,
    )
    grid = config.grid
    solver = IceMomentumSolver(config)
    x, y = grid.center_coordinates()
    thickness = 0.35 + 0.25 * x / config.length_x + 0.15 * y / config.length_y
    zeros_u = np.zeros(grid.u_shape)
    zeros_v = np.zeros(grid.v_shape)

    _, rhs = solver.assemble_linear_system(
        thickness=thickness,
        old_u=zeros_u,
        old_v=zeros_v,
        water_u=zeros_u,
        water_v=zeros_v,
        iterate_u=zeros_u,
        iterate_v=zeros_v,
        inertia_weight=0.0,
    )
    concentration = np.clip(thickness / config.h_i0, 0.0, 1.0)
    pressure = config.p_star * thickness * np.exp(
        -config.concentration_decay * (1.0 - concentration)
    )
    pressure_flux = (-0.5 * thickness * pressure).ravel()
    expected_u = solver.ops.center_x_to_u @ pressure_flux
    expected_v = solver.ops.center_y_to_v @ pressure_flux
    expected = np.concatenate((expected_u, expected_v))

    np.testing.assert_allclose(rhs, expected, rtol=1e-13, atol=1e-13)


def test_m1_ice_matrix_removes_coriolis_cross_blocks_while_m0_keeps_them():
    common = dict(
        nx=6,
        ny=4,
        duration_hours=0.5,
        p_star=0.0,
        drag_coefficient=0.0,
        wind_stress_x=0.0,
        wind_stress_y=0.0,
    )
    matrices = {}
    for mode in (ModelMode.M0_FULL, ModelMode.M1_QUASI_STATIC):
        config = Problem2Config(**common, mode=mode)
        grid = config.grid
        zeros_u = np.zeros(grid.u_shape)
        zeros_v = np.zeros(grid.v_shape)
        solver = IceMomentumSolver(config)
        matrix, _ = solver.assemble_linear_system(
            thickness=np.ones(grid.center_shape),
            old_u=zeros_u,
            old_v=zeros_v,
            water_u=zeros_u,
            water_v=zeros_v,
            iterate_u=zeros_u,
            iterate_v=zeros_v,
            inertia_weight=config.ice_inertia_weight,
        )
        matrices[mode] = (solver, matrix)

    m0_solver, m0 = matrices[ModelMode.M0_FULL]
    m1_solver, m1 = matrices[ModelMode.M1_QUASI_STATIC]
    m0_uv = m0[: m0_solver.ops.n_u, m0_solver.ops.n_u :].toarray()
    m0_vu = m0[m0_solver.ops.n_u :, : m0_solver.ops.n_u].toarray()
    m1_uv = m1[: m1_solver.ops.n_u, m1_solver.ops.n_u :].toarray()
    m1_vu = m1[m1_solver.ops.n_u :, : m1_solver.ops.n_u].toarray()

    assert np.linalg.norm(m0_uv) > 0.0
    assert np.linalg.norm(m0_vu) > 0.0
    np.testing.assert_array_equal(m1_uv, 0.0)
    np.testing.assert_array_equal(m1_vu, 0.0)


def test_picard_matrix_and_physical_residual_share_the_weighted_vp_operator():
    for mode in (ModelMode.M0_FULL, ModelMode.M1_QUASI_STATIC):
        config = Problem2Config(nx=7, ny=5, duration_hours=0.5, mode=mode)
        grid = config.grid
        solver = IceMomentumSolver(config)
        xc, yc = grid.center_coordinates()
        xu, yu = grid.u_coordinates()
        xv, yv = grid.v_coordinates()
        thickness = 0.45 + 0.2 * xc / config.length_x + 0.1 * yc / config.length_y
        candidate_u = 0.03 * np.sin(np.pi * xu / config.length_x) * np.sin(
            np.pi * yu / config.length_y
        )
        candidate_v = -0.02 * np.sin(np.pi * xv / config.length_x) * np.sin(
            np.pi * yv / config.length_y
        )
        old_u = 0.8 * candidate_u
        old_v = 0.8 * candidate_v
        water_u = -0.4 * candidate_u
        water_v = -0.4 * candidate_v
        matrix, rhs = solver.assemble_linear_system(
            thickness=thickness,
            old_u=old_u,
            old_v=old_v,
            water_u=water_u,
            water_v=water_v,
            iterate_u=candidate_u,
            iterate_v=candidate_v,
            inertia_weight=config.ice_inertia_weight,
        )
        drag = compute_interface_drag(
            grid,
            thickness=thickness,
            h_min=config.h_min,
            ice_u=candidate_u,
            ice_v=candidate_v,
            water_u=water_u,
            water_v=water_v,
            rho_water=config.rho_water,
            drag_coefficient=config.drag_coefficient,
        )
        residual_u, residual_v, _, _ = solver.physical_residual_components(
            thickness=thickness,
            old_u=old_u,
            old_v=old_v,
            water_u=water_u,
            water_v=water_v,
            candidate_u=candidate_u,
            candidate_v=candidate_v,
            applied_drag=drag,
            inertia_weight=config.ice_inertia_weight,
        )
        algebraic = matrix @ solver.ops.ice_velocity_vector(candidate_u, candidate_v) - rhs
        _, active_u, active_v = ice_active_face_masks(grid, thickness, h_min=config.h_min)
        active_u[:, (0, -1)] = False
        active_v[(0, -1), :] = False
        active = np.concatenate((active_u.ravel(), active_v.ravel()))
        physical = np.concatenate((residual_u.ravel(), residual_v.ravel()))

        np.testing.assert_allclose(algebraic[active], physical[active], rtol=1e-10, atol=1e-10)


def test_picard_gmres_solves_both_inertial_and_quasi_static_ice_momentum():
    for mode in (ModelMode.M0_FULL, ModelMode.M1_QUASI_STATIC):
        config = Problem2Config(nx=8, ny=4, duration_hours=0.5, mode=mode)
        state = CoupledState.initial(config)
        result = IceMomentumSolver(config).solve(
            thickness=state.thickness,
            old_u=state.ice_u,
            old_v=state.ice_v,
            water_u=state.ocean_u,
            water_v=state.ocean_v,
            inertia_weight=config.ice_inertia_weight,
            max_picard_iterations=60,
            preconditioner="robust_ilu",
        )

        assert result.converged
        assert result.nonlinear_residual <= config.ice_picard_tolerance
        assert np.all(np.isfinite(result.u))
        assert np.all(np.isfinite(result.v))
        assert np.all(result.u[:, (0, -1)] == 0.0)
        assert np.all(result.v[(0, -1), :] == 0.0)


def test_one_coupled_physical_step_passes_frozen_hard_checks(tmp_path):
    config = Problem2Config(
        nx=8,
        ny=4,
        duration_hours=0.5,
        mode=ModelMode.M1_QUASI_STATIC,
    )
    solver = Problem2Solver(config, failure_directory=tmp_path / "failures")
    result = solver.advance_one_physical_step(1)

    assert result.state.time_seconds == config.dt
    assert result.diagnostics.chi == 0.0
    assert result.diagnostics.chi_coriolis == 0.0
    assert result.diagnostics.ocean_coriolis_iterations >= 1
    assert result.diagnostics.ice_residual <= config.ice_picard_tolerance
    assert result.diagnostics.coupling_residual <= config.coupling_tolerance
    assert result.diagnostics.gmres_residual <= config.gmres_relative_tolerance
    assert result.diagnostics.drag_closure_error <= 1.0e-12
    assert result.diagnostics.drag_face_l1_closure_error == 0.0
    assert result.diagnostics.drag_object_reused
    assert result.diagnostics.maximum_inactive_drag == 0.0
    assert result.diagnostics.ice_volume_relative_error <= 1.0e-6
    assert result.diagnostics.minimum_thickness >= 0.0


def test_force_budget_uses_actual_discrete_terms_and_m1_deleted_forces_are_zero():
    config = Problem2Config(
        nx=8,
        ny=4,
        duration_hours=0.5,
        mode=ModelMode.M1_QUASI_STATIC,
    )
    state = CoupledState.initial(config)
    solver = IceMomentumSolver(config)
    drag = compute_interface_drag(
        config.grid,
        thickness=state.thickness,
        h_min=config.h_min,
        ice_u=state.ice_u,
        ice_v=state.ice_v,
        water_u=state.ocean_u,
        water_v=state.ocean_v,
        rho_water=config.rho_water,
        drag_coefficient=config.drag_coefficient,
    )

    budget = solver.force_budget(
        thickness=state.thickness,
        old_u=state.ice_u,
        old_v=state.ice_v,
        water_u=state.ocean_u,
        water_v=state.ocean_v,
        candidate_u=state.ice_u,
        candidate_v=state.ice_v,
        applied_drag=drag,
        inertia_weight=config.ice_inertia_weight,
    )

    assert set(budget.face_arrays) == {
        "F_I_u",
        "F_I_v",
        "F_C_u",
        "F_C_v",
        "F_sigma_u",
        "F_sigma_v",
        "F_D_u",
        "F_D_v",
        "F_W_u",
        "F_W_v",
    }
    assert np.count_nonzero(budget.face_arrays["F_I_u"]) == 0
    assert np.count_nonzero(budget.face_arrays["F_C_v"]) == 0
    for name in (
        "r_IC_corrected",
        "r_Sigma",
        "r_net_corrected",
        "combined_missing_abs",
        "undirected_IC_abs_sum",
        "retained_abs_sum",
        "retained_net_abs",
    ):
        assert name in budget.center_arrays
        assert np.all(np.isfinite(budget.center_arrays[name]))
        assert {"median", "p95", "maximum"} <= set(budget.summary[name])

    assert "deprecated_FI_plus_FC_abs" in budget.center_arrays


def test_force_budget_corrected_combined_missing_matches_physical_residual_sign():
    config = Problem2Config(
        nx=8,
        ny=4,
        dt=300.0,
        duration_hours=300.0 / 3600.0,
        mode=ModelMode.M0_FULL,
    )
    state = CoupledState.initial(config)
    solver = IceMomentumSolver(config)
    drag = compute_interface_drag(
        config.grid,
        thickness=state.thickness,
        h_min=config.h_min,
        ice_u=state.ice_u,
        ice_v=state.ice_v,
        water_u=state.ocean_u,
        water_v=state.ocean_v,
        rho_water=config.rho_water,
        drag_coefficient=config.drag_coefficient,
    )

    budget = solver.force_budget(
        thickness=state.thickness,
        old_u=state.ice_u,
        old_v=state.ice_v,
        water_u=state.ocean_u,
        water_v=state.ocean_v,
        candidate_u=state.ice_u,
        candidate_v=state.ice_v,
        applied_drag=drag,
        inertia_weight=config.ice_inertia_weight,
    )
    residual_u, residual_v, _, _ = solver.physical_residual_components(
        thickness=state.thickness,
        old_u=state.ice_u,
        old_v=state.ice_v,
        water_u=state.ocean_u,
        water_v=state.ocean_v,
        candidate_u=state.ice_u,
        candidate_v=state.ice_v,
        applied_drag=drag,
        inertia_weight=config.ice_inertia_weight,
    )
    residual_center_u, residual_center_v = config.grid.faces_to_center(
        residual_u, residual_v
    )
    residual_abs = np.hypot(residual_center_u, residual_center_v)
    active = budget.center_arrays["active_center"]

    assert np.allclose(
        budget.center_arrays["combined_missing_u"]
        - budget.center_arrays["retained_net_u"],
        residual_center_u,
    )
    assert np.allclose(
        budget.center_arrays["combined_missing_v"]
        - budget.center_arrays["retained_net_v"],
        residual_center_v,
    )
    assert np.all(
        np.abs(
            budget.center_arrays["combined_missing_abs"][active]
            - budget.center_arrays["retained_net_abs"][active]
        )
        <= residual_abs[active] + 1.0e-12
    )


def test_force_budget_m1_zeroes_inertia_and_ice_coriolis_center_terms():
    config = Problem2Config(
        nx=8,
        ny=4,
        dt=300.0,
        duration_hours=300.0 / 3600.0,
        mode=ModelMode.M1_QUASI_STATIC,
    )
    state = CoupledState.initial(config)
    solver = IceMomentumSolver(config)
    drag = compute_interface_drag(
        config.grid,
        thickness=state.thickness,
        h_min=config.h_min,
        ice_u=state.ice_u,
        ice_v=state.ice_v,
        water_u=state.ocean_u,
        water_v=state.ocean_v,
        rho_water=config.rho_water,
        drag_coefficient=config.drag_coefficient,
    )

    budget = solver.force_budget(
        thickness=state.thickness,
        old_u=state.ice_u,
        old_v=state.ice_v,
        water_u=state.ocean_u,
        water_v=state.ocean_v,
        candidate_u=state.ice_u,
        candidate_v=state.ice_v,
        applied_drag=drag,
        inertia_weight=config.ice_inertia_weight,
    )

    for name in (
        "F_I_u",
        "F_I_v",
        "F_C_u",
        "F_C_v",
    ):
        assert np.count_nonzero(budget.face_arrays[name]) == 0
    for name in (
        "F_I_center_u",
        "F_I_center_v",
        "F_C_center_u",
        "F_C_center_v",
        "combined_missing_abs",
        "undirected_IC_abs_sum",
    ):
        assert np.count_nonzero(budget.center_arrays[name]) == 0
