import json

import numpy as np

from problem2_core.config import ModelMode, Problem2Config
from problem2_core.coupling import compute_interface_drag
from problem2_core.diagnostic_modes import (
    DIAGNOSTIC_MODEL_MODES,
    DiagnosticMode,
    DiagnosticRunSpec,
    diagnostic_run_spec,
)
from problem2_core.ice_momentum import IceMomentumSolver
from problem2_core.runner import FORMAL_MODEL_MODES, run_simulation, select_model
from problem2_core.solver import Problem2Solver
from problem2_core.state import CoupledState


def test_v2_1_diagnostic_modes_have_the_four_frozen_weight_pairs():
    actual = {
        mode.value: (
            diagnostic_run_spec(mode).inertia_weight,
            diagnostic_run_spec(mode).ice_coriolis_weight,
        )
        for mode in DIAGNOSTIC_MODEL_MODES
    }

    assert actual == {
        "M0": (1.0, 1.0),
        "M_I": (0.0, 1.0),
        "M_C": (1.0, 0.0),
        "M1": (0.0, 0.0),
    }


def test_diagnostic_spec_rejects_weight_drift_and_formal_eligibility():
    try:
        DiagnosticRunSpec(DiagnosticMode.MI_NO_INERTIA, 1.0, 1.0)
    except ValueError as error:
        assert "冻结权重" in str(error)
    else:  # pragma: no cover
        raise AssertionError("漂移权重未被拒绝")

    try:
        DiagnosticRunSpec(
            DiagnosticMode.MI_NO_INERTIA,
            0.0,
            1.0,
            eligible_for_model_selection=True,
        )
    except ValueError as error:
        assert "不得进入正式选模" in str(error)
    else:  # pragma: no cover
        raise AssertionError("诊断模式错误进入正式选模")


def test_matrix_blocks_isolate_inertia_and_ice_coriolis_switches():
    config = Problem2Config(
        nx=6,
        ny=4,
        duration_hours=0.5,
        mode=ModelMode.M0_FULL,
        p_star=0.0,
        drag_coefficient=0.0,
        wind_stress_x=0.0,
        wind_stress_y=0.0,
    )
    grid = config.grid
    thickness = np.ones(grid.center_shape)
    zero_u = np.zeros(grid.u_shape)
    zero_v = np.zeros(grid.v_shape)
    matrices = {}
    for mode in DIAGNOSTIC_MODEL_MODES:
        spec = diagnostic_run_spec(mode)
        solver = IceMomentumSolver(
            config, ice_coriolis_weight=spec.ice_coriolis_weight
        )
        matrix, _ = solver.assemble_linear_system(
            thickness=thickness,
            old_u=zero_u,
            old_v=zero_v,
            water_u=zero_u,
            water_v=zero_v,
            iterate_u=zero_u,
            iterate_v=zero_v,
            inertia_weight=spec.inertia_weight,
        )
        matrices[mode.value] = (solver, matrix)

    m0_solver, m0 = matrices["M0"]
    _, mi = matrices["M_I"]
    _, mc = matrices["M_C"]
    _, m1 = matrices["M1"]
    n_u = m0_solver.ops.n_u
    assert np.linalg.norm(m0[:n_u, n_u:].toarray()) > 0.0
    np.testing.assert_array_equal(mc[:n_u, n_u:].toarray(), 0.0)
    np.testing.assert_array_equal(m1[:n_u, n_u:].toarray(), 0.0)
    np.testing.assert_allclose(m0[:n_u, :n_u].toarray(), mc[:n_u, :n_u].toarray())
    np.testing.assert_allclose(mi[:n_u, :n_u].toarray(), m1[:n_u, :n_u].toarray())
    assert not np.allclose(m0[:n_u, :n_u].toarray(), mi[:n_u, :n_u].toarray())


def test_force_budget_and_physical_residual_use_the_same_diagnostic_switches():
    config = Problem2Config(nx=7, ny=5, duration_hours=0.5, mode=ModelMode.M0_FULL)
    grid = config.grid
    thickness = np.full(grid.center_shape, 0.6)
    xu, yu = grid.u_coordinates()
    xv, yv = grid.v_coordinates()
    candidate_u = 0.03 * np.sin(np.pi * xu / config.length_x) * np.sin(
        np.pi * yu / config.length_y
    )
    candidate_v = -0.02 * np.sin(np.pi * xv / config.length_x) * np.sin(
        np.pi * yv / config.length_y
    )
    old_u = np.zeros(grid.u_shape)
    old_v = np.zeros(grid.v_shape)
    water_u = -0.2 * candidate_u
    water_v = -0.2 * candidate_v
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

    for mode in DIAGNOSTIC_MODEL_MODES:
        spec = diagnostic_run_spec(mode)
        solver = IceMomentumSolver(
            config, ice_coriolis_weight=spec.ice_coriolis_weight
        )
        budget = solver.force_budget(
            thickness=thickness,
            old_u=old_u,
            old_v=old_v,
            water_u=water_u,
            water_v=water_v,
            candidate_u=candidate_u,
            candidate_v=candidate_v,
            applied_drag=drag,
            inertia_weight=spec.inertia_weight,
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
            inertia_weight=spec.inertia_weight,
        )
        faces = budget.face_arrays
        np.testing.assert_allclose(
            residual_u,
            faces["F_I_u"]
            - faces["F_C_u"]
            - faces["F_sigma_u"]
            - faces["F_D_u"]
            - faces["F_W_u"],
        )
        np.testing.assert_allclose(
            residual_v,
            faces["F_I_v"]
            - faces["F_C_v"]
            - faces["F_sigma_v"]
            - faces["F_D_v"]
            - faces["F_W_v"],
        )
        assert (np.count_nonzero(faces["F_I_u"]) > 0) == bool(
            spec.inertia_weight
        )
        assert (np.count_nonzero(faces["F_C_u"]) > 0) == bool(
            spec.ice_coriolis_weight
        )


def test_diagnostic_injection_does_not_mutate_formal_modes_or_ocean_coriolis():
    config = Problem2Config(nx=4, ny=2, duration_hours=0.5, mode=ModelMode.M0_FULL)
    diagnostic = Problem2Solver(
        config, diagnostic_spec=diagnostic_run_spec(DiagnosticMode.MC_NO_ICE_CORIOLIS)
    )
    formal = Problem2Solver(config)

    assert diagnostic.ice_inertia_weight == 1.0
    assert diagnostic.ice_coriolis_weight == 0.0
    assert formal.ice_inertia_weight == 1.0
    assert formal.ice_coriolis_weight == 1.0
    assert config.ice_coriolis_weight == 1.0
    assert diagnostic.ocean_solver.config.ocean_coriolis_enabled
    assert diagnostic.ocean_solver.config.coriolis == formal.ocean_solver.config.coriolis
    assert FORMAL_MODEL_MODES == (ModelMode.M0_FULL, ModelMode.M1_QUASI_STATIC)
    assert select_model(
        accuracy_passed={"M_I": True},
        stability_passed={"M_I": True},
        speedup_vs_m0={"M_I": 2.0},
    ) == "m1_engineering_validation_failed"


def test_diagnostic_run_marks_machine_readable_outputs_as_ineligible(tmp_path):
    config = Problem2Config(
        nx=4,
        ny=2,
        dt=300.0,
        duration_hours=300.0 / 3600.0,
        mode=ModelMode.M0_FULL,
    )
    output = tmp_path / "M_I"
    summary = run_simulation(
        config,
        output,
        snapshot_hours=(0.0, config.duration_hours),
        diagnostic_spec=diagnostic_run_spec(DiagnosticMode.MI_NO_INERTIA),
    )

    assert summary["status"] == "passed"
    assert summary["mode"] == "M_I"
    assert summary["diagnostic_only"] is True
    assert summary["eligible_for_model_selection"] is False
    for name in ("config.json", "summary.json", "test_report.json"):
        payload = json.loads((output / name).read_text(encoding="utf-8"))
        assert payload["diagnostic_only"] is True
        assert payload["eligible_for_model_selection"] is False
    snapshots = dict(np.load(output / "snapshots.npz"))
    assert bool(snapshots["diagnostic_only"])
    assert not bool(snapshots["eligible_for_model_selection"])
    assert CoupledState.initial(config).time_seconds == 0.0
