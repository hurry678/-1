import numpy as np

from problem2_core.config import Problem2Config
from problem2_core.coupling import (
    compute_interface_drag,
    diagnose_applied_drag_closure,
)
from problem2_core.grid import CGrid
from problem2_core.ice_momentum import IceMomentumSolver
from problem2_core.ocean_swe import OceanIMEXSolver, OceanState
from problem2_core.state import CoupledState


def test_ice_water_drag_uses_one_shared_stress_and_closes_momentum_exactly():
    grid = CGrid(nx=6, ny=4, dx=100.0, dy=100.0)
    ice_u = np.full(grid.u_shape, 0.05)
    ice_v = np.full(grid.v_shape, -0.02)
    water_u = np.full(grid.u_shape, 0.18)
    water_v = np.full(grid.v_shape, 0.04)

    drag = compute_interface_drag(
        grid,
        thickness=np.full(grid.center_shape, 0.5),
        h_min=1.0e-6,
        ice_u=ice_u,
        ice_v=ice_v,
        water_u=water_u,
        water_v=water_v,
        rho_water=1025.0,
        drag_coefficient=0.005,
    )
    balance = diagnose_applied_drag_closure(
        grid,
        ice_force_u=drag.tau_u,
        ice_force_v=drag.tau_v,
        water_force_u=-drag.tau_u,
        water_force_v=-drag.tau_v,
        dt=300.0,
    )

    assert np.max(np.abs(drag.tau_u)) > 0.0
    assert np.max(np.abs(drag.tau_v)) > 0.0
    np.testing.assert_array_equal(balance.ice_u, -balance.water_u)
    np.testing.assert_array_equal(balance.ice_v, -balance.water_v)
    assert balance.absolute_closure_error == 0.0
    assert balance.relative_closure_error == 0.0

    broken = diagnose_applied_drag_closure(
        grid,
        ice_force_u=drag.tau_u,
        ice_force_v=drag.tau_v,
        water_force_u=-0.9 * drag.tau_u,
        water_force_v=-drag.tau_v,
        dt=300.0,
    )
    assert broken.relative_closure_error > 0.0


def test_drag_closure_uses_resultant_impulse_norm_and_reports_face_l1_auxiliary():
    grid = CGrid(nx=4, ny=3, dx=50.0, dy=80.0)
    ice_u = np.zeros(grid.u_shape)
    ice_v = np.zeros(grid.v_shape)
    ice_u[1, 1] = 2.0
    ice_u[1, 2] = 1.0
    ice_v[1, 1] = -4.0
    water_u = -ice_u.copy()
    water_v = -ice_v.copy()
    water_u[1, 1] += 0.2
    water_v[1, 1] += 0.3

    balance = diagnose_applied_drag_closure(
        grid,
        ice_force_u=ice_u,
        ice_force_v=ice_v,
        water_force_u=water_u,
        water_force_v=water_v,
        dt=300.0,
    )
    scale = grid.cell_area * 300.0
    expected = np.hypot(0.2 * scale, 0.3 * scale) / np.hypot(3.0 * scale, -4.0 * scale)
    expected_l1 = (0.2 + 0.3) / (2.0 + 1.0 + 4.0)

    assert np.isclose(balance.relative_closure_error, expected)
    assert np.isclose(balance.face_l1_relative_closure_error, expected_l1)


def test_zero_ice_faces_apply_exactly_zero_drag_to_nonzero_water():
    grid = CGrid(nx=6, ny=4, dx=100.0, dy=100.0)
    drag = compute_interface_drag(
        grid,
        thickness=np.zeros(grid.center_shape),
        h_min=1.0e-6,
        ice_u=np.zeros(grid.u_shape),
        ice_v=np.zeros(grid.v_shape),
        water_u=np.full(grid.u_shape, 0.2),
        water_v=np.full(grid.v_shape, -0.1),
        rho_water=1025.0,
        drag_coefficient=0.005,
    )

    np.testing.assert_array_equal(drag.tau_u, 0.0)
    np.testing.assert_array_equal(drag.tau_v, 0.0)


def test_ice_and_water_solvers_reuse_the_exact_same_drag_object():
    config = Problem2Config(nx=8, ny=4, duration_hours=0.5)
    state = CoupledState.initial(config)
    ice = IceMomentumSolver(config).solve(
        thickness=state.thickness,
        old_u=state.ice_u,
        old_v=state.ice_v,
        water_u=state.ocean_u,
        water_v=state.ocean_v,
        inertia_weight=1.0,
        max_picard_iterations=60,
        preconditioner="robust_ilu",
    )
    ocean = OceanIMEXSolver(config).advance(
        OceanState(xi=state.sea_surface, u=state.ocean_u, v=state.ocean_v),
        ice.applied_drag,
    )

    assert ice.converged
    assert ocean.converged
    assert ice.ice_drag_force_u is ice.applied_drag.tau_u
    assert ice.ice_drag_force_v is ice.applied_drag.tau_v
    assert ocean.applied_drag is ice.applied_drag
    assert all(not hasattr(record, "aitken_factor") for record in ice.records)
    balance = diagnose_applied_drag_closure(
        config.grid,
        ice_force_u=ice.ice_drag_force_u,
        ice_force_v=ice.ice_drag_force_v,
        water_force_u=ocean.water_drag_force_u,
        water_force_v=ocean.water_drag_force_v,
        dt=config.dt,
    )
    assert balance.relative_closure_error == 0.0
