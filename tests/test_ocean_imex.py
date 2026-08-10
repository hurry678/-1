import numpy as np

from problem2_core.config import ModelMode, Problem2Config
from problem2_core.coupling import InterfaceDrag
from problem2_core.ocean_swe import OceanIMEXSolver, OceanState


def test_imex_helmholtz_preserves_lake_at_rest_and_closed_surface_mean():
    config = Problem2Config(nx=12, ny=6, duration_hours=0.5)
    grid = config.grid
    state = OceanState(
        xi=np.zeros(grid.center_shape),
        u=np.zeros(grid.u_shape),
        v=np.zeros(grid.v_shape),
    )
    drag = InterfaceDrag(
        tau_u=np.zeros(grid.u_shape), tau_v=np.zeros(grid.v_shape)
    )
    result = OceanIMEXSolver(config).advance(state, drag)

    assert result.converged
    assert result.helmholtz_residual <= config.water_residual_tolerance
    assert result.equation_residual <= config.water_residual_tolerance
    assert result.residual_xi <= config.water_residual_tolerance
    assert result.residual_u <= config.water_residual_tolerance
    assert result.residual_v <= config.water_residual_tolerance
    assert result.coriolis_iterations >= 1
    assert np.max(np.abs(result.state.xi)) == 0.0
    assert np.max(np.abs(result.state.u)) == 0.0
    assert np.max(np.abs(result.state.v)) == 0.0
    assert abs(float(np.mean(result.state.xi))) <= 1.0e-15


def test_nonzero_ocean_step_converges_each_frozen_equation_with_current_coriolis():
    config = Problem2Config(nx=12, ny=6, duration_hours=0.5)
    grid = config.grid
    x = np.linspace(0.0, 1.0, grid.nx + 1)[None, :]
    y = np.linspace(0.0, 1.0, grid.ny + 1)[:, None]
    u = np.repeat(0.03 * np.sin(np.pi * x), grid.ny, axis=0)
    v = np.repeat(-0.02 * np.sin(np.pi * y), grid.nx, axis=1)
    state = OceanState(
        xi=np.zeros(grid.center_shape),
        u=u,
        v=v,
    )
    drag = InterfaceDrag(
        tau_u=np.full(grid.u_shape, 0.01),
        tau_v=np.full(grid.v_shape, -0.02),
    )

    result = OceanIMEXSolver(config).advance(state, drag)

    assert result.converged
    assert result.coriolis_iterations > 1
    assert result.residual_xi <= config.water_residual_tolerance
    assert result.residual_u <= config.water_residual_tolerance
    assert result.residual_v <= config.water_residual_tolerance
    assert result.equation_residual == max(
        result.residual_xi, result.residual_u, result.residual_v
    )
    assert result.applied_drag is drag


def test_m1_keeps_the_same_nonzero_ocean_coriolis_solution_as_m0():
    common = dict(nx=10, ny=6, duration_hours=0.5)
    m0 = Problem2Config(**common, mode=ModelMode.M0_FULL)
    m1 = Problem2Config(**common, mode=ModelMode.M1_QUASI_STATIC)
    grid = m0.grid
    xu, yu = grid.u_coordinates()
    xv, yv = grid.v_coordinates()
    state = OceanState(
        xi=np.zeros(grid.center_shape),
        u=0.02 * np.sin(np.pi * xu / m0.length_x),
        v=-0.015 * np.sin(np.pi * yv / m0.length_y),
    )
    drag = InterfaceDrag(
        tau_u=np.zeros(grid.u_shape), tau_v=np.zeros(grid.v_shape)
    )

    result_m0 = OceanIMEXSolver(m0).advance(state, drag)
    result_m1 = OceanIMEXSolver(m1).advance(state, drag)

    assert result_m0.converged and result_m1.converged
    assert result_m0.coriolis_iterations > 1
    assert result_m1.coriolis_iterations > 1
    np.testing.assert_allclose(result_m1.state.u, result_m0.state.u)
    np.testing.assert_allclose(result_m1.state.v, result_m0.state.v)
    np.testing.assert_allclose(result_m1.state.xi, result_m0.state.xi)
