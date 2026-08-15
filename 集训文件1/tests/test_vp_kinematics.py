import numpy as np

from problem2_core.grid import CGrid
from problem2_core.rheology_vp import strain_invariant_squared, strain_rates


def analytic_velocity_and_ghosts(grid, u_function, v_function):
    xu, yu = grid.u_coordinates()
    xv, yv = grid.v_coordinates()
    u_face = u_function(xu, yu)
    v_face = v_function(xv, yv)

    x_u = xu[0, :]
    y_v = yv[:, 0]
    u_bottom = u_function(x_u, np.full_like(x_u, grid.y0 - 0.5 * grid.dy))
    u_top = u_function(
        x_u, np.full_like(x_u, grid.y0 + (grid.ny + 0.5) * grid.dy)
    )
    v_left = v_function(np.full_like(y_v, grid.x0 - 0.5 * grid.dx), y_v)
    v_right = v_function(
        np.full_like(y_v, grid.x0 + (grid.nx + 0.5) * grid.dx), y_v
    )
    return u_face, v_face, (u_bottom, u_top), (v_left, v_right)


def test_constant_velocity_has_zero_symmetric_strain_everywhere():
    grid = CGrid(nx=8, ny=6, dx=100.0, dy=100.0)
    values = analytic_velocity_and_ghosts(
        grid,
        lambda x, y: np.zeros_like(x) + 0.3,
        lambda x, y: np.zeros_like(x) - 0.2,
    )

    strain = strain_rates(grid, *values)

    np.testing.assert_allclose(strain.exx, 0.0, atol=1e-14)
    np.testing.assert_allclose(strain.eyy, 0.0, atol=1e-14)
    np.testing.assert_allclose(strain.exy_corner, 0.0, atol=1e-14)


def test_rigid_rotation_has_vorticity_but_zero_symmetric_strain():
    grid = CGrid(nx=10, ny=7, dx=80.0, dy=120.0)
    omega = 2.5e-4
    xc = grid.x0 + 0.5 * grid.nx * grid.dx
    yc = grid.y0 + 0.5 * grid.ny * grid.dy
    values = analytic_velocity_and_ghosts(
        grid,
        lambda x, y: -omega * (y - yc),
        lambda x, y: omega * (x - xc),
    )

    strain = strain_rates(grid, *values)
    invariant_sq = strain_invariant_squared(strain, ellipse_ratio=2.0)

    np.testing.assert_allclose(strain.exx, 0.0, atol=1e-14)
    np.testing.assert_allclose(strain.eyy, 0.0, atol=1e-14)
    np.testing.assert_allclose(strain.exy_corner, 0.0, atol=1e-14)
    np.testing.assert_allclose(invariant_sq, 0.0, atol=1e-28)


def test_linear_shear_places_half_shear_rate_at_all_corners():
    grid = CGrid(nx=9, ny=5, dx=70.0, dy=90.0)
    gamma = 3.0e-4
    yc = grid.y0 + 0.5 * grid.ny * grid.dy
    values = analytic_velocity_and_ghosts(
        grid,
        lambda x, y: gamma * (y - yc),
        lambda x, y: np.zeros_like(x),
    )

    strain = strain_rates(grid, *values)

    np.testing.assert_allclose(strain.exx, 0.0, atol=1e-14)
    np.testing.assert_allclose(strain.eyy, 0.0, atol=1e-14)
    np.testing.assert_allclose(strain.exy_corner, 0.5 * gamma, atol=1e-14)

    expected_center_square = np.full(grid.center_shape, (0.5 * gamma) ** 2)
    np.testing.assert_allclose(strain.exy_square_center, expected_center_square)
