import numpy as np

from problem2_core.grid import CGrid
from problem2_core.rheology_vp import (
    compute_vp_stress,
    strain_rates,
    stress_divergence,
)


def test_uniform_stress_and_unit_thickness_have_zero_divergence_on_all_faces():
    grid = CGrid(nx=7, ny=5, dx=100.0, dy=100.0)
    thickness = np.ones(grid.center_shape)
    sigma_xx = np.full(grid.center_shape, 13.0)
    sigma_yy = np.full(grid.center_shape, -4.0)
    sigma_xy = np.full(grid.corner_shape, 2.5)

    divergence_u, divergence_v = stress_divergence(
        grid, thickness, sigma_xx, sigma_yy, sigma_xy
    )

    assert divergence_u.shape == grid.u_shape
    assert divergence_v.shape == grid.v_shape
    np.testing.assert_allclose(divergence_u, 0.0, atol=1e-14)
    np.testing.assert_allclose(divergence_v, 0.0, atol=1e-14)


def test_constant_thickness_scales_the_unit_thickness_stress_divergence():
    grid = CGrid(nx=9, ny=6, dx=80.0, dy=120.0)
    rng = np.random.default_rng(20260807)
    sigma_xx = rng.normal(size=grid.center_shape)
    sigma_yy = rng.normal(size=grid.center_shape)
    sigma_xy = rng.normal(size=grid.corner_shape)
    unit_u, unit_v = stress_divergence(
        grid, np.ones(grid.center_shape), sigma_xx, sigma_yy, sigma_xy
    )
    h0 = 2.3
    scaled_u, scaled_v = stress_divergence(
        grid, np.full(grid.center_shape, h0), sigma_xx, sigma_yy, sigma_xy
    )

    np.testing.assert_allclose(scaled_u, h0 * unit_u, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(scaled_v, h0 * unit_v, rtol=1e-13, atol=1e-13)


def test_constant_stress_with_variable_thickness_has_nonzero_product_divergence():
    grid = CGrid(nx=10, ny=7, dx=0.1, dy=0.2)
    x, y = grid.center_coordinates()
    thickness = 1.0 + 0.4 * x + 0.7 * y
    sigma_xx = np.full(grid.center_shape, 3.0)
    sigma_yy = np.full(grid.center_shape, -2.0)
    sigma_xy = np.zeros(grid.corner_shape)

    divergence_u, divergence_v = stress_divergence(
        grid, thickness, sigma_xx, sigma_yy, sigma_xy
    )

    np.testing.assert_allclose(divergence_u[:, 1:-1], 3.0 * 0.4, atol=1e-13)
    np.testing.assert_allclose(divergence_v[1:-1, :], -2.0 * 0.7, atol=1e-13)
    assert np.linalg.norm(divergence_u[:, 1:-1]) > 0.0
    assert np.linalg.norm(divergence_v[1:-1, :]) > 0.0


def test_constant_thickness_weighted_stress_has_zero_divergence():
    grid = CGrid(nx=8, ny=5, dx=100.0, dy=100.0)
    x, y = grid.center_coordinates()
    thickness = 0.4 + 0.1 * x / x.max() + 0.2 * y / y.max()
    h_corner = grid.center_to_corner(thickness)
    sigma_xx = 7.0 / thickness
    sigma_yy = -3.0 / thickness
    sigma_xy = 2.0 / h_corner

    divergence_u, divergence_v = stress_divergence(
        grid, thickness, sigma_xx, sigma_yy, sigma_xy
    )

    np.testing.assert_allclose(divergence_u, 0.0, atol=1e-14)
    np.testing.assert_allclose(divergence_v, 0.0, atol=1e-14)


def test_product_divergence_is_not_face_thickness_times_bare_divergence():
    grid = CGrid(nx=12, ny=8, dx=1.0 / 12.0, dy=1.0 / 8.0)
    xc, yc = grid.center_coordinates()
    xk, yk = grid.corner_coordinates()
    thickness = 0.8 + 0.15 * np.sin(2.0 * np.pi * xc) * np.cos(np.pi * yc)
    sigma_xx = 1.0 + xc**2
    sigma_yy = -0.5 + yc**2
    sigma_xy = 0.2 * np.sin(np.pi * xk) * np.sin(np.pi * yk)

    product_u, product_v = stress_divergence(
        grid, thickness, sigma_xx, sigma_yy, sigma_xy
    )
    bare_u, bare_v = stress_divergence(
        grid, np.ones(grid.center_shape), sigma_xx, sigma_yy, sigma_xy
    )
    incorrect_u = grid.center_to_u(thickness) * bare_u
    incorrect_v = grid.center_to_v(thickness) * bare_v

    assert np.linalg.norm(product_u[:, 1:-1] - incorrect_u[:, 1:-1]) > 1.0e-3
    assert np.linalg.norm(product_v[1:-1, :] - incorrect_v[1:-1, :]) > 1.0e-3


def test_corner_shear_flux_uses_four_point_and_boundary_one_sided_thickness():
    grid = CGrid(nx=2, ny=2, dx=2.0, dy=4.0)
    thickness = np.array([[1.0, 3.0], [5.0, 7.0]])
    sigma_xy = np.ones(grid.corner_shape)
    divergence_u, divergence_v = stress_divergence(
        grid,
        thickness,
        np.zeros(grid.center_shape),
        np.zeros(grid.center_shape),
        sigma_xy,
    )
    expected_h_corner = np.array(
        [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0]]
    )
    expected_u = (expected_h_corner[1:, :] - expected_h_corner[:-1, :]) / grid.dy
    expected_v = (expected_h_corner[:, 1:] - expected_h_corner[:, :-1]) / grid.dx

    np.testing.assert_allclose(divergence_u, expected_u)
    np.testing.assert_allclose(divergence_v, expected_v)


def test_zero_and_thin_ice_weight_stress_without_dividing_by_ice_mass():
    grid = CGrid(nx=6, ny=4, dx=100.0, dy=100.0)
    rng = np.random.default_rng(7)
    sigma_xx = rng.normal(size=grid.center_shape)
    sigma_yy = rng.normal(size=grid.center_shape)
    sigma_xy = rng.normal(size=grid.corner_shape)
    unit_u, unit_v = stress_divergence(
        grid, np.ones(grid.center_shape), sigma_xx, sigma_yy, sigma_xy
    )
    zero_u, zero_v = stress_divergence(
        grid, np.zeros(grid.center_shape), sigma_xx, sigma_yy, sigma_xy
    )
    thin = 1.0e-10
    thin_u, thin_v = stress_divergence(
        grid, np.full(grid.center_shape, thin), sigma_xx, sigma_yy, sigma_xy
    )

    np.testing.assert_array_equal(zero_u, 0.0)
    np.testing.assert_array_equal(zero_v, 0.0)
    np.testing.assert_allclose(thin_u, thin * unit_u, rtol=1e-12, atol=1e-20)
    np.testing.assert_allclose(thin_v, thin * unit_v, rtol=1e-12, atol=1e-20)


def test_zero_velocity_vp_stress_is_finite_and_uses_frozen_regularization():
    grid = CGrid(nx=6, ny=4, dx=100.0, dy=100.0)
    thickness = np.full(grid.center_shape, 0.5)
    u_face = np.zeros(grid.u_shape)
    v_face = np.zeros(grid.v_shape)
    strain = strain_rates(
        grid,
        u_face,
        v_face,
        (np.zeros(grid.nx + 1), np.zeros(grid.nx + 1)),
        (np.zeros(grid.ny + 1), np.zeros(grid.ny + 1)),
    )

    stress = compute_vp_stress(
        grid,
        thickness,
        strain,
        delta_min=1.0e-8,
        zeta_max=1.0e6,
    )

    expected_pressure = 5000.0 * 0.5 * np.exp(-20.0 * 0.5)
    np.testing.assert_allclose(stress.pressure, expected_pressure)
    np.testing.assert_allclose(stress.delta_epsilon, 1.0e-8)
    assert np.all(np.isfinite(stress.zeta))
    assert np.all(stress.zeta <= 1.0e6)
    np.testing.assert_allclose(stress.sigma_xx, -0.5 * expected_pressure)
    np.testing.assert_allclose(stress.sigma_yy, -0.5 * expected_pressure)
    np.testing.assert_allclose(stress.sigma_xy_corner, 0.0)


def manufactured_weighted_stress_divergence_error(nx, ny):
    grid = CGrid(nx=nx, ny=ny, dx=1.0 / nx, dy=1.0 / ny)
    xc, yc = grid.center_coordinates()
    xq, yq = grid.corner_coordinates()
    xu, yu = grid.u_coordinates()
    xv, yv = grid.v_coordinates()

    thickness = 1.0 + 0.2 * np.sin(np.pi * xc) * np.cos(np.pi * yc)
    sigma_xx = np.sin(2.0 * np.pi * xc) * np.cos(np.pi * yc)
    sigma_yy = np.cos(np.pi * xc) * np.sin(2.0 * np.pi * yc)
    sigma_xy = 0.2 * np.sin(np.pi * xq) * np.sin(np.pi * yq)
    discrete_u, discrete_v = stress_divergence(
        grid, thickness, sigma_xx, sigma_yy, sigma_xy
    )
    h_u = 1.0 + 0.2 * np.sin(np.pi * xu) * np.cos(np.pi * yu)
    dh_dx_u = 0.2 * np.pi * np.cos(np.pi * xu) * np.cos(np.pi * yu)
    dh_dy_u = -0.2 * np.pi * np.sin(np.pi * xu) * np.sin(np.pi * yu)
    sxx_u = np.sin(2.0 * np.pi * xu) * np.cos(np.pi * yu)
    sxy_u = 0.2 * np.sin(np.pi * xu) * np.sin(np.pi * yu)
    exact_u = (
        dh_dx_u * sxx_u
        + h_u * 2.0 * np.pi * np.cos(2.0 * np.pi * xu) * np.cos(np.pi * yu)
        + dh_dy_u * sxy_u
        + h_u * 0.2 * np.pi * np.sin(np.pi * xu) * np.cos(np.pi * yu)
    )
    h_v = 1.0 + 0.2 * np.sin(np.pi * xv) * np.cos(np.pi * yv)
    dh_dx_v = 0.2 * np.pi * np.cos(np.pi * xv) * np.cos(np.pi * yv)
    dh_dy_v = -0.2 * np.pi * np.sin(np.pi * xv) * np.sin(np.pi * yv)
    sxy_v = 0.2 * np.sin(np.pi * xv) * np.sin(np.pi * yv)
    syy_v = np.cos(np.pi * xv) * np.sin(2.0 * np.pi * yv)
    exact_v = (
        dh_dx_v * sxy_v
        + h_v * 0.2 * np.pi * np.cos(np.pi * xv) * np.sin(np.pi * yv)
        + dh_dy_v * syy_v
        + h_v * 2.0 * np.pi * np.cos(np.pi * xv) * np.cos(2.0 * np.pi * yv)
    )
    error_u = np.sqrt(np.mean((discrete_u[:, 1:-1] - exact_u[:, 1:-1]) ** 2))
    error_v = np.sqrt(np.mean((discrete_v[1:-1, :] - exact_v[1:-1, :]) ** 2))
    return np.hypot(error_u, error_v)


def test_smooth_variable_thickness_weighted_divergence_is_second_order_interior():
    error_coarse = manufactured_weighted_stress_divergence_error(24, 12)
    error_fine = manufactured_weighted_stress_divergence_error(48, 24)
    observed_order = np.log2(error_coarse / error_fine)

    assert error_fine < error_coarse
    assert observed_order > 1.8
