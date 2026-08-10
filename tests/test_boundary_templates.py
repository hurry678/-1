import numpy as np

from problem2_core.boundary import (
    apply_ice_no_slip,
    apply_water_free_slip,
    homogeneous_neumann_ghost,
    ice_active_face_masks,
    ice_tangential_ghosts,
    thickness_even_ghost,
    water_tangential_ghosts,
    zero_normal_face_fluxes,
)
from problem2_core.grid import CGrid
from problem2_core.ocean_swe import OceanState, lake_at_rest_residual


def test_ice_no_slip_sets_normal_faces_to_zero_and_uses_odd_tangential_ghosts():
    grid = CGrid(nx=5, ny=4, dx=100.0, dy=100.0)
    u_face = np.arange(np.prod(grid.u_shape), dtype=float).reshape(grid.u_shape)
    v_face = np.arange(np.prod(grid.v_shape), dtype=float).reshape(grid.v_shape)

    u_bounded, v_bounded = apply_ice_no_slip(grid, u_face, v_face)
    ghosts = ice_tangential_ghosts(grid, u_bounded, v_bounded)

    np.testing.assert_allclose(u_bounded[:, 0], 0.0)
    np.testing.assert_allclose(u_bounded[:, -1], 0.0)
    np.testing.assert_allclose(v_bounded[0, :], 0.0)
    np.testing.assert_allclose(v_bounded[-1, :], 0.0)
    np.testing.assert_allclose(0.5 * (ghosts.u_bottom + u_bounded[0, :]), 0.0)
    np.testing.assert_allclose(0.5 * (ghosts.u_top + u_bounded[-1, :]), 0.0)
    np.testing.assert_allclose(0.5 * (ghosts.v_left + v_bounded[:, 0]), 0.0)
    np.testing.assert_allclose(0.5 * (ghosts.v_right + v_bounded[:, -1]), 0.0)


def test_water_free_slip_sets_normal_faces_zero_and_tangential_derivative_zero():
    grid = CGrid(nx=5, ny=4, dx=100.0, dy=100.0)
    u_face = np.arange(np.prod(grid.u_shape), dtype=float).reshape(grid.u_shape)
    v_face = np.arange(np.prod(grid.v_shape), dtype=float).reshape(grid.v_shape)

    u_bounded, v_bounded = apply_water_free_slip(grid, u_face, v_face)
    ghosts = water_tangential_ghosts(grid, u_bounded, v_bounded)

    np.testing.assert_allclose(u_bounded[:, (0, -1)], 0.0)
    np.testing.assert_allclose(v_bounded[(0, -1), :], 0.0)
    np.testing.assert_allclose(ghosts.u_bottom - u_bounded[0, :], 0.0)
    np.testing.assert_allclose(ghosts.u_top - u_bounded[-1, :], 0.0)
    np.testing.assert_allclose(ghosts.v_left - v_bounded[:, 0], 0.0)
    np.testing.assert_allclose(ghosts.v_right - v_bounded[:, -1], 0.0)


def test_thickness_and_helmholtz_neumann_templates_are_even_extensions():
    grid = CGrid(nx=4, ny=3, dx=100.0, dy=100.0)
    center = np.arange(np.prod(grid.center_shape), dtype=float).reshape(
        grid.center_shape
    )

    thickness_ghost = thickness_even_ghost(grid, center)
    xi_ghost = homogeneous_neumann_ghost(grid, center)

    np.testing.assert_allclose(thickness_ghost[0, 1:-1], center[0, :])
    np.testing.assert_allclose(thickness_ghost[-1, 1:-1], center[-1, :])
    np.testing.assert_allclose(xi_ghost[1:-1, 0], center[:, 0])
    np.testing.assert_allclose(xi_ghost[1:-1, -1], center[:, -1])

    flux_u = np.ones(grid.u_shape)
    flux_v = np.ones(grid.v_shape)
    bounded_u, bounded_v = zero_normal_face_fluxes(grid, flux_u, flux_v)
    np.testing.assert_allclose(bounded_u[:, (0, -1)], 0.0)
    np.testing.assert_allclose(bounded_v[(0, -1), :], 0.0)


def test_lake_at_rest_has_zero_discrete_shallow_water_residual():
    grid = CGrid(nx=8, ny=4, dx=100.0, dy=100.0)
    state = OceanState(
        xi=np.zeros(grid.center_shape),
        u=np.zeros(grid.u_shape),
        v=np.zeros(grid.v_shape),
    )

    residual = lake_at_rest_residual(grid, state, water_depth=30.0, gravity=9.8)

    np.testing.assert_allclose(residual.xi, 0.0, atol=1e-14)
    np.testing.assert_allclose(residual.u, 0.0, atol=1e-14)
    np.testing.assert_allclose(residual.v, 0.0, atol=1e-14)


def test_ice_face_activity_uses_logical_or_across_an_ice_edge():
    grid = CGrid(nx=4, ny=2, dx=100.0, dy=100.0)
    thickness = np.array([[0.5, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0]])

    center_mask, u_mask, v_mask = ice_active_face_masks(
        grid, thickness, h_min=1.0e-3
    )

    assert np.all(center_mask[:, 0])
    assert not np.any(center_mask[:, 1:])
    assert np.all(u_mask[:, 1])  # 冰/无冰交界面仍保持活动，允许冰缘外传。
    assert not np.any(u_mask[:, 2:])
    assert np.all(v_mask[:, 0])
