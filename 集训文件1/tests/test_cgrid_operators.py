import numpy as np

from problem2_core.grid import CGrid


def test_cgrid_shapes_and_coordinates_match_frozen_locations():
    grid = CGrid(nx=4, ny=3, dx=100.0, dy=200.0)

    assert grid.center_shape == (3, 4)
    assert grid.u_shape == (3, 5)
    assert grid.v_shape == (4, 4)
    assert grid.corner_shape == (4, 5)

    xc, yc = grid.center_coordinates()
    xu, yu = grid.u_coordinates()
    xv, yv = grid.v_coordinates()
    xq, yq = grid.corner_coordinates()
    assert (xc[0, 0], yc[0, 0]) == (50.0, 100.0)
    assert (xu[0, 0], yu[0, 0]) == (0.0, 100.0)
    assert (xv[0, 0], yv[0, 0]) == (50.0, 0.0)
    assert (xq[0, 0], yq[0, 0]) == (0.0, 0.0)


def test_constant_fields_remain_constant_under_all_basic_interpolations():
    grid = CGrid(nx=5, ny=4, dx=2.0, dy=3.0)
    center = np.full(grid.center_shape, 2.5)
    u_face = np.full(grid.u_shape, -0.4)
    v_face = np.full(grid.v_shape, 0.7)

    np.testing.assert_allclose(grid.center_to_u(center), 2.5)
    np.testing.assert_allclose(grid.center_to_v(center), 2.5)
    corner = grid.center_to_corner(center)
    assert corner.shape == grid.corner_shape
    np.testing.assert_allclose(corner, 2.5)
    np.testing.assert_allclose(grid.v_to_u(v_face), 0.7)
    np.testing.assert_allclose(grid.u_to_v(u_face), -0.4)

    u_center, v_center = grid.faces_to_center(u_face, v_face)
    np.testing.assert_allclose(u_center, -0.4)
    np.testing.assert_allclose(v_center, 0.7)
