import numpy as np

from problem2_core.grid import CGrid
from problem2_core.restriction import (
    restrict_center_area_average,
    restrict_u_face_length_average,
    restrict_v_face_length_average,
)


def nested_grids():
    fine = CGrid(nx=8, ny=6, dx=50.0, dy=50.0)
    coarse = CGrid(nx=4, ny=3, dx=100.0, dy=100.0)
    return fine, coarse


def test_center_restriction_is_area_average_and_preserves_integral():
    fine, coarse = nested_grids()
    rng = np.random.default_rng(7)
    fine_field = rng.normal(size=fine.center_shape)

    coarse_field = restrict_center_area_average(fine, coarse, fine_field)
    expected = fine_field.reshape(coarse.ny, 2, coarse.nx, 2).mean(axis=(1, 3))

    np.testing.assert_allclose(coarse_field, expected)
    fine_integral = np.sum(fine_field) * fine.cell_area
    coarse_integral = np.sum(coarse_field) * coarse.cell_area
    np.testing.assert_allclose(coarse_integral, fine_integral, atol=1e-12)


def test_u_face_restriction_averages_fine_segments_on_same_vertical_face():
    fine, coarse = nested_grids()
    x, y = fine.u_coordinates()
    fine_u = 2.0 + 0.01 * x - 0.02 * y

    coarse_u = restrict_u_face_length_average(fine, coarse, fine_u)
    expected = fine_u[:, ::2].reshape(coarse.ny, 2, coarse.nx + 1).mean(axis=1)

    assert coarse_u.shape == coarse.u_shape
    np.testing.assert_allclose(coarse_u, expected)


def test_v_face_restriction_averages_fine_segments_on_same_horizontal_face():
    fine, coarse = nested_grids()
    x, y = fine.v_coordinates()
    fine_v = -1.0 + 0.03 * x + 0.005 * y

    coarse_v = restrict_v_face_length_average(fine, coarse, fine_v)
    expected = fine_v[::2, :].reshape(coarse.ny + 1, coarse.nx, 2).mean(axis=2)

    assert coarse_v.shape == coarse.v_shape
    np.testing.assert_allclose(coarse_v, expected)
