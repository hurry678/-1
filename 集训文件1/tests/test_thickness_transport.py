import numpy as np

from problem2_core.grid import CGrid
from problem2_core.thickness_transport import (
    advect_thickness_upwind,
    total_ice_volume,
)


def test_closed_domain_upwind_transport_conserves_total_ice_volume():
    grid = CGrid(nx=12, ny=7, dx=100.0, dy=100.0)
    rng = np.random.default_rng(20260806)
    thickness = 0.2 + 0.6 * rng.random(grid.center_shape)
    u_face = np.full(grid.u_shape, 0.12)
    v_face = np.full(grid.v_shape, -0.04)
    u_face[:, (0, -1)] = 0.0
    v_face[(0, -1), :] = 0.0
    volume_before = total_ice_volume(grid, thickness)

    updated, diagnostics = advect_thickness_upwind(
        grid, thickness, u_face, v_face, dt=300.0, cfl_limit=0.8
    )

    volume_after = total_ice_volume(grid, updated)
    assert diagnostics.substeps >= 1
    assert diagnostics.boundary_flux == 0.0
    assert np.min(updated) >= 0.0
    np.testing.assert_allclose(volume_after, volume_before, rtol=0.0, atol=1e-8)


def test_ice_edge_can_advance_into_an_initially_ice_free_cell():
    grid = CGrid(nx=4, ny=1, dx=100.0, dy=100.0)
    thickness = np.array([[1.0, 0.0, 0.0, 0.0]])
    u_face = np.zeros(grid.u_shape)
    v_face = np.zeros(grid.v_shape)
    u_face[0, 1] = 0.1

    updated, _ = advect_thickness_upwind(
        grid, thickness, u_face, v_face, dt=100.0, cfl_limit=0.8
    )

    assert updated[0, 1] > 0.0
    assert updated[0, 0] < thickness[0, 0]
    np.testing.assert_allclose(
        total_ice_volume(grid, updated),
        total_ice_volume(grid, thickness),
        rtol=0.0,
        atol=1e-10,
    )
