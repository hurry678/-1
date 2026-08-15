import numpy as np

from problem2_core.config import ModelMode, Problem2Config
from problem2_core.state import CoupledState
from problem2_core.thickness_transport import total_ice_volume


def test_frozen_initial_condition_has_zero_velocities_and_exact_total_ice_volume():
    config = Problem2Config(nx=20, ny=10, mode=ModelMode.M0_FULL)
    state = CoupledState.initial(config)
    state.validate(config)

    grid = config.grid
    np.testing.assert_allclose(state.ice_u, 0.0)
    np.testing.assert_allclose(state.ice_v, 0.0)
    np.testing.assert_allclose(state.ocean_u, 0.0)
    np.testing.assert_allclose(state.ocean_v, 0.0)
    np.testing.assert_allclose(state.sea_surface, 0.0)
    np.testing.assert_allclose(np.mean(state.thickness), 0.5, atol=1e-14)
    np.testing.assert_allclose(
        total_ice_volume(grid, state.thickness), 1.0e8, rtol=0.0, atol=1e-6
    )
