"""Q2-R3 P4+线性求解器：BiCGSTAB 与内层受保护 Anderson 测试。"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import bicgstab, gmres

from problem2_core.config import ModelMode, Problem2Config
from problem2_core.ice_momentum import IceMomentumSolver
from problem2_core.runner import run_simulation
from problem2_core.state import CoupledState


def test_bicgstab_and_gmres_reach_same_linear_solution():
    rng = np.random.default_rng(7)
    size = 120
    dense = np.eye(size) * (3.0 + rng.uniform(-1, 1, size=size))
    for offset in (-1, 1):
        for i in range(max(0, -offset), size - max(0, offset)):
            dense[i, i + offset] = rng.uniform(-0.5, 0.5)
    matrix = sparse.csr_matrix(dense)
    rhs = rng.normal(size=size)
    solution_g, info_g = gmres(matrix, rhs, rtol=1.0e-10, atol=0.0, maxiter=200)
    solution_b, info_b = bicgstab(matrix, rhs, rtol=1.0e-10, atol=0.0, maxiter=200)
    assert info_g == 0 and info_b == 0
    np.testing.assert_allclose(solution_g, solution_b, rtol=1.0e-5, atol=1.0e-6)


def test_sparse_direct_lu_and_bicgstab_reach_same_m1_fixed_point():
    common = dict(
        nx=10,
        ny=5,
        duration_hours=0.5,
        mode=ModelMode.M1_QUASI_STATIC,
        picard_iterations_standard=100,
    )
    state = CoupledState.initial(Problem2Config(**common))
    results = {}
    for method in ("bicgstab", "splu"):
        config = Problem2Config(**common, linear_solver=method)
        results[method] = IceMomentumSolver(config).solve(
            thickness=state.thickness,
            old_u=state.ice_u,
            old_v=state.ice_v,
            water_u=state.ocean_u,
            water_v=state.ocean_v,
            inertia_weight=0.0,
            max_picard_iterations=100,
            preconditioner="robust_ilu",
        )
        assert results[method].converged
        assert results[method].nonlinear_residual <= config.ice_picard_tolerance

    np.testing.assert_allclose(
        results["splu"].u, results["bicgstab"].u, rtol=1.0e-5, atol=1.0e-8
    )
    np.testing.assert_allclose(
        results["splu"].v, results["bicgstab"].v, rtol=1.0e-5, atol=1.0e-8
    )


def test_inner_anderson_combine_recovers_linear_fixed_point():
    config = Problem2Config(nx=8, ny=4, duration_hours=0.5)
    solver = IceMomentumSolver(config)
    size = config.grid.u_shape[0] * config.grid.u_shape[1] + config.grid.v_shape[0] * config.grid.v_shape[1]
    active = np.ones(size, dtype=bool)
    y0 = np.zeros(size)
    y1 = np.ones(size) * 0.3
    y2 = np.ones(size) * 0.7
    f0 = np.full(size, 0.3)
    f1 = np.full(size, 0.4)
    f2 = np.full(size, 0.3)
    candidate = solver._inner_anderson_combine([y0, y1, y2], [f0, f1, f2], 3, active)
    assert np.all(np.isfinite(candidate))
    candidate = solver._inner_anderson_combine([y0], [f0], 3, active)
    np.testing.assert_array_equal(candidate, y0)
    # 秩亏时回退最近迭代
    fallback = solver._inner_anderson_combine([y0, y1], [f0, f0], 3, active)
    np.testing.assert_array_equal(fallback, y1)


def test_inner_anderson_and_bicgstab_small_run_passes_and_matches():
    common = dict(
        nx=20,
        ny=10,
        duration_hours=0.5,
        mode=ModelMode.M1_QUASI_STATIC,
        temporal_predictor_enabled=True,
        outer_anderson_enabled=True,
        anderson_depth=3,
        anderson_damping=0.8,
        ilu_reuse_enabled=True,
        ilu_reuse_max=2,
        picard_iterations_reset_aitken=300,
        picard_iterations_robust=500,
    )
    baseline = run_simulation(
        Problem2Config(**common),
        "output/problem2/tests_opt/baseline",
        snapshot_hours=(0.0, 0.5),
    )
    optimized = run_simulation(
        Problem2Config(
            **common,
            linear_solver="bicgstab",
            inner_anderson_enabled=True,
            inner_anderson_depth=3,
            inner_anderson_residual_rise_factor=1.2,
        ),
        "output/problem2/tests_opt/optimized",
        snapshot_hours=(0.0, 0.5),
    )
    assert baseline["status"] == "passed"
    assert optimized["status"] == "passed"
    with np.load("output/problem2/tests_opt/baseline/snapshots.npz") as left:
        left_thickness = left["thickness"][-1]
        left_ice_u = left["ice_u"][-1]
    with np.load("output/problem2/tests_opt/optimized/snapshots.npz") as right:
        right_thickness = right["thickness"][-1]
        right_ice_u = right["ice_u"][-1]
    np.testing.assert_allclose(left_thickness, right_thickness, rtol=2.0e-3, atol=1.0e-9)
    np.testing.assert_allclose(left_ice_u, right_ice_u, rtol=2.0e-3, atol=1.0e-9)


def test_inner_anderson_disabled_by_default():
    config = Problem2Config(nx=8, ny=4, duration_hours=0.5)
    payload = config.as_dict()
    assert payload["linear_solver"] == "gmres"
    assert payload["inner_anderson_enabled"] is False
