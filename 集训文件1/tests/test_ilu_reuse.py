"""Q2-R3 P3：受保护 ILU 复用（安全缓存）单元与集成测试。"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from problem2_core.config import ModelMode, Problem2Config
from problem2_core.ice_momentum import IceMomentumSolver
from problem2_core.runner import run_simulation
from problem2_core.state import CoupledState


def _solver(*, reuse: bool, reuse_max: int = 2) -> IceMomentumSolver:
    config = Problem2Config(
        nx=20,
        ny=10,
        duration_hours=0.5,
        mode=ModelMode.M1_QUASI_STATIC,
        ilu_reuse_enabled=reuse,
        ilu_reuse_max=reuse_max,
    )
    return IceMomentumSolver(config)


def _matrix(nx: int = 20, ny: int = 10, scale: float = 1.0) -> sparse.csr_matrix:
    size = 2 * ny * (nx + 1) + 2 * (ny + 1) * nx
    diagonal = scale * np.ones(size)
    return sparse.diags(diagonal, format="csr")


def test_ilu_reuse_disabled_builds_on_every_picard_iteration():
    solver = _solver(reuse=False)
    config = solver.config
    state = CoupledState.initial(config)
    result = solver.solve(
        thickness=state.thickness,
        old_u=state.ice_u,
        old_v=state.ice_v,
        water_u=state.ocean_u,
        water_v=state.ocean_v,
        inertia_weight=config.ice_inertia_weight,
        max_picard_iterations=50,
        preconditioner=config.preconditioner_standard,
    )

    assert result.converged
    assert result.picard_iterations >= 2
    assert result.ilu_builds == result.picard_iterations
    assert result.ilu_reuses == 0
    assert result.ilu_rebuilds == 0
    assert all(record.cache_invalidation_reason == "cache_disabled" for record in result.records)
    assert all(record.ilu_builds == 1 and record.ilu_reuses == 0 for record in result.records)


def test_ilu_reuse_cache_is_per_coupling_round():
    solver = _solver(reuse=True)
    config = solver.config
    state = CoupledState.initial(config)
    first = solver.solve(
        thickness=state.thickness,
        old_u=state.ice_u,
        old_v=state.ice_v,
        water_u=state.ocean_u,
        water_v=state.ocean_v,
        inertia_weight=config.ice_inertia_weight,
        max_picard_iterations=50,
        preconditioner=config.preconditioner_standard,
    )
    second = solver.solve(
        thickness=state.thickness,
        old_u=state.ice_u,
        old_v=state.ice_v,
        water_u=state.ocean_u,
        water_v=state.ocean_v,
        inertia_weight=config.ice_inertia_weight,
        max_picard_iterations=50,
        preconditioner=config.preconditioner_standard,
    )

    assert first.converged and second.converged
    assert first.records[0].cache_invalidation_reason == "new_picard_round"
    assert second.records[0].cache_invalidation_reason == "new_picard_round"
    assert first.ilu_builds + first.ilu_reuses == first.picard_iterations
    assert second.ilu_builds + second.ilu_reuses == second.picard_iterations


def test_ilu_reuse_decision_rules():
    solver = _solver(reuse=True, reuse_max=2)
    matrix = _matrix()
    kind = "ilu"
    reference_op = object()
    cache: dict = {
        "ref_matrix": matrix,
        "ref_kind": kind,
        "ref_op": reference_op,
        "ref_gmres": 10,
        "consecutive_reuse": 0,
    }

    op, reason = solver._ilu_reuse_decision(matrix, kind, cache, previous_gmres_iterations=8)
    assert op is reference_op and reason == "reuse"

    op, reason = solver._ilu_reuse_decision(_matrix(scale=2.0), kind, cache, previous_gmres_iterations=8)
    assert op is None and reason == "matrix_drift_exceeded"

    op, reason = solver._ilu_reuse_decision(_matrix(nx=21), kind, cache, previous_gmres_iterations=8)
    assert op is None and reason == "shape_changed"

    op, reason = solver._ilu_reuse_decision(matrix, "robust_ilu", cache, previous_gmres_iterations=8)
    assert op is None and reason == "kind_changed"

    op, reason = solver._ilu_reuse_decision(matrix, kind, cache, previous_gmres_iterations=16)
    assert op is None and reason == "gmres_degraded"

    cache["consecutive_reuse"] = 2
    op, reason = solver._ilu_reuse_decision(matrix, kind, cache, previous_gmres_iterations=8)
    assert op is None and reason == "max_reuse_reached"

    solver_max_one = _solver(reuse=True, reuse_max=1)
    cache["consecutive_reuse"] = 1
    op, reason = solver_max_one._ilu_reuse_decision(
        matrix, kind, cache, previous_gmres_iterations=8
    )
    assert op is None and reason == "max_reuse_reached"

    op, reason = solver._ilu_reuse_decision(matrix, kind, {}, previous_gmres_iterations=8)
    assert op is None and reason == "new_picard_round"


def test_ilu_reuse_allows_small_pattern_change_when_matrix_drift_is_safe():
    """旧 ILU 可作为任意同形矩阵的预条件器，是否复用应由漂移与收敛决定。"""

    solver = _solver(reuse=True)
    matrix = _matrix()
    cache: dict = {
        "ref_matrix": matrix,
        "ref_kind": "ilu",
        "ref_op": object(),
        "ref_gmres": 5,
        "consecutive_reuse": 0,
    }
    different_pattern = (
        _matrix()
        + sparse.diags([1.0e-4], offsets=[1], shape=matrix.shape)
    ).tocsr()

    op, reason = solver._ilu_reuse_decision(different_pattern, "ilu", cache, previous_gmres_iterations=4)
    assert op is cache["ref_op"] and reason == "reuse"


def test_ilu_reuse_maximum_one_rebuilds_every_other_iteration():
    solver = _solver(reuse=True, reuse_max=1)
    config = solver.config
    state = CoupledState.initial(config)
    result = solver.solve(
        thickness=state.thickness,
        old_u=state.ice_u,
        old_v=state.ice_v,
        water_u=state.ocean_u,
        water_v=state.ocean_v,
        inertia_weight=config.ice_inertia_weight,
        max_picard_iterations=50,
        preconditioner=config.preconditioner_standard,
    )

    assert result.converged
    assert result.ilu_builds + result.ilu_reuses == result.picard_iterations
    assert result.ilu_rebuilds >= 1


def test_ilu_reuse_full_run_keeps_same_fixed_point_and_reports_counts():
    common = dict(
        nx=20,
        ny=10,
        duration_hours=0.5,
        mode=ModelMode.M1_QUASI_STATIC,
        temporal_predictor_enabled=True,
        outer_anderson_enabled=True,
        anderson_depth=3,
        anderson_damping=0.8,
    )
    disabled = run_simulation(
        Problem2Config(**common, ilu_reuse_enabled=False),
        "output/problem2/tests_ilu/disabled",
        snapshot_hours=(0.0, 0.5),
    )
    enabled = run_simulation(
        Problem2Config(**common, ilu_reuse_enabled=True, ilu_reuse_max=2),
        "output/problem2/tests_ilu/enabled",
        snapshot_hours=(0.0, 0.5),
    )

    assert disabled["status"] == "passed"
    assert enabled["status"] == "passed"
    assert enabled["total_ilu_builds"] > 0
    assert enabled["total_ilu_reuses"] > 0
    assert enabled["total_ilu_rebuilds"] >= 0
    assert (
        enabled["total_ilu_builds"] + enabled["total_ilu_reuses"]
        == enabled["all_attempts_total_picard_iterations"]
    )
    assert enabled["ilu_invalidation_reason_counts"].get("reuse", 0) == enabled["total_ilu_reuses"]
    assert enabled["checks"]["source_unchanged_during_run"] is True

    with np.load("output/problem2/tests_ilu/disabled/snapshots.npz") as left:
        left_thickness = left["thickness"][-1]
        left_ice_u = left["ice_u"][-1]
        left_ice_v = left["ice_v"][-1]
    with np.load("output/problem2/tests_ilu/enabled/snapshots.npz") as right:
        right_thickness = right["thickness"][-1]
        right_ice_u = right["ice_u"][-1]
        right_ice_v = right["ice_v"][-1]

    np.testing.assert_allclose(left_thickness, right_thickness, rtol=1.0e-6, atol=1.0e-9)
    np.testing.assert_allclose(left_ice_u, right_ice_u, rtol=1.0e-6, atol=1.0e-9)
    np.testing.assert_allclose(left_ice_v, right_ice_v, rtol=1.0e-6, atol=1.0e-9)
