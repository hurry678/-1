import numpy as np
import pandas as pd

from problem2_core.acceleration import ProtectedAnderson, build_temporal_initial_guess
from problem2_core.config import ModelMode, Problem2Config
from problem2_core.state import CoupledState
from problem2_core.solver import Problem2Solver
from problem2_core.runner import run_simulation


def test_phase2_acceleration_is_explicit_and_disabled_by_default():
    config = Problem2Config(nx=8, ny=4, duration_hours=0.5, mode=ModelMode.M1_QUASI_STATIC)

    payload = config.as_dict()

    assert payload["temporal_predictor_enabled"] is False
    assert payload["outer_anderson_enabled"] is False
    assert payload["outer_acceleration"] == "aitken"
    assert payload["ilu_reuse_enabled"] is False


def test_phase2_anderson_guard_parameters_are_serialized_when_enabled():
    config = Problem2Config(
        nx=8,
        ny=4,
        duration_hours=0.5,
        mode=ModelMode.M1_QUASI_STATIC,
        outer_anderson_enabled=True,
        anderson_depth=4,
        anderson_damping=0.8,
    )

    payload = config.as_dict()

    assert payload["outer_acceleration"] == "protected_anderson"
    assert payload["anderson_depth"] == 4
    assert payload["anderson_damping"] == 0.8


def test_temporal_predictor_extrapolates_only_the_initial_guess_and_enforces_masks():
    config = Problem2Config(
        nx=8,
        ny=4,
        duration_hours=0.5,
        mode=ModelMode.M1_QUASI_STATIC,
        temporal_predictor_enabled=True,
    )
    previous = CoupledState.initial(config)
    current = previous.copy()
    previous.time_seconds = 0.0
    current.time_seconds = config.dt
    current.ice_u[:] = 0.01
    current.ice_v[:] = -0.02
    current.ocean_u[:] = 0.03
    current.ocean_v[:] = -0.04
    current.sea_surface[:] = 0.005

    guess = build_temporal_initial_guess(config, current=current, previous=previous)

    assert guess.strategy == "two_state_linear_predictor"
    np.testing.assert_allclose(guess.ice_u[:, 1:-1], 0.02)
    np.testing.assert_allclose(guess.ice_v[1:-1, :], -0.04)
    np.testing.assert_allclose(guess.ocean_u[:, 1:-1], 0.06)
    np.testing.assert_allclose(guess.ocean_v[1:-1, :], -0.08)
    np.testing.assert_allclose(guess.sea_surface, 0.01)
    assert np.all(guess.ice_u[:, (0, -1)] == 0.0)
    assert np.all(guess.ice_v[(0, -1), :] == 0.0)


def test_temporal_predictor_falls_back_when_the_active_ice_mask_changed():
    config = Problem2Config(
        nx=8,
        ny=4,
        duration_hours=0.5,
        mode=ModelMode.M1_QUASI_STATIC,
        temporal_predictor_enabled=True,
    )
    previous = CoupledState.initial(config)
    current = previous.copy()
    current.time_seconds = config.dt
    current.thickness[0, 0] = 0.0
    current.ice_u[:] = 0.01

    guess = build_temporal_initial_guess(config, current=current, previous=previous)

    assert guess.strategy == "previous_accepted_state"
    assert guess.fallback_reason == "active_ice_mask_changed"
    np.testing.assert_array_equal(guess.ice_u, current.ice_u)


def test_solver_uses_temporal_prediction_only_after_two_states_are_accepted(tmp_path):
    config = Problem2Config(
        nx=8,
        ny=4,
        dt=300.0,
        duration_hours=0.5,
        mode=ModelMode.M1_QUASI_STATIC,
        temporal_predictor_enabled=True,
    )
    solver = Problem2Solver(config, failure_directory=tmp_path / "failures")

    first = solver.advance_one_physical_step(1)
    second = solver.advance_one_physical_step(2)

    assert first.diagnostics.initial_guess_strategy == "previous_accepted_state"
    assert second.diagnostics.initial_guess_strategy == "two_state_linear_predictor"
    assert second.diagnostics.ilu_reuses == 0


def test_restarted_solver_can_receive_the_previous_accepted_state_for_prediction(tmp_path):
    config = Problem2Config(
        nx=8,
        ny=4,
        dt=300.0,
        duration_hours=0.5,
        mode=ModelMode.M1_QUASI_STATIC,
        temporal_predictor_enabled=True,
    )
    previous = CoupledState.initial(config)
    current = previous.copy()
    current.time_seconds = config.dt
    solver = Problem2Solver(
        config,
        current,
        previous_accepted_state=previous,
        failure_directory=tmp_path / "failures",
    )

    result = solver.advance_one_physical_step(2)

    assert result.diagnostics.initial_guess_strategy == "two_state_linear_predictor"


def test_solver_applies_protected_anderson_only_to_outer_coupling(tmp_path):
    config = Problem2Config(
        nx=8,
        ny=4,
        dt=300.0,
        duration_hours=0.5,
        mode=ModelMode.M1_QUASI_STATIC,
        outer_anderson_enabled=True,
        anderson_depth=3,
        anderson_damping=0.8,
        anderson_step_ratio_limit=20.0,
    )
    solver = Problem2Solver(config, failure_directory=tmp_path / "failures")

    result = solver.advance_one_physical_step(1)

    assert result.diagnostics.outer_acceleration == "protected_anderson"
    assert any(
        record.outer_acceleration_method == "anderson"
        for record in result.diagnostics.coupling_records
    )
    assert result.diagnostics.ilu_reuses == 0
    assert all(record.ilu_reuses == 0 for record in result.diagnostics.coupling_records)


def test_explicitly_disabled_phase2_options_preserve_the_original_m1_path(tmp_path):
    common = dict(
        nx=8,
        ny=4,
        dt=300.0,
        duration_hours=0.5,
        mode=ModelMode.M1_QUASI_STATIC,
    )
    baseline = Problem2Solver(
        Problem2Config(**common), failure_directory=tmp_path / "baseline"
    )
    disabled = Problem2Solver(
        Problem2Config(
            **common,
            temporal_predictor_enabled=False,
            outer_anderson_enabled=False,
        ),
        failure_directory=tmp_path / "disabled",
    )

    baseline_step = baseline.advance_one_physical_step(1)
    disabled_step = disabled.advance_one_physical_step(1)

    for field in (
        "thickness",
        "ice_u",
        "ice_v",
        "ocean_u",
        "ocean_v",
        "sea_surface",
    ):
        np.testing.assert_array_equal(
            getattr(baseline_step.state, field), getattr(disabled_step.state, field)
        )
    assert (
        baseline_step.diagnostics.coupling_iterations
        == disabled_step.diagnostics.coupling_iterations
    )
    assert baseline_step.diagnostics.outer_acceleration == "aitken"
    assert disabled_step.diagnostics.outer_acceleration == "aitken"


def test_failed_attempt_clears_anderson_history_before_approved_retry(tmp_path):
    config = Problem2Config(
        nx=8,
        ny=4,
        dt=300.0,
        duration_hours=0.5,
        mode=ModelMode.M1_QUASI_STATIC,
        outer_anderson_enabled=True,
        anderson_step_ratio_limit=20.0,
        coupling_iterations_standard=1,
        coupling_iterations_reset_aitken=100,
        coupling_iterations_robust=100,
    )
    solver = Problem2Solver(config, failure_directory=tmp_path / "failures")

    result = solver.advance_one_physical_step(1)

    assert len(result.diagnostics.retry_attempts) >= 2
    first = result.diagnostics.retry_attempts[0]
    accepted = result.diagnostics.retry_attempts[-1]
    assert not first["accepted"]
    assert "failed_attempt_rollback" in first["diagnostics"][
        "acceleration_clear_reasons"
    ]
    assert accepted["accepted"]
    assert "failed_attempt_rollback" not in accepted["diagnostics"][
        "acceleration_clear_reasons"
    ]


def test_phase2_runner_logs_prediction_and_every_anderson_guard_event(tmp_path):
    config = Problem2Config(
        nx=8,
        ny=4,
        dt=300.0,
        duration_hours=600.0 / 3600.0,
        mode=ModelMode.M1_QUASI_STATIC,
        temporal_predictor_enabled=True,
        outer_anderson_enabled=True,
        anderson_step_ratio_limit=20.0,
    )
    output = tmp_path / "run"

    summary = run_simulation(config, output, snapshot_hours=(0.0,))
    residual = pd.read_csv(output / "residual_history.csv")
    attempts = pd.read_csv(output / "attempt_history.csv")

    assert summary["total_ilu_reuses"] == 0
    assert summary["total_anderson_updates"] > 0
    assert summary["total_acceleration_fallbacks"] >= 0
    assert summary["total_acceleration_history_clears"] >= 0
    assert summary["temporal_predictor_steps"] == 1
    assert "outer_acceleration_method" in residual
    assert "acceleration_fallback_reason" in residual
    assert "acceleration_history_cleared" in residual
    assert "acceleration_history_depth" in residual
    assert "initial_guess_fallback_reason" in attempts
    assert "acceleration_clear_reasons" in attempts


def test_protected_anderson_accelerates_a_contracting_public_fixed_point_map():
    matrix = np.array([[0.92, 0.03], [0.02, 0.88]])
    forcing = np.array([0.08, -0.04])
    accelerated = ProtectedAnderson(depth=4, damping=1.0, step_ratio_limit=50.0)
    x_anderson = np.zeros(2)
    x_relaxed = np.zeros(2)

    for _ in range(8):
        raw_anderson = matrix @ x_anderson + forcing
        residual = float(np.linalg.norm(raw_anderson - x_anderson))
        fallback = x_anderson + 0.5 * (raw_anderson - x_anderson)
        decision = accelerated.propose(
            current=x_anderson,
            raw=raw_anderson,
            residual=residual,
            fallback=fallback,
            mask_signature=np.array([True, True]),
        )
        x_anderson = decision.vector
        raw_relaxed = matrix @ x_relaxed + forcing
        x_relaxed = x_relaxed + 0.5 * (raw_relaxed - x_relaxed)

    fixed = np.linalg.solve(np.eye(2) - matrix, forcing)
    assert np.linalg.norm(x_anderson - fixed) < np.linalg.norm(x_relaxed - fixed)
    assert accelerated.accepted_steps > 0


def test_protected_anderson_clears_history_and_uses_aitken_when_residual_rises():
    accelerated = ProtectedAnderson(depth=3, residual_rise_factor=1.1)
    mask = np.array([True, True])
    first = accelerated.propose(
        current=np.zeros(2),
        raw=np.array([1.0, 0.0]),
        residual=1.0,
        fallback=np.array([0.5, 0.0]),
        mask_signature=mask,
    )
    second = accelerated.propose(
        current=first.vector,
        raw=np.array([2.0, 0.0]),
        residual=1.5,
        fallback=np.array([0.75, 0.0]),
        mask_signature=mask,
    )

    assert second.method == "aitken_fallback"
    assert second.fallback_reason == "coupling_residual_increased"
    assert second.history_cleared
    np.testing.assert_array_equal(second.vector, np.array([0.75, 0.0]))
    assert accelerated.clear_reasons[-1] == "coupling_residual_increased"


def test_protected_anderson_clears_history_when_active_ice_mask_changes():
    accelerated = ProtectedAnderson(depth=3)
    accelerated.propose(
        current=np.zeros(2),
        raw=np.array([0.2, -0.1]),
        residual=0.2,
        fallback=np.array([0.1, -0.05]),
        mask_signature=np.array([True, True]),
    )

    decision = accelerated.propose(
        current=np.array([0.1, -0.05]),
        raw=np.array([0.15, -0.08]),
        residual=0.05,
        fallback=np.array([0.125, -0.065]),
        mask_signature=np.array([True, False]),
    )

    assert decision.fallback_reason == "active_ice_mask_changed"
    assert decision.history_cleared
    assert accelerated.clear_reasons[-1] == "active_ice_mask_changed"


def test_protected_anderson_rejects_rank_deficient_least_squares_history():
    accelerated = ProtectedAnderson(depth=3)
    mask = np.array([True, True])
    accelerated.propose(
        current=np.array([0.0, 0.0]),
        raw=np.array([1.0, 0.0]),
        residual=1.0,
        fallback=np.array([0.5, 0.0]),
        mask_signature=mask,
    )
    accelerated.propose(
        current=np.array([0.5, 0.0]),
        raw=np.array([1.0, 0.0]),
        residual=0.5,
        fallback=np.array([0.75, 0.0]),
        mask_signature=mask,
    )

    decision = accelerated.propose(
        current=np.array([0.75, 0.0]),
        raw=np.array([1.0, 0.0]),
        residual=0.25,
        fallback=np.array([0.875, 0.0]),
        mask_signature=mask,
    )

    assert decision.method == "aitken_fallback"
    assert decision.fallback_reason == "ill_conditioned_or_rank_deficient"
    assert decision.history_cleared


def test_protected_anderson_rejects_nonfinite_values_and_records_retry_clear():
    accelerated = ProtectedAnderson(depth=3)
    decision = accelerated.propose(
        current=np.zeros(2),
        raw=np.array([np.nan, 0.0]),
        residual=np.nan,
        fallback=np.array([0.0, 0.0]),
        mask_signature=np.array([True, True]),
    )

    assert decision.fallback_reason == "nonfinite_value"
    assert decision.history_cleared
    accelerated.clear("failed_attempt_rollback")
    assert accelerated.history_depth == 0
    assert accelerated.clear_reasons[-1] == "failed_attempt_rollback"
