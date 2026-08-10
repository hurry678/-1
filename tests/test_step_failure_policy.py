import json

import numpy as np
import pytest

from problem2_core.config import Problem2Config
from problem2_core.failure import (
    PhysicalStepFailure,
    RetryPolicy,
    StepAttempt,
    execute_physical_step,
)


def test_failed_attempt_is_rolled_back_before_approved_retry(tmp_path):
    initial = {"thickness": np.array([[1.0]])}
    seen = []

    def solver(state, *, stage, target_residual):
        seen.append((stage.name, float(state["thickness"][0, 0]), target_residual))
        if stage.name == "standard":
            state["thickness"][0, 0] = 99.0
            return StepAttempt(
                converged=False,
                state=state,
                residual_history=[1.0, 0.2],
                iterations=stage.max_coupling_iterations,
                message="not converged",
            )
        state["thickness"][0, 0] = 2.0
        return StepAttempt(
            converged=True,
            state=state,
            residual_history=[1.0, target_residual * 0.5],
            iterations=12,
            message="converged",
        )

    policy = RetryPolicy.default(target_residual=1.0e-3)
    result = execute_physical_step(
        initial,
        solver,
        policy=policy,
        step_index=4,
        failure_directory=tmp_path,
        config={"dt": 300.0},
    )

    assert result.status == "success"
    assert [item[0] for item in seen] == ["standard", "reset_aitken"]
    assert [item[1] for item in seen] == [1.0, 1.0]
    assert all(item[2] == 1.0e-3 for item in seen)
    assert initial["thickness"][0, 0] == 1.0
    assert result.state["thickness"][0, 0] == 2.0


def test_exhausted_retries_abort_run_and_save_last_converged_state_and_logs(tmp_path):
    initial = {"thickness": np.array([[0.5, 0.6]]), "time": 900.0}

    def always_fail(state, *, stage, target_residual):
        state["thickness"][:] = -1.0
        return StepAttempt(
            converged=False,
            state=state,
            residual_history=[1.0, 0.3, 0.1],
            iterations=stage.max_coupling_iterations,
            message=f"{stage.name} failed",
        )

    policy = RetryPolicy.default(target_residual=3.0e-3)
    with pytest.raises(PhysicalStepFailure) as captured:
        execute_physical_step(
            initial,
            always_fail,
            policy=policy,
            step_index=7,
            failure_directory=tmp_path,
            config={"dt": 300.0, "model": "M0"},
        )

    failure = captured.value.result
    assert failure.status == "failed"
    assert len(failure.attempts) == len(policy.stages)
    np.testing.assert_allclose(initial["thickness"], [[0.5, 0.6]])

    report_path = tmp_path / "step_000007_failure.json"
    state_path = tmp_path / "step_000007_last_converged_state.npz"
    assert report_path.exists()
    assert state_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["target_residual"] == 3.0e-3
    assert all(item["target_residual"] == 3.0e-3 for item in report["attempts"])
    saved = np.load(state_path)
    np.testing.assert_allclose(saved["thickness"], [[0.5, 0.6]])


def test_config_serializes_the_actual_separate_retry_limits():
    payload = Problem2Config(nx=8, ny=4, duration_hours=0.5).as_dict()

    assert "coupling_max_iterations" not in payload
    assert payload["inner_picard_acceleration"] == "fixed_relaxation_only_no_aitken"
    assert payload["retry_policy"] == [
        {
            "name": "standard",
            "max_picard_iterations": 100,
            "max_coupling_iterations": 100,
            "outer_aitken_initial": 0.5,
            "preconditioner": "ilu",
        },
        {
            "name": "reset_aitken",
            "max_picard_iterations": 300,
            "max_coupling_iterations": 200,
            "outer_aitken_initial": 0.2,
            "preconditioner": "ilu",
        },
        {
            "name": "robust_preconditioner",
            "max_picard_iterations": 500,
            "max_coupling_iterations": 300,
            "outer_aitken_initial": 0.2,
            "preconditioner": "robust_ilu",
        },
    ]


def test_retry_records_preserve_diagnostics_for_failed_and_accepted_attempts(tmp_path):
    attempts = 0

    def solver(state, *, stage, target_residual):
        nonlocal attempts
        attempts += 1
        accepted = attempts == 2
        return StepAttempt(
            converged=accepted,
            state=state,
            residual_history=[target_residual * (0.5 if accepted else 2.0)],
            iterations=7 if accepted else 11,
            message="accepted" if accepted else "failed",
            diagnostics={
                "coupling_iterations": 7 if accepted else 11,
                "picard_iterations": 19 if accepted else 31,
                "gmres_iterations": 47 if accepted else 83,
                "timing_seconds": {"outer_coupling": 0.7 if accepted else 1.1},
            },
        )

    policy = RetryPolicy.default(target_residual=1.0e-5)
    result = execute_physical_step(
        {"thickness": np.ones((1, 1))},
        solver,
        policy=policy,
        step_index=3,
        failure_directory=tmp_path,
        config={},
    )

    assert [item.stage for item in result.attempts] == ["standard", "reset_aitken"]
    assert result.attempts[0].diagnostics["picard_iterations"] == 31
    assert result.attempts[0].diagnostics["gmres_iterations"] == 83
    assert result.attempts[1].diagnostics["timing_seconds"]["outer_coupling"] == 0.7
