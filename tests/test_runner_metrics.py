from types import SimpleNamespace
import importlib.util
import json
from pathlib import Path

import numpy as np

from problem2_core.config import ModelMode, Problem2Config
from problem2_core.runner import (
    FORMAL_MODEL_MODES,
    _history_row,
    _source_hashes,
    compute_model_error_history,
    compute_aligned_sensitivity,
    run_simulation,
    select_model,
    select_model_for_stage,
)
from problem2_core.state import CoupledState
from problem2_core.thickness_transport import total_ice_volume


def _synthetic_snapshots(reference: bool) -> dict[str, np.ndarray]:
    time = np.array([0.0, 300.0, 600.0])
    u = np.zeros((3, 1, 2))
    u[1, :, :] = 1.0 if reference else 2.0
    u[2, :, :] = 10.0 if reference else 11.0
    v = np.zeros((3, 2, 1))
    thickness = np.zeros((3, 1, 1))
    thickness[0, 0, 0] = 1.0
    thickness[1, 0, 0] = 1.0 if reference else 2.0
    thickness[2, 0, 0] = 10.0 if reference else 11.0
    return {
        "time_seconds": time,
        "time_hours": time / 3600.0,
        "thickness": thickness,
        "ice_u": u,
        "ice_v": v,
        "ocean_u": np.zeros_like(u),
        "ocean_v": np.zeros_like(v),
        "sea_surface": np.zeros_like(thickness),
    }


def test_v2_formal_driver_contains_only_m0_and_m1():
    assert FORMAL_MODEL_MODES == (ModelMode.M0_FULL, ModelMode.M1_QUASI_STATIC)


def test_phase1_source_hashes_cover_active_drivers_and_exclude_unrelated_q2_driver(
    tmp_path,
):
    (tmp_path / "problem2_core").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "problem2_core" / "runner.py").write_text("# core\n", encoding="utf-8")
    (tmp_path / "tests" / "test_runner.py").write_text("# test\n", encoding="utf-8")
    (tmp_path / "问题二_v2_数值验证.py").write_text("# active formal\n", encoding="utf-8")
    (tmp_path / "问题二_Q2_R2删项诊断.py").write_text("# unrelated q2r2\n", encoding="utf-8")
    (tmp_path / "问题二_M1严格诊断.py").write_text("# active phase1\n", encoding="utf-8")
    (tmp_path / "问题二_M1_Anderson第一轮.py").write_text(
        "# active phase2\n", encoding="utf-8"
    )
    (tmp_path / "问题二_运行基础测试.py").write_text("# test harness\n", encoding="utf-8")

    hashes = _source_hashes(tmp_path)

    assert "problem2_core/runner.py" in hashes
    assert "tests/test_runner.py" in hashes
    assert "问题二_v2_数值验证.py" in hashes
    assert "问题二_M1严格诊断.py" in hashes
    assert "问题二_M1_Anderson第一轮.py" in hashes
    assert "问题二_运行基础测试.py" in hashes
    assert "问题二_Q2_R2删项诊断.py" not in hashes


def test_base_test_report_uses_phase1_authoritative_source_hashes():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "问题二_运行基础测试.py"
    spec = importlib.util.spec_from_file_location("problem2_test_harness", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    report = module.build_report([], exit_code=0)

    assert report["source_sha256"] == _source_hashes(root)


def test_spatiotemporal_error_uses_frozen_ratio_of_energy_sums():
    history, aggregate = compute_model_error_history(
        _synthetic_snapshots(True),
        _synthetic_snapshots(False),
        h_min=0.0,
        epsilon=0.0,
        mode="M1",
    )

    expected = np.sqrt(2.0 / 101.0)
    assert len(history) == 2
    assert aggregate["velocity_spatiotemporal_error"] == expected
    assert aggregate["thickness_spatiotemporal_error"] == expected
    assert history.iloc[-1]["cumulative_velocity_spatiotemporal_error"] == expected
    assert history.iloc[-1]["maximum_center_speed_absolute_error_mps"] == 1.0
    assert history.iloc[-1]["maximum_thickness_absolute_error_m"] == 1.0
    assert aggregate["velocity_full_domain_spatiotemporal_error"] == expected
    assert aggregate["velocity_mass_weighted_spatiotemporal_error"] == np.sqrt(
        11.0 / 1001.0
    )


def test_model_selection_rejects_accurate_stable_model_without_speedup():
    selected = select_model(
        accuracy_passed={"M1": True},
        stability_passed={"M1": True},
        speedup_vs_m0={"M1": 0.8},
    )

    assert selected == "m1_accuracy_passed_no_efficiency_gain"


def test_v2_keeps_teacher_required_m1_when_accuracy_target_is_not_met():
    selected = select_model(
        accuracy_passed={"M1": False},
        stability_passed={"M1": True},
        speedup_vs_m0={"M1": 0.8},
    )

    assert selected == "m1_required_accuracy_target_not_met"


def test_model_selection_is_not_run_on_smoke_or_six_hour_regression():
    selected = select_model_for_stage(
        "smoke",
        accuracy_passed={"M1": True},
        stability_passed={"M1": True},
        speedup_vs_m0={"M1": 1.1},
    )

    assert selected == "not_applicable_non_24h_stage"


def test_history_records_both_v2_ice_weights_that_generated_the_state():
    config = Problem2Config(
        nx=4,
        ny=2,
        duration_hours=2.0,
        mode=ModelMode.M1_QUASI_STATIC,
    )
    state = CoupledState.initial(config)
    state.time_seconds = 3600.0
    diagnostics = SimpleNamespace(
        step_index=12,
        chi=0.0,
        chi_coriolis=0.0,
        ice_residual=0.0,
        gmres_residual=0.0,
        coupling_residual=0.0,
        water_residual=0.0,
        water_residual_xi=0.0,
        water_residual_u=0.0,
        water_residual_v=0.0,
        helmholtz_residual=0.0,
        drag_closure_error=0.0,
        drag_face_l1_closure_error=0.0,
        drag_object_reused=True,
        maximum_inactive_drag=0.0,
        thickness_cfl_macro=0.0,
        thickness_substeps=1,
    )

    row = _history_row(
        config,
        state,
        total_ice_volume(config.grid, state.thickness),
        diagnostics,
    )

    assert row["ice_inertia_weight"] == 0.0
    assert row["ice_coriolis_weight"] == 0.0


def test_run_simulation_writes_attempt_timing_and_force_budget_diagnostics(tmp_path):
    config = Problem2Config(
        nx=8,
        ny=4,
        dt=300.0,
        duration_hours=300.0 / 3600.0,
        mode=ModelMode.M1_QUASI_STATIC,
    )
    output = tmp_path / "diagnostic_run"

    summary = run_simulation(
        config,
        output,
        snapshot_hours=(0.0, config.duration_hours),
    )

    assert summary["status"] == "passed"
    for name in (
        "attempt_history.csv",
        "timing_history.csv",
        "force_budget_history.csv",
        "force_budget_arrays.npz",
    ):
        assert (output / name).exists()
    attempt_rows = np.genfromtxt(
        output / "attempt_history.csv", delimiter=",", names=True, encoding="utf-8-sig"
    )
    assert int(attempt_rows["coupling_iterations"]) >= 1
    assert float(attempt_rows["matrix_assembly_seconds"]) >= 0.0
    assert float(attempt_rows["preconditioner_build_seconds"]) >= 0.0
    assert float(attempt_rows["gmres_seconds"]) >= 0.0
    payload = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert payload["all_attempts_total_coupling_iterations"] >= 1
    assert payload["layered_timing_seconds"]["ice_solve"] >= 0.0
    assert payload["source_sha256_start"] == payload["source_sha256_end"]
    assert payload["checks"]["source_unchanged_during_run"] is True
    force_arrays = dict(np.load(output / "force_budget_arrays.npz"))
    assert "F_sigma_u" in force_arrays
    assert "F_sigma_center_u" in force_arrays
    assert "combined_missing_abs" in force_arrays
    assert "undirected_IC_abs_sum" in force_arrays
    assert "r_IC_corrected" in force_arrays
    assert force_arrays["F_sigma_u"].shape[0] == 1
    assert force_arrays["F_sigma_u"].shape[1:] == config.grid.u_shape
    assert force_arrays["F_sigma_center_u"].shape[1:] == config.grid.center_shape


def test_aligned_sensitivity_uses_only_common_physical_times():
    fine = _synthetic_snapshots(reference=True)
    fine["time_seconds"] = np.array([0.0, 150.0, 300.0])
    fine["time_hours"] = fine["time_seconds"] / 3600.0
    coarse = {key: value[[0, 2]] for key, value in fine.items()}

    sensitivity = compute_aligned_sensitivity(
        fine,
        coarse,
        h_min=1.0e-6,
        epsilon=1.0e-12,
    )

    assert sensitivity["common_time_count"] == 2
    assert sensitivity["terminal_velocity_relative_error"] == 0.0
    assert sensitivity["velocity_spatiotemporal_relative_error"] == 0.0
    assert sensitivity["terminal_thickness_relative_error"] == 0.0
    assert sensitivity["thickness_spatiotemporal_relative_error"] == 0.0
