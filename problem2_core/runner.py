"""数值规格 v2 的 M0/M1 运行、逐步验收、对照与诊断输出。"""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import psutil
except ImportError:  # pragma: no cover - 仅影响可选峰值内存诊断
    psutil = None

from .config import ModelMode, Problem2Config
from .diagnostic_modes import DiagnosticRunSpec
from .failure import PhysicalStepFailure
from .solver import PhysicalStepDiagnostics, Problem2Solver
from .state import CoupledState
from .thickness_transport import total_ice_volume


FORMAL_MODEL_MODES = (ModelMode.M0_FULL, ModelMode.M1_QUASI_STATIC)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"无法序列化 {type(value)!r}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _center_point_thickness(thickness: np.ndarray) -> float:
    ny, nx = thickness.shape
    if nx % 2 == 0 and ny % 2 == 0:
        return float(np.mean(thickness[ny // 2 - 1 : ny // 2 + 1, nx // 2 - 1 : nx // 2 + 1]))
    return float(thickness[ny // 2, nx // 2])


def _history_row(
    config: Problem2Config,
    state: CoupledState,
    initial_volume: float,
    diagnostics: PhysicalStepDiagnostics | None,
    *,
    ice_inertia_weight: float | None = None,
    ice_coriolis_weight: float | None = None,
) -> dict[str, Any]:
    grid = config.grid
    ice_uc, ice_vc = grid.faces_to_center(state.ice_u, state.ice_v)
    water_uc, water_vc = grid.faces_to_center(state.ocean_u, state.ocean_v)
    volume = total_ice_volume(grid, state.thickness)
    return {
        "step": 0 if diagnostics is None else diagnostics.step_index,
        "time_seconds": state.time_seconds,
        "time_hours": state.time_seconds / 3600.0,
        "chi": (
            config.ice_inertia_weight
            if diagnostics is None and ice_inertia_weight is None
            else ice_inertia_weight
            if diagnostics is None
            else diagnostics.chi
        ),
        "ice_inertia_weight": (
            config.ice_inertia_weight
            if diagnostics is None and ice_inertia_weight is None
            else ice_inertia_weight
            if diagnostics is None
            else diagnostics.chi
        ),
        "ice_coriolis_weight": (
            config.ice_coriolis_weight
            if diagnostics is None and ice_coriolis_weight is None
            else ice_coriolis_weight
            if diagnostics is None
            else diagnostics.chi_coriolis
        ),
        "mean_thickness_m": float(np.mean(state.thickness)),
        "center_thickness_m": _center_point_thickness(state.thickness),
        "minimum_thickness_m": float(np.min(state.thickness)),
        "maximum_thickness_m": float(np.max(state.thickness)),
        "ice_volume_m3": volume,
        "ice_volume_relative_error": abs(volume - initial_volume) / initial_volume,
        "maximum_ice_speed_mps": float(np.max(np.hypot(ice_uc, ice_vc))),
        "maximum_ocean_speed_mps": float(np.max(np.hypot(water_uc, water_vc))),
        "mean_sea_surface_m": float(np.mean(state.sea_surface)),
        "ice_residual": np.nan if diagnostics is None else diagnostics.ice_residual,
        "gmres_residual": np.nan if diagnostics is None else diagnostics.gmres_residual,
        "coupling_residual": np.nan if diagnostics is None else diagnostics.coupling_residual,
        "water_residual": np.nan if diagnostics is None else diagnostics.water_residual,
        "water_residual_xi": np.nan if diagnostics is None else diagnostics.water_residual_xi,
        "water_residual_u": np.nan if diagnostics is None else diagnostics.water_residual_u,
        "water_residual_v": np.nan if diagnostics is None else diagnostics.water_residual_v,
        "helmholtz_residual": np.nan if diagnostics is None else diagnostics.helmholtz_residual,
        "drag_closure_error": np.nan if diagnostics is None else diagnostics.drag_closure_error,
        "drag_face_l1_closure_error": (
            np.nan if diagnostics is None else diagnostics.drag_face_l1_closure_error
        ),
        "drag_object_reused": True if diagnostics is None else diagnostics.drag_object_reused,
        "maximum_inactive_drag": 0.0 if diagnostics is None else diagnostics.maximum_inactive_drag,
        "thickness_cfl_macro": np.nan if diagnostics is None else diagnostics.thickness_cfl_macro,
        "thickness_substeps": 0 if diagnostics is None else diagnostics.thickness_substeps,
    }


def _snapshot_payload(states: list[CoupledState]) -> dict[str, np.ndarray]:
    return {
        "time_seconds": np.array([state.time_seconds for state in states], dtype=float),
        "time_hours": np.array([state.time_seconds / 3600.0 for state in states], dtype=float),
        "thickness": np.stack([state.thickness for state in states]),
        "ice_u": np.stack([state.ice_u for state in states]),
        "ice_v": np.stack([state.ice_v for state in states]),
        "ocean_u": np.stack([state.ocean_u for state in states]),
        "ocean_v": np.stack([state.ocean_v for state in states]),
        "sea_surface": np.stack([state.sea_surface for state in states]),
    }


def _plot_history(history: pd.DataFrame, config: Problem2Config, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    t = history["time_hours"]
    axes[0, 0].plot(t, history["mean_thickness_m"], label="mean h")
    axes[0, 0].plot(t, history["center_thickness_m"], label="center h")
    axes[0, 0].axhline(0.5, color="k", linestyle="--", linewidth=1)
    axes[0, 0].set_ylabel("Thickness (m)")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].semilogy(t, np.maximum(history["ice_volume_relative_error"], 1.0e-18))
    axes[0, 1].axhline(1.0e-6, color="r", linestyle="--", linewidth=1)
    axes[0, 1].set_ylabel("Relative ice-volume error")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(t, history["maximum_ice_speed_mps"], label="ice")
    axes[1, 0].plot(t, history["maximum_ocean_speed_mps"], label="water")
    axes[1, 0].set_xlabel("Time (h)")
    axes[1, 0].set_ylabel("Maximum speed (m/s)")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    for column, label in (
        ("ice_residual", "ice"),
        ("gmres_residual", "GMRES"),
        ("coupling_residual", "coupling"),
        ("water_residual", "water"),
    ):
        values = np.maximum(history[column].fillna(np.nan).to_numpy(dtype=float), 1.0e-18)
        axes[1, 1].semilogy(t, values, label=label)
    axes[1, 1].set_xlabel("Time (h)")
    axes[1, 1].set_ylabel("Residual")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    fig.suptitle(f"{config.mode.value} diagnostics ({config.nx}x{config.ny})")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_final_fields(state: CoupledState, config: Problem2Config, path: Path) -> None:
    grid = config.grid
    x, y = grid.center_coordinates()
    ice_uc, ice_vc = grid.faces_to_center(state.ice_u, state.ice_v)
    speed = np.hypot(ice_uc, ice_vc)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    image0 = axes[0].pcolormesh(x / 1000.0, y / 1000.0, state.thickness, shading="auto", cmap="viridis")
    fig.colorbar(image0, ax=axes[0], label="Ice thickness (m)")
    axes[0].set_title(f"h at {state.time_seconds / 3600.0:g} h")
    stride = max(1, min(config.nx, config.ny) // 10)
    image1 = axes[1].pcolormesh(x / 1000.0, y / 1000.0, speed, shading="auto", cmap="magma")
    axes[1].quiver(
        x[::stride, ::stride] / 1000.0,
        y[::stride, ::stride] / 1000.0,
        ice_uc[::stride, ::stride],
        ice_vc[::stride, ::stride],
        color="white",
        scale=None,
    )
    fig.colorbar(image1, ax=axes[1], label="Ice speed (m/s)")
    axes[1].set_title("C-grid center velocity")
    for axis in axes:
        axis.set_xlabel("x (km)")
        axis.set_ylabel("y (km)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_snapshots(states: list[CoupledState], config: Problem2Config, path: Path) -> None:
    grid = config.grid
    x, y = grid.center_coordinates()
    columns = min(3, len(states))
    rows = int(np.ceil(len(states) / columns))
    vmin = min(float(np.min(state.thickness)) for state in states)
    vmax = max(float(np.max(state.thickness)) for state in states)
    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.4 * rows), squeeze=False)
    last_image = None
    for axis, state in zip(axes.ravel(), states):
        last_image = axis.pcolormesh(
            x / 1000.0,
            y / 1000.0,
            state.thickness,
            shading="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_title(f"t={state.time_seconds / 3600.0:g} h")
        axis.set_xlabel("x (km)")
        axis.set_ylabel("y (km)")
    for axis in axes.ravel()[len(states) :]:
        axis.axis("off")
    if last_image is not None:
        fig.colorbar(last_image, ax=axes.ravel().tolist(), label="Ice thickness (m)", shrink=0.85)
    fig.suptitle(f"{config.mode.value} thickness snapshots")
    fig.subplots_adjust(top=0.88, wspace=0.25, hspace=0.35)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _source_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted((root / "problem2_core").glob("*.py")):
        hashes[str(path.relative_to(root)).replace("\\", "/")] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted((root / "tests").glob("test_*.py")):
        hashes[str(path.relative_to(root)).replace("\\", "/")] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in (
        "问题二_v2_数值验证.py",
        "问题二_M1严格诊断.py",
        "问题二_M1_Anderson第一轮.py",
        "问题二_运行基础测试.py",
    ):
        path = root / name
        if path.exists():
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def run_simulation(
    config: Problem2Config,
    output_directory: str | Path,
    *,
    snapshot_hours: Iterable[float],
    diagnostic_spec: DiagnosticRunSpec | None = None,
) -> dict[str, Any]:
    total_started = time.perf_counter()
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace = Path(__file__).resolve().parents[1]
    source_hashes_start = _source_hashes(workspace)
    effective_inertia = (
        config.ice_inertia_weight
        if diagnostic_spec is None
        else diagnostic_spec.inertia_weight
    )
    effective_coriolis = (
        config.ice_coriolis_weight
        if diagnostic_spec is None
        else diagnostic_spec.ice_coriolis_weight
    )
    mode_label = config.mode.value if diagnostic_spec is None else diagnostic_spec.mode.value
    config_payload = config.as_dict()
    config_payload.update(
        {
            "mode": mode_label,
            "base_formal_mode": config.mode.value,
            "ice_inertia_weight": effective_inertia,
            "ice_coriolis_weight": effective_coriolis,
            "diagnostic_only": bool(diagnostic_spec is not None),
            "eligible_for_model_selection": False if diagnostic_spec is not None else True,
        }
    )
    _write_json(output_dir / "config.json", config_payload)
    initial_state = CoupledState.initial(config)
    initial_volume = total_ice_volume(config.grid, initial_state.thickness)
    solver = Problem2Solver(
        config,
        initial_state,
        failure_directory=output_dir / "failures",
        diagnostic_spec=diagnostic_spec,
    )
    history_rows = [
        _history_row(
            config,
            initial_state,
            initial_volume,
            None,
            ice_inertia_weight=effective_inertia,
            ice_coriolis_weight=effective_coriolis,
        )
    ]
    residual_rows: list[dict[str, Any]] = []
    iteration_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    force_budget_rows: list[dict[str, Any]] = []
    force_budget_arrays: list[dict[str, np.ndarray]] = []
    snapshot_steps = {int(round(hour * 3600.0 / config.dt)) for hour in snapshot_hours}
    snapshot_steps.add(0)
    snapshot_steps.add(config.physical_steps)
    all_states = [initial_state.copy()]
    plot_snapshots = [initial_state.copy()]
    diagnostics_list: list[PhysicalStepDiagnostics] = []
    status = "passed"
    failure_message = ""
    solver_started = time.perf_counter()
    for step in range(1, config.physical_steps + 1):
        try:
            advance = solver.advance_one_physical_step(step)
        except PhysicalStepFailure as exc:
            status = "failed"
            failure_message = str(exc)
            break
        diagnostics = advance.diagnostics
        diagnostics_list.append(
            replace(
                diagnostics,
                coupling_records=(),
                residual_records=(),
                force_budget_face_arrays={},
                force_budget_center_arrays={},
            )
        )
        output_capture_started = time.perf_counter()
        history_rows.append(
            _history_row(config, advance.state, initial_volume, diagnostics)
        )
        for record in diagnostics.residual_records:
            residual_rows.append({"record_type": "ice_picard", **record})
        for record in diagnostics.coupling_records:
            residual_rows.append(
                {
                    "record_type": "coupling",
                    "step": step,
                    "stage": diagnostics.retry_stage,
                    "coupling_iteration": record.iteration,
                    "picard_iteration": 0,
                    "ice_residual": record.ice_residual,
                    "gmres_residual": record.gmres_residual,
                    "gmres_iterations": record.gmres_iterations,
                    "outer_aitken_factor_applied": record.outer_aitken_factor_applied,
                    "outer_acceleration_method": record.outer_acceleration_method,
                    "acceleration_fallback_reason": record.acceleration_fallback_reason,
                    "acceleration_history_cleared": record.acceleration_history_cleared,
                    "acceleration_history_depth": record.acceleration_history_depth,
                    "preconditioner": "",
                    "coupling_residual": record.coupling_residual,
                    "water_fixed_point_residual": record.water_fixed_point_residual,
                    "water_equation_residual": record.water_equation_residual,
                    "water_residual_xi": record.water_residual_xi,
                    "water_residual_u": record.water_residual_u,
                    "water_residual_v": record.water_residual_v,
                    "helmholtz_residual": record.helmholtz_residual,
                    "ocean_coriolis_iterations": record.ocean_coriolis_iterations,
                    "drag_object_reused": record.drag_object_reused,
                    "ice_solve_seconds": record.ice_solve_seconds,
                    "ocean_solve_seconds": record.ocean_solve_seconds,
                    "coupling_iteration_seconds": record.coupling_iteration_seconds,
                    "coupling_overhead_seconds": record.coupling_overhead_seconds,
                    "matrix_assembly_seconds": record.matrix_assembly_seconds,
                    "preconditioner_build_seconds": record.preconditioner_build_seconds,
                    "gmres_seconds": record.gmres_seconds,
                    "ice_residual_evaluation_seconds": record.ice_residual_evaluation_seconds,
                    "ilu_builds": record.ilu_builds,
                    "ilu_reuses": record.ilu_reuses,
                    "ilu_rebuilds": record.ilu_rebuilds,
                    "cache_invalidation_reason": record.cache_invalidation_reason,
                    "preconditioner_fallbacks": record.preconditioner_fallbacks,
                }
            )
        iteration_rows.append(
            {
                "step": step,
                "time_hours": advance.state.time_seconds / 3600.0,
                "chi": diagnostics.chi,
                "ice_inertia_weight": diagnostics.chi,
                "ice_coriolis_weight": diagnostics.chi_coriolis,
                "retry_stage": diagnostics.retry_stage,
                "retry_attempts": len(diagnostics.retry_attempts),
                "coupling_iterations": diagnostics.coupling_iterations,
                "picard_iterations": diagnostics.total_picard_iterations,
                "gmres_iterations": diagnostics.total_gmres_iterations,
                "thickness_substeps": diagnostics.thickness_substeps,
                "all_attempts_coupling_iterations": sum(
                    int(item["diagnostics"].get("coupling_iterations", 0))
                    for item in diagnostics.retry_attempts
                ),
                "all_attempts_picard_iterations": sum(
                    int(item["diagnostics"].get("picard_iterations", 0))
                    for item in diagnostics.retry_attempts
                ),
                "all_attempts_gmres_iterations": sum(
                    int(item["diagnostics"].get("gmres_iterations", 0))
                    for item in diagnostics.retry_attempts
                ),
            }
        )
        for attempt_index, attempt in enumerate(diagnostics.retry_attempts, start=1):
            detail = dict(attempt.get("diagnostics", {}))
            timing_detail = dict(detail.get("timing_seconds", {}))
            attempt_rows.append(
                {
                    "step": step,
                    "time_hours": advance.state.time_seconds / 3600.0,
                    "attempt_index": attempt_index,
                    "stage": attempt["stage"],
                    "accepted": bool(attempt["accepted"]),
                    "converged": bool(attempt["converged"]),
                    "message": attempt["message"],
                    "failure_reason": detail.get("failure_reason", ""),
                    "preconditioner": attempt["preconditioner"],
                    "outer_aitken_initial": attempt["outer_aitken_initial"],
                    "coupling_iterations": int(detail.get("coupling_iterations", 0)),
                    "picard_iterations": int(detail.get("picard_iterations", 0)),
                    "gmres_iterations": int(detail.get("gmres_iterations", 0)),
                    "ice_residual": detail.get("ice_residual", np.nan),
                    "gmres_residual": detail.get("gmres_residual", np.nan),
                    "coupling_residual": detail.get("coupling_residual", np.nan),
                    "water_residual": detail.get("water_residual", np.nan),
                    "helmholtz_residual": detail.get("helmholtz_residual", np.nan),
                    "attempt_total_seconds": timing_detail.get("attempt_total", 0.0),
                    "outer_coupling_seconds": timing_detail.get("outer_coupling", 0.0),
                    "coupling_overhead_seconds": timing_detail.get("coupling_overhead", 0.0),
                    "ice_solve_seconds": timing_detail.get("ice_solve", 0.0),
                    "matrix_assembly_seconds": timing_detail.get("matrix_assembly", 0.0),
                    "preconditioner_build_seconds": timing_detail.get(
                        "preconditioner_build", 0.0
                    ),
                    "gmres_seconds": timing_detail.get("gmres", 0.0),
                    "ice_residual_evaluation_seconds": timing_detail.get(
                        "ice_residual_evaluation", 0.0
                    ),
                    "ocean_solve_seconds": timing_detail.get("ocean_solve", 0.0),
                    "thickness_transport_seconds": timing_detail.get(
                        "thickness_transport", 0.0
                    ),
                    "force_budget_seconds": timing_detail.get("force_budget", 0.0),
                    "postprocess_seconds": timing_detail.get("postprocess", 0.0),
                    "ilu_builds": int(detail.get("ilu_builds", 0)),
                    "ilu_reuses": int(detail.get("ilu_reuses", 0)),
                    "ilu_rebuilds": int(detail.get("ilu_rebuilds", 0)),
                    "preconditioner_fallbacks": int(
                        detail.get("preconditioner_fallbacks", 0)
                    ),
                    "initial_guess_strategy": detail.get(
                        "initial_guess_strategy", "previous_accepted_state"
                    ),
                    "initial_guess_fallback_reason": detail.get(
                        "initial_guess_fallback_reason", ""
                    ),
                    "outer_acceleration": detail.get("outer_acceleration", "aitken"),
                    "acceleration_fallbacks": int(
                        detail.get("acceleration_fallbacks", 0)
                    ),
                    "acceleration_history_clears": int(
                        detail.get("acceleration_history_clears", 0)
                    ),
                    "acceleration_clear_reasons": "|".join(
                        detail.get("acceleration_clear_reasons", ())
                    ),
                    "cache_invalidation_reason": detail.get(
                        "cache_invalidation_reason", "no_cache_phase1"
                    ),
                    "active_center_count_before": int(
                        detail.get("active_center_count_before", 0)
                    ),
                    "active_center_count_after": int(
                        detail.get("active_center_count_after", 0)
                    ),
                    "active_center_mask_changes": int(
                        detail.get("active_center_mask_changes", 0)
                    ),
                    "active_u_mask_changes": int(detail.get("active_u_mask_changes", 0)),
                    "active_v_mask_changes": int(detail.get("active_v_mask_changes", 0)),
                }
            )
        force_row: dict[str, Any] = {
            "step": step,
            "time_seconds": advance.state.time_seconds,
            "time_hours": advance.state.time_seconds / 3600.0,
        }
        for metric, stats in diagnostics.force_budget_summary.items():
            for statistic, value in stats.items():
                force_row[f"{metric}_{statistic}"] = value
        force_budget_rows.append(force_row)
        force_budget_arrays.append(
            {
                **diagnostics.force_budget_face_arrays,
                **diagnostics.force_budget_center_arrays,
            }
        )
        all_states.append(advance.state.copy())
        if step in snapshot_steps:
            plot_snapshots.append(advance.state.copy())
        output_capture_seconds = time.perf_counter() - output_capture_started
        accepted_timing = dict(diagnostics.timing_seconds)
        timing_rows.append(
            {
                "step": step,
                "time_hours": advance.state.time_seconds / 3600.0,
                **{f"accepted_{key}_seconds": value for key, value in accepted_timing.items()},
                "output_capture_seconds": output_capture_seconds,
                "all_attempts_total_seconds": sum(
                    float(item["diagnostics"].get("timing_seconds", {}).get("attempt_total", 0.0))
                    for item in diagnostics.retry_attempts
                ),
            }
        )
    solver_elapsed = time.perf_counter() - solver_started

    history = pd.DataFrame(history_rows)
    residual_history = pd.DataFrame(residual_rows)
    iteration_counts = pd.DataFrame(iteration_rows)
    attempt_history = pd.DataFrame(attempt_rows)
    timing_history = pd.DataFrame(timing_rows)
    force_budget_history = pd.DataFrame(force_budget_rows)
    if diagnostic_spec is not None:
        for frame in (
            history,
            residual_history,
            iteration_counts,
            attempt_history,
            timing_history,
            force_budget_history,
        ):
            frame.insert(0, "eligible_for_model_selection", False)
            frame.insert(0, "diagnostic_only", True)
            frame.insert(0, "diagnostic_mode", mode_label)
    data_output_started = time.perf_counter()
    history.to_csv(output_dir / "history.csv", index=False, encoding="utf-8-sig")
    residual_history.to_csv(output_dir / "residual_history.csv", index=False, encoding="utf-8-sig")
    iteration_counts.to_csv(output_dir / "iteration_counts.csv", index=False, encoding="utf-8-sig")
    attempt_history.to_csv(output_dir / "attempt_history.csv", index=False, encoding="utf-8-sig")
    timing_history.to_csv(output_dir / "timing_history.csv", index=False, encoding="utf-8-sig")
    force_budget_history.to_csv(
        output_dir / "force_budget_history.csv", index=False, encoding="utf-8-sig"
    )
    snapshot_payload = _snapshot_payload(all_states)
    if diagnostic_spec is not None:
        snapshot_payload.update(
            {
                "diagnostic_mode": np.array(mode_label),
                "diagnostic_only": np.array(True),
                "eligible_for_model_selection": np.array(False),
            }
        )
    np.savez_compressed(output_dir / "snapshots.npz", **snapshot_payload)
    if force_budget_arrays:
        force_payload = {
            "step": np.arange(1, len(force_budget_arrays) + 1, dtype=int),
            "time_seconds": np.array(
                [item.time_end_seconds for item in diagnostics_list], dtype=float
            ),
        }
        if diagnostic_spec is not None:
            force_payload.update(
                {
                    "diagnostic_mode": np.array(mode_label),
                    "diagnostic_only": np.array(True),
                    "eligible_for_model_selection": np.array(False),
                }
            )
        for key in force_budget_arrays[0]:
            force_payload[key] = np.stack([item[key] for item in force_budget_arrays])
        np.savez_compressed(output_dir / "force_budget_arrays.npz", **force_payload)
    else:
        empty_force_payload: dict[str, np.ndarray] = {
            "step": np.array([], dtype=int),
            "time_seconds": np.array([], dtype=float),
        }
        if diagnostic_spec is not None:
            empty_force_payload.update(
                {
                    "diagnostic_mode": np.array(mode_label),
                    "diagnostic_only": np.array(True),
                    "eligible_for_model_selection": np.array(False),
                }
            )
        np.savez_compressed(
            output_dir / "force_budget_arrays.npz", **empty_force_payload
        )
    data_output_seconds = time.perf_counter() - data_output_started

    final_state = solver.state
    final_row = history.iloc[-1]
    retry_attempts = (
        iteration_counts["retry_attempts"].to_numpy(dtype=float)
        if not iteration_counts.empty
        else np.array([], dtype=float)
    )
    maximum_ice_volume_error = float(history["ice_volume_relative_error"].max())
    initial_mean_sea_surface = float(history.iloc[0]["mean_sea_surface_m"])
    maximum_mean_sea_surface_error = float(
        np.max(np.abs(history["mean_sea_surface_m"] - initial_mean_sea_surface))
    )
    cfl_values = history["thickness_cfl_macro"].dropna().to_numpy(dtype=float)
    maximum_thickness_cfl = float(np.max(cfl_values)) if cfl_values.size else 0.0
    all_states_finite = all(
        np.all(np.isfinite(state.thickness))
        and np.all(np.isfinite(state.ice_u))
        and np.all(np.isfinite(state.ice_v))
        and np.all(np.isfinite(state.ocean_u))
        and np.all(np.isfinite(state.ocean_v))
        and np.all(np.isfinite(state.sea_surface))
        for state in all_states
    )
    ice_boundary_flux_zero = all(
        np.all(state.ice_u[:, (0, -1)] == 0.0)
        and np.all(state.ice_v[(0, -1), :] == 0.0)
        for state in all_states
    )
    ocean_boundary_flux_zero = all(
        np.all(state.ocean_u[:, (0, -1)] == 0.0)
        and np.all(state.ocean_v[(0, -1), :] == 0.0)
        for state in all_states
    )
    checks = {
        "completed_all_physical_steps": len(diagnostics_list) == config.physical_steps,
        "no_nan_or_inf": bool(all_states_finite),
        "no_abnormal_speed": bool(
            history["maximum_ice_speed_mps"].max() <= config.maximum_speed
            and history["maximum_ocean_speed_mps"].max() <= config.maximum_speed
        ),
        "thickness_nonnegative": bool(history["minimum_thickness_m"].min() >= 0.0),
        "unconverged_physical_steps_zero": status == "passed",
        "mean_thickness_near_0p500000": bool(abs(final_row["mean_thickness_m"] - 0.5) <= 5.0e-7),
        "ice_volume_error_le_1e_6": maximum_ice_volume_error <= 1.0e-6,
        "mean_sea_surface_error_le_1e_10_m": maximum_mean_sea_surface_error <= 1.0e-10,
        "thickness_cfl_le_0p8": maximum_thickness_cfl <= config.thickness_cfl_limit,
        "ice_boundary_normal_flux_zero": bool(ice_boundary_flux_zero),
        "ocean_boundary_normal_flux_zero": bool(ocean_boundary_flux_zero),
        "drag_impulse_closed": bool(
            not diagnostics_list
            or max(item.drag_closure_error for item in diagnostics_list) <= 1.0e-12
        ),
        "drag_object_reused_all_steps": bool(
            not diagnostics_list
            or all(item.drag_object_reused for item in diagnostics_list)
        ),
        "inactive_drag_exact_zero": bool(
            not diagnostics_list
            or max(item.maximum_inactive_drag for item in diagnostics_list) == 0.0
        ),
        "picard_residual_frozen": bool(
            not diagnostics_list
            or max(item.ice_residual for item in diagnostics_list) <= config.ice_picard_tolerance
        ),
        "gmres_residual_frozen": bool(
            not diagnostics_list
            or max(item.gmres_residual for item in diagnostics_list) <= config.gmres_relative_tolerance
        ),
        "coupling_residual_frozen": bool(
            not diagnostics_list
            or max(item.coupling_residual for item in diagnostics_list) <= config.coupling_tolerance
        ),
        "water_equation_residual_frozen": bool(
            not diagnostics_list
            or max(item.water_residual for item in diagnostics_list) <= config.water_residual_tolerance
        ),
        "helmholtz_residual_frozen": bool(
            not diagnostics_list
            or max(item.helmholtz_residual for item in diagnostics_list) <= config.water_residual_tolerance
        ),
    }
    if not all(checks.values()):
        status = "failed"
        if not failure_message:
            failure_message = "one or more frozen acceptance checks failed"
    source_hashes_end = _source_hashes(workspace)
    checks["source_unchanged_during_run"] = source_hashes_start == source_hashes_end
    if not all(checks.values()):
        status = "failed"
        if not failure_message:
            failure_message = "one or more frozen acceptance checks failed"
    def numeric_sum(frame: pd.DataFrame, column: str) -> float:
        if frame.empty or column not in frame:
            return 0.0
        return float(frame[column].fillna(0.0).sum())

    layered_timing = {
        "attempt_total": numeric_sum(attempt_history, "attempt_total_seconds"),
        "outer_coupling": numeric_sum(attempt_history, "outer_coupling_seconds"),
        "coupling_overhead": numeric_sum(attempt_history, "coupling_overhead_seconds"),
        "ice_solve": numeric_sum(attempt_history, "ice_solve_seconds"),
        "matrix_assembly": numeric_sum(attempt_history, "matrix_assembly_seconds"),
        "preconditioner_build": numeric_sum(
            attempt_history, "preconditioner_build_seconds"
        ),
        "gmres": numeric_sum(attempt_history, "gmres_seconds"),
        "ice_residual_evaluation": numeric_sum(
            attempt_history, "ice_residual_evaluation_seconds"
        ),
        "ocean_solve": numeric_sum(attempt_history, "ocean_solve_seconds"),
        "thickness_transport": numeric_sum(
            attempt_history, "thickness_transport_seconds"
        ),
        "force_budget": numeric_sum(attempt_history, "force_budget_seconds"),
        "postprocess": numeric_sum(attempt_history, "postprocess_seconds"),
        "output_capture": numeric_sum(timing_history, "output_capture_seconds"),
        "data_output": data_output_seconds,
        "plot_output": 0.0,
        "total_output": 0.0,
    }
    aggregate_force_budget: dict[str, dict[str, float]] = {}
    if force_budget_arrays:
        active_values = [item["active_center"].astype(bool) for item in force_budget_arrays]
        for metric in (
            "F_I_abs",
            "F_C_abs",
            "F_sigma_abs",
            "F_D_abs",
            "F_W_abs",
            "combined_missing_abs",
            "undirected_IC_abs_sum",
            "retained_abs_sum",
            "retained_net_abs",
            "r_IC_corrected",
            "r_Sigma",
            "r_net_corrected",
            "deprecated_FI_plus_FC_abs",
            "r_IC_deprecated_FI_plus_FC",
            "r_net_deprecated_FI_plus_FC",
        ):
            values = np.concatenate(
                [item[metric][mask] for item, mask in zip(force_budget_arrays, active_values)]
            )
            aggregate_force_budget[metric] = {
                "median": float(np.median(values)) if values.size else 0.0,
                "p95": float(np.percentile(values, 95.0)) if values.size else 0.0,
                "maximum": float(np.max(values)) if values.size else 0.0,
            }

    coupling_method_counts: dict[str, int] = {}
    fallback_reason_counts: dict[str, int] = {}
    if not residual_history.empty and "outer_acceleration_method" in residual_history:
        coupling_rows = residual_history.loc[
            residual_history["record_type"] == "coupling"
        ]
        coupling_method_counts = {
            str(key): int(value)
            for key, value in coupling_rows["outer_acceleration_method"]
            .dropna()
            .value_counts()
            .items()
        }
        fallback_reasons = coupling_rows["acceleration_fallback_reason"].dropna()
        fallback_reasons = fallback_reasons.loc[fallback_reasons.astype(str) != ""]
        fallback_reason_counts = {
            str(key): int(value)
            for key, value in fallback_reasons.value_counts().items()
        }
    temporal_predictor_steps = 0
    if not attempt_history.empty and "initial_guess_strategy" in attempt_history:
        accepted_attempts = attempt_history.loc[attempt_history["accepted"].astype(bool)]
        temporal_predictor_steps = int(
            np.count_nonzero(
                accepted_attempts["initial_guess_strategy"]
                == "two_state_linear_predictor"
            )
        )

    summary = {
        "schema_version": 3,
        "equation_specification": "problem2_numerical_spec_v2",
        "status": status,
        "mode": mode_label,
        "base_formal_mode": config.mode.value,
        "ice_inertia_weight": effective_inertia,
        "ice_coriolis_weight": effective_coriolis,
        "ocean_coriolis_enabled": config.ocean_coriolis_enabled,
        "diagnostic_only": bool(diagnostic_spec is not None),
        "eligible_for_model_selection": False if diagnostic_spec is not None else True,
        "grid": {"nx": config.nx, "ny": config.ny, "dx": config.grid.dx, "dy": config.grid.dy},
        "duration_hours": config.duration_hours,
        "completed_steps": len(diagnostics_list),
        "planned_steps": config.physical_steps,
        "wall_clock_seconds": None,
        "solver_wall_clock_seconds": solver_elapsed,
        "mean_thickness_m": float(final_row["mean_thickness_m"]),
        "center_thickness_m": float(final_row["center_thickness_m"]),
        "ice_volume_m3": float(final_row["ice_volume_m3"]),
        "ice_volume_relative_error": maximum_ice_volume_error,
        "final_ice_volume_relative_error": float(final_row["ice_volume_relative_error"]),
        "minimum_thickness_m": float(history["minimum_thickness_m"].min()),
        "maximum_thickness_m": float(history["maximum_thickness_m"].max()),
        "maximum_ice_speed_mps": float(history["maximum_ice_speed_mps"].max()),
        "maximum_ocean_speed_mps": float(history["maximum_ocean_speed_mps"].max()),
        "mean_sea_surface_m": float(final_row["mean_sea_surface_m"]),
        "maximum_mean_sea_surface_error_m": maximum_mean_sea_surface_error,
        "maximum_thickness_cfl_macro": maximum_thickness_cfl,
        "maximum_ice_residual": max((item.ice_residual for item in diagnostics_list), default=None),
        "maximum_gmres_residual": max((item.gmres_residual for item in diagnostics_list), default=None),
        "maximum_coupling_residual": max((item.coupling_residual for item in diagnostics_list), default=None),
        "maximum_water_residual": max((item.water_residual for item in diagnostics_list), default=None),
        "maximum_water_residual_xi": max((item.water_residual_xi for item in diagnostics_list), default=None),
        "maximum_water_residual_u": max((item.water_residual_u for item in diagnostics_list), default=None),
        "maximum_water_residual_v": max((item.water_residual_v for item in diagnostics_list), default=None),
        "maximum_helmholtz_residual": max(
            (item.helmholtz_residual for item in diagnostics_list), default=None
        ),
        "maximum_drag_closure_error": max((item.drag_closure_error for item in diagnostics_list), default=None),
        "maximum_drag_face_l1_closure_error": max(
            (item.drag_face_l1_closure_error for item in diagnostics_list),
            default=None,
        ),
        "maximum_inactive_drag": max((item.maximum_inactive_drag for item in diagnostics_list), default=None),
        "total_picard_iterations": sum(item.total_picard_iterations for item in diagnostics_list),
        "total_gmres_iterations": sum(item.total_gmres_iterations for item in diagnostics_list),
        "total_coupling_iterations": sum(item.coupling_iterations for item in diagnostics_list),
        "all_attempts_total_picard_iterations": int(
            numeric_sum(attempt_history, "picard_iterations")
        ),
        "all_attempts_total_gmres_iterations": int(
            numeric_sum(attempt_history, "gmres_iterations")
        ),
        "all_attempts_total_coupling_iterations": int(
            numeric_sum(attempt_history, "coupling_iterations")
        ),
        "total_ilu_builds": int(numeric_sum(attempt_history, "ilu_builds")),
        "total_ilu_reuses": int(numeric_sum(attempt_history, "ilu_reuses")),
        "total_ilu_rebuilds": int(numeric_sum(attempt_history, "ilu_rebuilds")),
        "ilu_invalidation_reason_counts": (
            {
                str(key): int(value)
                for key, value in residual_history.loc[
                    residual_history["record_type"] == "ice_picard",
                    "cache_invalidation_reason",
                ]
                .fillna("")
                .astype(str)
                .value_counts()
                .items()
            }
            if not residual_history.empty and "cache_invalidation_reason" in residual_history
            else {}
        ),
        "total_preconditioner_fallbacks": int(
            numeric_sum(attempt_history, "preconditioner_fallbacks")
        ),
        "temporal_predictor_enabled": config.temporal_predictor_enabled,
        "outer_anderson_enabled": config.outer_anderson_enabled,
        "temporal_predictor_steps": temporal_predictor_steps,
        "total_anderson_updates": coupling_method_counts.get("anderson", 0),
        "total_acceleration_fallbacks": int(
            numeric_sum(attempt_history, "acceleration_fallbacks")
        ),
        "total_acceleration_history_clears": int(
            numeric_sum(attempt_history, "acceleration_history_clears")
        ),
        "outer_acceleration_method_counts": coupling_method_counts,
        "acceleration_fallback_reason_counts": fallback_reason_counts,
        "layered_timing_seconds": layered_timing,
        "force_budget_aggregate": aggregate_force_budget,
        "maximum_picard_iterations_per_step": (
            int(iteration_counts["picard_iterations"].max())
            if not iteration_counts.empty
            else None
        ),
        "maximum_gmres_iterations_per_step": (
            int(iteration_counts["gmres_iterations"].max())
            if not iteration_counts.empty
            else None
        ),
        "maximum_coupling_iterations_per_step": (
            int(iteration_counts["coupling_iterations"].max())
            if not iteration_counts.empty
            else None
        ),
        "maximum_ocean_coriolis_iterations_per_step": max(
            (item.ocean_coriolis_iterations for item in diagnostics_list),
            default=None,
        ),
        "steps_using_retry": int(np.count_nonzero(retry_attempts > 1.0)),
        "extra_retry_attempts": int(
            np.sum(np.maximum(retry_attempts - 1.0, 0.0))
        ),
        "failure_message": failure_message,
        "checks": checks,
        "generated_at": datetime.now().astimezone().isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "source_sha256": source_hashes_end,
        "source_sha256_start": source_hashes_start,
        "source_sha256_end": source_hashes_end,
    }
    report = {
        "suite": (
            "problem2_v2_1_deletion_diagnostics"
            if diagnostic_spec is not None
            else "problem2_v2_engineering_validation"
        ),
        "status": status,
        "mode": mode_label,
        "diagnostic_only": bool(diagnostic_spec is not None),
        "eligible_for_model_selection": False if diagnostic_spec is not None else True,
        "checks": checks,
        "failure_message": failure_message,
        "forbidden_actions": {
            "ran_100x50": False,
            "ran_200x100": False,
            "searched_t_sw": False,
            "optimized_cd_or_alpha": False,
            "implemented_m3": False,
            "implemented_mevp": False,
            "implemented_muscl": False,
        },
    }
    plot_output_started = time.perf_counter()
    _plot_history(history, config, output_dir / "diagnostic_history.png")
    _plot_final_fields(final_state, config, output_dir / "final_fields.png")
    _plot_snapshots(plot_snapshots, config, output_dir / "thickness_snapshots.png")
    layered_timing["plot_output"] = time.perf_counter() - plot_output_started
    layered_timing["total_output"] = (
        layered_timing["output_capture"]
        + layered_timing["data_output"]
        + layered_timing["plot_output"]
    )
    peak_process_memory = None
    if psutil is not None:
        memory_info = psutil.Process().memory_info()
        peak_process_memory = getattr(memory_info, "peak_wset", None)
    summary["peak_process_memory_bytes"] = (
        None if peak_process_memory is None else int(peak_process_memory)
    )
    summary["wall_clock_seconds"] = time.perf_counter() - total_started
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "test_report.json", report)
    return summary


def _center_velocity(snapshot: dict[str, np.ndarray], index: int) -> tuple[np.ndarray, np.ndarray]:
    u = snapshot["ice_u"][index]
    v = snapshot["ice_v"][index]
    return 0.5 * (u[:, :-1] + u[:, 1:]), 0.5 * (v[:-1, :] + v[1:, :])


def compute_model_error_history(
    reference: dict[str, np.ndarray],
    current: dict[str, np.ndarray],
    *,
    h_min: float,
    epsilon: float = 1.0e-12,
    mode: str = "",
) -> tuple[pd.DataFrame, dict[str, float]]:
    """按冻结平方和比值，在每个物理步累计 M0 参照误差。"""

    if reference["time_seconds"].shape != current["time_seconds"].shape or not np.allclose(
        reference["time_seconds"], current["time_seconds"]
    ):
        raise ValueError("模型误差要求逐物理步时间集合完全一致")
    cumulative_u_num = cumulative_u_den = 0.0
    cumulative_h_num = cumulative_h_den = 0.0
    cumulative_full_u_num = cumulative_full_u_den = 0.0
    cumulative_mass_u_num = cumulative_mass_u_den = 0.0
    rows: list[dict[str, Any]] = []
    for index in range(1, len(reference["time_seconds"])):
        ref_u, ref_v = _center_velocity(reference, index)
        cur_u, cur_v = _center_velocity(current, index)
        ref_h = reference["thickness"][index]
        cur_h = current["thickness"][index]
        ref_mask = ref_h >= h_min
        cur_mask = cur_h >= h_min
        intersection = ref_mask & cur_mask
        union = ref_mask | cur_mask
        du = cur_u - ref_u
        dv = cur_v - ref_v
        u_num = float(np.sum(du[intersection] ** 2 + dv[intersection] ** 2))
        u_den = float(np.sum(ref_u[intersection] ** 2 + ref_v[intersection] ** 2))
        h_num = float(np.sum((cur_h - ref_h)[union] ** 2))
        h_den = float(np.sum(ref_h[union] ** 2))
        full_u_num = float(np.sum(du**2 + dv**2))
        full_u_den = float(np.sum(ref_u**2 + ref_v**2))
        mass_u_num = float(np.sum(ref_h * (du**2 + dv**2)))
        mass_u_den = float(np.sum(ref_h * (ref_u**2 + ref_v**2)))
        cumulative_u_num += u_num
        cumulative_u_den += u_den
        cumulative_h_num += h_num
        cumulative_h_den += h_den
        cumulative_full_u_num += full_u_num
        cumulative_full_u_den += full_u_den
        cumulative_mass_u_num += mass_u_num
        cumulative_mass_u_den += mass_u_den
        union_count = int(np.count_nonzero(union))
        ref_count = int(np.count_nonzero(ref_mask))
        rows.append(
            {
                "mode": mode,
                "step": index,
                "time_seconds": float(reference["time_seconds"][index]),
                "time_hours": float(reference["time_seconds"][index] / 3600.0),
                "velocity_error_energy": u_num,
                "velocity_reference_energy": u_den,
                "thickness_error_energy": h_num,
                "thickness_reference_energy": h_den,
                "velocity_relative_error": float(
                    np.sqrt(u_num) / (np.sqrt(u_den) + epsilon)
                ),
                "thickness_relative_error": float(
                    np.sqrt(h_num) / (np.sqrt(h_den) + epsilon)
                ),
                "velocity_full_domain_error": float(
                    np.sqrt(full_u_num) / (np.sqrt(full_u_den) + epsilon)
                ),
                "velocity_mass_weighted_error": float(
                    np.sqrt(mass_u_num / (mass_u_den + epsilon))
                ),
                "maximum_center_speed_absolute_error_mps": float(
                    np.max(np.sqrt(du**2 + dv**2))
                ),
                "maximum_thickness_absolute_error_m": float(
                    np.max(np.abs(cur_h - ref_h))
                ),
                "cumulative_velocity_spatiotemporal_error": float(
                    np.sqrt(cumulative_u_num / (cumulative_u_den + epsilon))
                ),
                "cumulative_thickness_spatiotemporal_error": float(
                    np.sqrt(cumulative_h_num / (cumulative_h_den + epsilon))
                ),
                "cumulative_velocity_full_domain_spatiotemporal_error": float(
                    np.sqrt(
                        cumulative_full_u_num
                        / (cumulative_full_u_den + epsilon)
                    )
                ),
                "cumulative_velocity_mass_weighted_spatiotemporal_error": float(
                    np.sqrt(
                        cumulative_mass_u_num
                        / (cumulative_mass_u_den + epsilon)
                    )
                ),
                "ice_area_relative_error": abs(
                    int(np.count_nonzero(cur_mask)) - ref_count
                )
                / max(ref_count, 1),
                "ice_jaccard": (
                    float(np.count_nonzero(intersection) / union_count)
                    if union_count
                    else 1.0
                ),
            }
        )
    history = pd.DataFrame(rows)
    aggregate = {
        "velocity_spatiotemporal_error": float(
            np.sqrt(cumulative_u_num / (cumulative_u_den + epsilon))
        ),
        "thickness_spatiotemporal_error": float(
            np.sqrt(cumulative_h_num / (cumulative_h_den + epsilon))
        ),
        "velocity_full_domain_spatiotemporal_error": float(
            np.sqrt(cumulative_full_u_num / (cumulative_full_u_den + epsilon))
        ),
        "velocity_mass_weighted_spatiotemporal_error": float(
            np.sqrt(cumulative_mass_u_num / (cumulative_mass_u_den + epsilon))
        ),
    }
    return history, aggregate


def compute_aligned_sensitivity(
    reference: dict[str, np.ndarray],
    current: dict[str, np.ndarray],
    *,
    h_min: float,
    epsilon: float = 1.0e-12,
) -> dict[str, Any]:
    """在两组不同时间步结果的共同物理时刻使用冻结误差定义。"""

    reference_times = np.asarray(reference["time_seconds"], dtype=float)
    current_times = np.asarray(current["time_seconds"], dtype=float)
    reference_lookup = {round(float(value), 9): index for index, value in enumerate(reference_times)}
    current_lookup = {round(float(value), 9): index for index, value in enumerate(current_times)}
    common_keys = sorted(set(reference_lookup) & set(current_lookup))
    if len(common_keys) < 2:
        raise ValueError("同模型敏感性至少需要初值和一个共同非零物理时刻")
    reference_indices = np.array([reference_lookup[key] for key in common_keys], dtype=int)
    current_indices = np.array([current_lookup[key] for key in common_keys], dtype=int)

    def subset(payload: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        source_length = len(payload["time_seconds"])
        for key, values in payload.items():
            array = np.asarray(values)
            result[key] = array[indices] if array.ndim >= 1 and len(array) == source_length else array
        return result

    aligned_reference = subset(reference, reference_indices)
    aligned_current = subset(current, current_indices)
    history, aggregate = compute_model_error_history(
        aligned_reference,
        aligned_current,
        h_min=h_min,
        epsilon=epsilon,
        mode="sensitivity",
    )
    terminal = history.iloc[-1]
    return {
        "common_time_count": len(common_keys),
        "common_time_seconds": [float(value) for value in common_keys],
        "terminal_velocity_relative_error": float(terminal["velocity_relative_error"]),
        "velocity_spatiotemporal_relative_error": float(
            aggregate["velocity_spatiotemporal_error"]
        ),
        "terminal_thickness_relative_error": float(
            terminal["thickness_relative_error"]
        ),
        "thickness_spatiotemporal_relative_error": float(
            aggregate["thickness_spatiotemporal_error"]
        ),
    }


def select_model(
    *,
    accuracy_passed: dict[str, bool],
    stability_passed: dict[str, bool],
    speedup_vs_m0: dict[str, float],
) -> str:
    """按 v2 口径报告 M1 状态；精度不足时仍保留教师指定方程。"""

    stable = stability_passed.get("M1", False)
    accurate = accuracy_passed.get("M1", False)
    speedup = speedup_vs_m0.get("M1", 0.0)
    if not stable:
        return "m1_engineering_validation_failed"
    if accurate and speedup > 1.0:
        return "M1"
    if accurate:
        return "m1_accuracy_passed_no_efficiency_gain"
    return "m1_required_accuracy_target_not_met"


def select_model_for_stage(
    stage_name: str,
    *,
    accuracy_passed: dict[str, bool],
    stability_passed: dict[str, bool],
    speedup_vs_m0: dict[str, float],
) -> str:
    """只有 40×20、24 h 工程阶段执行结构选择。"""

    if stage_name != "forward_40x20_24h":
        return "not_applicable_non_24h_stage"
    return select_model(
        accuracy_passed=accuracy_passed,
        stability_passed=stability_passed,
        speedup_vs_m0=speedup_vs_m0,
    )


def write_model_comparison(directory: str | Path) -> dict[str, Any]:
    root = Path(directory)
    modes = ("M0", "M1")
    summaries = {mode: json.loads((root / mode / "summary.json").read_text(encoding="utf-8")) for mode in modes}
    configs = {mode: json.loads((root / mode / "config.json").read_text(encoding="utf-8")) for mode in modes}
    snapshots = {mode: dict(np.load(root / mode / "snapshots.npz")) for mode in modes}
    reference = snapshots["M0"]
    rows: list[dict[str, Any]] = []
    error_histories: list[pd.DataFrame] = []
    segment_contributions: dict[str, dict[str, dict[str, float]]] = {}
    for mode in modes:
        current = snapshots[mode]
        h_min = float(configs[mode]["h_min"])
        error_history, aggregate = compute_model_error_history(
            reference,
            current,
            h_min=h_min,
            epsilon=float(configs[mode]["norm_epsilon"]),
            mode=mode,
        )
        error_histories.append(error_history)
        terminal = error_history.iloc[-1]
        total_u_error_energy = float(error_history["velocity_error_energy"].sum())
        total_h_error_energy = float(error_history["thickness_error_energy"].sum())
        startup_mask = error_history["time_seconds"] <= 2.0 * 3600.0
        later_mask = error_history["time_seconds"] > 2.0 * 3600.0

        def contribution(mask: pd.Series, column: str, total: float) -> float:
            return float(error_history.loc[mask, column].sum() / total) if total > 0.0 else 0.0

        mode_segments: dict[str, dict[str, float]] = {
            "startup_0_2h": {
                "velocity_error_energy_fraction": contribution(
                    startup_mask, "velocity_error_energy", total_u_error_energy
                ),
                "thickness_error_energy_fraction": contribution(
                    startup_mask, "thickness_error_energy", total_h_error_energy
                ),
            },
            "later_after_2h": {
                "velocity_error_energy_fraction": contribution(
                    later_mask, "velocity_error_energy", total_u_error_energy
                ),
                "thickness_error_energy_fraction": contribution(
                    later_mask, "thickness_error_energy", total_h_error_energy
                ),
            },
        }
        segment_contributions[mode] = mode_segments
        rows.append(
            {
                "mode": mode,
                "status": summaries[mode]["status"],
                "terminal_velocity_relative_error": float(terminal["velocity_relative_error"]),
                "terminal_velocity_full_domain_error": float(terminal["velocity_full_domain_error"]),
                "terminal_velocity_mass_weighted_error": float(terminal["velocity_mass_weighted_error"]),
                "terminal_thickness_relative_error": float(terminal["thickness_relative_error"]),
                "velocity_spatiotemporal_relative_error": aggregate["velocity_spatiotemporal_error"],
                "thickness_spatiotemporal_relative_error": aggregate["thickness_spatiotemporal_error"],
                "velocity_full_domain_spatiotemporal_error": aggregate[
                    "velocity_full_domain_spatiotemporal_error"
                ],
                "velocity_mass_weighted_spatiotemporal_error": aggregate[
                    "velocity_mass_weighted_spatiotemporal_error"
                ],
                "terminal_maximum_center_speed_absolute_error_mps": float(
                    terminal["maximum_center_speed_absolute_error_mps"]
                ),
                "terminal_maximum_thickness_absolute_error_m": float(
                    terminal["maximum_thickness_absolute_error_m"]
                ),
                "terminal_ice_area_relative_error": float(terminal["ice_area_relative_error"]),
                "terminal_ice_jaccard": float(terminal["ice_jaccard"]),
                "minimum_ice_jaccard_over_time": float(
                    error_history["ice_jaccard"].min()
                ),
                "wall_clock_seconds": summaries[mode]["wall_clock_seconds"],
                "speedup_vs_m0": summaries["M0"]["wall_clock_seconds"] / max(summaries[mode]["wall_clock_seconds"], 1.0e-12),
                "total_picard_iterations": summaries[mode]["total_picard_iterations"],
                "total_gmres_iterations": summaries[mode]["total_gmres_iterations"],
                "total_coupling_iterations": summaries[mode]["total_coupling_iterations"],
                "maximum_picard_iterations_per_step": summaries[mode][
                    "maximum_picard_iterations_per_step"
                ],
                "maximum_gmres_iterations_per_step": summaries[mode][
                    "maximum_gmres_iterations_per_step"
                ],
                "maximum_coupling_iterations_per_step": summaries[mode][
                    "maximum_coupling_iterations_per_step"
                ],
                "steps_using_retry": summaries[mode]["steps_using_retry"],
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(root / "model_comparison.csv", index=False, encoding="utf-8-sig")
    combined_error_history = pd.concat(error_histories, ignore_index=True)
    combined_error_history.to_csv(
        root / "model_error_history.csv", index=False, encoding="utf-8-sig"
    )
    histories = []
    residuals = []
    iterations = []
    aggregate_snapshots: dict[str, np.ndarray] = {}
    for mode in modes:
        mode_history = pd.read_csv(root / mode / "history.csv")
        mode_history.insert(0, "mode", mode)
        histories.append(mode_history)
        mode_residual = pd.read_csv(root / mode / "residual_history.csv")
        mode_residual.insert(0, "mode", mode)
        residuals.append(mode_residual)
        mode_iterations = pd.read_csv(root / mode / "iteration_counts.csv")
        mode_iterations.insert(0, "mode", mode)
        iterations.append(mode_iterations)
        for key, value in snapshots[mode].items():
            aggregate_snapshots[f"{mode}_{key}"] = value
    pd.concat(histories, ignore_index=True).to_csv(
        root / "history.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(residuals, ignore_index=True).to_csv(
        root / "residual_history.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(iterations, ignore_index=True).to_csv(
        root / "iteration_counts.csv", index=False, encoding="utf-8-sig"
    )
    np.savez_compressed(root / "snapshots.npz", **aggregate_snapshots)
    _write_json(
        root / "config.json",
        {
            "schema_version": 3,
            "equation_specification": "problem2_numerical_spec_v2",
            "stage": root.name,
            "shared_numerical_path": True,
            "only_mode_difference": "ice inertia and ice Coriolis weights",
            "modes": configs,
        },
    )
    accuracy = {
        row["mode"]: {
            "terminal_velocity_le_5pct": row["terminal_velocity_relative_error"] <= 0.05,
            "terminal_thickness_le_5pct": row["terminal_thickness_relative_error"] <= 0.05,
            "spatiotemporal_velocity_le_5pct": row["velocity_spatiotemporal_relative_error"] <= 0.05,
            "spatiotemporal_thickness_le_5pct": row["thickness_spatiotemporal_relative_error"] <= 0.05,
        }
        for row in rows
        if row["mode"] != "M0"
    }
    accuracy_pass = {
        mode: all(values.values()) for mode, values in accuracy.items()
    }
    stability_pass = {
        mode: bool(
            summaries[mode]["status"] == "passed"
            and all(summaries[mode]["checks"].values())
        )
        for mode in ("M1",)
    }
    speedups = {
        row["mode"]: float(row["speedup_vs_m0"])
        for row in rows
        if row["mode"] != "M0"
    }
    efficiency_pass = {mode: value > 1.0 for mode, value in speedups.items()}
    clear_efficiency_pass = {
        mode: value >= 1.10 for mode, value in speedups.items()
    }
    model_selection = select_model_for_stage(
        root.name,
        accuracy_passed=accuracy_pass,
        stability_passed=stability_pass,
        speedup_vs_m0=speedups,
    )
    workspace = Path(__file__).resolve().parents[1]
    current_hashes = _source_hashes(workspace)
    source_hashes_consistent = all(
        summaries[mode]["source_sha256"] == current_hashes for mode in modes
    )
    payload = {
        "status": (
            "passed"
            if all(value["status"] == "passed" for value in summaries.values())
            and source_hashes_consistent
            else "failed"
        ),
        "rows": rows,
        "model_accuracy_checks": accuracy,
        "model_accuracy_passed": accuracy_pass,
        "model_stability_passed": stability_pass,
        "model_efficiency_passed": efficiency_pass,
        "model_clear_efficiency_passed": clear_efficiency_pass,
        "model_feasible": {
            "M1": accuracy_pass["M1"]
            and stability_pass["M1"]
            and efficiency_pass["M1"]
        },
        "model_selection": model_selection,
        "model_selection_stage_eligible": root.name == "forward_40x20_24h",
        "error_energy_segment_contributions": segment_contributions,
        "source_hashes_consistent": source_hashes_consistent,
        "c_grid_difference_note": "本轮使用 Arakawa C 交错网格、隐式迎风冰动量、IMEX/Helmholtz 浅水和同一界面拖曳对象双向耦合；与旧共点网格/人工动量扩散结果的差异来自变量位置、边界离散、压力-速度耦合、平流处理、拖曳掩膜与非线性收敛路径，旧结果未被用作拟合目标。",
    }
    _write_json(root / "test_report.json", payload)
    _write_json(
        root / "summary.json",
        {
            "schema_version": 3,
            "equation_specification": "problem2_numerical_spec_v2",
            "status": payload["status"],
            "stage": root.name,
            "model_selection": payload["model_selection"],
            "model_accuracy_passed": accuracy_pass,
            "model_stability_passed": stability_pass,
            "model_efficiency_passed": efficiency_pass,
            "model_clear_efficiency_passed": clear_efficiency_pass,
            "source_hashes_consistent": source_hashes_consistent,
            "modes": summaries,
        },
    )
    m1_row = comparison.loc[comparison["mode"] == "M1"].iloc[0]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    core_labels = ("U(T)", "h(T)", "U(st)", "h(st)")
    core_values = 100.0 * np.array(
        [
            m1_row["terminal_velocity_relative_error"],
            m1_row["terminal_thickness_relative_error"],
            m1_row["velocity_spatiotemporal_relative_error"],
            m1_row["thickness_spatiotemporal_relative_error"],
        ]
    )
    axes[0].bar(core_labels, core_values)
    axes[0].axhline(5.0, color="r", linestyle="--", label="5% target")
    axes[0].set_ylabel("M1 relative error (%)")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)
    auxiliary_labels = ("full U(T)", "mass U(T)", "full U(st)", "mass U(st)")
    auxiliary_values = 100.0 * np.array(
        [
            m1_row["terminal_velocity_full_domain_error"],
            m1_row["terminal_velocity_mass_weighted_error"],
            m1_row["velocity_full_domain_spatiotemporal_error"],
            m1_row["velocity_mass_weighted_spatiotemporal_error"],
        ]
    )
    axes[1].bar(auxiliary_labels, auxiliary_values)
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set_ylabel("Auxiliary velocity error (%)")
    axes[1].grid(axis="y", alpha=0.3)
    axes[2].bar(comparison["mode"], comparison["speedup_vs_m0"])
    axes[2].axhline(1.0, color="k", linestyle="--", linewidth=1)
    axes[2].axhline(1.10, color="r", linestyle=":", linewidth=1)
    axes[2].set_ylabel("Single-run speedup vs M0")
    axes[2].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(root / "model_comparison.png", dpi=180)
    plt.close(fig)

    m1_error_history = combined_error_history.loc[
        combined_error_history["mode"] == "M1"
    ]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].semilogy(
        m1_error_history["time_hours"],
        np.maximum(m1_error_history["velocity_error_energy"], 1.0e-30),
    )
    axes[0].set_ylabel("Velocity error energy")
    axes[0].grid(alpha=0.3)
    axes[1].semilogy(
        m1_error_history["time_hours"],
        np.maximum(m1_error_history["thickness_error_energy"], 1.0e-30),
    )
    axes[1].set_xlabel("Time (h)")
    axes[1].set_ylabel("Thickness error energy")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(root / "model_error_energy.png", dpi=180)
    plt.close(fig)

    final_index = len(reference["time_seconds"]) - 1
    ref_u, ref_v = _center_velocity(reference, final_index)
    cur_u, cur_v = _center_velocity(snapshots["M1"], final_index)
    speed_difference = np.hypot(cur_u - ref_u, cur_v - ref_v)
    thickness_difference = snapshots["M1"]["thickness"][final_index] - reference[
        "thickness"
    ][final_index]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    image_u = axes[0].imshow(speed_difference, origin="lower", aspect="auto", cmap="magma")
    fig.colorbar(image_u, ax=axes[0], label="Center-speed absolute error (m/s)")
    axes[0].set_title("M1 - M0 velocity at final time")
    image_h = axes[1].imshow(thickness_difference, origin="lower", aspect="auto", cmap="coolwarm")
    fig.colorbar(image_h, ax=axes[1], label="Thickness difference (m)")
    axes[1].set_title("M1 - M0 thickness at final time")
    for axis in axes:
        axis.set_xlabel("i")
        axis.set_ylabel("j")
    fig.tight_layout()
    fig.savefig(root / "final_model_error_fields.png", dpi=180)
    plt.close(fig)
    return payload


def stage_configuration(stage: str) -> tuple[Problem2Config, tuple[float, ...]]:
    if stage == "smoke":
        return Problem2Config(nx=20, ny=10, duration_hours=0.5), (0.0, 0.5)
    if stage == "regression":
        return Problem2Config(nx=40, ny=20, duration_hours=6.0), (0.0, 1.0, 3.0, 6.0)
    if stage == "forward":
        return Problem2Config(nx=40, ny=20, duration_hours=24.0), (0.0, 6.0, 12.0, 18.0, 24.0)
    raise ValueError(f"未知运行阶段: {stage}")


def run_stage(stage: str, output_directory: str | Path) -> dict[str, Any]:
    base_config, snapshot_hours = stage_configuration(stage)
    root = Path(output_directory)
    root.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}
    for mode in FORMAL_MODEL_MODES:
        config = replace(base_config, mode=mode)
        summaries[mode.value] = run_simulation(
            config, root / mode.value, snapshot_hours=snapshot_hours
        )
    comparison = write_model_comparison(root)
    return {"stage": stage, "summaries": summaries, "comparison": comparison}
