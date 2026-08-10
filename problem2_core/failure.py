"""物理时间步失败、回滚、批准重试与中止记录框架。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np


@dataclass(frozen=True)
class RetryStage:
    name: str
    max_picard_iterations: int
    max_coupling_iterations: int
    outer_aitken_initial: float
    preconditioner: str


@dataclass(frozen=True)
class RetryPolicy:
    target_residual: float
    stages: tuple[RetryStage, ...]

    @classmethod
    def default(
        cls,
        *,
        target_residual: float,
        picard_iterations: tuple[int, int, int] = (100, 200, 300),
        coupling_iterations: tuple[int, int, int] = (100, 200, 300),
        outer_aitken_initials: tuple[float, float, float] = (0.5, 0.2, 0.2),
        preconditioners: tuple[str, str, str] = ("ilu", "ilu", "robust_ilu"),
    ) -> "RetryPolicy":
        if target_residual <= 0.0:
            raise ValueError("target_residual 必须为正")
        if any(value <= 0 for value in (*picard_iterations, *coupling_iterations)):
            raise ValueError("Picard 与耦合迭代上限必须为正")
        return cls(
            target_residual=target_residual,
            stages=(
                RetryStage(
                    "standard",
                    picard_iterations[0],
                    coupling_iterations[0],
                    outer_aitken_initials[0],
                    preconditioners[0],
                ),
                RetryStage(
                    "reset_aitken",
                    picard_iterations[1],
                    coupling_iterations[1],
                    outer_aitken_initials[1],
                    preconditioners[1],
                ),
                RetryStage(
                    "robust_preconditioner",
                    picard_iterations[2],
                    coupling_iterations[2],
                    outer_aitken_initials[2],
                    preconditioners[2],
                ),
            ),
        )


@dataclass(frozen=True)
class StepAttempt:
    converged: bool
    state: Mapping[str, Any]
    residual_history: list[float]
    iterations: int
    message: str = ""
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AttemptRecord:
    stage: str
    max_picard_iterations: int
    max_coupling_iterations: int
    outer_aitken_initial: float
    preconditioner: str
    target_residual: float
    converged: bool
    accepted: bool
    final_residual: float | None
    iterations: int
    message: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PhysicalStepResult:
    status: str
    state: Mapping[str, Any]
    attempts: tuple[AttemptRecord, ...]
    step_index: int


class PhysicalStepFailure(RuntimeError):
    def __init__(self, result: PhysicalStepResult):
        super().__init__(f"物理时间步 {result.step_index} 在预定重试后仍未收敛")
        self.result = result


Solver = Callable[..., StepAttempt]


def _last_residual(history: list[float]) -> float | None:
    return float(history[-1]) if history else None


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _write_failure_artifacts(
    directory: Path,
    *,
    result: PhysicalStepResult,
    policy: RetryPolicy,
    config: Mapping[str, Any],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    prefix = f"step_{result.step_index:06d}"
    array_state = {
        key: value
        for key, value in result.state.items()
        if isinstance(value, np.ndarray)
    }
    np.savez_compressed(directory / f"{prefix}_last_converged_state.npz", **array_state)
    scalar_state = {
        key: value
        for key, value in result.state.items()
        if not isinstance(value, np.ndarray)
    }
    payload = {
        "status": "failed",
        "step_index": result.step_index,
        "target_residual": policy.target_residual,
        "config": _json_value(config),
        "last_converged_scalar_state": _json_value(scalar_state),
        "attempts": [asdict(record) for record in result.attempts],
    }
    (directory / f"{prefix}_failure.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def execute_physical_step(
    last_converged_state: Mapping[str, Any],
    solver: Solver,
    *,
    policy: RetryPolicy,
    step_index: int,
    failure_directory: str | Path,
    config: Mapping[str, Any],
) -> PhysicalStepResult:
    """执行一个物理步；所有重试均从同一最后收敛状态开始。

    目标残差由 ``RetryPolicy.target_residual`` 固定传入每次尝试。批准的重试
    只允许重置 Aitken 初值、增加迭代上限或更换稳健预条件器。
    """

    if step_index < 0:
        raise ValueError("step_index 不得为负")
    immutable_start = deepcopy(last_converged_state)
    records: list[AttemptRecord] = []

    for stage in policy.stages:
        trial_state = deepcopy(immutable_start)
        try:
            attempt = solver(
                trial_state,
                stage=stage,
                target_residual=policy.target_residual,
            )
            final_residual = _last_residual(attempt.residual_history)
            accepted = bool(
                attempt.converged
                and final_residual is not None
                and np.isfinite(final_residual)
                and final_residual <= policy.target_residual
            )
            record = AttemptRecord(
                stage=stage.name,
                max_picard_iterations=stage.max_picard_iterations,
                max_coupling_iterations=stage.max_coupling_iterations,
                outer_aitken_initial=stage.outer_aitken_initial,
                preconditioner=stage.preconditioner,
                target_residual=policy.target_residual,
                converged=bool(attempt.converged),
                accepted=accepted,
                final_residual=final_residual,
                iterations=int(attempt.iterations),
                message=attempt.message,
                diagnostics=deepcopy(attempt.diagnostics),
            )
            records.append(record)
            if accepted:
                return PhysicalStepResult(
                    status="success",
                    state=deepcopy(attempt.state),
                    attempts=tuple(records),
                    step_index=step_index,
                )
        except Exception as exc:  # 求解器异常也视为本次尝试失败并进入固定重试链
            records.append(
                AttemptRecord(
                    stage=stage.name,
                    max_picard_iterations=stage.max_picard_iterations,
                    max_coupling_iterations=stage.max_coupling_iterations,
                    outer_aitken_initial=stage.outer_aitken_initial,
                    preconditioner=stage.preconditioner,
                    target_residual=policy.target_residual,
                    converged=False,
                    accepted=False,
                    final_residual=None,
                    iterations=0,
                    message=f"{type(exc).__name__}: {exc}",
                    diagnostics={},
                )
            )

    failed = PhysicalStepResult(
        status="failed",
        state=deepcopy(immutable_start),
        attempts=tuple(records),
        step_index=step_index,
    )
    _write_failure_artifacts(
        Path(failure_directory), result=failed, policy=policy, config=config
    )
    raise PhysicalStepFailure(failed)
