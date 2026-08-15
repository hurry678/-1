"""问题二 v2.1 删项归因专用模式。

本模块刻意与正式 ``ModelMode`` 隔离：这里的四种组合只用于同离散归因，
不得进入正式模型选择器或替代教师指定的 M1。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class DiagnosticMode(StrEnum):
    M0_REFERENCE = "M0"
    MI_NO_INERTIA = "M_I"
    MC_NO_ICE_CORIOLIS = "M_C"
    M1_TEACHER = "M1"


DIAGNOSTIC_MODEL_MODES = (
    DiagnosticMode.M0_REFERENCE,
    DiagnosticMode.MI_NO_INERTIA,
    DiagnosticMode.MC_NO_ICE_CORIOLIS,
    DiagnosticMode.M1_TEACHER,
)


_FROZEN_WEIGHTS = {
    DiagnosticMode.M0_REFERENCE: (1.0, 1.0),
    DiagnosticMode.MI_NO_INERTIA: (0.0, 1.0),
    DiagnosticMode.MC_NO_ICE_CORIOLIS: (1.0, 0.0),
    DiagnosticMode.M1_TEACHER: (0.0, 0.0),
}


@dataclass(frozen=True)
class DiagnosticRunSpec:
    """不可进入正式选模的冻结诊断权重。"""

    mode: DiagnosticMode
    inertia_weight: float
    ice_coriolis_weight: float
    diagnostic_only: bool = True
    eligible_for_model_selection: bool = False

    def __post_init__(self) -> None:
        expected = _FROZEN_WEIGHTS[self.mode]
        actual = (float(self.inertia_weight), float(self.ice_coriolis_weight))
        if actual != expected:
            raise ValueError(
                f"{self.mode.value} 冻结权重应为 {expected}，实际为 {actual}"
            )
        if not self.diagnostic_only or self.eligible_for_model_selection:
            raise ValueError("删项模式必须保持 diagnostic_only 且不得进入正式选模")

    def as_dict(self) -> dict[str, object]:
        return {
            "diagnostic_mode": self.mode.value,
            "ice_inertia_weight": self.inertia_weight,
            "ice_coriolis_weight": self.ice_coriolis_weight,
            "diagnostic_only": self.diagnostic_only,
            "eligible_for_model_selection": self.eligible_for_model_selection,
        }


def diagnostic_run_spec(mode: DiagnosticMode | str) -> DiagnosticRunSpec:
    resolved = mode if isinstance(mode, DiagnosticMode) else DiagnosticMode(mode)
    inertia, coriolis = _FROZEN_WEIGHTS[resolved]
    return DiagnosticRunSpec(
        mode=resolved,
        inertia_weight=inertia,
        ice_coriolis_weight=coriolis,
    )
