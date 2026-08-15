"""问题二数值规格 v2 的冻结配置与海冰模式开关。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum

import numpy as np

from .grid import CGrid


class ModelMode(StrEnum):
    M0_FULL = "M0"
    M1_QUASI_STATIC = "M1"
    M2_STARTUP_WINDOW = "M2"


@dataclass(frozen=True)
class Problem2Config:
    nx: int = 200
    ny: int = 100
    length_x: float = 20_000.0
    length_y: float = 10_000.0
    dt: float = 300.0
    duration_hours: float = 24.0
    rho_ice: float = 917.0
    rho_water: float = 1025.0
    water_depth: float = 30.0
    gravity: float = 9.8
    coriolis: float = 1.4e-4
    ocean_coriolis_enabled: bool = True
    drag_coefficient: float = 0.005
    p_star: float = 5000.0
    h_i0: float = 1.0
    ellipse_ratio: float = 2.0
    concentration_decay: float = 20.0
    wind_stress_x: float = 0.12
    wind_stress_y: float = 0.04
    delta_min: float = 1.0e-9
    zeta_max: float = 1.0e8
    h_min: float = 1.0e-6
    thickness_cfl_limit: float = 0.8
    ice_picard_tolerance: float = 1.0e-3
    inner_picard_relaxation: float = 0.5
    gmres_relative_tolerance: float = 1.0e-8
    gmres_restart: int = 80
    gmres_max_iterations: int = 200
    coupling_tolerance: float = 1.0e-5
    picard_iterations_standard: int = 100
    picard_iterations_reset_aitken: int = 300
    picard_iterations_robust: int = 500
    coupling_iterations_standard: int = 100
    coupling_iterations_reset_aitken: int = 200
    coupling_iterations_robust: int = 300
    outer_aitken_standard: float = 0.5
    outer_aitken_reset: float = 0.2
    outer_aitken_robust: float = 0.2
    preconditioner_standard: str = "ilu"
    preconditioner_reset_aitken: str = "ilu"
    preconditioner_robust: str = "robust_ilu"
    water_residual_tolerance: float = 1.0e-8
    ocean_coriolis_max_iterations: int = 50
    aitken_min: float = 0.1
    aitken_max: float = 1.0
    maximum_speed: float = 5.0
    force_epsilon: float = 1.0e-14
    norm_epsilon: float = 1.0e-12
    temporal_predictor_enabled: bool = False
    outer_anderson_enabled: bool = False
    anderson_depth: int = 4
    anderson_damping: float = 0.8
    anderson_residual_rise_factor: float = 1.25
    anderson_condition_limit: float = 1.0e10
    anderson_step_ratio_limit: float = 5.0
    ilu_reuse_enabled: bool = False
    ilu_reuse_max: int = 2
    linear_solver: str = "gmres"
    inner_anderson_enabled: bool = False
    inner_anderson_depth: int = 3
    inner_anderson_residual_rise_factor: float = 1.2
    inner_backtracking_enabled: bool = False
    inner_backtracking_initial: float = 0.7
    inner_backtracking_min: float = 0.05
    inner_backtracking_max: float = 0.9
    inner_backtracking_factor: float = 0.5
    inner_backtracking_growth_limit: float = 1.1
    ilu_drop_tol: float = 1.0e-3
    ilu_fill_factor: float = 8.0
    robust_ilu_drop_tol: float = 1.0e-6
    robust_ilu_fill_factor: float = 20.0
    mode: ModelMode = ModelMode.M0_FULL
    startup_window_minutes: float = 60.0

    def __post_init__(self) -> None:
        if self.nx <= 0 or self.ny <= 0:
            raise ValueError("网格数必须为正")
        if self.length_x <= 0.0 or self.length_y <= 0.0:
            raise ValueError("区域尺度必须为正")
        if self.dt <= 0.0 or self.duration_hours <= 0.0:
            raise ValueError("时间规格必须为正")
        if not (0.0 < self.thickness_cfl_limit <= 0.8):
            raise ValueError("冻结规格要求冰厚输运 CFL 上限不超过 0.8")
        if self.delta_min <= 0.0 or self.zeta_max <= 0.0 or self.h_min < 0.0:
            raise ValueError("VP 正则化参数必须满足冻结定义")
        if not (
            self.ice_picard_tolerance > 0.0
            and self.gmres_relative_tolerance > 0.0
            and self.coupling_tolerance > 0.0
            and self.water_residual_tolerance > 0.0
        ):
            raise ValueError("所有冻结残差容差必须为正")
        if self.gmres_restart <= 0 or self.gmres_max_iterations <= 0:
            raise ValueError("GMRES 迭代参数必须为正")
        if not (0.0 < self.inner_picard_relaxation <= 1.0):
            raise ValueError("固定内层 Picard 松弛必须位于 (0, 1]")
        iteration_limits = (
            self.picard_iterations_standard,
            self.picard_iterations_reset_aitken,
            self.picard_iterations_robust,
            self.coupling_iterations_standard,
            self.coupling_iterations_reset_aitken,
            self.coupling_iterations_robust,
            self.ocean_coriolis_max_iterations,
        )
        if any(value <= 0 for value in iteration_limits):
            raise ValueError("Picard、耦合和科里奥利迭代上限必须为正")
        outer_aitken_values = (
            self.outer_aitken_standard,
            self.outer_aitken_reset,
            self.outer_aitken_robust,
        )
        if any(not self.aitken_min <= value <= self.aitken_max for value in outer_aitken_values):
            raise ValueError("外层 Aitken 初值必须位于冻结安全区间")
        if any(
            not value
            for value in (
                self.preconditioner_standard,
                self.preconditioner_reset_aitken,
                self.preconditioner_robust,
            )
        ):
            raise ValueError("预条件器名称不得为空")
        if not (0.0 < self.aitken_min <= self.aitken_max <= 1.0):
            raise ValueError("Aitken 安全区间必须位于 (0, 1]")
        if self.maximum_speed <= 0.0:
            raise ValueError("异常速度阈值必须为正")
        if not self.ocean_coriolis_enabled:
            raise ValueError("数值规格 v2 要求 M0、M1 的海水科里奥利始终开启")
        if self.ilu_reuse_enabled and not 1 <= self.ilu_reuse_max <= 4:
            raise ValueError("ILU 最大连续复用次数必须位于 1--4")
        if self.linear_solver not in ("gmres", "bicgstab", "splu"):
            raise ValueError("线性求解器必须为 gmres、bicgstab 或 splu")
        if not 2 <= self.inner_anderson_depth <= 5:
            raise ValueError("内层 Anderson 深度必须位于 2--5")
        if self.inner_anderson_residual_rise_factor <= 1.0:
            raise ValueError("内层 Anderson 残差上升保护因子必须大于 1")
        if not 1 <= self.anderson_depth <= 5:
            raise ValueError("Anderson 深度必须位于 1--5")
        if not 0.0 < self.anderson_damping <= 1.0:
            raise ValueError("Anderson 阻尼必须位于 (0, 1]")
        if self.anderson_residual_rise_factor <= 1.0:
            raise ValueError("Anderson 残差上升保护因子必须大于 1")
        if self.anderson_condition_limit <= 1.0:
            raise ValueError("Anderson 病态保护阈值必须大于 1")
        if self.anderson_step_ratio_limit <= 1.0:
            raise ValueError("Anderson 步长保护阈值必须大于 1")
        if self.mode is ModelMode.M2_STARTUP_WINDOW and self.startup_window_minutes < 0.0:
            raise ValueError("M2 启动窗口不得为负")

    @property
    def grid(self) -> CGrid:
        return CGrid(
            nx=self.nx,
            ny=self.ny,
            dx=self.length_x / self.nx,
            dy=self.length_y / self.ny,
        )

    @property
    def physical_steps(self) -> int:
        duration_seconds = self.duration_hours * 3600.0
        steps = round(duration_seconds / self.dt)
        if not np.isclose(steps * self.dt, duration_seconds):
            raise ValueError("duration_hours 必须对应整数个物理时间步")
        return int(steps)

    def inertia_weight(self, time_seconds: float) -> float:
        """返回海冰惯性权重；M2 仅为历史回归兼容路径。"""

        if time_seconds < 0.0:
            raise ValueError("time_seconds 不得为负")
        if self.mode is ModelMode.M0_FULL:
            return 1.0
        if self.mode is ModelMode.M1_QUASI_STATIC:
            return 0.0
        switch_seconds = self.startup_window_minutes * 60.0
        return 1.0 if time_seconds < switch_seconds else 0.0

    @property
    def ice_inertia_weight(self) -> float:
        """正式 M0/M1 在真实物理时间内使用的海冰惯性权重。"""

        return 1.0 if self.mode is ModelMode.M0_FULL else 0.0

    @property
    def ice_coriolis_weight(self) -> float:
        """只控制海冰科里奥利；海水仍使用题定 ``coriolis``。"""

        return 0.0 if self.mode is ModelMode.M1_QUASI_STATIC else 1.0

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["mode"] = self.mode.value
        payload["dx"] = self.grid.dx
        payload["dy"] = self.grid.dy
        payload["physical_steps"] = self.physical_steps
        payload["ice_inertia_weight"] = self.ice_inertia_weight
        payload["ice_coriolis_weight"] = self.ice_coriolis_weight
        payload["inner_picard_acceleration"] = "fixed_relaxation_only_no_aitken"
        payload["outer_acceleration"] = (
            "protected_anderson" if self.outer_anderson_enabled else "aitken"
        )
        payload["retry_policy"] = [
            {
                "name": "standard",
                "max_picard_iterations": self.picard_iterations_standard,
                "max_coupling_iterations": self.coupling_iterations_standard,
                "outer_aitken_initial": self.outer_aitken_standard,
                "preconditioner": self.preconditioner_standard,
            },
            {
                "name": "reset_aitken",
                "max_picard_iterations": self.picard_iterations_reset_aitken,
                "max_coupling_iterations": self.coupling_iterations_reset_aitken,
                "outer_aitken_initial": self.outer_aitken_reset,
                "preconditioner": self.preconditioner_reset_aitken,
            },
            {
                "name": "robust_preconditioner",
                "max_picard_iterations": self.picard_iterations_robust,
                "max_coupling_iterations": self.coupling_iterations_robust,
                "outer_aitken_initial": self.outer_aitken_robust,
                "preconditioner": self.preconditioner_robust,
            },
        ]
        return payload
