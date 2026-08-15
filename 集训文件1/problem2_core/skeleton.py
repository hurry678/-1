"""问题二统一求解器的阶段 A 程序骨架。

当前版本仅组织冻结配置、状态和基础算子；正式 Picard-GMRES、IMEX/Helmholtz
以及 24 h 时间循环将在后续阶段接入。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .boundary import ice_active_face_masks
from .config import Problem2Config
from .ocean_swe import OceanState, lake_at_rest_residual
from .rheology_vp import compute_vp_stress, strain_rates, stress_divergence
from .state import CoupledState
from .thickness_transport import total_ice_volume


@dataclass
class Problem2SolverSkeleton:
    config: Problem2Config
    state: CoupledState

    @classmethod
    def initialize(cls, config: Problem2Config | None = None) -> "Problem2SolverSkeleton":
        resolved = config or Problem2Config()
        state = CoupledState.initial(resolved)
        state.validate(resolved)
        return cls(config=resolved, state=state)

    def operator_diagnostics(self) -> dict[str, Any]:
        """在初始状态上执行不推进时间的算子诊断。"""

        grid = self.config.grid
        center_mask, u_mask, v_mask = ice_active_face_masks(
            grid, self.state.thickness, h_min=self.config.h_min
        )
        strain = strain_rates(
            grid,
            self.state.ice_u,
            self.state.ice_v,
            (np.zeros(grid.nx + 1), np.zeros(grid.nx + 1)),
            (np.zeros(grid.ny + 1), np.zeros(grid.ny + 1)),
        )
        stress = compute_vp_stress(
            grid,
            self.state.thickness,
            strain,
            h_i0=self.config.h_i0,
            p_star=self.config.p_star,
            concentration_decay=self.config.concentration_decay,
            ellipse_ratio=self.config.ellipse_ratio,
            delta_min=self.config.delta_min,
            zeta_max=self.config.zeta_max,
        )
        divergence_u, divergence_v = stress_divergence(
            grid,
            self.state.thickness,
            stress.sigma_xx,
            stress.sigma_yy,
            stress.sigma_xy_corner,
        )
        ocean_residual = lake_at_rest_residual(
            grid,
            OceanState(
                xi=self.state.sea_surface,
                u=self.state.ocean_u,
                v=self.state.ocean_v,
            ),
            water_depth=self.config.water_depth,
            gravity=self.config.gravity,
            coriolis=self.config.coriolis,
        )
        return {
            "ice_volume_m3": total_ice_volume(grid, self.state.thickness),
            "active_center_fraction": float(np.mean(center_mask)),
            "active_u_fraction": float(np.mean(u_mask)),
            "active_v_fraction": float(np.mean(v_mask)),
            "max_abs_strain_rate": float(
                max(
                    np.max(np.abs(strain.exx)),
                    np.max(np.abs(strain.eyy)),
                    np.max(np.abs(strain.exy_corner)),
                )
            ),
            "max_abs_stress_divergence_u": float(np.max(np.abs(divergence_u))),
            "max_abs_stress_divergence_v": float(np.max(np.abs(divergence_v))),
            "max_abs_lake_at_rest_residual": float(
                max(
                    np.max(np.abs(ocean_residual.xi)),
                    np.max(np.abs(ocean_residual.u)),
                    np.max(np.abs(ocean_residual.v)),
                )
            ),
        }

    def advance_one_physical_step(self) -> None:
        raise NotImplementedError(
            "阶段 A 仅交付基础算子骨架；正式 IMEX、Picard-GMRES 和耦合时间步尚未启用"
        )
