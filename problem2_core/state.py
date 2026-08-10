"""问题二耦合状态的数据结构与冻结初始条件。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from .config import Problem2Config

Array = NDArray[np.float64]


@dataclass
class CoupledState:
    thickness: Array
    ice_u: Array
    ice_v: Array
    ocean_u: Array
    ocean_v: Array
    sea_surface: Array
    time_seconds: float = 0.0

    @classmethod
    def initial(cls, config: Problem2Config) -> "CoupledState":
        grid = config.grid
        x_center, y_center = grid.center_coordinates()
        thickness = 0.5 + 0.3 * np.sin(np.pi * x_center / config.length_x) * np.cos(
            np.pi * y_center / config.length_y
        )
        return cls(
            thickness=thickness,
            ice_u=np.zeros(grid.u_shape),
            ice_v=np.zeros(grid.v_shape),
            ocean_u=np.zeros(grid.u_shape),
            ocean_v=np.zeros(grid.v_shape),
            sea_surface=np.zeros(grid.center_shape),
        )

    def validate(self, config: Problem2Config) -> None:
        grid = config.grid
        grid.require_center(self.thickness)
        grid.require_u(self.ice_u)
        grid.require_v(self.ice_v)
        grid.require_u(self.ocean_u)
        grid.require_v(self.ocean_v)
        grid.require_center(self.sea_surface)
        if self.time_seconds < 0.0:
            raise ValueError("time_seconds 不得为负")
        if not all(
            np.all(np.isfinite(field))
            for field in (
                self.thickness,
                self.ice_u,
                self.ice_v,
                self.ocean_u,
                self.ocean_v,
                self.sea_surface,
            )
        ):
            raise FloatingPointError("状态中存在 NaN 或无穷值")
        if np.min(self.thickness) < 0.0:
            raise FloatingPointError("冰厚必须非负")

    def copy(self) -> "CoupledState":
        return CoupledState(
            thickness=np.array(self.thickness, copy=True),
            ice_u=np.array(self.ice_u, copy=True),
            ice_v=np.array(self.ice_v, copy=True),
            ocean_u=np.array(self.ocean_u, copy=True),
            ocean_v=np.array(self.ocean_v, copy=True),
            sea_surface=np.array(self.sea_surface, copy=True),
            time_seconds=float(self.time_seconds),
        )

    def as_mapping(self) -> dict[str, Any]:
        return {
            "thickness": np.array(self.thickness, copy=True),
            "ice_u": np.array(self.ice_u, copy=True),
            "ice_v": np.array(self.ice_v, copy=True),
            "ocean_u": np.array(self.ocean_u, copy=True),
            "ocean_v": np.array(self.ocean_v, copy=True),
            "sea_surface": np.array(self.sea_surface, copy=True),
            "time_seconds": float(self.time_seconds),
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "CoupledState":
        return cls(
            thickness=np.array(values["thickness"], dtype=float, copy=True),
            ice_u=np.array(values["ice_u"], dtype=float, copy=True),
            ice_v=np.array(values["ice_v"], dtype=float, copy=True),
            ocean_u=np.array(values["ocean_u"], dtype=float, copy=True),
            ocean_v=np.array(values["ocean_v"], dtype=float, copy=True),
            sea_surface=np.array(values["sea_surface"], dtype=float, copy=True),
            time_seconds=float(values["time_seconds"]),
        )
