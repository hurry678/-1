"""Arakawa C 型网格及冻结规格中的基础插值算子。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class CGrid:
    """二维均匀 C 网格。

    中心量形状为 ``(ny, nx)``；东向面量为 ``(ny, nx + 1)``；
    北向面量为 ``(ny + 1, nx)``；角点量为 ``(ny + 1, nx + 1)``。
    """

    nx: int
    ny: int
    dx: float
    dy: float
    x0: float = 0.0
    y0: float = 0.0

    def __post_init__(self) -> None:
        if self.nx <= 0 or self.ny <= 0:
            raise ValueError("nx 和 ny 必须为正整数")
        if self.dx <= 0.0 or self.dy <= 0.0:
            raise ValueError("dx 和 dy 必须为正数")

    @property
    def center_shape(self) -> tuple[int, int]:
        return self.ny, self.nx

    @property
    def u_shape(self) -> tuple[int, int]:
        return self.ny, self.nx + 1

    @property
    def v_shape(self) -> tuple[int, int]:
        return self.ny + 1, self.nx

    @property
    def corner_shape(self) -> tuple[int, int]:
        return self.ny + 1, self.nx + 1

    @property
    def cell_area(self) -> float:
        return self.dx * self.dy

    def center_coordinates(self) -> tuple[Array, Array]:
        x = self.x0 + (np.arange(self.nx, dtype=float) + 0.5) * self.dx
        y = self.y0 + (np.arange(self.ny, dtype=float) + 0.5) * self.dy
        return np.meshgrid(x, y)

    def u_coordinates(self) -> tuple[Array, Array]:
        x = self.x0 + np.arange(self.nx + 1, dtype=float) * self.dx
        y = self.y0 + (np.arange(self.ny, dtype=float) + 0.5) * self.dy
        return np.meshgrid(x, y)

    def v_coordinates(self) -> tuple[Array, Array]:
        x = self.x0 + (np.arange(self.nx, dtype=float) + 0.5) * self.dx
        y = self.y0 + np.arange(self.ny + 1, dtype=float) * self.dy
        return np.meshgrid(x, y)

    def corner_coordinates(self) -> tuple[Array, Array]:
        x = self.x0 + np.arange(self.nx + 1, dtype=float) * self.dx
        y = self.y0 + np.arange(self.ny + 1, dtype=float) * self.dy
        return np.meshgrid(x, y)

    def require_center(self, field: Array) -> None:
        if field.shape != self.center_shape:
            raise ValueError(f"中心量形状应为 {self.center_shape}，实际为 {field.shape}")

    def require_u(self, field: Array) -> None:
        if field.shape != self.u_shape:
            raise ValueError(f"U 面量形状应为 {self.u_shape}，实际为 {field.shape}")

    def require_v(self, field: Array) -> None:
        if field.shape != self.v_shape:
            raise ValueError(f"V 面量形状应为 {self.v_shape}，实际为 {field.shape}")

    def require_corner(self, field: Array) -> None:
        if field.shape != self.corner_shape:
            raise ValueError(f"角点量形状应为 {self.corner_shape}，实际为 {field.shape}")

    def center_to_u(self, field: Array) -> Array:
        """中心量算术平均到 U 面，物理边界使用偶延拓。"""
        self.require_center(field)
        result = np.empty(self.u_shape, dtype=float)
        result[:, 1:-1] = 0.5 * (field[:, :-1] + field[:, 1:])
        result[:, 0] = field[:, 0]
        result[:, -1] = field[:, -1]
        return result

    def center_to_v(self, field: Array) -> Array:
        """中心量算术平均到 V 面，物理边界使用偶延拓。"""
        self.require_center(field)
        result = np.empty(self.v_shape, dtype=float)
        result[1:-1, :] = 0.5 * (field[:-1, :] + field[1:, :])
        result[0, :] = field[0, :]
        result[-1, :] = field[-1, :]
        return result

    def center_to_corner(self, field: Array) -> Array:
        """中心量算术平均到角点，边界采用相邻有效中心的单边平均。"""
        self.require_center(field)
        padded = np.pad(field, ((1, 1), (1, 1)), mode="edge")
        return 0.25 * (
            padded[:-1, :-1]
            + padded[:-1, 1:]
            + padded[1:, :-1]
            + padded[1:, 1:]
        )

    def faces_to_center(self, u_face: Array, v_face: Array) -> tuple[Array, Array]:
        """将两个方向的面速度分别算术平均回单元中心。"""
        self.require_u(u_face)
        self.require_v(v_face)
        u_center = 0.5 * (u_face[:, :-1] + u_face[:, 1:])
        v_center = 0.5 * (v_face[:-1, :] + v_face[1:, :])
        return u_center, v_center

    def v_to_u(self, v_face: Array) -> Array:
        """冻结规格中的四点插值：V 面量插值到 U 面。"""
        self.require_v(v_face)
        padded = np.pad(v_face, ((0, 0), (1, 1)), mode="edge")
        return 0.25 * (
            padded[:-1, :-1]
            + padded[1:, :-1]
            + padded[:-1, 1:]
            + padded[1:, 1:]
        )

    def u_to_v(self, u_face: Array) -> Array:
        """冻结规格中的四点插值：U 面量插值到 V 面。"""
        self.require_u(u_face)
        padded = np.pad(u_face, ((1, 1), (0, 0)), mode="edge")
        return 0.25 * (
            padded[:-1, :-1]
            + padded[:-1, 1:]
            + padded[1:, :-1]
            + padded[1:, 1:]
        )

    def u_dual_area_weights(self) -> Array:
        weights = np.full(self.u_shape, self.cell_area, dtype=float)
        weights[:, (0, -1)] *= 0.5
        return weights

    def v_dual_area_weights(self) -> Array:
        weights = np.full(self.v_shape, self.cell_area, dtype=float)
        weights[(0, -1), :] *= 0.5
        return weights
