"""C 网格稀疏差分、插值与一阶迎风矩阵。

这些矩阵只编码冻结的变量位置和边界模板，不包含任何模式判断。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from .grid import CGrid

Array = NDArray[np.float64]
SparseMatrix = sparse.csr_matrix


def _matrix(
    rows: list[int],
    cols: list[int],
    data: list[float],
    shape: tuple[int, int],
) -> SparseMatrix:
    return sparse.coo_matrix((data, (rows, cols)), shape=shape).tocsr()


@dataclass(frozen=True)
class SparseCGridOperators:
    grid: CGrid
    exx_from_u: SparseMatrix
    eyy_from_v: SparseMatrix
    exy_from_u: SparseMatrix
    exy_from_v: SparseMatrix
    center_x_to_u: SparseMatrix
    corner_y_to_u: SparseMatrix
    corner_x_to_v: SparseMatrix
    center_y_to_v: SparseMatrix
    divergence_u: SparseMatrix
    divergence_v: SparseMatrix
    v_to_u: SparseMatrix
    u_to_v: SparseMatrix
    ice_interior_mask: NDArray[np.bool_]

    @property
    def n_center(self) -> int:
        return self.grid.nx * self.grid.ny

    @property
    def n_u(self) -> int:
        return self.grid.ny * (self.grid.nx + 1)

    @property
    def n_v(self) -> int:
        return (self.grid.ny + 1) * self.grid.nx

    @property
    def n_corner(self) -> int:
        return (self.grid.ny + 1) * (self.grid.nx + 1)

    @classmethod
    def build(cls, grid: CGrid) -> "SparseCGridOperators":
        nx, ny, dx, dy = grid.nx, grid.ny, grid.dx, grid.dy
        n_c = nx * ny
        n_u = (nx + 1) * ny
        n_v = nx * (ny + 1)
        n_k = (nx + 1) * (ny + 1)

        def c(j: int, i: int) -> int:
            return j * nx + i

        def u(j: int, i: int) -> int:
            return j * (nx + 1) + i

        def v(j: int, i: int) -> int:
            return j * nx + i

        def k(j: int, i: int) -> int:
            return j * (nx + 1) + i

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for j in range(ny):
            for i in range(nx):
                row = c(j, i)
                rows.extend((row, row))
                cols.extend((u(j, i), u(j, i + 1)))
                data.extend((-1.0 / dx, 1.0 / dx))
        exx_from_u = _matrix(rows, cols, data, (n_c, n_u))

        rows, cols, data = [], [], []
        for j in range(ny):
            for i in range(nx):
                row = c(j, i)
                rows.extend((row, row))
                cols.extend((v(j, i), v(j + 1, i)))
                data.extend((-1.0 / dy, 1.0 / dy))
        eyy_from_v = _matrix(rows, cols, data, (n_c, n_v))

        rows, cols, data = [], [], []
        for j in range(ny + 1):
            for i in range(nx + 1):
                row = k(j, i)
                if j == 0:
                    rows.append(row)
                    cols.append(u(0, i))
                    data.append(1.0 / dy)
                elif j == ny:
                    rows.append(row)
                    cols.append(u(ny - 1, i))
                    data.append(-1.0 / dy)
                else:
                    rows.extend((row, row))
                    cols.extend((u(j - 1, i), u(j, i)))
                    data.extend((-0.5 / dy, 0.5 / dy))
        exy_from_u = _matrix(rows, cols, data, (n_k, n_u))

        rows, cols, data = [], [], []
        for j in range(ny + 1):
            for i in range(nx + 1):
                row = k(j, i)
                if i == 0:
                    rows.append(row)
                    cols.append(v(j, 0))
                    data.append(1.0 / dx)
                elif i == nx:
                    rows.append(row)
                    cols.append(v(j, nx - 1))
                    data.append(-1.0 / dx)
                else:
                    rows.extend((row, row))
                    cols.extend((v(j, i - 1), v(j, i)))
                    data.extend((-0.5 / dx, 0.5 / dx))
        exy_from_v = _matrix(rows, cols, data, (n_k, n_v))

        rows, cols, data = [], [], []
        for j in range(ny):
            for i in range(1, nx):
                row = u(j, i)
                rows.extend((row, row))
                cols.extend((c(j, i - 1), c(j, i)))
                data.extend((-1.0 / dx, 1.0 / dx))
        center_x_to_u = _matrix(rows, cols, data, (n_u, n_c))

        rows, cols, data = [], [], []
        for j in range(ny):
            for i in range(nx + 1):
                row = u(j, i)
                rows.extend((row, row))
                cols.extend((k(j, i), k(j + 1, i)))
                data.extend((-1.0 / dy, 1.0 / dy))
        corner_y_to_u = _matrix(rows, cols, data, (n_u, n_k))

        rows, cols, data = [], [], []
        for j in range(ny + 1):
            for i in range(nx):
                row = v(j, i)
                rows.extend((row, row))
                cols.extend((k(j, i), k(j, i + 1)))
                data.extend((-1.0 / dx, 1.0 / dx))
        corner_x_to_v = _matrix(rows, cols, data, (n_v, n_k))

        rows, cols, data = [], [], []
        for j in range(1, ny):
            for i in range(nx):
                row = v(j, i)
                rows.extend((row, row))
                cols.extend((c(j - 1, i), c(j, i)))
                data.extend((-1.0 / dy, 1.0 / dy))
        center_y_to_v = _matrix(rows, cols, data, (n_v, n_c))

        rows, cols, data = [], [], []
        for j in range(ny):
            for i in range(nx):
                row = c(j, i)
                rows.extend((row, row))
                cols.extend((u(j, i), u(j, i + 1)))
                data.extend((-1.0 / dx, 1.0 / dx))
        divergence_u = _matrix(rows, cols, data, (n_c, n_u))

        rows, cols, data = [], [], []
        for j in range(ny):
            for i in range(nx):
                row = c(j, i)
                rows.extend((row, row))
                cols.extend((v(j, i), v(j + 1, i)))
                data.extend((-1.0 / dy, 1.0 / dy))
        divergence_v = _matrix(rows, cols, data, (n_c, n_v))

        rows, cols, data = [], [], []
        for j in range(ny):
            for i in range(nx + 1):
                row = u(j, i)
                for jj in (j, j + 1):
                    for ii in (max(i - 1, 0), min(i, nx - 1)):
                        rows.append(row)
                        cols.append(v(jj, ii))
                        data.append(0.25)
        v_to_u = _matrix(rows, cols, data, (n_u, n_v))

        rows, cols, data = [], [], []
        for j in range(ny + 1):
            for i in range(nx):
                row = v(j, i)
                for jj in (max(j - 1, 0), min(j, ny - 1)):
                    for ii in (i, i + 1):
                        rows.append(row)
                        cols.append(u(jj, ii))
                        data.append(0.25)
        u_to_v = _matrix(rows, cols, data, (n_v, n_u))

        interior_u = np.ones(grid.u_shape, dtype=bool)
        interior_v = np.ones(grid.v_shape, dtype=bool)
        interior_u[:, (0, -1)] = False
        interior_v[(0, -1), :] = False
        ice_interior_mask = np.concatenate((interior_u.ravel(), interior_v.ravel()))
        return cls(
            grid=grid,
            exx_from_u=exx_from_u,
            eyy_from_v=eyy_from_v,
            exy_from_u=exy_from_u,
            exy_from_v=exy_from_v,
            center_x_to_u=center_x_to_u,
            corner_y_to_u=corner_y_to_u,
            corner_x_to_v=corner_x_to_v,
            center_y_to_v=center_y_to_v,
            divergence_u=divergence_u,
            divergence_v=divergence_v,
            v_to_u=v_to_u,
            u_to_v=u_to_v,
            ice_interior_mask=ice_interior_mask,
        )

    def ice_velocity_vector(self, u_face: Array, v_face: Array) -> Array:
        self.grid.require_u(u_face)
        self.grid.require_v(v_face)
        return np.concatenate((u_face.ravel(), v_face.ravel()))

    def split_ice_velocity(self, vector: Array) -> tuple[Array, Array]:
        if vector.size != self.n_u + self.n_v:
            raise ValueError("冰速向量尺寸与 C 网格不一致")
        u = vector[: self.n_u].reshape(self.grid.u_shape)
        v = vector[self.n_u :].reshape(self.grid.v_shape)
        return u, v

    def implicit_upwind_u(self, transport_u: Array, transport_v_at_u: Array) -> SparseMatrix:
        self.grid.require_u(transport_u)
        self.grid.require_u(transport_v_at_u)
        nx, ny = self.grid.nx, self.grid.ny
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []

        def index(j: int, i: int) -> int:
            return j * (nx + 1) + i

        for j in range(ny):
            for i in range(1, nx):
                row = index(j, i)
                ax = float(transport_u[j, i])
                ay = float(transport_v_at_u[j, i])
                ax_p, ax_m = max(ax, 0.0), min(ax, 0.0)
                ay_p, ay_m = max(ay, 0.0), min(ay, 0.0)
                diag = ax_p / self.grid.dx - ax_m / self.grid.dx
                rows.extend((row, row))
                cols.extend((index(j, i - 1), index(j, i + 1)))
                data.extend((-ax_p / self.grid.dx, ax_m / self.grid.dx))

                diag += ay_p / self.grid.dy - ay_m / self.grid.dy
                if j == 0:
                    diag += ay_p / self.grid.dy
                else:
                    rows.append(row)
                    cols.append(index(j - 1, i))
                    data.append(-ay_p / self.grid.dy)
                if j == ny - 1:
                    diag -= ay_m / self.grid.dy
                else:
                    rows.append(row)
                    cols.append(index(j + 1, i))
                    data.append(ay_m / self.grid.dy)
                rows.append(row)
                cols.append(row)
                data.append(diag)
        return _matrix(rows, cols, data, (self.n_u, self.n_u))

    def implicit_upwind_v(self, transport_u_at_v: Array, transport_v: Array) -> SparseMatrix:
        self.grid.require_v(transport_u_at_v)
        self.grid.require_v(transport_v)
        nx, ny = self.grid.nx, self.grid.ny
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []

        def index(j: int, i: int) -> int:
            return j * nx + i

        for j in range(1, ny):
            for i in range(nx):
                row = index(j, i)
                ax = float(transport_u_at_v[j, i])
                ay = float(transport_v[j, i])
                ax_p, ax_m = max(ax, 0.0), min(ax, 0.0)
                ay_p, ay_m = max(ay, 0.0), min(ay, 0.0)
                diag = ay_p / self.grid.dy - ay_m / self.grid.dy
                rows.extend((row, row))
                cols.extend((index(j - 1, i), index(j + 1, i)))
                data.extend((-ay_p / self.grid.dy, ay_m / self.grid.dy))

                diag += ax_p / self.grid.dx - ax_m / self.grid.dx
                if i == 0:
                    diag += ax_p / self.grid.dx
                else:
                    rows.append(row)
                    cols.append(index(j, i - 1))
                    data.append(-ax_p / self.grid.dx)
                if i == nx - 1:
                    diag -= ax_m / self.grid.dx
                else:
                    rows.append(row)
                    cols.append(index(j, i + 1))
                    data.append(ax_m / self.grid.dx)
                rows.append(row)
                cols.append(row)
                data.append(diag)
        return _matrix(rows, cols, data, (self.n_v, self.n_v))
