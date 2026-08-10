"""多重网格（MG）预条件器：C 网格 u/v 面变量的 2:1 嵌套。

用法：把 `mg_preconditioner(A, grid, levels)` 返回的 LinearOperator 作为
BiCGSTAB/GMRES 的 M 参数。V 循环在细网格上用 ILU 平滑，粗网格算子用
Galerkin 三乘积 A_c = R A P 构造。

当前实现针对问题二海冰动量线性系统（40300×40300 稀疏矩阵）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, spilu


@dataclass(frozen=True)
class GridSpec:
    nx: int
    ny: int

    @property
    def u_size(self) -> int:
        return self.ny * (self.nx + 1)

    @property
    def v_size(self) -> int:
        return (self.ny + 1) * self.nx

    @property
    def size(self) -> int:
        return self.u_size + self.v_size


def _u_index(ny: int, nx: int) -> tuple[np.ndarray, np.ndarray]:
    """u 面 (i, j) → 全局索引。"""
    i, j = np.meshgrid(np.arange(ny), np.arange(nx + 1), indexing="ij")
    return i.ravel(), j.ravel()


def _v_index(ny: int, nx: int) -> tuple[np.ndarray, np.ndarray]:
    """v 面 (i, j) → 全局索引。"""
    i, j = np.meshgrid(np.arange(ny + 1), np.arange(nx), indexing="ij")
    return i.ravel(), j.ravel()


def build_restriction_prolongation(fine: GridSpec, coarse: GridSpec) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """构造 C 网格 2:1 嵌套的限制 R（细→粗）与延拓 P（粗→细）。

    u 面 (ny, nx+1)：
      - x 方向：粗面与细面重合（j_c = 2*j_f 处的细面即粗面）；
      - y 方向：粗面覆盖 2 条细面，限制取长度平均，延拓线性复制。
    v 面 (ny+1, nx) 对称处理。
    """
    ratio_x = fine.nx // coarse.nx
    ratio_y = fine.ny // coarse.ny
    assert ratio_x == 2 and ratio_y == 2

    # u 面
    fi_u, fj_u = _u_index(fine.ny, fine.nx)
    ci_u, cj_u = _u_index(coarse.ny, coarse.nx)
    coarse_u_map = {
        (int(i), int(j)): idx for idx, (i, j) in enumerate(zip(ci_u, cj_u))
    }
    r_rows: list[int] = []
    r_cols: list[int] = []
    r_vals: list[float] = []
    p_rows: list[int] = []
    p_cols: list[int] = []
    p_vals: list[float] = []
    # u: 细面 (i_f, j_f)；粗面 (i_c, j_c)，j_c = j_f（重合），i_c = i_f//2
    for k in range(len(fi_u)):
        i_f, j_f = int(fi_u[k]), int(fj_u[k])
        if j_f % 2 != 0:
            continue  # x 方向非重合面不进入限制
        i_c, j_c = i_f // 2, j_f // 2
        c_key = (i_c, j_c)
        if c_key not in coarse_u_map:
            continue
        c_idx = coarse_u_map[c_key]
        r_rows.append(c_idx)
        r_cols.append(k)
        r_vals.append(0.5)  # y 方向 2 行长度平均
    # 延拓 P：粗 u 面 → 细 u 面（y 方向线性复制到 2 行，x 方向重合）
    for idx, (i_c, j_c) in enumerate(zip(ci_u, cj_u)):
        for i_f in (2 * i_c, 2 * i_c + 1):
            if i_f >= fine.ny:
                continue
            j_f = 2 * j_c
            k = int(np.flatnonzero((fi_u == i_f) & (fj_u == j_f))[0])
            p_rows.append(k)
            p_cols.append(idx)
            p_vals.append(1.0)

    # v 面
    fi_v, fj_v = _v_index(fine.ny, fine.nx)
    ci_v, cj_v = _v_index(coarse.ny, coarse.nx)
    coarse_v_map = {
        (int(i), int(j)): idx for idx, (i, j) in enumerate(zip(ci_v, cj_v))
    }
    for k in range(len(fi_v)):
        i_f, j_f = int(fi_v[k]), int(fj_v[k])
        if i_f % 2 != 0:
            continue  # y 方向非重合面不进入限制
        i_c, j_c = i_f // 2, j_f // 2
        c_key = (i_c, j_c)
        if c_key not in coarse_v_map:
            continue
        c_idx = coarse_v_map[c_key]
        r_rows.append(coarse.u_size + c_idx)
        r_cols.append(fine.u_size + k)
        r_vals.append(0.5)  # x 方向 2 列长度平均
    for idx, (i_c, j_c) in enumerate(zip(ci_v, cj_v)):
        i_f = 2 * i_c
        for j_f in (2 * j_c, 2 * j_c + 1):
            if j_f >= fine.nx:
                continue
            k = int(np.flatnonzero((fi_v == i_f) & (fj_v == j_f))[0])
            p_rows.append(fine.u_size + k)
            p_cols.append(coarse.u_size + idx)
            p_vals.append(1.0)

    r_matrix = sparse.coo_matrix(
        (r_vals, (r_rows, r_cols)),
        shape=(coarse.size, fine.size),
    ).tocsr()
    p_matrix = sparse.coo_matrix(
        (p_vals, (p_rows, p_cols)),
        shape=(fine.size, coarse.size),
    ).tocsr()
    return r_matrix, p_matrix


def build_hierarchy(fine: GridSpec, levels: int) -> list[tuple[GridSpec, sparse.csr_matrix, sparse.csr_matrix]]:
    """返回逐层 (GridSpec, R, P)。"""
    hierarchy: list[tuple[GridSpec, sparse.csr_matrix, sparse.csr_matrix]] = []
    current = fine
    for _ in range(levels):
        if current.nx % 2 != 0 or current.ny % 2 != 0:
            break
        coarse = GridSpec(nx=current.nx // 2, ny=current.ny // 2)
        r_matrix, p_matrix = build_restriction_prolongation(current, coarse)
        hierarchy.append((current, r_matrix, p_matrix))
        current = coarse
    return hierarchy


class MultigridPreconditioner:
    """MG V 循环预条件器（ILU 平滑 + Galerkin 粗算子）。"""

    def __init__(
        self,
        matrix: sparse.csr_matrix,
        fine: GridSpec,
        *,
        levels: int = 2,
        nu1: int = 1,
        nu2: int = 1,
        ilu_drop_tol: float = 1.0e-4,
        ilu_fill_factor: float = 16.0,
    ):
        self.fine = fine
        self.nu1 = nu1
        self.nu2 = nu2
        hierarchy = build_hierarchy(fine, levels)
        self.hierarchy = hierarchy
        # 每层：GridSpec、限制 R、延拓 P、平滑器、本层矩阵
        self.level_data: list[dict[str, Any]] = []
        current_matrix = matrix.tocsr()
        for grid, r_matrix, p_matrix in hierarchy:
            ilu = spilu(
                current_matrix.tocsc(),
                drop_tol=ilu_drop_tol,
                fill_factor=ilu_fill_factor,
            )
            smoother = LinearOperator(current_matrix.shape, matvec=ilu.solve)
            coarse_matrix = (r_matrix @ current_matrix @ p_matrix).tocsr()
            self.level_data.append(
                {
                    "grid": grid,
                    "R": r_matrix,
                    "P": p_matrix,
                    "smoother": smoother,
                    "A": current_matrix,
                }
            )
            current_matrix = coarse_matrix
        # 最粗层
        self.coarsest_matrix = current_matrix
        self.coarsest_ilu = spilu(
            current_matrix.tocsc(),
            drop_tol=1.0e-6,
            fill_factor=20.0,
        )
        self.coarsest_solve = LinearOperator(
            current_matrix.shape, matvec=self.coarsest_ilu.solve
        )

    def _smooth(
        self,
        matrix: sparse.csr_matrix,
        smoother: Any,
        rhs: np.ndarray,
        x: np.ndarray,
        nu: int,
    ) -> np.ndarray:
        for _ in range(nu):
            residual = rhs - matrix @ x
            x = x + smoother @ residual
        return x

    def v_cycle(self, rhs: np.ndarray, x: np.ndarray, level: int = 0) -> np.ndarray:
        if level == len(self.level_data):
            return self.coarsest_solve @ rhs
        data = self.level_data[level]
        a_matrix = data["A"]
        smoother = data["smoother"]
        r_matrix = data["R"]
        p_matrix = data["P"]
        x = self._smooth(a_matrix, smoother, rhs, x, self.nu1)
        residual = rhs - a_matrix @ x
        coarse_rhs = r_matrix @ residual
        coarse_x = np.zeros(r_matrix.shape[0])
        coarse_x = self.v_cycle(coarse_rhs, coarse_x, level + 1)
        x = x + p_matrix @ coarse_x
        x = self._smooth(a_matrix, smoother, rhs, x, self.nu2)
        return x


def mg_preconditioner(matrix: sparse.csr_matrix, fine: GridSpec, **kwargs: Any) -> LinearOperator:
    """构造 MG 预条件器 LinearOperator。"""
    preconditioner = MultigridPreconditioner(matrix, fine, **kwargs)
    size = fine.size

    def apply(rhs: np.ndarray) -> np.ndarray:
        x = np.zeros(size)
        return preconditioner.v_cycle(rhs, x, 0)

    return LinearOperator((size, size), matvec=apply)
