"""Schwarz 区域分解（条带）原型与并行求解器。

设计：把 C 网格沿 y 方向切成若干条带，每个条带在各自进程内做局部
Picard--BiCGSTAB（以 ILU 为局部预条件器），条带之间交换 1 层 halo 的
冰速，迭代到界面一致。当前先提供串行原型 `schwarz_solve_serial` 用于
验证收敛性，再提供多进程版本 `schwarz_solve_parallel`。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, bicgstab, spilu


@dataclass(frozen=True)
class StripPartition:
    """沿 y 方向的条带划分。

    u 面速度形状 (ny, nx+1)，v 面速度形状 (ny+1, nx)。
    状态向量 x = [u.ravel(), v.ravel()]。
    """

    ny: int
    nx: int
    n_strips: int
    halo: int = 1

    @property
    def u_offset(self) -> int:
        return self.ny * (self.nx + 1)

    @property
    def size(self) -> int:
        return self.u_offset + (self.ny + 1) * self.nx

    def strip_rows(self, strip: int) -> tuple[np.ndarray, np.ndarray]:
        """返回该条带的 (u 行索引, v 行索引)，含 halo。"""
        base = self.ny // self.n_strips
        i0 = strip * base
        i1 = i0 + base  # 不含
        if strip == self.n_strips - 1:
            i1 = self.ny
        u_lo = max(i0 - self.halo, 0)
        u_hi = min(i1 + self.halo, self.ny)
        v_lo = max(i0 - self.halo, 0)
        v_hi = min(i1 + self.halo, self.ny + 1)
        u_rows = np.arange(u_lo * (self.nx + 1), u_hi * (self.nx + 1))
        v_rows = self.u_offset + np.arange(v_lo * self.nx, v_hi * self.nx)
        return u_rows, v_rows

    def interior_mask(self, strip: int, u_rows: np.ndarray, v_rows: np.ndarray) -> np.ndarray:
        """标记条带内部自由度（非 halo），用于加性 Schwarz 组合。"""
        base = self.ny // self.n_strips
        i0 = strip * base
        i1 = i0 + base
        if strip == self.n_strips - 1:
            i1 = self.ny
        mask = np.ones(len(u_rows) + len(v_rows), dtype=bool)
        u_n = self.nx + 1
        v_n = self.nx
        for j, row in enumerate(u_rows):
            row_index = row // u_n
            if row_index < i0 or row_index >= i1:
                mask[j] = False
        for j, row in enumerate(v_rows):
            row_index = (row - self.u_offset) // v_n
            if row_index < i0 or row_index >= i1:
                mask[len(u_rows) + j] = False
        return mask


def _extract_strip_system(
    matrix: sparse.csr_matrix,
    rhs: np.ndarray,
    partition: StripPartition,
    strip: int,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    """提取条带局部系统 A_ii x_i + A_ih x_h = b_i。"""
    u_rows, v_rows = partition.strip_rows(strip)
    rows = np.concatenate([u_rows, v_rows])
    mask = partition.interior_mask(strip, u_rows, v_rows)
    interior = rows[mask]
    halo = rows[~mask]
    local = matrix[interior][:, interior].tocsr()
    cross = matrix[interior][:, halo].tocsr()
    local_rhs = rhs[interior]
    return local, cross, local_rhs, interior, halo


def schwarz_solve_serial(
    matrix: sparse.csr_matrix,
    rhs: np.ndarray,
    partition: StripPartition,
    *,
    rtol: float = 1.0e-8,
    max_iter: int = 200,
    ilu_drop_tol: float = 1.0e-4,
    ilu_fill_factor: float = 16.0,
    relaxation: float = 1.0,
) -> dict[str, Any]:
    """串行加性 Schwarz（块 Jacobi）求解 A x = b。"""
    systems = [
        _extract_strip_system(matrix, rhs, partition, strip)
        for strip in range(partition.n_strips)
    ]
    preconditioners = []
    for local, _cross, _b, _interior, _halo in systems:
        ilu = spilu(local.tocsc(), drop_tol=ilu_drop_tol, fill_factor=ilu_fill_factor)
        preconditioners.append(LinearOperator(local.shape, matvec=ilu.solve))
    x = np.zeros(partition.size)
    b_norm = float(np.linalg.norm(rhs))
    if b_norm <= 1.0e-14:
        return {"converged": True, "iterations": 0, "x": x, "residual": 0.0}
    for iteration in range(1, max_iter + 1):
        new_x = x.copy()
        for strip in range(partition.n_strips):
            local, cross, local_rhs, interior, halo = systems[strip]
            rhs_local = local_rhs - cross @ x[halo]
            solution, info = bicgstab(
                local,
                rhs_local,
                rtol=1.0e-8,
                atol=0.0,
                maxiter=100,
                M=preconditioners[strip],
                x0=x[interior],
            )
            if info != 0:
                return {"converged": False, "iterations": iteration, "x": x, "residual": float("inf")}
            new_x[interior] = (1.0 - relaxation) * x[interior] + relaxation * solution
        x = new_x
        residual = float(np.linalg.norm(matrix @ x - rhs) / b_norm)
        if residual <= rtol:
            return {"converged": True, "iterations": iteration, "x": x, "residual": residual}
    residual = float(np.linalg.norm(matrix @ x - rhs) / b_norm)
    return {"converged": residual <= rtol, "iterations": max_iter, "x": x, "residual": residual}
