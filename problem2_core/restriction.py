"""嵌套 C 网格间的守恒限制算子。"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .grid import CGrid

Array = NDArray[np.float64]


def _nested_ratios(fine: CGrid, coarse: CGrid) -> tuple[int, int]:
    if fine.nx % coarse.nx != 0 or fine.ny % coarse.ny != 0:
        raise ValueError("细粗网格单元数必须为整数嵌套关系")
    ratio_x = fine.nx // coarse.nx
    ratio_y = fine.ny // coarse.ny
    if ratio_x <= 0 or ratio_y <= 0:
        raise ValueError("限制算子要求 fine 网格不粗于 coarse 网格")
    if not np.isclose(coarse.dx, ratio_x * fine.dx) or not np.isclose(
        coarse.dy, ratio_y * fine.dy
    ):
        raise ValueError("细粗网格间距与单元数比例不一致")
    if not np.isclose(fine.x0, coarse.x0) or not np.isclose(fine.y0, coarse.y0):
        raise ValueError("细粗网格原点必须一致")
    return ratio_x, ratio_y


def restrict_center_area_average(
    fine: CGrid, coarse: CGrid, fine_field: Array
) -> Array:
    """中心量按粗单元覆盖面积平均。"""

    fine.require_center(fine_field)
    ratio_x, ratio_y = _nested_ratios(fine, coarse)
    return fine_field.reshape(
        coarse.ny, ratio_y, coarse.nx, ratio_x
    ).mean(axis=(1, 3))


def restrict_u_face_length_average(
    fine: CGrid, coarse: CGrid, fine_u: Array
) -> Array:
    """东向面速度沿同一粗竖直面的细面段作长度平均。"""

    fine.require_u(fine_u)
    ratio_x, ratio_y = _nested_ratios(fine, coarse)
    coincident_faces = fine_u[:, ::ratio_x]
    return coincident_faces.reshape(
        coarse.ny, ratio_y, coarse.nx + 1
    ).mean(axis=1)


def restrict_v_face_length_average(
    fine: CGrid, coarse: CGrid, fine_v: Array
) -> Array:
    """北向面速度沿同一粗水平面的细面段作长度平均。"""

    fine.require_v(fine_v)
    ratio_x, ratio_y = _nested_ratios(fine, coarse)
    coincident_faces = fine_v[::ratio_y, :]
    return coincident_faces.reshape(
        coarse.ny + 1, coarse.nx, ratio_x
    ).mean(axis=2)
