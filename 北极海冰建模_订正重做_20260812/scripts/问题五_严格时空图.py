"""问题五：到最近可通行点的严格时间展开图最短路。

每个状态为 (时间层, 空间格)，移动边先按物理航速完成航行，再等待到下一个
时间层。所有边耗油非负，因此 Dijkstra 给出该离散时空图上的全局最优解。
"""

from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DELIVERY_ROOT = Path(__file__).resolve().parents[1]
VMAX = 15.0 * 0.5144
A_LIMIT = 0.6
Q0 = 0.5
Q1 = 0.005
T_START_H = 24.0
T_END_H = 72.0
LX = 20_000.0
LY = 10_000.0


def load_forecast(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        thickness = np.asarray(archive["thickness"], dtype=float)
        times = np.asarray(archive["time_hours"], dtype=float)
    return {
        "concentration": np.clip(thickness, 0.0, 1.0),
        "time_hours": times,
    }


def concentration_at(fields: dict[str, np.ndarray], time_h: float) -> np.ndarray:
    times = fields["time_hours"]
    upper = int(np.searchsorted(times, time_h, side="right"))
    if upper <= 0:
        return fields["concentration"][0]
    if upper >= times.size:
        return fields["concentration"][-1]
    lower = upper - 1
    weight = (time_h - times[lower]) / (times[upper] - times[lower])
    return (
        (1.0 - weight) * fields["concentration"][lower]
        + weight * fields["concentration"][upper]
    )


def concentration_xy(
    fields: dict[str, np.ndarray],
    time_h: float,
    x: float,
    y: float,
) -> float:
    field = concentration_at(fields, time_h)
    ny, nx = field.shape
    dx, dy = LX / nx, LY / ny
    fx = np.clip(x / dx - 0.5, 0.0, nx - 1.0)
    fy = np.clip(y / dy - 0.5, 0.0, ny - 1.0)
    j0, i0 = int(np.floor(fx)), int(np.floor(fy))
    j1, i1 = min(j0 + 1, nx - 1), min(i0 + 1, ny - 1)
    wx, wy = fx - j0, fy - i0
    return float(
        (1.0 - wy) * ((1.0 - wx) * field[i0, j0] + wx * field[i0, j1])
        + wy * ((1.0 - wx) * field[i1, j0] + wx * field[i1, j1])
    )


def select_nodes(
    fields: dict[str, np.ndarray],
    start: tuple[float, float],
    destination: tuple[float, float],
) -> dict[str, Any]:
    a0 = concentration_at(fields, T_START_H)
    ny, nx = a0.shape
    dx, dy = LX / nx, LY / ny
    xs = (np.arange(nx) + 0.5) * dx
    ys = (np.arange(ny) + 0.5) * dy
    if not concentration_xy(fields, T_START_H, *start) < A_LIMIT:
        raise ValueError("起点在 t=24 h 不可通行")
    nearest_i = np.argsort(np.abs(ys - start[1]))[:2]
    nearest_j = np.argsort(np.abs(xs - start[0]))[:2]
    start_candidates = [(int(i), int(j)) for i in nearest_i for j in nearest_j]
    candidates = []
    for i in range(ny):
        for j in range(nx):
            if a0[i, j] < A_LIMIT:
                distance = float(np.hypot(xs[j] - destination[0], ys[i] - destination[1]))
                candidates.append((distance, -xs[j], abs(ys[i] - destination[1]), i, j))
    if not candidates:
        raise ValueError("t=24 h 不存在 A<0.6 的可通行格")
    distance, _, _, i_target, j_target = min(candidates)
    return {
        "xs": xs,
        "ys": ys,
        "dx": dx,
        "dy": dy,
        "start_candidates": start_candidates,
        "target_node": (i_target, j_target),
        "target_distance_to_E_m": distance,
    }


def edge_physics(
    fields: dict[str, np.ndarray],
    node: tuple[int, int],
    neighbor: tuple[int, int],
    depart_h: float,
    distance_m: float,
) -> tuple[float, float, float] | None:
    """返回 (航行时间 h, 航行耗油 t, 保守代表 A)。"""
    a_edge = max(
        float(concentration_at(fields, depart_h)[node]),
        float(concentration_at(fields, depart_h)[neighbor]),
    )
    for _ in range(4):
        if a_edge >= A_LIMIT:
            return None
        speed = VMAX * (1.0 - a_edge)
        travel_h = distance_m / speed / 3600.0
        middle_h = depart_h + 0.5 * travel_h
        arrival_h = depart_h + travel_h
        a_edge = max(
            a_edge,
            float(concentration_at(fields, middle_h)[node]),
            float(concentration_at(fields, middle_h)[neighbor]),
            float(concentration_at(fields, arrival_h)[node]),
            float(concentration_at(fields, arrival_h)[neighbor]),
        )
    if a_edge >= A_LIMIT:
        return None
    speed = VMAX * (1.0 - a_edge)
    travel_h = distance_m / speed / 3600.0
    fuel_t = (Q0 + Q1 * speed**2) * travel_h
    return travel_h, fuel_t, a_edge


def virtual_start_edge_physics(
    fields: dict[str, np.ndarray],
    start: tuple[float, float],
    neighbor: tuple[int, int],
    xs: np.ndarray,
    ys: np.ndarray,
) -> tuple[float, float, float] | None:
    distance_m = float(
        np.hypot(xs[neighbor[1]] - start[0], ys[neighbor[0]] - start[1])
    )
    a_edge = max(
        concentration_xy(fields, T_START_H, *start),
        float(concentration_at(fields, T_START_H)[neighbor]),
    )
    for _ in range(4):
        if a_edge >= A_LIMIT:
            return None
        speed = VMAX * (1.0 - a_edge)
        travel_h = distance_m / speed / 3600.0
        arrival_h = T_START_H + travel_h
        middle_h = T_START_H + 0.5 * travel_h
        a_edge = max(
            a_edge,
            concentration_xy(fields, middle_h, *start),
            float(concentration_at(fields, middle_h)[neighbor]),
            float(concentration_at(fields, arrival_h)[neighbor]),
        )
    if a_edge >= A_LIMIT:
        return None
    speed = VMAX * (1.0 - a_edge)
    travel_h = distance_m / speed / 3600.0
    return travel_h, (Q0 + Q1 * speed**2) * travel_h, a_edge


def solve(
    fields: dict[str, np.ndarray],
    *,
    tau_seconds: float,
    start: tuple[float, float] = (1000.0, 5000.0),
    destination: tuple[float, float] = (19000.0, 5000.0),
) -> dict[str, Any]:
    nodes = select_nodes(fields, start, destination)
    xs, ys = nodes["xs"], nodes["ys"]
    ny, nx = len(ys), len(xs)
    target_node = nodes["target_node"]
    total_layers = int(np.floor((T_END_H - T_START_H) * 3600.0 / tau_seconds))

    def state_index(layer: int, i: int, j: int) -> int:
        return (layer * ny + i) * nx + j

    def decode(index: int) -> tuple[int, int, int]:
        layer, rem = divmod(index, ny * nx)
        i, j = divmod(rem, nx)
        return layer, i, j

    fuel: dict[int, float] = {}
    previous: dict[int, int] = {}
    edge_meta: dict[int, dict[str, float]] = {}
    heap: list[tuple[float, int]] = []
    for candidate_node in nodes["start_candidates"]:
        physics = virtual_start_edge_physics(
            fields, start, candidate_node, xs, ys
        )
        if physics is None:
            continue
        travel_h, travel_fuel, a_edge = physics
        layer_jump = max(1, int(np.ceil(travel_h * 3600.0 / tau_seconds)))
        snapped_h = layer_jump * tau_seconds / 3600.0
        wait_h = snapped_h - travel_h
        snapped_arrival_h = T_START_H + snapped_h
        if (
            float(concentration_at(fields, snapped_arrival_h)[candidate_node])
            >= A_LIMIT
        ):
            continue
        candidate_index = state_index(layer_jump, *candidate_node)
        candidate_fuel = travel_fuel + Q0 * wait_h
        if candidate_fuel < fuel.get(candidate_index, np.inf):
            fuel[candidate_index] = candidate_fuel
            previous[candidate_index] = -1
            edge_meta[candidate_index] = {
                "travel_h": travel_h,
                "wait_h": wait_h,
                "a_edge": a_edge,
                "distance_m": float(
                    np.hypot(
                        xs[candidate_node[1]] - start[0],
                        ys[candidate_node[0]] - start[1],
                    )
                ),
            }
            heapq.heappush(heap, (candidate_fuel, candidate_index))
    target_index = None
    neighbor_offsets = [
        (di, dj)
        for di in (-1, 0, 1)
        for dj in (-1, 0, 1)
        if not (di == 0 and dj == 0)
    ]

    # 任意路径的基础耗油不低于 Q0*T；已有可行解出现后可据此严格剪枝。
    incumbent = np.inf
    while heap:
        current_fuel, current = heapq.heappop(heap)
        if current_fuel != fuel.get(current):
            continue
        layer, i, j = decode(current)
        elapsed_h = layer * tau_seconds / 3600.0
        if current_fuel + Q0 * 0.0 >= incumbent:
            continue
        if (i, j) == target_node:
            incumbent = current_fuel
            target_index = current
            break
        if layer >= total_layers:
            continue
        depart_h = T_START_H + elapsed_h
        for di, dj in neighbor_offsets:
            ni, nj = i + di, j + dj
            if not (0 <= ni < ny and 0 <= nj < nx):
                continue
            distance = float(np.hypot(dj * nodes["dx"], di * nodes["dy"]))
            physics = edge_physics(fields, (i, j), (ni, nj), depart_h, distance)
            if physics is None:
                continue
            travel_h, travel_fuel, a_edge = physics
            travel_seconds = travel_h * 3600.0
            layer_jump = max(1, int(np.ceil(travel_seconds / tau_seconds)))
            next_layer = layer + layer_jump
            if next_layer > total_layers:
                continue
            snapped_seconds = layer_jump * tau_seconds
            wait_h = (snapped_seconds - travel_seconds) / 3600.0
            arrival_h = T_START_H + next_layer * tau_seconds / 3600.0
            if float(concentration_at(fields, arrival_h)[ni, nj]) >= A_LIMIT:
                continue
            added_fuel = travel_fuel + Q0 * wait_h
            candidate = current_fuel + added_fuel
            if candidate >= incumbent:
                continue
            next_index = state_index(next_layer, ni, nj)
            if candidate + 1.0e-14 < fuel.get(next_index, np.inf):
                fuel[next_index] = candidate
                previous[next_index] = current
                edge_meta[next_index] = {
                    "travel_h": travel_h,
                    "wait_h": wait_h,
                    "a_edge": a_edge,
                    "distance_m": distance,
                }
                heapq.heappush(heap, (candidate, next_index))

        # 显式等待边。
        next_layer = layer + 1
        wait_arrival_h = T_START_H + next_layer * tau_seconds / 3600.0
        if (
            next_layer <= total_layers
            and float(concentration_at(fields, wait_arrival_h)[i, j]) < A_LIMIT
        ):
            candidate = current_fuel + Q0 * tau_seconds / 3600.0
            next_index = state_index(next_layer, i, j)
            if candidate + 1.0e-14 < fuel.get(next_index, np.inf):
                fuel[next_index] = candidate
                previous[next_index] = current
                edge_meta[next_index] = {
                    "travel_h": 0.0,
                    "wait_h": tau_seconds / 3600.0,
                    "a_edge": float(concentration_at(fields, depart_h)[i, j]),
                    "distance_m": 0.0,
                }
                heapq.heappush(heap, (candidate, next_index))

    if target_index is None:
        return {"success": False, "message": "离散时空图内无可行路径"}

    path = []
    cursor = target_index
    while cursor >= 0:
        layer, i, j = decode(cursor)
        time_h = T_START_H + layer * tau_seconds / 3600.0
        path.append(
            {
                "x_m": float(xs[j]),
                "y_m": float(ys[i]),
                "time_h": time_h,
                "concentration": float(concentration_at(fields, time_h)[i, j]),
                "cumulative_fuel_t": float(fuel[cursor]),
                "edge": edge_meta.get(cursor),
            }
        )
        cursor = previous[cursor]
    path.reverse()
    path.insert(
        0,
        {
            "x_m": float(start[0]),
            "y_m": float(start[1]),
            "time_h": T_START_H,
            "concentration": concentration_xy(fields, T_START_H, *start),
            "cumulative_fuel_t": 0.0,
            "edge": None,
        },
    )
    arrival_h = path[-1]["time_h"]
    return {
        "success": True,
        "algorithm": "full_time_expanded_dijkstra",
        "discrete_global_optimum": True,
        "tau_seconds": tau_seconds,
        "total_fuel_t": float(fuel[target_index]),
        "total_time_h": float(arrival_h - T_START_H),
        "arrival_time_h": arrival_h,
        "start_requested_m": list(start),
        "first_grid_m": [path[1]["x_m"], path[1]["y_m"]],
        "destination_E_m": list(destination),
        "target_grid_m": [path[-1]["x_m"], path[-1]["y_m"]],
        "target_distance_to_E_m": nodes["target_distance_to_E_m"],
        "target_definition": "t=24 h 时距 E 最近且 A<0.6 的网格中心",
        "waypoints": path,
        "states_settled": len(fuel),
    }


def plot_result(
    fields: dict[str, np.ndarray], result: dict[str, Any], output: Path
) -> None:
    if not result.get("success"):
        return
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharex=True, sharey=True)
    path_x = [row["x_m"] for row in result["waypoints"]]
    path_y = [row["y_m"] for row in result["waypoints"]]
    for ax, time_h in zip(axes, (24.0, 48.0, 72.0)):
        field = concentration_at(fields, time_h)
        ny, nx = field.shape
        xs = (np.arange(nx) + 0.5) * LX / nx
        ys = (np.arange(ny) + 0.5) * LY / ny
        mesh = ax.pcolormesh(xs, ys, field, shading="auto", cmap="Blues", vmin=0, vmax=1)
        ax.contour(xs, ys, field, levels=[A_LIMIT], colors="red", linewidths=1.2)
        ax.plot(path_x, path_y, "k-", lw=1.8)
        ax.plot(path_x[0], path_y[0], "go", label="S")
        ax.plot(path_x[-1], path_y[-1], "m*", ms=12, label="nearest A<0.6")
        ax.plot(19_000, 5_000, "rX", ms=9, label="E blocked")
        ax.set_title(f"t={time_h:.0f} h")
        ax.set_xlabel("x / m")
        ax.set_ylabel("y / m")
        ax.legend(fontsize=8)
        fig.colorbar(mesh, ax=ax, label="A")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--forecast",
        type=Path,
        default=DELIVERY_ROOT / "results" / "统一72h预报.npz",
    )
    parser.add_argument("--tau-seconds", type=float, default=30.0)
    args = parser.parse_args()
    fields = load_forecast(args.forecast)
    result = solve(fields, tau_seconds=args.tau_seconds)
    result_path = DELIVERY_ROOT / "results" / "问题五_严格时空图.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    plot_result(fields, result, DELIVERY_ROOT / "figures" / "问题五_最优航线.png")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
