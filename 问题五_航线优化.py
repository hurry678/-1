"""问题五：时变冰情下 LNG 船全局最优航线（时变 Dijkstra）。

输入：72 h 正演预报的逐小时冰厚场（npz），输出 t∈[24,72] h 的冰密集度
场；船舶 S(1000,5000) → E(19000,5000)，Vmax=15 kn，A>0.6 禁行，
Vs=Vmax(1-A)，油耗 q=q0+q1*Vs²。最小燃油，可等待。
"""

from __future__ import annotations

import argparse
import heapq
from datetime import datetime
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "output" / "problem5"

VMAX = 15.0 * 0.5144  # m/s
A_LIMIT = 0.6
Q0 = 0.5  # t/h
Q1 = 0.005  # t*h/m^2
T_START_H = 24.0
T_END_H = 72.0
TAU_MIN = 5.0  # min 时间层
FORECAST_HOURS = np.arange(0.0, 73.0, 1.0)


def load_forecast(path: Path) -> dict[str, Any]:
    """读取 72 h 预报 npz：thickness (nt, ny, nx)，time_hours (nt,)。"""
    with np.load(path) as archive:
        thickness = np.asarray(archive["thickness"], dtype=float)
        time_hours = np.asarray(archive["time_hours"], dtype=float)
    if thickness.ndim == 3:
        thickness = np.asarray(thickness)
    else:
        raise ValueError("thickness 必须为三维 (nt, ny, nx)")
    concentration = np.clip(thickness / 1.0, 0.0, 1.0)
    return {"thickness": thickness, "concentration": concentration, "time_hours": time_hours}


def concentration_at(fields: dict[str, Any], t_hours: float) -> np.ndarray:
    times = fields["time_hours"]
    index = min(max(int(np.searchsorted(times, t_hours)), 0), len(times) - 1)
    if index >= len(times) - 1 or times[index] == t_hours:
        return fields["concentration"][index]
    t0, t1 = times[index], times[index + 1]
    w = (t_hours - t0) / max(t1 - t0, 1.0e-12)
    return (1.0 - w) * fields["concentration"][index] + w * fields["concentration"][index + 1]


def build_graph(
    fields: dict[str, Any],
    *,
    tau_min: float = TAU_MIN,
    start: tuple[float, float] = (1000.0, 5000.0),
    end: tuple[float, float] = (19000.0, 5000.0),
) -> dict[str, Any]:
    concentration = concentration_at(fields, T_START_H)
    ny, nx = concentration.shape
    xs = 250.0 + 500.0 * np.arange(nx)
    ys = 250.0 + 500.0 * np.arange(ny)
    passable = concentration <= A_LIMIT
    start_node = (int(np.argmin(np.abs(ys - start[1]))), int(np.argmin(np.abs(xs - start[0]))))
    end_node = (int(np.argmin(np.abs(ys - end[1]))), int(np.argmin(np.abs(xs - end[0]))))
    if not passable[start_node]:
        raise ValueError("起点所在网格不可通行")
    true_end = end_node
    endpoint_deviation_m = 0.0
    if not passable[end_node]:
        candidates = [
            (np.hypot(xs[j] - xs[end_node[1]], ys[i] - ys[end_node[0]]), i, j)
            for i in range(ny)
            for j in range(nx)
            if passable[i, j]
        ]
        if not candidates:
            raise ValueError("终点附近无可通行网格")
        _, i, j = min(candidates, key=lambda item: item[0])
        end_node = (i, j)
        endpoint_deviation_m = float(
            np.hypot(xs[j] - xs[true_end[1]], ys[i] - ys[true_end[0]])
        )
    neighbors = []
    for i in range(ny):
        for j in range(nx):
            moves = []
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < ny and 0 <= nj < nx:
                        moves.append((ni, nj, np.hypot(di * 500.0, dj * 500.0)))
            neighbors.append(moves)
    return {
        "xs": xs,
        "ys": ys,
        "passable": passable,
        "start_node": start_node,
        "end_node": end_node,
        "true_end_node": true_end,
        "endpoint_deviation_m": endpoint_deviation_m,
        "neighbors": neighbors,
        "ny": ny,
        "nx": nx,
        "tau_min": tau_min,
    }


def route_optimize(
    fields: dict[str, Any],
    *,
    tau_min: float = TAU_MIN,
    start: tuple[float, float] = (1000.0, 5000.0),
    end: tuple[float, float] = (19000.0, 5000.0),
) -> dict[str, Any]:
    graph = build_graph(fields, tau_min=tau_min, start=start, end=end)
    start_node = graph["start_node"]
    end_node = graph["end_node"]
    ny, nx = graph["ny"], graph["nx"]
    tau_seconds = tau_min * 60.0
    t_start = T_START_H * 3600.0
    t_end = T_END_H * 3600.0
    layers = int(np.ceil((t_end - t_start) / tau_seconds))
    index_of = lambda node, layer: (node[0] * nx + node[1]) * (layers + 1) + layer
    total_nodes = ny * nx * (layers + 1)
    fuel = np.full(total_nodes, np.inf)
    best_time = np.full(total_nodes, np.inf)
    prev = np.full(total_nodes, -1, dtype=np.int64)
    start_index = index_of(start_node, 0)
    fuel[start_index] = 0.0
    best_time[start_index] = t_start
    heap = [(0.0, start_index, t_start)]
    while heap:
        current_fuel, current_index, current_time = heapq.heappop(heap)
        if current_fuel > fuel[current_index] + 1.0e-12 or current_time > best_time[current_index] + 1.0e-6:
            continue
        block = current_index // (layers + 1)
        layer = current_index % (layers + 1)
        node = (block // nx, block % nx)
        if node == end_node:
            break
        if layer >= layers:
            continue
        t_now = current_time
        a_departure = float(concentration_at(fields, t_now / 3600.0)[node])
        if a_departure > A_LIMIT:
            continue
        speed = VMAX * (1.0 - a_departure)
        for ni, nj, distance in graph["neighbors"][node[0] * nx + node[1]]:
            move_time = distance / speed
            arrival_time = t_now + move_time
            if arrival_time > t_end:
                continue
            a_mid = float(concentration_at(fields, (t_now + 0.5 * move_time) / 3600.0)[ni, nj])
            a_arrival = float(concentration_at(fields, arrival_time / 3600.0)[ni, nj])
            if a_arrival > A_LIMIT:
                continue
            if max(a_departure, a_mid, a_arrival) > A_LIMIT:
                continue
            arrival_layer = int(np.floor((arrival_time - t_start) / tau_seconds))
            arrival_layer = max(0, min(arrival_layer, layers))
            vs = VMAX * (1.0 - a_arrival)
            fuel_add = (Q0 + Q1 * vs**2) * (move_time / 3600.0)
            next_index = index_of((ni, nj), arrival_layer)
            candidate = current_fuel + fuel_add
            if candidate < fuel[next_index] - 1.0e-12 or (
                abs(candidate - fuel[next_index]) <= 1.0e-12
                and arrival_time < best_time[next_index] - 1.0e-6
            ):
                fuel[next_index] = candidate
                best_time[next_index] = arrival_time
                prev[next_index] = current_index
                heapq.heappush(heap, (candidate, next_index, arrival_time))
        wait_index = index_of(node, layer + 1)
        wait_end_time = t_now + tau_seconds
        if wait_end_time <= t_end:
            a_wait_end = float(concentration_at(fields, wait_end_time / 3600.0)[node])
            if a_wait_end <= A_LIMIT:
                wait_fuel = current_fuel + Q0 * (tau_seconds / 3600.0)
                if wait_fuel < fuel[wait_index] - 1.0e-12 or (
                    abs(wait_fuel - fuel[wait_index]) <= 1.0e-12
                    and wait_end_time < best_time[wait_index] - 1.0e-6
                ):
                    fuel[wait_index] = wait_fuel
                    best_time[wait_index] = wait_end_time
                    prev[wait_index] = current_index
                    heapq.heappush(heap, (wait_fuel, wait_index, wait_end_time))
    best_indices = [
        index_of(end_node, layer) for layer in range(layers + 1)
        if np.isfinite(fuel[index_of(end_node, layer)])
    ]
    if not best_indices:
        return {"success": False, "message": "72 h 内无法到达终点"}
    end_index = min(best_indices, key=lambda idx: (fuel[idx], best_time[idx]))
    path_indices = []
    cursor = end_index
    while cursor >= 0:
        path_indices.append(cursor)
        cursor = prev[cursor]
        if len(path_indices) > total_nodes:
            raise RuntimeError("路径回溯异常")
    path_indices.reverse()
    waypoints = []
    for idx in path_indices:
        block = idx // (layers + 1)
        node = (block // nx, block % nx)
        t_h = best_time[idx] / 3600.0
        a = float(concentration_at(fields, t_h)[node])
        waypoints.append(
            {
                "x_m": float(graph["xs"][node[1]]),
                "y_m": float(graph["ys"][node[0]]),
                "time_h": t_h,
                "concentration": a,
                "fuel_t": float(fuel[idx]),
                "index": int(idx),
            }
        )
    total_fuel = float(fuel[end_index])
    total_time_h = waypoints[-1]["time_h"] - T_START_H
    return {
        "success": True,
        "total_fuel_t": total_fuel,
        "total_time_h": total_time_h,
        "arrival_time_h": waypoints[-1]["time_h"],
        "waypoints": waypoints,
        "path_x": [w["x_m"] for w in waypoints],
        "path_y": [w["y_m"] for w in waypoints],
        "path_time_h": [w["time_h"] for w in waypoints],
        "path_fuel_t": [w["fuel_t"] for w in waypoints],
        "tau_min": tau_min,
        "graph": {
            key: graph[key]
            for key in ("ny", "nx", "start_node", "end_node", "true_end_node", "endpoint_deviation_m")
        },
        "endpoint_note": (
            ""
            if graph["endpoint_deviation_m"] == 0.0
            else (
                f"题设终点 E 在 M1 预报中 A>0.6 禁行（24-72 h 持续）；"
                f"航线终点取距 E 最近的可通行网格，偏差 "
                f"{graph['endpoint_deviation_m']:.0f} m"
            )
        ),
    }


def plot_route(fields: dict[str, Any], result: dict[str, Any], *, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.0))
    for ax, t_h in ((axes[0], 24.0), (axes[1], 48.0), (axes[2], 72.0)):
        a = concentration_at(fields, t_h)
        ny, nx = a.shape
        xs = 250.0 + 500.0 * np.arange(nx)
        ys = 250.0 + 500.0 * np.arange(ny)
        mesh = ax.pcolormesh(xs, ys, a, shading="auto", cmap="Blues", vmin=0.0, vmax=1.0)
        fig.colorbar(mesh, ax=ax, label="A")
        ax.contour(xs, ys, a, levels=[A_LIMIT], colors="red", linewidths=1.5)
        if result.get("success"):
            ax.plot(result["path_x"], result["path_y"], color="black", lw=1.8, label="route")
            ax.plot(result["path_x"][0], result["path_y"][0], "go", ms=8, label="S")
            ax.plot(result["path_x"][-1], result["path_y"][-1], "r*", ms=12, label="E")
            if result["graph"].get("endpoint_deviation_m", 0.0) > 0.0:
                true_end = result["graph"]["true_end_node"]
                ax.plot(
                    250.0 + 500.0 * true_end[1],
                    250.0 + 500.0 * true_end[0],
                    "rX",
                    ms=10,
                    label="E (blocked)",
                )
        ax.set_title(f"t = {t_h:.0f} h")
        ax.set_xlabel("x / m")
        ax.set_ylabel("y / m")
        ax.legend(loc="upper right")
    fig.suptitle("LNG route over time-varying ice concentration")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast", type=Path, required=True, help="72 h 预报 npz")
    parser.add_argument("--tau", type=float, default=TAU_MIN)
    args = parser.parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    fields = load_forecast(args.forecast)
    result = route_optimize(fields, tau_min=args.tau)
    result["wall_clock_seconds"] = time.perf_counter() - started
    result["generated_at"] = datetime.now().astimezone().isoformat()
    if result["success"]:
        plot_route(fields, result, path=OUTPUT_ROOT / f"route_tau{args.tau:g}.png")
    payload_path = OUTPUT_ROOT / f"route_result_tau{args.tau:g}.json"
    payload_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))
    return 0


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"无法序列化 {type(value)!r}")


if __name__ == "__main__":
    raise SystemExit(main())
