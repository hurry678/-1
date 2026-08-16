from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import openpyxl

DT = 0.2
VEHICLE_LENGTH = 950.0
MIN_CLEARANCE = 300.0
MIN_REF_GAP = VEHICLE_LENGTH + MIN_CLEARANCE
ACC_MAX = 2000.0
DEC_MAX = 3000.0
POS_TOL = 0.001
NUM_TOL = 1e-7
MODEL_TOL = 0.02


@dataclass(frozen=True)
class Link:
    link_id: int
    u: int
    v: int
    length: float
    vmax: float


@dataclass
class Segment:
    start: float
    end: float
    s0: float
    v0: float
    a: float

    def state(self, t: float) -> tuple[float, float, float]:
        u = min(max(t, self.start), self.end) - self.start
        return self.s0 + self.v0 * u + 0.5 * self.a * u * u, max(0.0, self.v0 + self.a * u), self.a


class Profile:
    def __init__(self, before: dict[str, str], after: dict[str, str], links: dict[int, Link]):
        self.vehicle = before["VehicleID"]
        self.before = before
        self.after = after
        self.l0 = int(before["CurrentEdgeID"])
        self.l1 = int(after["CurrentEdgeID"])
        self.p0 = float(before["Position"])
        self.p1 = float(after["Position"])
        self.v0 = float(before["Speed"])
        self.v1 = float(after["Speed"])
        self.next0 = int(before["NextEdgeID"]) if before["NextEdgeID"].strip() else None
        self.next1 = int(after["NextEdgeID"]) if after["NextEdgeID"].strip() else None
        self.transition = self.l0 != self.l1
        if not self.transition:
            self.distance = self.p1 - self.p0
            self.boundary_distance = math.inf
        else:
            if links[self.l0].v != links[self.l1].u:
                raise ValueError(f"{self.vehicle}: Link{self.l0} 到 Link{self.l1} 不连续")
            self.boundary_distance = links[self.l0].length - self.p0
            self.distance = self.boundary_distance + self.p1
        if self.distance < -MODEL_TOL:
            raise ValueError(f"{self.vehicle}: 出现反向位移 {self.distance:.6f}")
        self.distance = max(0.0, self.distance)
        self.segments, self.model = self._build_segments()
        self.breakpoints = sorted({0.0, DT, *(x.start for x in self.segments), *(x.end for x in self.segments)})
        if self.transition:
            bt = self._time_for_distance(self.boundary_distance)
            if bt is None:
                raise ValueError(f"{self.vehicle}: 无法重建跨 Link 时刻")
            self.breakpoints.append(bt)
            self.breakpoints = sorted(set(round(x, 12) for x in self.breakpoints))

    def _build_segments(self) -> tuple[list[Segment], str]:
        trap = 0.5 * (self.v0 + self.v1) * DT
        a = (self.v1 - self.v0) / DT
        if abs(self.distance - trap) <= MODEL_TOL:
            return [Segment(0.0, DT, 0.0, self.v0, a)], "常加速度"
        if self.v1 <= 0.001 and self.v0 > 0.001:
            min_stop = self.v0 * self.v0 / (2.0 * DEC_MAX)
            if self.distance < min_stop - MODEL_TOL:
                raise ValueError(
                    f"{self.vehicle}: 停车位移{self.distance:.6f}小于最短制动距离{min_stop:.6f}"
                )
            coast = max(0.0, (self.distance - min_stop) / self.v0)
            brake = self.v0 / DEC_MAX
            if coast + brake <= DT + MODEL_TOL:
                coast = min(coast, DT)
                brake_end = min(DT, coast + brake)
                result: list[Segment] = []
                s = 0.0
                if coast > NUM_TOL:
                    result.append(Segment(0.0, coast, 0.0, self.v0, 0.0))
                    s = self.v0 * coast
                if brake_end > coast + NUM_TOL:
                    result.append(Segment(coast, brake_end, s, self.v0, -DEC_MAX))
                    s += self.v0 * (brake_end - coast) - 0.5 * DEC_MAX * (brake_end - coast) ** 2
                if brake_end < DT - NUM_TOL:
                    result.append(Segment(brake_end, DT, self.distance, 0.0, 0.0))
                return result, "滑行-最大制动-静止"
        raise ValueError(
            f"{self.vehicle}: 位移与速度端点不符合允许模型，位移={self.distance:.6f}, "
            f"梯形位移={trap:.6f}, v0={self.v0:.6f}, v1={self.v1:.6f}"
        )

    def state(self, t: float) -> tuple[float, float, float]:
        for seg in self.segments:
            if t <= seg.end + NUM_TOL:
                return seg.state(t)
        return self.distance, self.v1, 0.0

    def _time_for_distance(self, target: float) -> float | None:
        if target <= NUM_TOL:
            return 0.0
        for seg in self.segments:
            s_end, _, _ = seg.state(seg.end)
            if target > s_end + MODEL_TOL:
                continue
            c = seg.s0 - target
            if abs(seg.a) <= NUM_TOL:
                if seg.v0 <= NUM_TOL:
                    return seg.start if abs(c) <= MODEL_TOL else None
                u = -c / seg.v0
            else:
                disc = seg.v0 * seg.v0 - 2.0 * seg.a * c
                if disc < -MODEL_TOL:
                    return None
                roots = [
                    (-seg.v0 + math.sqrt(max(0.0, disc))) / seg.a,
                    (-seg.v0 - math.sqrt(max(0.0, disc))) / seg.a,
                ]
                valid = [x for x in roots if -NUM_TOL <= x <= seg.end - seg.start + NUM_TOL]
                if not valid:
                    return None
                u = min(valid)
            return min(DT, max(0.0, seg.start + u))
        return None

    def location(self, t: float) -> tuple[int, float, int | None]:
        s, _, _ = self.state(t)
        if not self.transition or s < self.boundary_distance - NUM_TOL:
            return self.l0, self.p0 + s, self.next0
        return self.l1, max(0.0, s - self.boundary_distance), self.next1


def load_network(path: Path) -> tuple[dict[int, Link], dict[int, list[tuple[int, float, int]]]]:
    ws = openpyxl.load_workbook(path, data_only=True, read_only=True)["Link"]
    links: dict[int, Link] = {}
    outgoing: dict[int, list[tuple[int, float, int]]] = defaultdict(list)
    for row in ws.iter_rows(min_row=2, values_only=True):
        if int(row[1]) != 1:
            continue
        link = Link(int(row[0]), int(row[2]), int(row[3]), float(row[4]), float(row[5]))
        links[link.link_id] = link
        outgoing[link.u].append((link.v, link.length, link.link_id))
    return links, outgoing


class NetworkDistance:
    def __init__(self, links: dict[int, Link], outgoing: dict[int, list[tuple[int, float, int]]]):
        self.links = links
        self.outgoing = outgoing
        self.cache: dict[tuple[int, int], tuple[float, list[int]]] = {}
        self.blank_next_unique_queries = 0
        self.blank_next_unique_hits = 0

    def shortest(self, start: int, target: int, cutoff: float = 3000.0) -> tuple[float, list[int]]:
        key = (start, target)
        if key in self.cache:
            return self.cache[key]
        if start == target:
            self.cache[key] = (0.0, [])
            return self.cache[key]
        pq = [(0.0, start, [])]
        best = {start: 0.0}
        while pq:
            d, node, path = heapq.heappop(pq)
            if d != best[node] or d > cutoff:
                continue
            if node == target:
                self.cache[key] = (d, path)
                return self.cache[key]
            for nxt, length, lid in self.outgoing.get(node, []):
                nd = d + length
                if nd < best.get(nxt, math.inf) and nd <= cutoff:
                    best[nxt] = nd
                    heapq.heappush(pq, (nd, nxt, path + [lid]))
        self.cache[key] = (math.inf, [])
        return self.cache[key]

    def gap(self, back: Profile, front: Profile, t: float) -> tuple[float, list[int]]:
        bl, bp, bn = back.location(t)
        fl, fp, _ = front.location(t)
        if bl == fl:
            return (fp - bp, [bl]) if fp > bp + NUM_TOL else (math.inf, [])
        if bn is None:
            # Port 作业帧的 NextEdgeID 为空并不表示物理轨道终止。先走完当前
            # Link 剩余段，再只沿唯一后继链追踪；遇到分支即停止，绝不猜路线。
            self.blank_next_unique_queries += 1
            total = self.links[bl].length - bp
            route = [bl]
            node = self.links[bl].v
            seen = {bl}
            while total <= 3000.0 + NUM_TOL:
                options = self.outgoing.get(node, [])
                if len(options) != 1:
                    return math.inf, []
                nxt_node, length, link_id = options[0]
                if link_id in seen:
                    return math.inf, []
                route.append(link_id)
                seen.add(link_id)
                if link_id == fl:
                    self.blank_next_unique_hits += 1
                    return total + fp, route
                total += length
                node = nxt_node
            return math.inf, []
        if bn == fl:
            return self.links[bl].length - bp + fp, [bl, fl]
        if bn not in self.links or self.links[bl].v != self.links[bn].u:
            return math.inf, []
        middle, path = self.shortest(self.links[bn].v, self.links[fl].u)
        if math.isinf(middle):
            return math.inf, []
        return self.links[bl].length - bp + self.links[bn].length + middle + fp, [bl, bn] + path + [fl]


def read_steps(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        current = None
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            step = int(row["StepNo"])
            if current is None:
                current = step
            if step != current:
                yield current, rows
                current, rows = step, {}
            rows[row["VehicleID"]] = row
        if current is not None:
            yield current, rows


def audit_trace(path: Path, links: dict[int, Link], outgoing) -> dict[str, object]:
    distance = NetworkDistance(links, outgoing)
    previous_step = None
    previous_rows = None
    min_event = {
        "reference_gap_mm": math.inf,
        "clearance_mm": math.inf,
    }
    model_errors: list[dict[str, object]] = []
    violations: list[dict[str, object]] = []
    profile_counts: dict[str, int] = defaultdict(int)
    intervals = 0
    service_vehicle_pair_checks = 0

    for step, rows in read_steps(path):
        if len(rows) != 20:
            model_errors.append({"step": step, "error": f"该步只有{len(rows)}辆车"})
        if previous_rows is None:
            previous_step, previous_rows = step, rows
            continue
        if step != previous_step + 1:
            model_errors.append({"step": step, "error": f"StepNo不连续，上一帧为{previous_step}"})
        profiles: dict[str, Profile] = {}
        for vid in sorted(set(previous_rows) & set(rows)):
            try:
                profile = Profile(previous_rows[vid], rows[vid], links)
                profiles[vid] = profile
                profile_counts[profile.model] += 1
                for seg in profile.segments:
                    if seg.a > ACC_MAX + MODEL_TOL or seg.a < -DEC_MAX - MODEL_TOL:
                        model_errors.append({"step": step, "vehicle": vid, "error": f"加速度{seg.a:.6f}越界"})
            except ValueError as exc:
                model_errors.append({"step": step, "vehicle": vid, "error": str(exc)})
        points = {0.0, DT}
        for profile in profiles.values():
            points.update(profile.breakpoints)
        points = sorted(points)
        vids = sorted(profiles)
        for left, right in zip(points, points[1:]):
            if right <= left + NUM_TOL:
                continue
            probe = 0.5 * (left + right)
            for back_id in vids:
                back = profiles[back_id]
                for front_id in vids:
                    if back_id == front_id:
                        continue
                    front = profiles[front_id]
                    probe_gap, route = distance.gap(back, front, probe)
                    if not (0.0 < probe_gap <= 3000.0):
                        continue
                    if back.before["VehicleState"] in {"取货", "放货"}:
                        service_vehicle_pair_checks += 1
                    candidates = [left + 1e-10, right - 1e-10]
                    _, bv, ba = back.state(probe)
                    _, fv, fa = front.state(probe)
                    rel_a = fa - ba
                    if abs(rel_a) > NUM_TOL:
                        root = probe - (fv - bv) / rel_a
                        if left + NUM_TOL < root < right - NUM_TOL:
                            candidates.append(root)
                    for local_t in candidates:
                        gap, actual_route = distance.gap(back, front, local_t)
                        if not (0.0 < gap < math.inf):
                            continue
                        if gap < min_event["reference_gap_mm"]:
                            min_event = {
                                "step": step,
                                "sim_time_s": float(rows[back_id]["SimTime"]) - DT + local_t,
                                "local_time_s": local_t,
                                "back_vehicle": back_id,
                                "front_vehicle": front_id,
                                "route": actual_route or route,
                                "reference_gap_mm": gap,
                                "clearance_mm": gap - VEHICLE_LENGTH,
                            }
                        if gap < MIN_REF_GAP - MODEL_TOL:
                            violations.append({
                                "step": step,
                                "local_time_s": local_t,
                                "back_vehicle": back_id,
                                "front_vehicle": front_id,
                                "reference_gap_mm": gap,
                                "clearance_mm": gap - VEHICLE_LENGTH,
                                "route": actual_route or route,
                            })
            intervals += 1
        previous_step, previous_rows = step, rows

    observed = float(min_event["clearance_mm"])
    conservative = observed - 2.0 * POS_TOL - MODEL_TOL
    return {
        "trace": str(path),
        "dt_s": DT,
        "vehicle_length_mm": VEHICLE_LENGTH,
        "minimum_clearance_mm": MIN_CLEARANCE,
        "profile_counts": dict(profile_counts),
        "topology_intervals_checked": intervals,
        "blank_next_unique_successor_queries": distance.blank_next_unique_queries,
        "blank_next_unique_successor_hits": distance.blank_next_unique_hits,
        "service_vehicle_pair_checks": service_vehicle_pair_checks,
        "motion_model_error_count": len(model_errors),
        "motion_model_errors": model_errors[:100],
        "continuous_gap_violation_count": len(violations),
        "continuous_gap_violations": violations[:100],
        "minimum_reconstructed_event": min_event,
        "conservative_clearance_lower_bound_mm": conservative,
        "passed": not model_errors and not violations and conservative >= MIN_CLEARANCE,
        "tolerance_note": "保守下界从重建净空扣除双车位置舍入误差及运动模型容差。",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--links", type=Path, required=True)
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    links, outgoing = load_network(args.links)
    result = {"audits": [audit_trace(path, links, outgoing) for path in args.trace]}
    result["passed"] = all(item["passed"] for item in result["audits"])
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "passed": result["passed"],
        "summaries": [{
            "trace": item["trace"],
            "passed": item["passed"],
            "motion_model_error_count": item["motion_model_error_count"],
            "continuous_gap_violation_count": item["continuous_gap_violation_count"],
            "minimum_reconstructed_event": item["minimum_reconstructed_event"],
            "conservative_clearance_lower_bound_mm": item["conservative_clearance_lower_bound_mm"],
        } for item in result["audits"]],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
