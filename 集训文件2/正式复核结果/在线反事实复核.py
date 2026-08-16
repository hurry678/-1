from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).with_name("在线反事实验证.json")


def clear_solver_modules() -> None:
    for name in ["q1_solver", "q2_solver", "q3_solver"]:
        sys.modules.pop(name, None)


def decision_hash(decisions: list[dict[str, object]]) -> str:
    cleaned = []
    for item in decisions:
        row = {
            key: value
            for key, value in item.items()
            if key not in {"solver_runtime_ms"}
        }
        cleaned.append(row)
    payload = json.dumps(
        cleaned, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def perturb_future(data, core, cutoff_s: float) -> None:
    base = min(task.install_time for task in data.tasks.values())
    ports = sorted(data.ports)
    changed = {}
    ordered = list(data.tasks.items())
    for index, (task_id, task) in enumerate(reversed(ordered)):
        release_s = (task.install_time - base).total_seconds()
        if release_s <= cutoff_s:
            changed[task_id] = task
            continue
        source = ports[(index * 7 + 3) % len(ports)]
        destination = ports[(index * 11 + 5) % len(ports)]
        if destination == source:
            destination = ports[(ports.index(source) + 1) % len(ports)]
        changed[task_id] = core.Task(
            command_id=task.command_id,
            install_time=task.install_time,
            source=source,
            destination=destination,
            priority=99 if task.priority != 99 else 50,
            carrier_id=f"CF-{index:04d}",
        )
    data.tasks = changed


def run_q2(cutoff_s: float, perturbed: bool) -> str:
    clear_solver_modules()
    folder = ROOT / "问题2答案"
    sys.path.insert(0, str(folder))
    try:
        core = importlib.import_module("q1_solver")
        online = importlib.import_module("q2_solver")
        data = online.load_q2_data(ROOT)
        if perturbed:
            perturb_future(data, core, cutoff_s)
        sim = online.OnlineTrafficSimulator(
            data, core.GraphEngine(data), "balanced", capture_trajectory=False
        )
        sim.run(max_time_s=cutoff_s)
        return decision_hash(sim.decision_log)
    finally:
        sys.path.remove(str(folder))
        clear_solver_modules()


def run_q3(cutoff_s: float, perturbed: bool) -> str:
    clear_solver_modules()
    folder = ROOT / "问题3答案"
    sys.path.insert(0, str(folder))
    try:
        core = importlib.import_module("q1_solver")
        online = importlib.import_module("q2_solver")
        high = importlib.import_module("q3_solver")
        data = high.load_q3_data(ROOT)
        if perturbed:
            perturb_future(data, core, cutoff_s)
        sim = high.HighDensitySimulator(
            data, core.GraphEngine(data), "microbatch", capture_trajectory=False
        )
        sim.run(max_time_s=cutoff_s)
        return decision_hash(sim.decision_log)
    finally:
        sys.path.remove(str(folder))
        clear_solver_modules()


def main() -> None:
    tests = []
    for question, cutoffs, runner in [
        (2, [889.704, 1779.408, 2669.112], run_q2),
        (3, [449.25, 898.5, 1347.75], run_q3),
    ]:
        for cutoff in cutoffs:
            original = runner(cutoff, False)
            perturbed = runner(cutoff, True)
            tests.append({
                "question": question,
                "cutoff_s": cutoff,
                "original_hash": original,
                "perturbed_hash": perturbed,
                "passed": original == perturbed,
            })
    result = {
        "passed": all(item["passed"] for item in tests),
        "method": (
            "在3个释放跨度分位点确定性扰动全部未来任务的Source、Destination、"
            "Priority、CarrierID并反转任务字典顺序；比较截止时刻前去除运行时字段后的决策日志SHA-256"
        ),
        "tests": tests,
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
