from datetime import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


q1 = read(ROOT / "outputs/q1/问题1_方案.json")
q2 = read(ROOT / "outputs/q2/问题2_方案.json")
q3 = read(ROOT / "outputs/q3/问题3_方案.json")
verification = read(HERE / "统一验证证据.json")
independent = read(HERE / "二次独立复核证据.json")
continuous = read(HERE / "独立连续净空复核证据.json")
old_path = HERE / "正式复核汇总.json"
old = read(old_path) if old_path.exists() else {}
core_paths = [ROOT / f"问题{q}答案/q1_solver.py" for q in (1, 2, 3)]
core_hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in core_paths]

sources = {
    1: {
        "baseline": q1["baseline_summary"],
        "insertion": q1["insertion_summary"],
        "ALNS": q1["improved_summary"],
    },
    2: q2["summaries"],
    3: q3["summaries"],
}
selected = {1: "ALNS", 2: q2["selected_mode"], 3: q3["selected_mode"]}
expected = {1: 32, 2: 190, 3: 600}


def row(question: int, name: str, summary: dict):
    independent_trace = independent["questions"][str(question)]["trace"]
    return {
        "QuestionNo": question,
        "Candidate": name,
        "CompletedTasks": summary["completed_tasks"],
        "AvgTransferTime_s": summary["avg_transfer_time_s"],
        "Makespan_s": summary["makespan_s"],
        "MinContinuousClearance_mm": summary["min_continuous_clearance_mm"],
        "HardViolations": summary["hard_violation_count"],
        "FutureLeakCount": summary.get("future_leak_count", 0),
        "IndependentBrakingViolations": (
            independent_trace["braking_distance_violation_count"]
            if name == selected[question] else None
        ),
        "HardGate": (
            "通过"
            if summary["completed_tasks"] == expected[question]
            and summary["hard_violation_count"] == 0
            and summary.get("future_leak_count", 0) == 0
            and summary["min_continuous_clearance_mm"] >= 300 - 1e-4
            else "失败"
        ),
        "Selected": "是" if name == selected[question] else "否",
    }


candidate_rows = [
    row(question, name, summary)
    for question in (1, 2, 3)
    for name, summary in sources[question].items()
]
selected_rows = [item for item in candidate_rows if item["Selected"] == "是"]

result = {
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "overall_passed": (
        verification.get("passed") is True
        and continuous.get("passed") is True
        and len(set(core_hashes)) == 1
        and all(
            independent["questions"][str(q)]["trace"]["braking_distance_violation_count"] == 0
            for q in (1, 2, 3)
        )
    ),
    "hard_gate": {
        "minimum_clearance_mm": 300.0,
        "all_tasks_completed": True,
        "independent_braking_distance_required": True,
        "resource_carrier_online_required": True,
        "independent_continuous_clearance_required": True,
        "common_core_hash_required": True,
    },
    "selected_solutions": selected_rows,
    "candidate_results": candidate_rows,
    "verification": verification,
    "independent_verification": independent,
    "independent_continuous_clearance": continuous,
    "common_core_sha256": core_hashes[0] if len(set(core_hashes)) == 1 else core_hashes,
    "provenance": old.get("provenance", {}),
    "commands": [
        'python3 "问题1答案/q1_solver.py" --root "<工作目录>" --iterations 1200',
        'python3 "问题2答案/q2_solver.py" --root "<工作目录>"',
        'python3 "问题3答案/q3_solver.py" --root "<工作目录>"',
        'python3 "正式复核结果/在线反事实复核.py"',
        'python3 "正式复核结果/二次独立复核.py"',
        'python3 "正式复核结果/复核验证.py"',
    ],
}
old_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"overall_passed": result["overall_passed"], "selected": selected_rows}, ensure_ascii=False, indent=2))
