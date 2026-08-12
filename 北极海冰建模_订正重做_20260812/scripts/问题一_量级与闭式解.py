"""问题一：五项力量级与局部三力平衡闭式解。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RHO_I = 917.0
RHO_W = 1025.0
L0 = 20_000.0
U0 = 0.5
H0 = 1.0
F = 1.4e-4
CD = 0.005
P_STAR = 5_000.0
TAU_A = np.array([0.12, 0.04])


def main() -> None:
    t0 = L0 / U0
    # 教师参考答案按 VP 各向同性压力项 P/2 估计内力尺度。
    magnitudes = {
        "inertia": RHO_I * H0 * U0 / t0,
        "coriolis": RHO_I * H0 * F * U0,
        "water_ice_drag": RHO_W * CD * U0**2,
        "rheology": H0 * (P_STAR * H0 / 2.0) / L0,
        "wind": float(np.linalg.norm(TAU_A)),
    }
    orders = {
        name: int(np.floor(np.log10(value))) for name, value in magnitudes.items()
    }

    # 若局部内力散度可忽略且海水静止，三力平衡退化为拖曳-风应力平衡。
    tau_norm = float(np.linalg.norm(TAU_A))
    wind_only_velocity = TAU_A / np.sqrt(RHO_W * CD * tau_norm)
    wind_only_speed = float(np.linalg.norm(wind_only_velocity))

    result = {
        "characteristic_time_s": t0,
        "force_magnitudes_N_per_m2": magnitudes,
        "scientific_orders": orders,
        "deleted_terms": ["inertia", "coriolis"],
        "retained_terms": ["water_ice_drag", "rheology", "wind"],
        "wind_drag_balance": {
            "ice_velocity_mps": wind_only_velocity.tolist(),
            "ice_speed_mps": wind_only_speed,
        },
    }

    output = Path(__file__).resolve().parents[1] / "results" / "问题一_量级.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    labels = ["Inertia", "Coriolis", "Drag", "VP stress", "Wind"]
    values = [
        magnitudes["inertia"],
        magnitudes["coriolis"],
        magnitudes["water_ice_drag"],
        magnitudes["rheology"],
        magnitudes["wind"],
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bars = ax.bar(labels, values, color=["#999999", "#999999", "#2166ac", "#67a9cf", "#d6604d"])
    ax.set_yscale("log")
    ax.set_ylabel("Characteristic magnitude / (N m$^{-2}$)")
    ax.set_title("Orders of the five force terms")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.12,
            f"{value:.3e}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(
        Path(__file__).resolve().parents[1] / "figures" / "问题一_五力量级.png",
        dpi=180,
    )
    plt.close(fig)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
