from pathlib import Path
import json

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import networkx as nx
import numpy as np
import openpyxl
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures_oht"
OUT.mkdir(exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 160
COLORS = {"blue": "#4C78A8", "orange": "#F58518", "green": "#54A24B",
          "red": "#E45756", "purple": "#7A5195", "gray": "#9D9D9D"}


def read_sheet(path, sheet):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[sheet]
    rows = list(ws.iter_rows(values_only=True))
    return pd.DataFrame(rows[1:], columns=rows[0])


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


nodes = read_sheet(ROOT / "附件1_轨道节点数据.xlsx", "Node")
links = read_sheet(ROOT / "附件2_轨道连接数据.xlsx", "Link")
ports = read_sheet(ROOT / "附件3_Port位置数据.xlsx", "Port")
vehicles = read_sheet(ROOT / "附件4_OHT初始位置数据.xlsx", "Vehicle")
ports.loc[ports["LocationID"] == "P17", ["NodeID", "LinkID", "Distance"]] = [70, 507, 303.0]

pos = {int(r.NodeID): (float(r.MonitorX), float(r.MonitorY)) for r in nodes.itertuples()}
fig, ax = plt.subplots(figsize=(10.5, 5.8))
for r in links[links["UseFlag"] == 1].itertuples():
    x1, y1 = pos[int(r.FromNodeID)]
    x2, y2 = pos[int(r.ToNodeID)]
    color = COLORS["orange"] if str(r.TrackClass).upper() == "CURVE" else "#A7B6C2"
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=0.75,
                                shrinkA=0, shrinkB=0, mutation_scale=5), zorder=1)
roles = {"NORMAL": (COLORS["gray"], 9), "FORK": (COLORS["blue"], 28),
         "MERGE": (COLORS["red"], 28), "TERMINAL": ("black", 40)}
for role, (c, size) in roles.items():
    part = nodes[nodes["NodeRole"] == role]
    ax.scatter(part["MonitorX"], part["MonitorY"], s=size, c=c, label=role, zorder=3,
               edgecolors="white" if role != "NORMAL" else "none", linewidths=0.4)
link_map = links.set_index("LinkID")
port_xy = []
for r in ports.itertuples():
    e = link_map.loc[int(r.LinkID)]
    a = pos[int(e.FromNodeID)]
    b = pos[int(e.ToNodeID)]
    ratio = min(max(float(r.Distance) / float(e.Distance), 0), 1)
    port_xy.append((a[0] + ratio * (b[0] - a[0]), a[1] + ratio * (b[1] - a[1])))
px, py = zip(*port_xy)
ax.scatter(px, py, marker="s", s=22, c=COLORS["green"], label="Port", zorder=4,
           edgecolors="white", linewidths=0.4)
veh_xy = []
for r in vehicles.itertuples():
    e = link_map.loc[int(r.LinkID)]
    a = pos[int(e.FromNodeID)]
    b = pos[int(e.ToNodeID)]
    ratio = min(max(float(r.Distance) / float(e.Distance), 0), 1)
    veh_xy.append((a[0] + ratio * (b[0] - a[0]), a[1] + ratio * (b[1] - a[1])))
vx, vy = zip(*veh_xy)
ax.scatter(vx, vy, marker="^", s=32, c=COLORS["purple"], label="OHT初始位置", zorder=5)
ax.set_title("OHT有向轨道拓扑、Port与初始车辆位置")
ax.set_xlabel("标准化横坐标")
ax.set_ylabel("标准化纵坐标")
ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08))
ax.grid(alpha=0.15)
save(fig, "network_topology.png")


q = {}
for i in (1, 2, 3):
    f = ROOT / "outputs" / f"q{i}" / f"问题{i}_任务结果.csv"
    q[i] = pd.read_csv(f)
    q[i]["InstallTime"] = pd.to_datetime(q[i]["InstallTime"], format="mixed")
    q[i]["TransferCompletedTime"] = pd.to_datetime(q[i]["TransferCompletedTime"], format="mixed")

fig, ax = plt.subplots(figsize=(9.2, 5.4))
for i, color, label in [(2, COLORS["blue"], "问题二：190项动态任务"),
                        (3, COLORS["red"], "问题三：600项高密任务")]:
    t = (q[i]["InstallTime"].sort_values() - q[i]["InstallTime"].min()).dt.total_seconds().to_numpy()
    ax.step(t, np.arange(1, len(t) + 1), where="post", lw=2, color=color, label=label)
ax.set_xlabel("相对释放时间/s")
ax.set_ylabel("累计释放任务数")
ax.set_title("动态场景任务累计到达曲线")
ax.grid(alpha=0.25, linestyle="--")
ax.legend()
save(fig, "arrival_curves.png")


fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
data = [q[i]["TransferTime"].astype(float).to_numpy() for i in (1, 2, 3)]
bp = axes[0].boxplot(data, tick_labels=["问题一", "问题二", "问题三"], patch_artist=True,
                     showfliers=False, widths=0.55)
for patch, color in zip(bp["boxes"], [COLORS["green"], COLORS["blue"], COLORS["red"]]):
    patch.set_facecolor(color); patch.set_alpha(0.65)
axes[0].set_ylabel("任务执行时间/s")
axes[0].set_title("三问任务执行时间分布")
axes[0].grid(axis="y", alpha=0.25, linestyle="--")
for i, color, label in [(2, COLORS["blue"], "问题二"), (3, COLORS["red"], "问题三")]:
    done = (q[i]["TransferCompletedTime"].sort_values() - q[i]["InstallTime"].min()).dt.total_seconds().to_numpy()
    axes[1].step(done, np.arange(1, len(done) + 1) / len(done), where="post", lw=2,
                 color=color, label=label)
axes[1].set_xlabel("相对完成时刻/s")
axes[1].set_ylabel("累计完成比例")
axes[1].set_title("在线任务完成经验分布")
axes[1].grid(alpha=0.25, linestyle="--")
axes[1].legend()
save(fig, "time_distribution_and_completion.png")


fig, ax = plt.subplots(figsize=(10.2, 5.2))
vehicles_order = [f"OHT{i:02d}" for i in range(1, 21)]
x = np.arange(20)
width = 0.25
for idx, i in enumerate((1, 2, 3)):
    counts = q[i]["VehicleID"].value_counts().reindex(vehicles_order, fill_value=0)
    ax.bar(x + (idx - 1) * width, counts, width=width,
           label=f"问题{i}", color=[COLORS["green"], COLORS["blue"], COLORS["red"]][idx], alpha=0.82)
ax.set_xticks(x)
ax.set_xticklabels(vehicles_order, rotation=55, ha="right", fontsize=8)
ax.set_ylabel("分配任务数")
ax.set_title("三问各OHT承担任务数量")
ax.grid(axis="y", alpha=0.22, linestyle="--")
ax.legend(ncol=3)
save(fig, "vehicle_workload.png")


fig, ax = plt.subplots(figsize=(8.8, 5.2))
components = ["分配等待", "取货响应", "取放货服务", "载货运输"]
q2 = np.array([1.713674, 35.904211, 16.0, 89.328421])
q3 = np.array([260.212087, 489.112000, 16.0, 89.227333])
bottom = np.zeros(2)
for name, color, vals in zip(components,
                             [COLORS["purple"], COLORS["orange"], COLORS["green"], COLORS["blue"]],
                             np.vstack([q2, q3]).T):
    ax.bar([0, 1], vals, bottom=bottom, width=0.58, label=name, color=color, alpha=0.86)
    bottom += vals
ax.set_xticks([0, 1], ["问题二", "问题三"])
ax.set_ylabel("平均任务执行时间分量/s")
ax.set_title("动态负载与高密负载的耗时构成")
ax.legend(ncol=2)
ax.grid(axis="y", alpha=0.22, linestyle="--")
for i, total in enumerate(bottom):
    ax.text(i, total + 12, f"合计 {total:.2f}s", ha="center", fontsize=9)
save(fig, "time_components.png")


fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
q2_modes = ["FCFS", "滚动插入", "均衡拍卖", "完整候选"]
q2_avg = [142.946305, 142.946305, 142.946305, 147.716832]
axes[0].bar(q2_modes, q2_avg, color=[COLORS["blue"]] * 3 + [COLORS["orange"]], alpha=0.82)
axes[0].set_ylim(138, 151)
axes[0].set_ylabel("平均任务执行时间/s")
axes[0].set_title("问题二候选性能比较")
axes[0].tick_params(axis="x", rotation=18)
for i, v in enumerate(q2_avg): axes[0].text(i, v + 0.25, f"{v:.2f}", ha="center", fontsize=8)
q3_modes = ["直接滚动", "普通微批", "压力微批", "完整模型"]
q3_avg = [854.551420, 843.893087, 843.259420, 841.457753]
clearance = [325, 298, 298, 298]
colors = [COLORS["green"] if c >= 300 else COLORS["red"] for c in clearance]
axes[1].bar(q3_modes, q3_avg, color=colors, alpha=0.82)
axes[1].set_ylim(835, 860)
axes[1].set_ylabel("平均任务执行时间/s")
axes[1].set_title("问题三性能与连续净空硬门")
axes[1].tick_params(axis="x", rotation=18)
for i, (v, c) in enumerate(zip(q3_avg, clearance)):
    axes[1].text(i, v + 0.55, f"{v:.2f}\n净空{c}mm", ha="center", fontsize=8)
save(fig, "candidate_comparisons.png")


fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8))
for ax, i, color in [(axes[0], 2, COLORS["blue"]), (axes[1], 3, COLORS["red"])]:
    xvals = q[i]["Priority"].astype(float)
    yvals = q[i]["TransferTime"].astype(float)
    ax.scatter(xvals, yvals, s=13, alpha=0.42, color=color, edgecolors="none")
    groups = q[i].groupby("Priority")["TransferTime"].mean().sort_index()
    ax.plot(groups.index, groups.values, color="black", lw=1.5, label="同优先级均值")
    ax.set_xlabel("Priority")
    ax.set_ylabel("任务执行时间/s")
    ax.set_title(f"问题{i}：优先级与执行时间")
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(fontsize=8)
save(fig, "priority_transfer_scatter.png")


summary = {
    "arrival_rate_q2": len(q[2]) / ((q[2]["InstallTime"].max() - q[2]["InstallTime"].min()).total_seconds()),
    "arrival_rate_q3": len(q[3]) / ((q[3]["InstallTime"].max() - q[3]["InstallTime"].min()).total_seconds()),
    "figures": sorted(p.name for p in OUT.glob("*.png")),
}
(OUT / "figure_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
