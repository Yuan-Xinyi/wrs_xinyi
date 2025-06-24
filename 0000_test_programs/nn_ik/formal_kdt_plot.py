import json
import matplotlib.pyplot as plt
from collections import defaultdict

# 读取 .jsonl 数据并按机器人归类
def parse_jsonl_group_by_robot(file_path):
    data = defaultdict(list)

    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            robot = entry["robot"]
            n_intervals = int(entry["n_intervals"])
            success_rate = float(entry["success_rate"].replace('%', ''))
            t_mean = float(entry["t_mean"].replace(' ms', ''))

            data[robot].append((n_intervals, success_rate, t_mean))

    # 按 n_intervals 排序
    for robot in data:
        data[robot] = sorted(data[robot], key=lambda x: x[0])

    return data

# 加载数据
data = parse_jsonl_group_by_robot("kdt_size_plot.jsonl")

# 定义颜色与标记
colors = {
    "Cobotta": "#FF9F57",
    "CobottaPro1300WithRobotiq140": "#6498B7",
    "UR3": "#6BAFAD",
    "YumiSglArm": "#E75A4E"
}
markers = {
    "Cobotta": 'o',
    "CobottaPro1300WithRobotiq140": 's',
    "UR3": '^',
    "YumiSglArm": 'd'
}

# ==== 图1：Success Rate vs n_intervals ====
plt.figure(figsize=(12, 6))
markersize = 10
for robot, values in data.items():
    n_vals = [v[0] for v in values]
    success_vals = [v[1] for v in values]
    plt.plot(n_vals, success_vals, marker=markers[robot], label=robot, color=colors[robot],
             linewidth=2.5, markersize=8)

plt.xticks(range(4, 13))
plt.ylim(30, 101)
plt.yticks([40, 60, 80, 100])
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.gca().set_xticklabels([])
# plt.tight_layout()
# plt.legend()
# plt.title("Success Rate vs Re-selection Intervals")
# plt.xlabel("n_intervals")
# plt.ylabel("Success Rate (%)")
plt.savefig(f'0000_test_programs/nn_ik/res_figs/0621_save/kdt_sr.png', dpi = 600, bbox_inches='tight')
plt.show()

# ==== 图2：Average Time vs n_intervals ====
plt.figure(figsize=(12, 6))
markersize = 10
for robot, values in data.items():
    n_vals = [v[0] for v in values]
    time_vals = [v[2] for v in values]
    plt.plot(n_vals, time_vals, marker=markers[robot], label=robot, color=colors[robot],
             linewidth=2.5, markersize=8)

plt.xticks(range(4, 13))
plt.ylim(1.2, 3.6)
plt.yticks([1.5, 2.0, 2.5, 3.0, 3.5])
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.gca().set_xticklabels([])
# plt.tight_layout()
# plt.legend()
# plt.title("Average Time vs Re-selection Intervals")
# plt.xlabel("n_intervals")
# plt.ylabel("Average Time (ms)")
plt.savefig(f'0000_test_programs/nn_ik/res_figs/0621_save/kdt_avgt.png', dpi = 600, bbox_inches='tight')
plt.show()
