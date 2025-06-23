import json
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

def load_jsonl_as_numpy_by_robot(file_path):
    robot_data = defaultdict(list)

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                robot = entry["robot"]

                # 提取你想转成 array 的字段
                row = [
                    float(entry["success_rate"].strip('%')),
                    float(entry["t_mean"].strip(' ms')),
                    float(entry["t_std"].strip(' ms')),
                    float(entry["t_min"].strip(' ms')),
                    float(entry["t_max"].strip(' ms')),
                    float(entry["pos_err_mean"].strip(' mm')),
                    float(entry["pos_err_std"].strip(' mm')),
                    float(entry["pos_err_min"].strip(' mm')),
                    float(entry["pos_err_q1"].strip(' mm')),
                    float(entry["pos_err_q3"].strip(' mm')),
                    float(entry["pos_err_max"].strip(' mm')),
                    float(entry["rot_err_mean"].strip(' deg')),
                    float(entry["rot_err_std"].strip(' deg')),
                    float(entry["rot_err_min"].strip(' deg')),
                    float(entry["rot_err_q1"].strip(' deg')),
                    float(entry["rot_err_q3"].strip(' deg')),
                    float(entry["rot_err_max"].strip(' deg')),
                ]
                robot_data[robot].append(row)

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"Skipped line due to error: {e}")

    # 转为 numpy.ndarray
    for robot in robot_data:
        robot_data[robot] = np.array(robot_data[robot])

    return robot_data

# 示例调用

data_by_robot = load_jsonl_as_numpy_by_robot("ReA_plot.jsonl")
for robot, arr in data_by_robot.items():
    print(f"{robot}: shape = {arr.shape}")

reselect_counts = np.arange(1, 21)
success_c0 = data_by_robot['Cobotta'][:20, 0]
success_c1 = data_by_robot['CobottaPro1300WithRobotiq140'][:20, 0]
success_ur3 = data_by_robot['UR3'][:20, 0]
success_irb = data_by_robot['YumiSglArm'][:20, 0]
colors = ["#FF9F57", "#6BAFAD", "#6498B7", "#E75A4E"]

'''
绘制成功率与重选次数的关系图
'''
plt.figure(figsize=(12, 6))
markersize = 10

# 绘制每条曲线
plt.plot(reselect_counts, success_c0, marker='o', label='C0', color=colors[0], linewidth=2.5, markersize=markersize)
plt.plot(reselect_counts, success_c1, marker='s', label='C1', color=colors[2], linewidth=2.5, markersize=markersize)
plt.plot(reselect_counts, success_ur3, marker='^', label='UR3', color=colors[1], linewidth=2.5, markersize=markersize)
plt.plot(reselect_counts, success_irb, marker='d', label='IRB', color=colors[3], linewidth=2.5, markersize=markersize)

# 设置标题和标签
# plt.xlabel("Re-selection Count", fontsize=12)
# plt.ylabel("Success Rate (%)", fontsize=12)
# plt.title("Success Rate vs. Re-selection Count", fontsize=13)

plt.xticks([1, 5, 10, 15, 20])
# 隐藏 x 和 y 轴刻度数字
# plt.gca().set_xticklabels([])
# plt.gca().set_yticklabels([])

# 显示 y 轴方向网格线
plt.grid(axis='y', linestyle='--', alpha=0.6)

# 图例
# plt.legend(title="Robot", fontsize=10)

# 自动调整布局防止遮挡
plt.tight_layout()
plt.savefig(f'0000_test_programs/nn_ik/res_figs/0621_save/sr2ReA.png', dpi = 1200, bbox_inches='tight')
# 显示图像
plt.show()

'''
绘制平均时间与重选次数的关系图
'''
# mean_time_c0 = data_by_robot['Cobotta'][:20, 1]
# mean_time_c1 = data_by_robot['CobottaPro1300WithRobotiq140'][:20, 1]
# mean_time_ur3 = data_by_robot['UR3'][:20, 1]
# mean_time_irb = data_by_robot['YumiSglArm'][:20, 1]
# plt.figure(figsize=(12, 6))
# markersize = 10
# # 绘制每条曲线
# plt.plot(reselect_counts, mean_time_c0, marker='o', label='C0', color=colors[0], linewidth=2.5, markersize=markersize)
# plt.plot(reselect_counts, mean_time_c1, marker='s', label='C1', color=colors[2], linewidth=2.5, markersize=markersize)  
# plt.plot(reselect_counts, mean_time_ur3, marker='^', label='UR3', color=colors[1], linewidth=2.5, markersize=markersize)
# plt.plot(reselect_counts, mean_time_irb, marker='d', label='IRB', color=colors[3], linewidth=2.5, markersize=markersize)
# #
# # plt.legend(['Cobotta', 'CobottaPro1300WithRobotiq140', 'UR3', 'YumiSglArm'], loc='upper right', fontsize=12)
# plt.xticks([1, 5, 10, 15, 20])
# plt.yticks([2, 3, 4, 5])
# plt.grid(axis='y', linestyle='--', alpha=0.6)
# plt.savefig(f'0000_test_programs/nn_ik/res_figs/0621_save/avgtime2ReA.png', dpi = 1200, bbox_inches='tight')
# plt.show()

