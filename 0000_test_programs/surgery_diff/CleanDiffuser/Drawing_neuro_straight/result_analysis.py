import matplotlib.pyplot as plt
import numpy as np

xcs = [
    [0.22667526, -0.11776538, 0],
    [0.21146455, 0.16981612, 0],
    [0.23369516, 0.14750803, 0],
    [0.40307793, 0.34884274, 0],
    [0.24711241, -0.10846166, 0],
    [0.14937423, 0.10832477, 0],
    [0.45417136, -0.3653374, 0],
    [0.17260942, -0.51152927, 0],
    [0.25486058, 0.03350953, 0],
    [0.2801755, 0.13957125, 0]
]

ds = [
    [0.36705586, 0.97624475, 0.        ],
    [0.68381244, -0.65156716, 0.        ],
    [0.4950527, -0.8490249, 0.       ],
    [-0.50366807, 0.52256763, 0.        ],
    [-0.30821067, -0.94474614, 0.        ],
    [-0.26670367,  0.9799813, 0.        ],
    [0.37302637, 0.48383456, 0.        ],
    [-0.5684415, -0.23839238, 0.        ],
    [-0.186502, 0.970524, 0.      ],
    [ 0.36474252, -0.9243927, 0.        ]
]

ls = [
    1.043,
    0.945,
    0.983,
    0.726,
    0.994,
    1.016,
    0.611,
    0.616,
    0.988,
    0.994
]

plt.figure(figsize=(10, 8))

# 核心修正：使用 numpy 处理数值运算
for i in range(len(ds)):
    xc = np.array(xcs[i][:2])  # 转换为 numpy 数组以支持数学运算
    d = np.array(ds[i][:2])
    l = ls[i]
    
    # 计算端点：P = xc ± (l/2)*d
    p1 = xc - (l / 2) * d
    p2 = xc + (l / 2) * d
    
    # 绘图：直线
    line, = plt.plot([p1[0], p2[0]], [p1[1], p2[1]], '-', linewidth=2, label=f'Line {i}')
    
    # 绘图：中心点
    plt.scatter(xc[0], xc[1], color=line.get_color(), s=40, zorder=5)
    
    # 绘图：方向箭头 (scale=1 表示原始长度，为了美观这里设为支持自适应缩放)
    plt.quiver(xc[0], xc[1], d[0], d[1], color=line.get_color(), 
               angles='xy', scale_units='xy', scale=3, width=0.005, alpha=0.5)

plt.axis('equal')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # 图例放在外侧
plt.title("Line Visualization: Center, Direction, and Length")
plt.tight_layout()
plt.xlim(-0.6, 0.6)
plt.ylim(-0.6, 0.6)
plt.show()