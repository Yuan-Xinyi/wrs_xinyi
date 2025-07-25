import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ======== 0. 配置 ======== #
npy_path = "wrs/robot_sim/ur3_configs_rrt_rtree_0724.npy"    # 你的节点文件
M = 200_000                    # 蒙特卡洛采样数
use_faiss = True               # 没有 faiss 就 False 用 KDTree
k_cand = 32                    # R-tree/Faiss 粗筛时候选，KDTree可忽略

# ======== 1. 载入节点 ======== #
Q = np.load(npy_path).astype(np.float32)  # N x d
N, d = Q.shape
print("Loaded", Q.shape)

# ======== 2. 构建 NN 索引 ======== #
if use_faiss:
    import faiss
    index = faiss.IndexFlatL2(d)
    index.add(Q)
    def nn(query):
        D,I = index.search(query, 1)
        return I[:,0]
else:
    from scipy.spatial import cKDTree
    tree = cKDTree(Q)
    def nn(query):
        _, I = tree.query(query, k=1, workers=-1)
        return I

# ======== 3. 采样点 ======== #
# 建议用机器人实际关节上下限；这里用数据集包围盒示例
jmin = Q.min(axis=0)
jmax = Q.max(axis=0)
X = np.random.uniform(jmin, jmax, size=(M, d)).astype(np.float32)

# ======== 4. 最近邻统计（蒙特卡洛） ======== #
batch = 10_000
hits = np.zeros(N, dtype=np.int64)

for i in range(0, M, batch):
    q = X[i:i+batch]
    idxs = nn(q)
    for idx in idxs:
        hits[idx] += 1

freq = hits / M  # 频率
print("sum(freq) =", freq.sum())

# ======== 5. 指标 ======== #
def stats_of_freq(freq):
    N = len(freq)
    u = np.ones(N)/N
    eps = 1e-12
    kl = np.sum(freq * np.log((freq+eps)/(u+eps)))
    m = 0.5*(freq+u)
    js = 0.5*np.sum(freq*np.log((freq+eps)/(m+eps))) + 0.5*np.sum(u*np.log((u+eps)/(m+eps)))
    chi2 = np.sum((freq-u)**2 / (u+eps))
    H = -np.sum(freq*np.log(freq+eps))
    Hratio = H / np.log(N)
    G = np.sum(np.abs(freq[:,None]-freq[None,:])) / (2*N*np.sum(freq))
    CV = np.std(freq) / np.mean(freq)
    return dict(KL=kl, JS=js, Chi2=chi2, Hratio=Hratio, Gini=G, CV=CV)

metrics = stats_of_freq(freq)
print(metrics)

# 置信区间（可选）
ci = 1.96*np.sqrt(freq*(1-freq)/M)

# ======== 6. 可视化 ======== #
# 6.1 排名曲线
order = np.argsort(freq)[::-1]
plt.figure()
plt.semilogy(freq[order] + 1e-12)
plt.xlabel('Node rank (desc)')
plt.ylabel('Frequency (log)')
plt.title('Sorted Voronoi frequency')
plt.tight_layout()

# 6.2 频率直方图（乘N后理想集中在1附近）
plt.figure()
plt.hist(freq * N, bins=60, alpha=0.8)
plt.xlabel('p_i * N')
plt.ylabel('count')
plt.title('Histogram of normalized frequency')
plt.tight_layout()

# 6.3 PCA 2D 投影散点
if d > 2 and Q.shape[0] > 5:
    X2 = PCA(n_components=2).fit_transform(Q)
    plt.figure(figsize=(6,5))
    sc = plt.scatter(X2[:,0], X2[:,1], c=freq, s=6, cmap='viridis')
    plt.colorbar(sc, label='freq')
    plt.title('PCA projection colored by Voronoi freq')
    plt.tight_layout()

plt.show()
