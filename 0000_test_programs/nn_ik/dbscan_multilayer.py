import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Generate synthetic 3D position data (or replace this with your real data)
np.random.seed(42)
n_points = 300
data = np.vstack((
    np.random.normal(loc=[0, 0, 0], scale=0.5, size=(n_points, 3)),
    np.random.normal(loc=[3, 3, 3], scale=0.5, size=(n_points, 3)),
    np.random.normal(loc=[6, 0, 3], scale=0.5, size=(n_points, 3))
))


dataset = np.load(f'0000_test_programs/nn_ik/datasets/effective_seedset_1M.npz') # ['source', 'target', 'seed_jnt_value', 'jnt_result']
_, tgt_pos = dataset['target'][:5000, :3], dataset['target'][:5000, 3:]

# Step 2: Apply DBSCAN clustering
# dbscan = DBSCAN(eps=0.05, min_samples=10)  # Adjust `eps` and `min_samples` as needed
dbscan = DBSCAN(eps=0.4, min_samples=8)  # Adjust `eps` and `min_samples` as needed
labels = dbscan.fit_predict(tgt_pos)

# Step 3: Visualize the results in 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Get unique labels and plot each cluster in a different color
unique_labels = set(labels)
colors = plt.cm.get_cmap("tab10", len(unique_labels))

for label in unique_labels:
    label_mask = labels == label
    color = 'k' if label == -1 else colors(label)  # Black for noise
    ax.scatter(tgt_pos[label_mask, 0], tgt_pos[label_mask, 1], tgt_pos[label_mask, 2],
               c=[color], label=f'Cluster {label}' if label != -1 else "Noise", s=20)

ax.set_title("3D DBSCAN Clustering")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.show()
