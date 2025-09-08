import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from skimage.graph import route_through_array
import networkx as nx
from scipy import interpolate

# 1. Binarize
img_path = "0000_test_programs/surgery_diff/CleanDiffuser/motion_planner/bezier_curve/kanji_1.png"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
_, bw = cv2.threshold(img, 128, 1, cv2.THRESH_BINARY_INV)

# 2. Skeleton
skeleton = skeletonize(bw.astype(bool))

# 3. Build graph from skeleton pixels
coords = np.argwhere(skeleton)  # (row, col)
coords_set = {tuple(p) for p in coords}

def neighbors(p):
    r, c = p
    for dr in [-1,0,1]:
        for dc in [-1,0,1]:
            if dr==0 and dc==0: continue
            q = (r+dr, c+dc)
            if q in coords_set:
                yield q

G = nx.Graph()
for p in coords_set:
    for q in neighbors(p):
        G.add_edge(p, q)

# 4. Find endpoints (degree=1)
endpoints = [n for n in G.nodes() if G.degree(n)==1]

# 5. Extract strokes (simple: shortest paths between endpoints)
strokes = []
visited = set()
for i, p in enumerate(endpoints):
    for q in endpoints[i+1:]:
        if nx.has_path(G, p, q):
            path = nx.shortest_path(G, p, q)
            if len(path) > 10:  # filter too short spurs
                strokes.append(np.array(path)[:, ::-1])  # (x,y)

# 6. Fit each stroke with B-spline
plt.figure(figsize=(6,6))
plt.imshow(img, cmap="gray")

for stroke in strokes:
    x, y = stroke[:,0], stroke[:,1]
    try:
        tck, u = interpolate.splprep([x, y], k=3, s=5.0)
        u_new = np.linspace(0,1,100)
        x_new, y_new = interpolate.splev(u_new, tck)
        plt.plot(x_new, y_new, linewidth=2)
    except Exception as e:
        # skip strokes with too few points
        continue

plt.axis("equal")
plt.title("Multi-stroke Bézier/B-spline modeling")
plt.show()
