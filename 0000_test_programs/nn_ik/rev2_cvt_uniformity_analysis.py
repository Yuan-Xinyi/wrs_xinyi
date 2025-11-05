import os, json, numpy as np, matplotlib.pyplot as plt
from scipy.spatial import cKDTree

ARTIFACT_DIR, OUTPUT_DIR = "artifacts", "uniformity_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SUMMARY_JSON = os.path.join(OUTPUT_DIR, "uniformity_summary.json")

def analyze_uniformity(data: np.ndarray):
    n, d = data.shape
    tree = cKDTree(data)
    dist, _ = tree.query(data, k=2)
    nearest = dist[:, 1]
    mean_nn, std_nn = nearest.mean(), nearest.std()
    cv_nn = std_nn / mean_nn if mean_nn > 0 else np.nan
    return nearest, dict(n=n, d=d,
        mean_nn=round(mean_nn, 2),
        std_nn=round(std_nn, 2),
        cv_nn=round(cv_nn, 2))

def main():
    npy_files = sorted([f for f in os.listdir(ARTIFACT_DIR) if f.endswith("r0.npy")])
    if not npy_files: return print("❌ No r0 .npy files found.")

    pairs, summary = {}, []
    for f in npy_files:
        parts = f.replace(".npy", "").split("_")
        if len(parts) < 4: continue
        method, n_str, d_str = parts[0], parts[1], parts[2]
        key = f"{n_str}_{d_str}"
        pairs.setdefault(key, {})[method] = f

    for key, group in pairs.items():
        if "random" not in group or "cvt" not in group: continue
        try:
            data_r, data_c = np.load(os.path.join(ARTIFACT_DIR, group["random"])), np.load(os.path.join(ARTIFACT_DIR, group["cvt"]))
        except Exception as e:
            print(f"⚠️ Skipping {key}: {e}"); continue
        nearest_r, stats_r = analyze_uniformity(data_r)
        nearest_c, stats_c = analyze_uniformity(data_c)
        stats_r["file"], stats_c["file"] = group["random"], group["cvt"]
        summary.extend([stats_r, stats_c])

        plt.figure(figsize=(6, 4))
        plt.hist(nearest_r, bins=50, color='orange', alpha=0.6, label='Random')
        plt.hist(nearest_c, bins=50, color='steelblue', alpha=0.6, label='CVT')
        plt.xlabel("Nearest Neighbor Distance"); plt.ylabel("Count")
        plt.title(f"NN Distance Distribution ({key})"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{key}_compare.png"), dpi=200); plt.close()

        print(f"{key} | Random mean={stats_r['mean_nn']:.2f} cv={stats_r['cv_nn']:.2f} | CVT mean={stats_c['mean_nn']:.2f} cv={stats_c['cv_nn']:.2f}")

    with open(SUMMARY_JSON, "w") as f: json.dump(summary, f, indent=4)
    print(f"\n📊 Saved summary JSON → {SUMMARY_JSON}")

if __name__ == "__main__":
    main()
