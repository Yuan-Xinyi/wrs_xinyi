import os
import time
import json
import threading
from statistics import median
from typing import Callable, Dict, Any, Tuple, List

import numpy as np
import psutil  # pip install psutil
import samply

# === wrs 相关 ===
import wrs.robot_sim.robots.cobotta.cobotta as cbt
import wrs.robot_sim.manipulators.rs007l.rs007l as rs007l
import wrs.robot_sim.manipulators.ur3.ur3 as ur3
import wrs.robot_sim.manipulators.ur3e.ur3e as ur3e
import wrs.basis.robot_math as rm
import wrs.robot_sim.robots.yumi.yumi_single_arm as yumi
import wrs.robot_sim.robots.cobotta_pro1300.cobotta_pro1300 as cbtpro1300
import wrs.robot_sim.robots.cobotta_pro900.cobotta_pro900_spine as cbtpro900

from tqdm import tqdm

# =========================================================
# 采样器实现
# =========================================================

def make_random_joint_samples(n: int, d: int, *, robot, seed: int = 42) -> np.ndarray:
    """
    高效随机关节采样：生成 n×d 的关节角，不构造笛卡尔积。
    """
    rng = np.random.default_rng(seed)
    jnt_min = robot.jnt_ranges[:, 0]
    jnt_max = robot.jnt_ranges[:, 1]
    sampled_qs = rng.uniform(low=jnt_min, high=jnt_max, size=(n, d)).astype(np.float32)
    print(f"[Random Joint Sampling] Total samples: {len(sampled_qs)} × {d}")
    return sampled_qs

def make_cvt_samples(n: int, d: int, *, robot, seed: int = 42) -> np.ndarray:
    # Step 1. 在 [0,1]^d 上生成均匀的 CVT 点
    pts = samply.hypercube.cvt(n, d).astype(np.float32)

    # Step 2. 映射到实际关节范围
    jnt_ranges = np.asarray(robot.jnt_ranges, dtype=np.float32)
    low = jnt_ranges[:, 0]
    high = jnt_ranges[:, 1]
    scaled_pts = low + (high - low) * pts  # 线性映射

    print(f"[CVT Sampling] Generated {n} samples in {d}D, mapped to joint limits.")
    return scaled_pts


import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import wrs.motion.probabilistic.rrt_rtree as rrt

def make_rrt_samples(n: int,
                     d: int,
                     *,
                     robot,
                     ext_dist: float = 0.3,
                     expand_ratio: float = 1.0,
                     seed: int = 42) -> np.ndarray:
    """
    Generate roughly uniform samples in robot joint space via RRT expansion.
    Structure aligned with make_cvt_samples().

    Args:
        n (int): number of configurations to sample
        d (int): joint dimension
        robot: robot instance with .jnt_ranges
        ext_dist (float): RRT extension distance
        expand_ratio (float): range expansion ratio beyond joint limits
        seed (int): random seed

    Returns:
        np.ndarray: (n, d) sampled joint configurations
    """
    np.random.seed(seed)

    # --- Step 1: initialize RRT planner ---
    planner = rrt.RRT(robot)
    planner.roadmap.clear()

    joint_lower, joint_upper = robot.jnt_ranges[:, 0], robot.jnt_ranges[:, 1]
    start_conf = (joint_lower + joint_upper) / 2
    planner.start_conf = start_conf
    planner.roadmap.add_node("start", conf=start_conf)

    # --- Step 2: iterative random expansion ---
    pbar = tqdm(total=n, desc=f"[{type(robot).__name__}] RRT Sampling")
    while len(planner.roadmap.nodes) < n + 1:
        prev_n = len(planner.roadmap.nodes)
        rand_conf = np.random.uniform(
            joint_lower - expand_ratio * (joint_upper - joint_lower),
            joint_upper + expand_ratio * (joint_upper - joint_lower)
        )
        planner._extend_roadmap(rand_conf, ext_dist, rand_conf, [], [])
        pbar.update(len(planner.roadmap.nodes) - prev_n)
    pbar.close()

    # --- Step 3: extract configurations ---
    conf_array = np.array(
        [data["conf"] for nid, data in planner.roadmap.nodes.items() if nid != "start"]
    )
    conf_array = conf_array[:n]

    print(f"[RRT Sampling] Generated {len(conf_array)} samples in {d}D for {type(robot).__name__}.")

    # --- Step 4: visualize histograms (optional) ---
    import math
    n_joints = conf_array.shape[1]
    n_cols = 3
    n_rows = math.ceil(n_joints / n_cols)

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()
    for i in range(n_joints):
        axes[i].hist(conf_array[:, i], bins=50, alpha=0.7, color='steelblue')
        axes[i].set_title(f"Joint {i}")
    for j in range(n_joints, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()

    out_dir = "0000_test_programs/nn_ik/res_figs/rrt_samples"
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{type(robot).__name__}_rrt_hist.png"), dpi=300)
    plt.close(fig)

    return conf_array


# =========================================================
# RAM 监控类
# =========================================================

class PeakRAMMonitor:
    """后台线程周期性采样进程 RSS，记录峰值 RAM"""
    def __init__(self, interval_sec: float = 0.02):
        self.interval = interval_sec
        self._stop = threading.Event()
        self._peak = 0
        self._thr = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        proc = psutil.Process(os.getpid())
        while not self._stop.is_set():
            rss = proc.memory_info().rss  # bytes
            if rss > self._peak:
                self._peak = rss
            time.sleep(self.interval)

    def __enter__(self):
        self._thr.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thr.join()

    @property
    def peak_bytes(self) -> int:
        return self._peak


# =========================================================
# 单次测量
# =========================================================

def measure_once(fn: Callable[..., np.ndarray], *,
                 n: int, d: int, out_prefix: str,
                 save_npz: bool = True, **fn_kwargs) -> Dict[str, Any]:
    """执行一次采样函数，测量时间、峰值 RAM、保存文件大小"""
    with PeakRAMMonitor(interval_sec=0.01) as mon:
        t0 = time.perf_counter()
        arr = fn(n=n, d=d, **fn_kwargs)
        t1 = time.perf_counter()
    elapsed = t1 - t0
    peak_ram = mon.peak_bytes

    os.makedirs("artifacts", exist_ok=True)
    npy_path = os.path.join("artifacts", f"{out_prefix}.npy")
    np.save(npy_path, arr)
    npy_size = os.path.getsize(npy_path)

    npz_size = None
    npz_path = None
    if save_npz:
        npz_path = os.path.join("artifacts", f"{out_prefix}.npz")
        np.savez_compressed(npz_path, data=arr)
        npz_size = os.path.getsize(npz_path)

    return dict(
        elapsed_s=elapsed,
        peak_ram_mb=peak_ram / (1024**2),
        file_npy_mb=npy_size / (1024**2),
        file_npz_mb=(npz_size / (1024**2)) if npz_size is not None else None,
    )


# =========================================================
# benchmark
# =========================================================

def benchmark(robot: Any,
              methods: Dict[str, Tuple[Callable, Dict[str, Any]]],
              n: int, d: int, repeats: int = 5, seed_base: int = 42,
              json_path: str = "cvt_cost_benchmark.json"):
    """对每种采样方法重复测量，结果保存到 JSON（仅均值与标准差）"""
    rows: List[Dict[str, Any]] = []

    for name, (fn, kwargs) in methods.items():
        print(f"== {name} (n={n}, d={d}) ==")
        elapsed_list, ram_list, npy_list, npz_list = [], [], [], []

        for r in range(repeats):
            kw = dict(kwargs)
            if "seed" in fn.__code__.co_varnames:
                kw["seed"] = seed_base + r

            res = measure_once(
                fn, n=n, d=d,
                out_prefix=f"{name}_n{n}_d{d}_r{r}",
                **kw
            )
            elapsed_list.append(res["elapsed_s"])
            ram_list.append(res["peak_ram_mb"])
            npy_list.append(res["file_npy_mb"])
            if res["file_npz_mb"] is not None:
                npz_list.append(res["file_npz_mb"])

            print(f"  run {r}: {res['elapsed_s']:.3f}s, "
                  f"peakRAM={res['peak_ram_mb']:.1f}MB, "
                  f"npy={res['file_npy_mb']:.1f}MB, "
                  f"npz={res['file_npz_mb']:.1f}MB")

        # === 仅保留 mean / std，保留两位小数 ===
        row = dict(
            method=name,
            n=n, d=d, repeats=repeats,
            time_mean_s=round(float(np.mean(elapsed_list)), 2),
            time_std_s=round(float(np.std(elapsed_list)), 2),
            peakRAM_mean_mb=round(float(np.mean(ram_list)), 2),
            peakRAM_std_mb=round(float(np.std(ram_list)), 2),
            file_npy_mean_mb=round(float(np.mean(npy_list)), 2),
            file_npy_std_mb=round(float(np.std(npy_list)), 2),
            file_npz_mean_mb=round(float(np.mean(npz_list)), 2) if npz_list else None,
            file_npz_std_mb=round(float(np.std(npz_list)), 2) if npz_list else None,
        )
        rows.append(row)

    # === 写 JSON（追加模式） ===
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            existing = json.load(f)
    else:
        existing = []
    existing.extend(rows)
    with open(json_path, "w") as f:
        json.dump(existing, f, indent=4)

    print(f"\n✅ Saved summary JSON → {json_path}\n")


# =========================================================
# 主入口
# =========================================================

if __name__ == "__main__":
    # N6 = [4096, 5120, 12000, 25200, 40320, 60480, 151200, 166320, 259200, 2592000]
    # N7 = [16384, 20480, 48000, 100800, 201600, 362880, 604800, 1663200, 1814400, 3628800]
    # N6 = [60480, 151200, 166320, 259200, 2592000]
    N7 = [16384, 20480, 48000, 100800, 201600, 362880, 604800, 1663200, 1814400, 3628800]

    # # 6 维
    # for n in N6:
    #     robot6 = cbt.Cobotta(pos=rm.vec(0.1, .3, .5), enable_cc=True)
    #     methods6 = {
    #         # "random": (make_random_joint_samples, {"robot": robot6}),
    #         # "cvt":    (make_cvt_samples, {"robot": robot6}),
    #         "rrt":    (make_rrt_samples, {"robot": robot6}),
    #     }
    #     benchmark(robot6, methods6, n=n, d=6, repeats=5)

    # 7 维
    for n in N7:
        robot7 = yumi.YumiSglArm(pos=rm.vec(0.1, .3, .5), enable_cc=True)
        methods7 = {
            # "random": (make_random_joint_samples, {"robot": robot7}),
            # "cvt":    (make_cvt_samples, {"robot": robot7}),
            "rrt":    (make_rrt_samples, {"robot": robot7}),
        }
        benchmark(robot7, methods7, n=n, d=7, repeats=5)
