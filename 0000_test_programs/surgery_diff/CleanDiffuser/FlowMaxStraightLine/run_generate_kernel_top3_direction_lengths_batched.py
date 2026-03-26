import subprocess
import sys
from pathlib import Path

import numpy as np


BASE_DIR = Path("0000_test_programs/surgery_diff/CleanDiffuser/FlowMaxStraightLine")
INPUT_NPY = BASE_DIR / "datasets" / "cvt_kernels_collision_free_10000.npy"
OUTPUT_H5 = BASE_DIR / "datasets" / "xarm_tcpz_orthogonal_kernel_top3_10000.h5"
GENERATE_SCRIPT = BASE_DIR / "generate_kernel_top3_direction_lengths.py"

ROBOT = "xarm"
SAMPLING_MODE = "orientation_cone"
BATCH_SIZE = 500
CHECKPOINT_INTERVAL = 100
PYTHON_EXE = Path(sys.executable)


def main():
    if not INPUT_NPY.exists():
        raise FileNotFoundError(f"Input kernel file not found: {INPUT_NPY}")
    if not GENERATE_SCRIPT.exists():
        raise FileNotFoundError(f"Generate script not found: {GENERATE_SCRIPT}")

    kernel_qs = np.load(INPUT_NPY)
    total_kernels = int(kernel_qs.shape[0])

    print(f"[BatchRunner] python={PYTHON_EXE}", flush=True)
    print(f"[BatchRunner] input={INPUT_NPY}", flush=True)
    print(f"[BatchRunner] output={OUTPUT_H5}", flush=True)
    print(f"[BatchRunner] robot={ROBOT} | sampling_mode={SAMPLING_MODE}", flush=True)
    print(
        f"[BatchRunner] batch_size={BATCH_SIZE} | checkpoint_interval={CHECKPOINT_INTERVAL} | total_kernels={total_kernels}",
        flush=True,
    )

    for batch_start in range(0, total_kernels, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_kernels)
        cmd = [
            str(PYTHON_EXE),
            str(GENERATE_SCRIPT),
            "--resume",
            "--input-npy",
            str(INPUT_NPY),
            "--output-h5",
            str(OUTPUT_H5),
            "--robot",
            ROBOT,
            "--sampling-mode",
            SAMPLING_MODE,
            "--checkpoint-interval",
            str(CHECKPOINT_INTERVAL),
            "--start-idx",
            str(batch_start),
            "--end-idx",
            str(batch_end),
        ]
        print("", flush=True)
        print(f"[BatchRunner] launch range=[{batch_start}, {batch_end})", flush=True)
        print(f"[BatchRunner] cmd={' '.join(cmd)}", flush=True)
        completed = subprocess.run(cmd, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"Subprocess failed for range [{batch_start}, {batch_end}) with return code {completed.returncode}"
            )
        print(f"[BatchRunner] finished range=[{batch_start}, {batch_end})", flush=True)

    print("", flush=True)
    print("[BatchRunner] all requested ranges finished.", flush=True)


if __name__ == "__main__":
    main()
