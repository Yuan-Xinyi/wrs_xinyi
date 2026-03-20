import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np


BASE_DIR = Path("0000_test_programs/surgery_diff/CleanDiffuser/FlowMaxStraightLine")
INPUT_NPY = BASE_DIR / "datasets" / "cvt_kernels_collision_free.npy"
OUTPUT_H5 = BASE_DIR / "datasets" / "xarm_trail1_large_scale_top10.h5"
GENERATE_SCRIPT = BASE_DIR / "generate_datasets.py"
BOOTSTRAP_H5 = BASE_DIR / "datasets" / "checkpoints_till4800" / "latest.h5"

START_IDX = 4800
BATCH_SIZE = 1000
CHECKPOINT_INTERVAL = 1000
PYTHON_EXE = Path(sys.executable)


def probe_h5(path):
    if not path.exists():
        return False, None
    try:
        with h5py.File(path, "r") as f:
            done_count = int(np.count_nonzero(f["done_mask"][:]))
        return True, done_count
    except Exception:
        return False, None


def ensure_bootstrap_output():
    output_ok, output_done = probe_h5(OUTPUT_H5)
    if output_ok and output_done is not None and output_done >= START_IDX:
        print(f"[BatchRunner] existing output is readable, done_count={output_done}", flush=True)
        return

    bootstrap_ok, bootstrap_done = probe_h5(BOOTSTRAP_H5)
    if not bootstrap_ok:
        raise RuntimeError(f"Bootstrap checkpoint is not readable: {BOOTSTRAP_H5}")
    if bootstrap_done is None or bootstrap_done < START_IDX:
        raise RuntimeError(
            f"Bootstrap checkpoint done_count={bootstrap_done} is smaller than START_IDX={START_IDX}: {BOOTSTRAP_H5}"
        )

    OUTPUT_H5.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BOOTSTRAP_H5, OUTPUT_H5)
    print(
        f"[BatchRunner] restored output from bootstrap checkpoint | "
        f"bootstrap={BOOTSTRAP_H5} | done_count={bootstrap_done}",
        flush=True,
    )


def main():
    if not INPUT_NPY.exists():
        raise FileNotFoundError(f"Input kernel file not found: {INPUT_NPY}")
    if not GENERATE_SCRIPT.exists():
        raise FileNotFoundError(f"Generate script not found: {GENERATE_SCRIPT}")

    kernel_qs = np.load(INPUT_NPY)
    total_kernels = int(kernel_qs.shape[0])
    ensure_bootstrap_output()

    print(f"[BatchRunner] python={PYTHON_EXE}", flush=True)
    print(f"[BatchRunner] input={INPUT_NPY}", flush=True)
    print(f"[BatchRunner] output={OUTPUT_H5}", flush=True)
    print(f"[BatchRunner] bootstrap={BOOTSTRAP_H5}", flush=True)
    print(
        f"[BatchRunner] start_idx={START_IDX} | batch_size={BATCH_SIZE} | "
        f"total_kernels={total_kernels}",
        flush=True,
    )

    for batch_start in range(START_IDX, total_kernels, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_kernels)
        cmd = [
            str(PYTHON_EXE),
            str(GENERATE_SCRIPT),
            "--resume",
            "--input-npy",
            str(INPUT_NPY),
            "--output-h5",
            str(OUTPUT_H5),
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
                f"Subprocess failed for range [{batch_start}, {batch_end}) "
                f"with return code {completed.returncode}"
            )
        print(f"[BatchRunner] finished range=[{batch_start}, {batch_end})", flush=True)

    print("", flush=True)
    print("[BatchRunner] all requested ranges finished.", flush=True)


if __name__ == "__main__":
    main()
