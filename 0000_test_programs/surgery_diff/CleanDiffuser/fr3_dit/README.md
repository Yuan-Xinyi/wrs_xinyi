# fr3_dit — Farsighted IK via Task-Conditioned Diffusion Transformer

Pipeline for FR3 plane-constrained trajectory generation, topological stitching, and task tokenization. Feeds a downstream DiT-based policy.

## Structure

```
fr3_dit/
├── core/                 # Robot model + visualization helpers
│   ├── pen_fr3_robot.py     # PEN_LENGTH, PenFrankaResearch3 / PenFrankaResearch3GPU
│   └── viz_utils.py         # visualize_anime_path
├── data_generation/      # Phase 0: collect raw plane-constrained straight trajectories
│   └── generate_fr3_plane_dataset.py
├── stitching/            # Phase 1+2: anchor-based stitching → composite tasks + tokens
│   └── stitch_composite_tasks.py
├── training/             # Phase 3: DiT training-side data
│   └── composite_task_dataset.py
├── visualization/        # Viewers for raw + composite trajectories
│   ├── visualize_fr3_plane_trajectory.py
│   └── visualize_composite_task.py
├── experiments/          # Ablations
│   ├── test_same_task_start_conf_gap.py
│   └── outputs/             # gitignored
├── data/                 # HDF5 datasets (gitignored)
│   ├── pen_fr3_plane_trajectories.hdf5
│   └── pen_fr3_composite_tasks.hdf5
└── README.md
```

## Run commands

All commands are run from the `CleanDiffuser/` directory using the `-m` module form.

```bash
cd /home/lqin/wrs_xinyi/0000_test_programs/surgery_diff/CleanDiffuser
```

### Phase 0 — Collect raw plane-constrained trajectories

```bash
python -m fr3_dit.data_generation.generate_fr3_plane_dataset \
    --num-trajectories 1000 \
    --batch-size 32
```

Writes `fr3_dit/data/pen_fr3_plane_trajectories.hdf5`.

Useful flags: `--theta-max-deg`, `--angle-margin-deg`, `--joint-margin-ratio`, `--max-steps`, `--seed`.

### Phase 1+2 — Stitch + tokenize composite tasks

Two-stage criterion, in priority order:

1. **Primary — Workspace intersection**: two anchors are candidates iff their TCP positions are within `--tcp-eps-m` meters. Guarantees composite trajectories are spatially connected.
2. **Secondary — C-space similarity**: among TCP-proximate candidates, keep only those with weighted joint-space distance

   $$\big\|W (q_A - q_B)\big\| < \epsilon_q, \qquad W = \mathrm{diag}(1/\mathrm{span}_i)$$

   This rejects pairs that share a TCP point but sit on different IK branches (elbow-up vs elbow-down etc.).
3. **Optional** — plane / seam physics, off by default.

```bash
python -m fr3_dit.stitching.stitch_composite_tasks \
    --tcp-eps-m 0.02 \
    --c-eps 0.5 \
    --anchor-stride 10 \
    --max-hops 3 \
    --include-singles

# Tighter — require co-planar trajectories + smooth seams
python -m fr3_dit.stitching.stitch_composite_tasks \
    --tcp-eps-m 0.02 --c-eps 0.3 \
    --enforce-plane --enforce-physics --include-singles

# Small-data fallback: split each long trajectory into N contiguous sub-segments
python -m fr3_dit.stitching.stitch_composite_tasks --self-slice 3 --include-singles
```

Reads `fr3_dit/data/pen_fr3_plane_trajectories.hdf5`,
writes `fr3_dit/data/pen_fr3_composite_tasks.hdf5`.

Key flags:
- `--tcp-eps-m` (default `0.02`) — **primary**: TCP-space seam radius in meters
- `--c-eps` (default `0.1`) — **secondary**: weighted C-space similarity bound (dimensionless). At `0.1` cross-trajectory stitches stay on the same IK branch; loosen (e.g. `0.3`) to admit branch switches and find more 3-hop composites at the cost of bigger joint jumps at the seam.
- `--anchor-stride` (default `10`) — sample an anchor every N steps along each trajectory
- `--enforce-plane` (off by default) — additionally require same plane normal (within `--plane-cos`) and same plane side
- `--enforce-physics` (off by default) — additionally require cubic-fit `v/a/jerk` at the seam under `--vel-ratio`/`--acc-ratio`/`--jerk-ratio`
- `--max-hops` (default `3`), `--max-per-seed` (default `16`)
- `--self-slice N` — fallback N-way split of each long trajectory (continuous, Δθ≈0)
- `--include-singles` — also emit single-segment tasks for curriculum

> **Data-density note**: with few independently-sampled trajectories, true 3-trajectory workspace hubs are rare. Expect tighter `--tcp-eps-m` (1–2 cm) to yield few 3-segment composites until the raw dataset grows to ≥ 5000 trajectories. Use `--self-slice 3` for continuous demo data, or loosen `--tcp-eps-m` to 5–10 cm to accept visible TCP gaps at seams.

### Phase 3 — Dataset sanity check

```bash
python -m fr3_dit.training.composite_task_dataset --batch-size 4 --num-batches 2
```

Use `CompositeTaskDataset` + `dit_collate` from `fr3_dit.training.composite_task_dataset`.

### Visualization

```bash
# Single raw trajectory
python -m fr3_dit.visualization.visualize_fr3_plane_trajectory --seed 42

# Composite task (any seg count)
python -m fr3_dit.visualization.visualize_composite_task --seed 42

# Only 3-segment composites
python -m fr3_dit.visualization.visualize_composite_task --min-segs 3 --seed 42

# Animate robot along the path
python -m fr3_dit.visualization.visualize_composite_task --min-segs 3 --seed 42 --toggle-animate

# Specific task index
python -m fr3_dit.visualization.visualize_composite_task --task-idx 200
```

### Experiments

```bash
python -m fr3_dit.experiments.test_same_task_start_conf_gap --num-starts 64
```

Writes `fr3_dit/experiments/outputs/same_task_start_conf_gap_curves.svg`.

## HDF5 layout (composite tasks)

```
/meta                                # hyperparams used at stitch time
/raw_trajs
    q_flat, tcp_flat, offset         # packed raw trajectories
    direction, plane_normal, plane_point, plane_side, length
/tasks
    token_flat    : (T_total, 32)    # packed tokens, all tasks concatenated
    token_offset  : (M+1,) int64     # task m tokens at [offset[m], offset[m+1])
    token_kind    : (T_total,) u8    # 0=start, 1=segment, 2=corner
    qtraj_flat    : (Q_total, 7)     # packed joint-space trajectory
    qtraj_offset  : (M+1,) int64
    tcp_flat      : (Q_total, 3)
    start_q       : (M, 7)
    local_frame   : (M, 3, 3)        # columns = (x̂, ŷ, ẑ)
    local_origin  : (M, 3)
    plane_normal  : (M, 3)           # world frame
    seg_count     : (M,) u16
    total_length  : (M,) f32
    subseg_meta_flat   : (S_total, 3) int32   # (traj_id, start, end)
    subseg_offset      : (M+1,) int64
    seg_step_counts_flat : (S_total,) int32
```

## Token layout (`token_dim = 32`)

| offset | size | channel | semantics |
|---|---|---|---|
| 0..3   | 3 | `kind_onehot`        | `[start, segment, corner]` |
| 3..6   | 3 | `dir_local`          | segment direction in local frame |
| 6      | 1 | `len_norm`           | segment length / `length_ref` |
| 7..9   | 2 | `delta_theta_sincos` | `[sin Δθ, cos Δθ]` for corner |
| 9..12  | 3 | `axis_local`         | corner rotation axis |
| 12..15 | 3 | `bisector_local`     | corner angle bisector |
| 15..18 | 3 | `plane_normal_local` | plane normal in local frame |
| 18     | 1 | `cum_len_norm`       | cumulative arc length / `length_ref` |
| 19..27 | 8 | `fourier_time`       | 4-band Fourier features of `cum_len_norm` |
| 27..32 | 5 | `pad`                | zero, reserved |

Local frame: `x̂ = v̂₁` (first segment direction), `ẑ = n̂` (plane normal after Gram-Schmidt vs `x̂`), `ŷ = ẑ × x̂`.

## Physical limits used at seam check

FR3 per-joint maxima (conservative):
- `q̇_max` = `[2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610]` rad/s
- `q̈_max` = `[15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0]` rad/s²
- `⃛q_max` = `[7500, 3750, 5000, 6250, 7500, 10000, 10000]` rad/s³

Seam check fits a cubic per joint over a `window_steps` window on each side, evaluates `(v, a)` at the endpoints, and compares to `{vel, acc, jerk}_ratio × FR3 limits`.
