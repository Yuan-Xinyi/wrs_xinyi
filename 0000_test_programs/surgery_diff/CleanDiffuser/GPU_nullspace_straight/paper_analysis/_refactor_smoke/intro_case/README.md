# Introduction Figure Candidate

This file lists the strongest candidate for a 'local greedy vs global better' introduction figure.

- Selected case: `986`
- Real-best length: `1.4689`
- Score-best length: `0.0000`
- Absolute gap: `1.4689`
- Retention ratio: `0.0000`
- Score advantage of score-best: `0.4052`

Recommended replay command:

```bash
/home/lqin/miniconda3/envs/wrs/bin/python /home/lqin/wrs_xinyi/0000_test_programs/surgery_diff/CleanDiffuser/GPU_nullspace_straight/length_prediction/vis_lnet_contrastive_fr3_extreme_failures.py --input /home/lqin/wrs_xinyi/0000_test_programs/surgery_diff/CleanDiffuser/GPU_nullspace_straight/length_prediction/lnet_contrastive_fr3_rotation_cone_eval_batch.jsonl --case-idx 986
```

Suggested narrative:

Under the same TCP task condition, the locally preferred solution receives a higher learned score,
but the globally better solution preserves substantially longer feasible rollout length.
This illustrates why one-step attractiveness is insufficient for future task potential.
