# Max Manipulability Framework

Modules:

- `pybullet_franka.py`: PyBullet Franka wrapper
- `nullspace_tracker.py`: low-level null-space straight-line executor
- `franka_single_point_rl.py`: goal-conditioned continuous-action PPO training pipeline
- `offline_bc.py`: offline random dataset generation and BC pretraining
- `visualize_policy_vs_random.py`: compare RL-selected q0 vs random q0 path lengths

Training:

```bash
python 0000_test_programs/surgery_diff/CleanDiffuser/max_manipulability/franka_single_point_rl.py
```

Visualization:

```bash
python 0000_test_programs/surgery_diff/CleanDiffuser/max_manipulability/visualize_policy_vs_random.py
```
