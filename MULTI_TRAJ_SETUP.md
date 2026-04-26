# Multi-Trajectory Mimic Setup

## Goal

Train one mimic policy on multiple reference motions instead of a single `.npz`, then compare against the Task 1 single-motion baseline on reward, episode length, tracking error, torque, and velocity.

## Motion Dataset

The multi-motion list is stored at:

```text
motions/filelist.txt
```

It contains 20 `.npz` files directly under `motions/`:

```text
jumps1_subject1.npz
jumps1_subject2.npz
jumps1_subject5.npz
push1_subject2.npz
run1_subject2.npz
run1_subject5.npz
run2_subject1.npz
run2_subject4.npz
sprint1_subject2.npz
sprint1_subject4.npz
walk1_subject1.npz
walk1_subject2.npz
walk1_subject5.npz
walk2_subject1.npz
walk2_subject3.npz
walk2_subject4.npz
walk3_subject2.npz
walk3_subject3.npz
walk3_subject5.npz
walk4_subject1.npz
```

Paths in `filelist.txt` are relative to the list file itself, so `walk1_subject1.npz` resolves to `motions/walk1_subject1.npz`.

## Code Changes

`scripts/rsl_rl/train.py`

- Added `--motion_files`, a text-file argument for multi-motion training.
- Kept the original `--motion_file` single-motion path for backward compatibility.
- For tracking tasks, motion priority is:
  1. `--motion_files`
  2. `--motion_file`
  3. `--registry_name`
- The loaded `motion_files` list is written into `env_cfg.commands.motion.motion_files` and saved in each run's `params/env.yaml`.

`source/era_okcc_humanoid_lab/era_okcc_humanoid_lab/tasks/mimic/mdp/commands.py`

- `MotionLoader` now accepts either one motion file or a list of motion files.
- All motions are loaded once at environment startup and concatenated into tensors.
- Per-motion `motion_lengths` and `motion_offsets` are stored so sampling never crosses motion boundaries.
- `MotionCommand` now keeps per-env `motion_ids` plus local `time_steps`.
- On reset/resampling, each env randomly chooses one motion id, then samples a valid timestep inside that motion.
- Added runtime evidence:
  - startup print: number of motion files, total frames, per-motion lengths
  - metrics: `motion_id`, `motion_id_min`, `motion_id_max`, `motion_id_unique_count`

## Training Commands

Single-GPU multi-motion run:

```bash
python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-L7_29Dof-v0 \
  --run_name multi_seed42 \
  --num_envs=8192 \
  --headless \
  --logger wandb \
  --log_project_name humanoid-lab \
  --seed 42 \
  --motion_files motions/filelist.txt
```

## Verification

Check the saved run config:

```text
logs/rsl_rl/mimic/<run_name>/params/env.yaml
```

Expected fields:

```yaml
motion_file: /home/liyixuan/humanoid-lab/motions/jumps1_subject1.npz
motion_files:
- /home/liyixuan/humanoid-lab/motions/jumps1_subject1.npz
- ...
- /home/liyixuan/humanoid-lab/motions/walk4_subject1.npz
```

During training, confirm these metrics:

- `motion_id_min` should be near `0`.
- `motion_id_max` should be near `19`.
- `motion_id_unique_count` should usually be close to `20` with many envs.
- `motion_id` should fluctuate around the middle of the id range.

## Notes

The per-step training time is expected to be close to single-motion training. The 20 motions are loaded once and indexed by `motion_id + time_step`; physics simulation and PPO updates remain the dominant cost.

Reward and episode length can overlap with the single-motion baseline if the selected motions are similar or the baseline motion is already representative. The main comparison should include tracking error, torque, velocity, and evaluation on the same Task 1 motion.
