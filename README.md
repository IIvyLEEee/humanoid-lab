# Humanoid Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.2-silver)](https://isaac-sim.github.io/IsaacLab)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)






## Overview

This project provides reinforcement learning pipeline for **locomotion** and **single-motion tracking** on **L7 robots**, built on **Isaac Lab**, covering the full workflow from policy training to **sim2sim** and **sim2real** deployment.

<table border="1" cellpadding="6" cellspacing="0">
  <tr>
    <th></th>
    <th>motion</th>
    <th>sim2sim</th>
    <th>sim2real</th>
  </tr>
  <tr>
    <td>locomotion</td>
    <td></td>
    <td><video width="320" height="240" src="https://github.com/user-attachments/assets/44acb152-8183-4763-b757-2fef5a2bcb9d"></video></td>
    <td><video width="320" height="240" src="https://github.com/user-attachments/assets/b2a6566c-84a8-4464-b605-6cfedfce697d"></video></td>
  </tr>
  <tr>
    <td>mimic</td>
    <td><video width="320" height="240" src="https://github.com/user-attachments/assets/93251bc4-d8cc-4028-a224-6ce0baa40701"></video></td>
    <td><video width="320" height="240" src="https://github.com/user-attachments/assets/9f06f9be-4f39-4bb2-b0af-033e5c552b44"></video></td>
    <td><video width="320" height="240" src="https://github.com/user-attachments/assets/d2e548c5-5580-4067-a163-5180c106339a"></video></td>
  </tr>
</table>



## Installation

- Install Isaac Lab v2.3.2 by following
  the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We recommend
  using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone this repository separately from the Isaac Lab installation (i.e., outside the `IsaacLab` directory):

- Using a Python interpreter that has Isaac Lab installed, install the library

```bash
python -m pip install -e source/era_okcc_humanoid_lab
```

## Motion Tracking

### Motion Preprocessing

We use [GMR](https://github.com/YanjieZe/GMR) to retarget collected mocap data to the L7 robot.

- Convert retargeted motions to include the maximum coordinates information (body pose, body velocity, and body
  acceleration) via forward kinematics,

```bash

python scripts/l7_csv_to_npz.py   --input_file {motion_name}.csv    --input_fps 30 --output_name {motion_name}  --save_to motions/{motion_name}.npz  --no_wandb

```

for example,


```bash

python scripts/l7_csv_to_npz.py   --input_file motions/csv/dance_7.csv   --input_fps 30 --output_name dance_7 --save_to motions/dance_7.npz  --no_wandb

```

This will automatically upload the processed motion file to the WandB registry with output name {motion_name}.

- Test if the npz file works properly by replaying the motion in Isaac Sim:

```bash
python scripts/replay_l7_npz.py --motion_file motions/{motion_name}.npz

```

for example,

```bash
python scripts/replay_l7_npz.py --motion_file motions/dance_7.npz

```

### Policy Training

- Train tracking policy by the following command:

```bash
 python scripts/rsl_rl/train.py   --task=Tracking-Flat-L7_29Dof-v0    --motion_file {motion_name}   --run_name  {run_name}   --num_envs=8192   --headless
```

for example,
```bash
 python scripts/rsl_rl/train.py   --task=Tracking-Flat-L7_29Dof-v0    --motion_file motions/long_motion_1.npz   --run_name Tracking-long_motion_1   --num_envs=8192   --headless
```
### Policy Evaluation

- Play the trained policy by the following command:

```bash
# play tracking policy
python scripts/rsl_rl/play.py --task=Tracking-Flat-L7_29Dof-v0  --motion_file motions/long_motion_1.npz --model_path {model_path} --num_envs 2
```

## Locomotion
### Policy Training
- Train locomotion policy by the following command

```bash
 python scripts/rsl_rl/train.py --task=Locomotion-Flat-L7_29Dof-v0 --run_name L7_locomotion   --num_envs=8192   --headless
```

### Policy Evaluation
- Play the trained policy by the following command:

```bash
python scripts/rsl_rl/play.py --task=Locomotion-Flat-L7_29Dof-v0  --model_path {model_path} --num_envs 2
```

## Deployment
### Prerequisites
- Ubuntu with ROS 2 Humble installed. See [ROS2 installation instructions](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html).
- Python packages required for inference and MuJoCo:
```bash
pip install onnx onnxruntime mujoco pynput
```

### Sim2Sim

- Launch controller node:

```bash
cd deploy/

# source ros2 environment
source /opt/ros/humble/setup.bash

# build era_rl_controller package
colcon build --packages-up-to era_rl_controller

# source the setup files
source install/setup.bash

# run controller node with the tracking configuration
ros2 run era_rl_controller era_rl_controller_node --config src/era_rl_controller/configs/mimic_dance_9.yaml --mode sim2sim

# or run controller node with the locomotion configuration
ros2 run era_rl_controller era_rl_controller_node --config src/era_rl_controller/configs/loco_walk_1.yaml --mode sim2sim
```

- Launch robot node (new terminal):

```bash
# source ros2 environment
source /opt/ros/humble/setup.bash

# source the setup files
source install/setup.bash

# launch era_robot node
ros2 launch era_robot era_robot_launch.py


```

### Sim2Real

Please follow the L7 robot's official instructions to start the robot, then run the following commands to deploy the policy:

```bash
# ros2 run era_rl_controller era_rl_controller_node --config path/to/config --mode real
ros2 run era_rl_controller era_rl_controller_node --config src/era_rl_controller/configs/mimic_dance_9.yaml --mode real

ros2 run era_rl_controller era_rl_controller_node --config src/era_rl_controller/configs/loco_walk_1.yaml --mode real
```



## Acknowledgements

This repository is built upon the support and contributions of the following open-source projects. Special thanks to:

- [IsaacLab](https://github.com/isaac-sim/IsaacLab): The foundation for training and running codes.
- [mujoco](https://github.com/google-deepmind/mujoco.git): Providing powerful simulation functionalities.
- [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking): Versatile humanoid control framework for motion tracking.
- [GMR](https://github.com/YanjieZe/GMR): The motion retargeting and processing pipeline.
