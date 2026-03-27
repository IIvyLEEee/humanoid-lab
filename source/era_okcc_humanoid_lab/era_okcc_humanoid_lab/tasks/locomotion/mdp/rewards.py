# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from whole_body_tracking.tasks.locomotion.mdp.commands import CustomUniformVelocityCommand

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def get_body_ids(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), name_keys: list = None
) -> torch.Tensor:
    """Retrieve body IDs for the specified asset from the environment scene."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.find_bodies(name_keys)


def traking_joint_pos(
    env: ManagerBasedRLEnv,
    command_name: str,
    joint_names: list,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sigma: float = 0.1,
) -> torch.Tensor:
    """Reward for tracking the desired joint positions based on the provided command."""
    asset: Articulation = env.scene[asset_cfg.name]
    isaac_joint_ids, _ = asset.find_joints(joint_names)
    command = env.command_manager.get_term(command_name)

    ref_joint_pos = command.ref_action[:, isaac_joint_ids]
    current_joint_pos = asset.data.joint_pos[:, isaac_joint_ids]

    jit_diff = ref_joint_pos - current_joint_pos
    return torch.exp(-sigma * torch.norm(jit_diff, dim=1))


def feet_contact_number(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Reward based on feet contact accurately aligning with the expected gait stance phase."""
    command: CustomUniformVelocityCommand = env.command_manager.get_term(command_name)
    stance_mask = command.stance_mask

    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    num_feet = stance_mask.shape[1]
    feet_contact = contact_sensor.data.net_forces_w[:, :num_feet, 2] > threshold

    reward = torch.where(feet_contact == stance_mask, 1.0, -0.2)
    return torch.mean(reward, dim=1)


def feet_contact_number_sum(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Reward matching the total number of feet in contact with the desired contact count."""
    command: CustomUniformVelocityCommand = env.command_manager.get_term(command_name)
    desired_contact_num = command.contact_number_des

    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    feet_contact = contact_sensor.data.net_forces_w[:, :2, 2] > threshold

    actual_contact_num = torch.sum(feet_contact.float(), dim=1)
    return torch.where(actual_contact_num == desired_contact_num, 1.0, -0.2)


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Penalize air time based on the last contact time of the feet."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.sum((last_contact_time < threshold) * first_air, dim=-1)


def jnt_powers(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_names: list = None,
    scale: float = 1.0,
) -> torch.Tensor:
    """Penalize high energy consumption, calculated as torque * velocity^2."""
    asset: Articulation = env.scene[asset_cfg.name]

    joint_ids = None
    if joint_names is not None:
        joint_ids = []
        for name in joint_names:
            ids = asset.find_joints(name)[0]
            if len(ids) > 0:
                joint_ids.extend(ids)

    joint_torques = asset.data.applied_torque
    joint_vel = asset.data.joint_vel

    if joint_ids is not None:
        energy = torch.sum(torch.abs(joint_torques[:, joint_ids]) * torch.square(joint_vel[:, joint_ids]), dim=1)
    else:
        energy = torch.sum(torch.abs(joint_torques) * torch.square(joint_vel), dim=1)

    return energy * scale


def swing_foot_clearance(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward for tracking the desired task-space foot height during the swing phase."""
    command: CustomUniformVelocityCommand = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[asset_cfg.name]

    body_ids, _ = asset.find_bodies(asset_cfg.body_names)
    feet_z = asset.data.body_pos_w[:, body_ids, 2]
    desired_z = command.feet_desired_z
    swing_mask = 1.0 - command.stance_mask.float()

    height_error = torch.abs(desired_z - feet_z) * swing_mask
    return torch.exp(-10.0 * torch.sum(height_error, dim=1))
