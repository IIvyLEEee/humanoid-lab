# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import (
    quat_apply_inverse,
    quat_error_magnitude,
)

from .commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    # print('motion_global_body_angular_velocity_error_exp body_names:',body_names)
    body_indexes = _get_body_indexes(command, body_names)
    # print('body_indexes in motion_global_body_angular_velocity_error_exp:',body_indexes)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward


def track_joint_pos(env: ManagerBasedRLEnv, std: float, command_name: str, joint_names="[.*]") -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    goal_joint_angles = command.joint_pos
    current_joint_angles = command.robot_joint_pos
    # print('track_joint_pos goal_joint_angles.shape:',goal_joint_angles.shape)
    # print('track_joint_pos current_joint_angles.shape:',current_joint_angles.shape)

    # print('track_joint_pos goal_joint_angles:\n',goal_joint_angles)
    # print('track_joint_pos current_joint_angles:\n:',current_joint_angles)

    joint_ang_error = torch.sum(torch.square(current_joint_angles - goal_joint_angles), dim=-1)
    # print("joint_ang_error.shape: ", joint_ang_error.shape)
    ret = torch.exp(-joint_ang_error / std**2)
    print("track_joint_pos ret: ", ret)
    return ret


def jnt_powers(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_names: list = None,
    scale: float = 1.0,
) -> torch.Tensor:
    """Penalize high energy consumption.

    This function computes the energy consumption as the product of joint torques and
    squared joint velocities. This is a common model for energy consumption in robotics,
    where energy is proportional to the product of force and velocity.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the asset to compute energy for.
        joint_names: Names of joints to include in computation. If None, all joints are used.
        scale: Scaling factor for the energy computation.

    Returns:
        torch.Tensor: Energy consumption penalty for each environment.
    """
    # Extract the asset
    asset = env.scene[asset_cfg.name]

    # Get joint IDs if joint_names is provided
    if joint_names is not None:
        joint_ids = []
        for name in joint_names:
            ids = asset.find_joints(name)[0]
            if len(ids) > 0:
                joint_ids.extend(ids)
    else:
        # Use all joints
        joint_ids = None

    # Get joint torques and velocities
    joint_torques = asset.data.applied_torque
    # joint_torques = asset.data.joint_torques
    joint_vel = asset.data.joint_vel

    # Calculate energy as torque * velocity^2
    if joint_ids is not None:
        # Only for specified joints
        energy = torch.sum(torch.abs(joint_torques[:, joint_ids]) * torch.square(joint_vel[:, joint_ids]), dim=1)
    else:
        # For all joints
        energy = torch.sum(torch.abs(joint_torques) * torch.square(joint_vel), dim=1)

    # Apply scaling
    ret = energy * scale
    # print('jnt_powers:',ret)
    return ret


def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: RigidObject = env.scene[asset_cfg.name]

    # feet_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    # reward = torch.sum(feet_vel.norm(dim=-1) * contacts, dim=1)

    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footvel_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footvel_translated[:, i, :])
    foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(
        env.num_envs, -1
    )
    reward = torch.sum(foot_leteral_vel * contacts, dim=1)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    # print('feet_slide:',reward)
    return reward


# def skate_feet_contact(
#     env: ManagerBasedRLEnv,
#     height_threshold = float
# ) -> torch.Tensor:
#     """Reward for feet contact with skateboard"""
#     FR_contact_sensor: ContactSensor = env.scene.sensors["FR_contact"]
#     FL_contact_sensor: ContactSensor = env.scene.sensors["FL_contact"]
#     RR_contact_sensor: ContactSensor = env.scene.sensors["RR_contact"]
#     RL_contact_sensor: ContactSensor = env.scene.sensors["RL_contact"]
#     FR_contact_sensor.data.force_matrix_w.squeeze(1)
#     cat = torch.cat([FR_contact_sensor.data.force_matrix_w.squeeze(1), FL_contact_sensor.data.force_matrix_w.squeeze(1),
#      RR_contact_sensor.data.force_matrix_w.squeeze(1), RL_contact_sensor.data.force_matrix_w.squeeze(1)], dim=1)
#     heigh_mask = torch.cat([FR_contact_sensor.data.pos_w[..., 2], FL_contact_sensor.data.pos_w[..., 2],
#      RR_contact_sensor.data.pos_w[..., 2], RL_contact_sensor.data.pos_w[..., 2]], dim=1)

#     heigh_mask = heigh_mask > height_threshold
#     contact_tensor = torch.any(cat != 0, dim=2)
#     contact_tensor &= heigh_mask
#     contact_tensor = contact_tensor.float()
#     # contact_tensor *= torch.tensor([1, 1, 1.5, 1.5], device = env.device)
#     reward = torch.sum(contact_tensor, dim=1)
#     # reward = torch.where(env.episode_length_buf > 300, reward, 0)
#     # reward = 2**reward
#     # reward = torch.where(reward > 3.5, reward, 0)
#     reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
#     reward = torch.clamp(reward, 0, 4)
#     return reward


def skate_orientation_tracking(
    env: ManagerBasedRLEnv,
    distance_threshold: float,
    std: float,
) -> torch.Tensor:
    """Reward for align robot and skateboard coordinate frames (orientation).
    Works only if distance between robot and skate less than "distance_threshold" arg
    """

    skate_rot_rel = env.scene["ball_transform"].data.target_quat_source.squeeze(1)
    skate_angle_rel = 2 * torch.acos(torch.clamp(torch.abs(skate_rot_rel[:, 0]), max=1.0))
    distance = torch.linalg.norm(env.scene["ball_transform"].data.target_pos_source.squeeze(1), dim=1)

    vicinity_mask = (distance < distance_threshold).float()
    error = skate_angle_rel / torch.pi
    error = torch.clamp(error, min=0)
    reward = torch.exp(-error / std**2)
    reward = reward * vicinity_mask
    # reward = torch.where(env.episode_length_buf > 300, reward, 0)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    print("skate_orientation_tracking:", reward)
    return reward


def skate_distance_tracking(
    env: ManagerBasedRLEnv,
    std: float,
) -> torch.Tensor:
    """Reward for align robot and skateboard coordinate frames (distance)."""

    distance = torch.linalg.norm(env.scene["ball_transform"].data.target_pos_source.squeeze(1)[:, :2], dim=1)
    reward = torch.exp(-distance / std**2)
    # reward = torch.where(env.episode_length_buf > 300, reward, 0)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    print("skate_distance_tracking:", reward)

    return reward


def skateboard_upward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty for skateboard being tilted or upside down."""
    skateboard = env.scene["ball"]
    upward = skateboard.data.projected_gravity_b[:, 2]
    reward = upward > 0
    print("skateboard_upward:", reward)

    return reward


def skate_velocity_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize skate velocity."""

    asset: Articulation = env.scene[asset_cfg.name]
    skate_vel = torch.linalg.norm(asset.data.root_lin_vel_b, dim=1)
    reward = torch.clamp(skate_vel, 0, 5)
    print("skate_velocity_penalty:", reward)

    return reward
