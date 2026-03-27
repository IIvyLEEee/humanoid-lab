# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import (
    euler_xyz_from_quat,
    matrix_from_quat,
    subtract_frame_transforms,
)

from .commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    # skate_pos_rel = ball_pos_rel(env)
    # print('skate_pos_rel:',skate_pos_rel)

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


def dr_state(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    # if not hasattr(env, "_dr_com_offset"):
    #     env._dr_com_offset = torch.zeros((env.num_envs, 3), device=env.device)
    # if not hasattr(env, "_dr_push"):
    #     env._dr_push = torch.zeros((env.num_envs, 6), device=env.device)
    # dr_state  = torch.cat([env._dr_com_offset, command._dr_push],dim=1)
    dr_state = command._dr_push

    # sum over contacts for each environment
    # print('dr_state:',dr_state)
    return dr_state


def ball_pos_rel(
    env: ManagerBasedEnv,
) -> torch.Tensor:
    """Position of skateboard in robot's frame"""
    skate_pos_rel = env.scene["ball_transform"].data.target_pos_source.squeeze(1)
    return skate_pos_rel


def ball_rot_rel(
    env: ManagerBasedEnv,
) -> torch.Tensor:
    """Orientation (yaw) of skateboard in robot's frame"""

    skate_rot_quat = env.scene["ball_transform"].data.target_quat_source.squeeze(1)
    skate_rot_euler = euler_xyz_from_quat(skate_rot_quat)
    # skate_rot_euler = torch.cat([skate_rot_euler[0].unsqueeze(1), skate_rot_euler[1].unsqueeze(1), skate_rot_euler[2].unsqueeze(1)], dim=1)
    return skate_rot_euler[2].unsqueeze(1)


# def skate_feet_contact_obs(
#     env: ManagerBasedRLEnv,
# ) -> torch.Tensor:
#     """Returns the presence of contact with the skate for each foot"""

#     FR_contact_sensor: ContactSensor = env.scene.sensors["FR_contact"]
#     FL_contact_sensor: ContactSensor = env.scene.sensors["FL_contact"]
#     RR_contact_sensor: ContactSensor = env.scene.sensors["RR_contact"]
#     RL_contact_sensor: ContactSensor = env.scene.sensors["RL_contact"]
#     FR_contact_sensor.data.force_matrix_w.squeeze(1)
#     cat = torch.cat([FR_contact_sensor.data.force_matrix_w.squeeze(1), FL_contact_sensor.data.force_matrix_w.squeeze(1),
#      RR_contact_sensor.data.force_matrix_w.squeeze(1), RL_contact_sensor.data.force_matrix_w.squeeze(1)], dim=1)

#     contact_tensor = torch.any(cat != 0, dim=2).float()
#     return contact_tensor

# def skate_point_cloud(
#     env: ManagerBasedEnv,
# ) -> torch.Tensor:
#     """Positions of points along the skateboard’s edge in robot's frame, spaced with "interval" """
#     # extract the used quantities (to enable type-hinting)
#     skate_pos = env.scene["ball_transform"].data.target_pos_source.squeeze(1)
#     skate_rot_quat = env.scene["ball_transform"].data.target_quat_source.squeeze(1)

#     # vectors = rectangle_perimeter_tensor(0.575, 0.43, 0.1).to(env.device)
#     length = 0.575
#     width = 0.25
#     interval = 0.083
#     vectors = rectangle_perimeter_tensor(length, width, interval).to(env.device)
#     vectors_transformed = transform_vectors_to_parent_frame(skate_pos, skate_rot_quat, vectors)

#     return vectors_transformed
