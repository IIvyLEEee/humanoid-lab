# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the velocity-based locomotion task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import omni.log

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.commands import UniformVelocityCommandCfg
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

PITCH_TO_FEET = 0.6164
LEG_LENGTH = 0.36


class CustomUniformVelocityCommand(CommandTerm):
    r"""Command generator that generates a velocity command in SE(2) from a uniform distribution."""

    cfg: UniformVelocityCommandCfg

    def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator."""
        super().__init__(cfg, env)

        if self.cfg.heading_command and self.cfg.ranges.heading is None:
            raise ValueError("Heading command is active but `ranges.heading` is None.")
        if self.cfg.ranges.heading and not self.cfg.heading_command:
            omni.log.warn("`ranges.heading` is set but heading command is not active.")

        self.robot: Articulation = env.scene[cfg.asset_name]

        # Initialize command buffers
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_standing_env = torch.zeros_like(self.is_heading_env)

        # Initialize metrics
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

        # Initialize gait properties
        self.gait_phase = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.gait_cycle = torch.full(
            (self.num_envs,), self.cfg.gait.gait_cycle, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.swing_ratio = torch.full(
            (self.num_envs,), float(self.cfg.gait.swing_ratio), device=self.device, dtype=torch.float32
        )
        self.stance_ratio = torch.full(
            (self.num_envs,), float(self.cfg.gait.stance_ratio), device=self.device, dtype=torch.float32
        )
        self.swing_phase = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)
        self.stance_phase = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)
        self.phase_offset = torch.tensor(
            [self.cfg.gait.gait_phase_offset_l, self.cfg.gait.gait_phase_offset_r],
            dtype=torch.float,
            device=self.device,
        ).repeat(self.num_envs, 1)

        self.ref_action = torch.zeros((self.num_envs, self.cfg.action.dim), device=self.device, dtype=torch.float32)
        self.stance_mask = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)
        self.contact_number_des = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.feet_desired_z = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)

        # Cache Isaac Lab's internal joint indices
        self.idx_pitch = [
            self.robot.find_joints("left_hip_pitch_joint")[0][0],
            self.robot.find_joints("right_hip_pitch_joint")[0][0],
        ]
        self.idx_knee = [
            self.robot.find_joints("left_knee_joint")[0][0],
            self.robot.find_joints("right_knee_joint")[0][0],
        ]
        self.idx_ankle_pitch = [
            self.robot.find_joints("left_ankle_pitch_joint")[0][0],
            self.robot.find_joints("right_ankle_pitch_joint")[0][0],
        ]

    def __str__(self) -> str:
        msg = "UniformVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        if self.cfg.heading_command:
            msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.vel_command_b

    @property
    def robot_lin_vel(self) -> torch.Tensor:
        return self.robot.data.root_lin_vel_b

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def desired_stance_mask(self) -> torch.Tensor:
        return self.stance_mask

    def _update_metrics(self):
        max_command_step = self.cfg.resampling_time_range[1] / self._env.step_dt
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]) / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)

        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self):
        """Post-processes the velocity command to handle heading control and standing envs."""
        if self.cfg.heading_command:
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            heading_error = math_utils.wrap_to_pi(self.heading_target[env_ids] - self.robot.data.heading_w[env_ids])
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )

        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0
        self._get_walk_traj()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0

        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)

        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)
        return arrow_scale, arrow_quat

    def parabolic_trajectory(self, t, t_total, y_start, y_mid, y_end):
        """Generates a parabolic trajectory for the swing foot."""
        T_safe = torch.clamp(t_total.unsqueeze(1), min=1e-6)
        delta = y_end - y_start
        B = (4.0 * (y_mid - y_start) - delta) / T_safe
        A = (delta - B * T_safe) / (T_safe**2)
        return A * (t**2) + B * t + y_start

    def calc_desired_feet_z(self, stance_phase, swing_phase, swing_ratio, cycle_time):
        """Calculates the desired task-space Z trajectory for the feet."""
        v_takeoff = torch.zeros_like(stance_phase)
        zeros = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float)

        # Swing phase
        t_total_swing = swing_ratio * cycle_time
        t_swing = swing_phase * t_total_swing.unsqueeze(1)
        y_mid_swing = self.cfg.foot_height * torch.ones_like(zeros)
        pos_swing = self.parabolic_trajectory(t_swing, t_total_swing, zeros, y_mid_swing, zeros)

        # Stance phase
        t_total_stance = (1.0 - swing_ratio) * cycle_time
        t_stance = stance_phase * t_total_stance.unsqueeze(1)
        y_mid_stance = (v_takeoff**2) / 30.0
        pos_stance = self.parabolic_trajectory(t_stance, t_total_stance, zeros, y_mid_stance, zeros)

        stance_mask = stance_phase > 0
        return torch.where(stance_mask, pos_stance, pos_swing)

    def calc_desired_feet_x(self, stance_phase, swing_phase, swing_ratio, cycle_time):
        """Calculates the desired task-space X trajectory for the feet."""
        t_total_stance = (1.0 - swing_ratio) * cycle_time
        vx_cmd_err = torch.clamp(self.vel_command_b[:, 0] - self.robot_lin_vel[:, 0], -0.6, 0.6)
        vx_cmd = self.robot_lin_vel[:, 0] + vx_cmd_err

        swing_end = torch.clamp(0.5 * vx_cmd * t_total_stance, -0.25, 0.25).unsqueeze(1).expand(-1, 2)
        swing_start = -swing_end

        t_total_swing = swing_ratio * cycle_time
        t_swing = swing_phase * t_total_swing.unsqueeze(1)
        y_mid_x = 0.5 * (swing_start + swing_end)

        pos_swing = self.parabolic_trajectory(t_swing, t_total_swing, swing_start, y_mid_x, swing_end)

        # Linear trajectory for stance
        pos_stance = swing_end * (1.0 - stance_phase) + swing_start * stance_phase
        stance_mask = stance_phase > 0
        return torch.where(stance_mask, pos_stance, pos_swing)

    def _calculate_ik_xz(self, x, z):
        """Analytic inverse kinematics mapping (X, Z) to hip/knee joint positions."""
        z = z.clone().detach()
        z_leg = PITCH_TO_FEET - z
        x_leg = x

        l_leg = torch.clamp(torch.sqrt(x_leg**2 + z_leg**2), 0.2, 0.64)
        theta_1 = torch.atan2(x_leg, z_leg)
        theta_2 = torch.acos((l_leg / 2) / LEG_LENGTH)

        pos_hip = -(theta_1 + theta_2 - torch.pi / 6)
        pos_knee = 2 * (theta_2 - torch.pi / 6)

        return pos_hip, pos_knee

    def _calc_gait_para(self) -> None:
        """Update normalized gait parameters based on simulation time."""
        t = self._env.episode_length_buf * self._env.step_dt / self.gait_cycle
        self.gait_phase[:, 0] = (t + self.phase_offset[:, 0]) % 1.0
        self.gait_phase[:, 1] = (t + self.phase_offset[:, 1]) % 1.0

    def _calc_walking_phase(self):
        """Compute masks and normalized phases for swing and stance."""
        phase = self.gait_phase
        swing_ratio = self.swing_ratio.unsqueeze(1)

        swing_mask = phase < swing_ratio
        stance_mask = ~swing_mask
        self.stance_mask = stance_mask

        self.swing_phase = torch.where(swing_mask, phase / swing_ratio, torch.zeros_like(phase))
        self.stance_phase = torch.where(
            stance_mask, (phase - swing_ratio) / (1.0 - swing_ratio), torch.zeros_like(phase)
        )
        self.contact_number_des = stance_mask.sum(dim=1)

    def calculate_ref_dof(self, feet_desired_x, feet_desired_z):
        """Calculate and apply the reference joint actions."""
        # Clear buffer to prevent indefinite accumulation per step
        self.ref_action.zero_()
        self.target_pitch_joint_l_r, self.target_knee_joint_l_r = self._calculate_ik_xz(feet_desired_x, feet_desired_z)

        # Apply IK targets directly to cached native joint indices
        self.ref_action[:, self.idx_pitch] += self.target_pitch_joint_l_r
        self.ref_action[:, self.idx_knee] += self.target_knee_joint_l_r
        self.ref_action[:, self.idx_ankle_pitch] += (
            -self.ref_action[:, self.idx_knee] - self.ref_action[:, self.idx_pitch]
        )

    def _get_walk_traj(self):
        """Main pipeline to compute desired walking trajectories and map them to joint references."""
        self._calc_gait_para()
        self._calc_walking_phase()
        self.feet_desired_x = self.calc_desired_feet_x(
            self.stance_phase, self.swing_phase, self.swing_ratio, self.gait_cycle
        )
        self.feet_desired_z = self.calc_desired_feet_z(
            self.stance_phase, self.swing_phase, self.swing_ratio, self.gait_cycle
        )
        self.calculate_ref_dof(self.feet_desired_x, self.feet_desired_z)


# =========================================================================
# URDF 29-DOF Joint Sequence Reference
# =========================================================================
#  0-5  : left_hip_roll, left_hip_yaw, left_hip_pitch, left_knee, left_ankle_pitch, left_ankle_roll
#  6-11 : right_hip_roll, right_hip_yaw, right_hip_pitch, right_knee, right_ankle_pitch, right_ankle_roll
# 12-14 : waist_roll, waist_yaw, waist_pitch
# 15-21 : left_shoulder_pitch, left_shoulder_roll, left_arm_yaw, left_elbow_pitch, left_elbow_yaw, left_wrist_pitch, left_wrist_roll
# 22-28 : right_shoulder_pitch, right_shoulder_roll, right_arm_yaw, right_elbow_pitch, right_elbow_yaw, right_wrist_pitch, right_wrist_roll
# =========================================================================
