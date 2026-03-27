import numpy as np
import onnxruntime as rt

from ..utils.math import (
    matrix_from_quat,
    quat_inv,
    quat_mul,
    yaw_quaternion,
)
from .base_rl_interface import BaseRLInterface


class MimicRLInterface(BaseRLInterface):
    def __init__(self, config, log_callback=None):
        super().__init__(config)
        self.log_callback = log_callback if log_callback else print

        motion_data = np.load(self.config["motion_path"])
        self.log_callback(f"==Loaded motion data from {list(motion_data.keys())}==")
        self.motion_joint_pos = motion_data["joint_pos"]  # [T, 29]
        self.motion_joint_vel = motion_data["joint_vel"]  # [T, 29]
        self.motion_body_quat_w = motion_data["body_quat_w"]  # [T, num_bodies, 4]
        self.log_callback(f"Motion joint pos shape: {self.motion_joint_pos.shape}")
        self.log_callback(f"Motion joint vel shape: {self.motion_joint_vel.shape}")
        self.log_callback(f"Motion body quat shape: {self.motion_body_quat_w.shape}")
        self.motion_length = self.motion_joint_pos.shape[0]

        self.world_to_init_motion_quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.init_flag = False
        self.run_flag = False

        self.parse_model()

    def parse_model(self):
        self.policy = rt.InferenceSession(self.config["policy_path"])
        meta_data = self.policy.get_modelmeta().custom_metadata_map

        for key in meta_data:
            if key == "joint_names":
                self.joint_seq = meta_data[key].split(",")
            if key == "default_joint_pos":
                self.default_joint_pos = np.array([float(x) for x in meta_data[key].split(",")], dtype=np.float32)
            if key == "joint_stiffness":
                self.Kp = np.array([float(x) for x in meta_data[key].split(",")], dtype=np.float32)
            if key == "joint_damping":
                self.Kd = np.array([float(x) for x in meta_data[key].split(",")], dtype=np.float32)
            if key == "action_scale":
                self.action_scale = np.array([float(x) for x in meta_data[key].split(",")], dtype=np.float32)
            if key == "anchor_body_name":
                self.anchor_body_name = meta_data[key]
            if key == "anchor_body_id":
                self.anchor_body_id = int(meta_data[key])
            self.log_callback(f"{key}: {meta_data[key]}")

        self.action_buffer = np.zeros((len(self.joint_seq),), dtype=np.float32)
        self.time_step = 0

    def perform_inference(self, root_quat, root_ang_vel, mes_q, mes_qdot):
        if not self.init_flag:
            init_world_yaw_quat = yaw_quaternion(root_quat)
            init_anchor_body_yaw_quat = yaw_quaternion(self.motion_body_quat_w[self.time_step, self.anchor_body_id])
            self.world_to_init_motion_quat = quat_mul(init_world_yaw_quat, quat_inv(init_anchor_body_yaw_quat))
            self.init_flag = True

        self.time_step = np.clip(self.time_step, 0, self.motion_length - 1)

        curr_motion_data = np.concatenate(
            (self.motion_joint_pos[self.time_step, :], self.motion_joint_vel[self.time_step, :]), axis=0
        ).astype(np.float32)

        motion_anchor_ori_quat = quat_mul(
            quat_mul(quat_inv(root_quat), self.world_to_init_motion_quat),
            self.motion_body_quat_w[self.time_step, self.anchor_body_id],
        )
        motion_anchor_ori_b = matrix_from_quat(motion_anchor_ori_quat).astype(np.float32)[..., :2]

        mes_qpos_rel = (mes_q - self.default_joint_pos).astype(np.float32)  # (29,)
        prev_action = self.action_buffer.copy().astype(np.float32)

        self.obs_hist.push_item("curr_motion_data", curr_motion_data)
        self.obs_hist.push_item("motion_anchor_ori_b", motion_anchor_ori_b)
        self.obs_hist.push_item("base_ang_vel", root_ang_vel)
        self.obs_hist.push_item("qpos_rel", mes_qpos_rel)
        self.obs_hist.push_item("qvel", mes_qdot)
        self.obs_hist.push_item("action", prev_action)

        obs = self.obs_hist.build_obs()  # shape = (num_obs, )

        action = self.policy.run(
            ["actions"],
            {"obs": obs.reshape(1, -1), "time_step": np.array([self.time_step], dtype=np.float32).reshape(1, 1)},
        )[0]

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        self.action_buffer = action.copy()

        joint_pos_des = action * self.action_scale + self.default_joint_pos

        self.time_step += 1

        return joint_pos_des
