import os
import select
import sys
import termios
import threading
import tty

import numpy as np
import onnxruntime

from ..utils.math import quat_rotate_inverse_np
from .base_rl_interface import BaseRLInterface


class LocomotionRLInterface(BaseRLInterface):
    def __init__(self, config, log_callback=None):
        super().__init__(config)
        self.log_callback = log_callback if log_callback else print

        self.command = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # 初始化内部 command
        self.step_v = 0.1
        self.step_w = 0.1

        self._kb_thread = None
        if os.isatty(sys.stdin.fileno()):
            self._orig_settings = termios.tcgetattr(sys.stdin)
        else:
            self._orig_settings = None

        self.parse_model()

        if self._orig_settings is not None:
            self._kb_thread = threading.Thread(target=self._tty_loop, daemon=True)
            self._kb_thread.start()
            self.log_callback("Keyboard listener started (tty)")
        else:
            self.log_callback("stdin is not a TTY; keyboard disabled. Use set_command() instead.")

    def _tty_loop(self):
        settings = self._orig_settings
        assert settings is not None
        try:
            while True:
                tty.setraw(sys.stdin.fileno())
                rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                if rlist:
                    key = sys.stdin.read(1)
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
                    if key == "\x03":  # CTRL-C
                        break
                    self._handle_key(key)
                else:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

    def _handle_key(self, key):
        if key == "w":
            self.command[0] += self.step_v
        elif key == "s":
            self.command[0] -= self.step_v
        elif key == "a":
            self.command[1] += self.step_v
        elif key == "d":
            self.command[1] -= self.step_v
        elif key == "q":
            self.command[2] -= self.step_w
        elif key == "e":
            self.command[2] += self.step_w
        elif key == " " or key == "x":
            self.command = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        else:
            return

        self.log_callback(
            f"Current Command => [v_x(前): {self.command[0]:.2f}, v_y(左): {self.command[1]:.2f}, yaw_z(旋):"
            f" {self.command[2]:.2f}]"
        )

    def set_command(self, command):
        # 接收外部传入的控制指令 [v_x, v_y, omega_yaw]
        self.command = np.array(command, dtype=np.float32)

    def parse_model(self):
        self.policy = onnxruntime.InferenceSession(self.config["policy_path"])
        meta_data = self.policy.get_modelmeta().custom_metadata_map

        for key in meta_data:
            if key == "joint_names":
                self.joint_seq = meta_data[key].split(",")
            if key == "default_joint_pos":
                self.default_joint_pos = np.array([float(x) for x in meta_data[key].split(",")])
            if key == "joint_stiffness":
                self.Kp = np.array([float(x) for x in meta_data[key].split(",")])
            if key == "joint_damping":
                self.Kd = np.array([float(x) for x in meta_data[key].split(",")])
            if key == "action_scale":
                self.action_scale = np.array([float(x) for x in meta_data[key].split(",")])

        self.action_buffer = np.zeros((len(self.joint_seq),), dtype=np.float32)
        self.time_step = 0

    def perform_inference(self, root_quat, root_ang_vel, mes_q, mes_qdot):
        # 使用类内部更新的 command
        command = self.command.copy()

        gvec2 = quat_rotate_inverse_np(root_quat, np.array([0.0, 0.0, -1.0], dtype=np.float32)).astype(
            np.float32
        )  # (3,)
        mes_qpos_rel = (mes_q - self.default_joint_pos).astype(np.float32)  # (29,)
        prev_action = self.action_buffer.copy().astype(np.float32)

        self.obs_hist.push_item("command", command)
        self.obs_hist.push_item("gvec2", gvec2)
        self.obs_hist.push_item("base_ang_vel", root_ang_vel)
        self.obs_hist.push_item("qpos_rel", mes_qpos_rel)
        self.obs_hist.push_item("qvel", mes_qdot)
        self.obs_hist.push_item("action", prev_action)

        obs = self.obs_hist.build_obs()  # shape = (num_obs, )

        action = self.policy.run(
            ["actions"],
            {
                "obs": obs.reshape(1, -1),
            },
        )[0]

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        self.action_buffer = action.copy()

        joint_pos_des = action * self.action_scale + self.default_joint_pos

        self.time_step += 1

        return joint_pos_des
