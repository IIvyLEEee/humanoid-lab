import argparse
import sys
from typing import List

import numpy as np
import rclpy
import yaml
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Bool, Float64MultiArray
from xbot_common_interfaces.msg import Imu

from .utils.utils import convert_joint_order


class EraRLController(Node):
    def __init__(self, config_path: str, mode: str):
        super().__init__("era_rl_controller")

        self.get_logger().info(f"Using config file: {config_path}")
        self.get_logger().info(f"Mode: {mode}")
        self.get_logger().info("Configuration loaded successfully")

        self.measured_q = None
        self.measured_qd = None

        with open(config_path) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        self.mode = mode
        self.robot_joint_sequence = self.config["robot"][self.mode]["joint_sequence"]

        self.init_rl_interface()
        self.Kp = convert_joint_order(self.rl_interface.get_Kp(), self.rl_joint_sequence, self.robot_joint_sequence)
        self.Kd = convert_joint_order(self.rl_interface.get_Kd(), self.rl_joint_sequence, self.robot_joint_sequence)

        if self.mode == "real":
            self.enable_rl_infer = False
            self.move_to_init_pose_duration = 3.0
            self.move_count = 0
            self.init_Kp = self.config["robot"][self.mode]["init_Kp"]
            self.init_Kd = self.config["robot"][self.mode]["init_Kd"]
            neck_yaw_index = self.robot_joint_sequence.index("neck_yaw_joint")
            neck_pitch_index = self.robot_joint_sequence.index("neck_pitch_joint")
            self.neck_indices = [neck_yaw_index, neck_pitch_index]
            self.Kp[self.neck_indices] = 90.0
            self.Kd[self.neck_indices] = 2.0

        else:
            self.enable_rl_infer = True

        self.action_scale = self.rl_interface.get_action_scale()
        self.N_JOINTS = len(self.robot_joint_sequence)

        self.get_logger().debug(f"Robot joint sequence: {self.robot_joint_sequence}")
        self.get_logger().debug(f"RL joint sequence: {self.rl_joint_sequence}")
        self.get_logger().debug(f"Kp: {self.Kp}")
        self.get_logger().debug(f"Kd: {self.Kd}")
        self.get_logger().debug(f"Action scale: {self.action_scale}")

        self.imu_msg: Imu | None = None
        self.imu_sub = self.create_subscription(Imu, "/imu_feedback", self.imu_callback, 10)

        self.motor_msg: Float64MultiArray | None = None
        self._motor_sub = self.create_subscription(Float64MultiArray, "/motor_feedback", self.motor_callback, 10)

        qos_profile = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._motor_command_pub = self.create_publisher(
            Float64MultiArray, "/rl_controller/policy_inference", qos_profile
        )

        self.is_release_sim_hold_pub = self.create_publisher(Bool, "/is_sim_hold_release", 1)

        self.control_timer = self.create_timer(self.config["control_dt"], self.timer_callback)

    def init_rl_interface(self):
        self.get_logger().info("Initializing RL Interface")
        if self.config["rl_model"]["type"] == "mimic":
            from .rl_interfaces.mimic_rl_interface import MimicRLInterface

            self.rl_interface = MimicRLInterface(
                self.config["rl_model"], log_callback=lambda msg: self.get_logger().info(msg)
            )
        elif self.config["rl_model"]["type"] == "locomotion":
            from .rl_interfaces.locomotion_rl_interface import LocomotionRLInterface

            self.rl_interface = LocomotionRLInterface(
                self.config["rl_model"], log_callback=lambda msg: self.get_logger().info(msg)
            )
        else:
            self.get_logger().error(f"Unsupported RL model type: {self.config['rl_model']['type']}")
            raise ValueError(f"Unsupported RL model type: {self.config['rl_model']['type']}")

        self.get_logger().info("RL Interface initialized successfully")
        self.rl_joint_sequence = self.rl_interface.get_joint_sequence()

    def timer_callback(self):
        if self.imu_msg is None or self.motor_msg is None:
            return

        self.measured_q = self.motor_msg.data[: self.N_JOINTS]
        self.measured_qd = self.motor_msg.data[self.N_JOINTS : 2 * self.N_JOINTS]
        mes_q = convert_joint_order(self.measured_q, self.robot_joint_sequence, self.rl_joint_sequence)
        mes_qd = convert_joint_order(self.measured_qd, self.robot_joint_sequence, self.rl_joint_sequence)

        self.root_quat = np.array(
            [
                self.imu_msg.orientation_w,
                self.imu_msg.orientation_x,
                self.imu_msg.orientation_y,
                self.imu_msg.orientation_z,
            ],
            dtype=np.float32,
        )
        self.root_ang_vel = np.array(
            [self.imu_msg.angular_vel_x, self.imu_msg.angular_vel_y, self.imu_msg.angular_vel_z], dtype=np.float32
        )

        if not hasattr(self, "start_mesured_q"):
            self.start_mesured_q = np.copy(self.measured_q)

        if not self.enable_rl_infer:
            reordered_joint_target_pos = self.move_to_init_pose()
        else:
            joint_target_pos = self.rl_interface.perform_inference(self.root_quat, self.root_ang_vel, mes_q, mes_qd)
            reordered_joint_target_pos = convert_joint_order(
                joint_target_pos, self.rl_joint_sequence, self.robot_joint_sequence
            )

        self.send_motor_command(
            reordered_joint_target_pos,
            [0.0] * len(reordered_joint_target_pos),
            self.Kp,
            self.Kd,
            [0.0] * len(reordered_joint_target_pos),
        )

    def move_to_init_pose(self):
        target_pos = self.rl_interface.get_default_joint_pos()
        reordered_target_pos = convert_joint_order(target_pos, self.rl_joint_sequence, self.robot_joint_sequence)
        if self.move_count < self.move_to_init_pose_duration / self.config["control_dt"]:
            alpha = (self.move_count * self.config["control_dt"]) / self.move_to_init_pose_duration
            current_target = (1 - alpha) * self.start_mesured_q + alpha * reordered_target_pos
            self.move_count += 1
            return current_target
        else:
            self.enable_rl_infer = True
            msg_bool = Bool()
            msg_bool.data = True
            self.is_release_sim_hold_pub.publish(msg_bool)
            return reordered_target_pos

    def imu_callback(self, msg: Imu):
        self.imu_msg = msg

    def motor_callback(self, msg: Float64MultiArray):
        self.motor_msg = msg

    def send_motor_command(
        self, q_des: List[float], qd_des: List[float], kp: List[float], kd: List[float], torque: List[float]
    ):
        if (
            len(q_des) != self.N_JOINTS
            or len(qd_des) != self.N_JOINTS
            or len(kp) != self.N_JOINTS
            or len(kd) != self.N_JOINTS
            or len(torque) != self.N_JOINTS
        ):
            self.get_logger().error("Invalid command dimensions")
            return
        msg = Float64MultiArray()
        msg.data.extend(q_des)
        msg.data.extend(qd_des)
        msg.data.extend(kp)
        msg.data.extend(kd)
        msg.data.extend(torque)
        self._motor_command_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser(description="BYD Mimic Controller")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration YAML file")
    parser.add_argument(
        "--mode", type=str, choices=["sim2sim", "real"], default="sim2sim", help="Mode: sim2sim or real"
    )
    ros_args = rclpy.utilities.remove_ros_args(sys.argv)
    args = parser.parse_args(ros_args[1:])
    node = EraRLController(args.config, args.mode)

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        sys.exit(1)
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
