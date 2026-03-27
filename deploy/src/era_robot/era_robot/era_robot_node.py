import argparse
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Float64MultiArray
from xbot_common_interfaces.msg import Imu


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


class EraRobot(Node):
    def __init__(self, xml_path: str):
        super().__init__("era_robot")
        self.get_logger().info(f"Using XML file: {xml_path}")

        self.max_duration = 200.0  # seconds #
        self.get_logger().info("Running in simulation mode")
        self.imu_publisher = self.create_publisher(Imu, "/imu_feedback", 10)

        self.robot_state_publisher = self.create_publisher(Float64MultiArray, "/motor_feedback", 10)

        qos_profile = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.motor_command_subscriber = self.create_subscription(
            Float64MultiArray, "/rl_controller/policy_inference", self.motor_command_callback, qos_profile
        )

        # Load MuJoCo model
        self.get_logger().info(f"Loading MuJoCo XML from {xml_path}")
        self.m = mujoco.MjModel.from_xml_path(xml_path)
        self.get_logger().info(f"Model loaded: nq={self.m.nq}, nv={self.m.nv}, na={self.m.na}")
        self.d = mujoco.MjData(self.m)
        self.get_logger().info("MjData created")
        self.m.opt.timestep = 0.002  # Simulation timestep
        self.d.qpos[7:] = np.zeros(self.m.nq - 7)
        self.d.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.d.qpos[:3] = np.array([0.0, 0.0, 0.977054])

        self.N_JOINTS = self.m.nq - 7  # Assuming first 7 are base
        self.q_des = np.zeros(self.N_JOINTS)
        self.qd_des = np.zeros(self.N_JOINTS)
        self.Kp = np.zeros(self.N_JOINTS)
        self.Kd = np.zeros(self.N_JOINTS)
        self.torque = np.zeros(self.N_JOINTS)
        self.motor_cmd_msg = Float64MultiArray(data=[0.0] * (5 * self.N_JOINTS))

        self.viewer = mujoco.viewer.launch_passive(self.m, self.d, show_left_ui=False, show_right_ui=False)
        self.viewer_dt = 0.02
        self.sim_count = 0

        # Simulation timer
        self.start_time = time.time()
        self.sim_timer = self.create_timer(self.m.opt.timestep, self.simulation_callback)

    def motor_command_callback(self, msg: Float64MultiArray):
        # self.get_logger().info(f"Received motor command: {msg.data}")
        self.motor_cmd_msg = msg

    def simulation_callback(self):
        test_a = time.time()
        # self.get_logger().info(f"Simulation step callback at time: {test_a - self.start_time:.4f} seconds")
        if self.viewer.is_running() and time.time() - self.start_time < self.max_duration:
            # Get robot state
            joint_positions = self.d.qpos[7:].copy()  # Assuming first 7 are base
            joint_velocities = self.d.qvel[6:].copy()  # Assuming first 6 are base vel

            # Publish robot state
            robot_state_msg = Float64MultiArray()
            robot_state_msg.data = np.concatenate([joint_positions, joint_velocities]).tolist()
            self.robot_state_publisher.publish(robot_state_msg)

            # Get IMU data (base orientation and angular velocity)
            imu_msg = Imu()
            imu_msg.orientation_w = self.d.qpos[3]
            imu_msg.orientation_x = self.d.qpos[4]
            imu_msg.orientation_y = self.d.qpos[5]
            imu_msg.orientation_z = self.d.qpos[6]
            imu_msg.angular_vel_x = self.d.qvel[3]
            imu_msg.angular_vel_y = self.d.qvel[4]
            imu_msg.angular_vel_z = self.d.qvel[5]
            self.imu_publisher.publish(imu_msg)

            num_joints = self.N_JOINTS
            self.q_des = np.array(self.motor_cmd_msg.data[:num_joints])  # Hold current position to stand steady
            self.qd_des = np.array(self.motor_cmd_msg.data[num_joints : 2 * num_joints])  # Set desired velocity to zero
            self.Kp = np.array(self.motor_cmd_msg.data[2 * num_joints : 3 * num_joints])
            self.Kd = np.array(self.motor_cmd_msg.data[3 * num_joints : 4 * num_joints])
            self.torque = np.zeros(num_joints)  # Ignore RL torque for steady standing

            self.d.ctrl[:] = (
                pd_control(self.q_des, self.d.qpos[7:], self.Kp, self.qd_des, self.d.qvel[6:], self.Kd) + self.torque
            )
            mujoco.mj_step(self.m, self.d)
            if self.sim_count % (self.viewer_dt / self.m.opt.timestep) == 0:
                self.viewer.sync()
            self.sim_count += 1


def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser(description="Era Robot")
    parser.add_argument("--xml_path", type=str, required=True, help="Path to XML file")

    ros_args = rclpy.utilities.remove_ros_args(sys.argv)
    args = parser.parse_args(ros_args[1:])
    node = EraRobot(args.xml_path)

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
