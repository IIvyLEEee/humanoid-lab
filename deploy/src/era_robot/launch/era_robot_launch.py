from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    xml_path = LaunchConfiguration("xml_path", default="l7_29dof_neck_fixed/l7_29dof_neck_fixed.xml")

    nodes = [
        Node(
            package="era_robot",
            namespace="era_robot",
            executable="era_robot_node",
            name="era_robot",
            arguments=["--xml_path", xml_path],
        ),
    ]

    return LaunchDescription(nodes)
