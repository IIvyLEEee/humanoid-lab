# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.commands import UniformVelocityCommandCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp

##
# Pre-defined configs
##

VELOCITY_RANGE = {
    "x": (-0.8, 0.8),
    "y": (-0.8, 0.8),
    "z": (-0.3, 0.3),
    "roll": (-0.82, 0.82),
    "pitch": (-0.82, 0.82),
    "yaw": (-1.0, 1.0),
}


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )

    # robots
    robot: ArticulationCfg = MISSING
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=10.0,
        debug_vis=True,
    )

    # sensors
    height_scanner = None


##
# MDP settings
##


@configclass
class CustomUniformVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for CustomUniformVelocityCommand with gait and action parameters."""

    class_type: type = mdp.CustomUniformVelocityCommand

    @configclass
    class GaitCfg:
        """Gait parameters for locomotion."""

        gait_cycle: float = 0.8  # gait cycle time in seconds
        gait_phase_offset_l: float = 0.0  # left leg phase offset
        gait_phase_offset_r: float = 0.5  # right leg phase offset
        swing_ratio: float = 0.4  # ratio of swing phase in the gait cycle
        stance_ratio: float = 0.6  # ratio of stance phase in the

    @configclass
    class ActionCfg:
        """Action parameters."""

        dim: int = 29  # dimension of action space

    gait: GaitCfg = GaitCfg()
    action: ActionCfg = ActionCfg()
    foot_height: float = 0.1  # desired foot height during swing phase in meters
    anchor_body_name: str = "pelvis"
    # body name used for anchor position and orientation


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    loco_command = CustomUniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0, 3.0),
        debug_vis=True,
        heading_command=False,
        rel_standing_envs=0.0,
        ranges=CustomUniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
            heading=None,
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "loco_command"},
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,  # 自定义函数：获取机器人基座线速度
            noise=Unoise(n_min=-0.05, n_max=0.05),  # 加性噪声：±0.5m/s（模拟IMU线速度噪声）
            clip=(-100, 100),
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.3, n_max=0.3),
            clip=(-100, 100),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100, 100),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.0, n_max=1.0),
            clip=(-100, 100),
        )
        actions = ObsTerm(func=mdp.last_action, clip=(-100, 100))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5

    @configclass
    class PrivilegedCfg(ObsGroup):
        command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "loco_command"},
        )
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-100, 100))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, clip=(-100, 100))
        actions = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))

        def __post_init__(self):
            self.history_length = 5

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 2.0),  # (0.3, 1.6)
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.05, 0.05),  # (-0.01, 0.01)
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {
                "x": (-0.04, 0.04),
                "y": (-0.05, 0.05),
                "z": (-0.05, 0.05),
            },
        },
    )

    # interval
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity2,
    #      mode="interval",
    #      interval_range_s=(1.0, 3.0),
    #      params={"velocity_range": VELOCITY_RANGE,
    #             "command_name": "loco_motion"},
    #  )

    scale_body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-3.0, 4.0),
            "operation": "add",
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    random_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.5, 2.0),  # (0.5, 1.25)
            "operation": "scale",
        },
    )

    random_joint_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "armature_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )

    random_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


leg_joint_names = [
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",  # 0-5
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",  # 6-11
]

req_jnt_names = [
    # 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_hip_pitch_joint',
    # 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    # 0-5
    # 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_hip_pitch_joint',
    # 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    # 6-11
    "waist_roll_joint",
    "waist_yaw_joint",
    "waist_pitch_joint",  # 12,13,14
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_arm_yaw_joint",
    "left_elbow_pitch_joint",
    "left_elbow_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",  # 15-21
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_arm_yaw_joint",
    "right_elbow_pitch_joint",
    "right_elbow_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",  # 22-28
]


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"std": 0.5, "command_name": "loco_command"},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"std": 0.5, "command_name": "loco_command"},
    )

    contact_match = RewTerm(
        func=mdp.feet_contact_number,
        weight=1.0,
        params={
            "command_name": "loco_command",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[".*ankle_roll_link"],
            ),
        },
    )
    contact_num_match = RewTerm(
        func=mdp.feet_contact_number_sum,
        weight=1.0,
        params={
            "command_name": "loco_command",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[".*ankle_roll_link"],
            ),
        },
    )

    track_joint_pos = RewTerm(
        func=mdp.traking_joint_pos,
        weight=1.0,
        params={
            "command_name": "loco_command",
            "joint_names": leg_joint_names,
            "asset_cfg": SceneEntityCfg("robot"),
            "sigma": 0.1,
        },
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-4e-1)

    arm_waist_deviation_l1 = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=req_jnt_names)},
    )

    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    body_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="pelvis")},
        weight=-2.0,
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)

    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    ".*_ankle_roll_link",
                    ".*_wrist_roll_link",
                    ".*_elbow_yaw_link",
                    ".*_hip_yaw_link",
                ],
            ),
            "threshold": 1.0,
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    bad_contact = DoneTerm(
        func=mdp.bad_contacts_task,
        params={
            "sensor_cfg": SceneEntityCfg(
                name="contact_forces",
                body_names=[
                    "torso_link",
                ],
            ),
            "threshold": 30.0,
        },
    )
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


##
# Environment configuration
##


@configclass
class LocomotionEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    foot_link_name = ".*_ankle_roll_link"

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0

        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # viewer settings
        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        self.disable_zero_weight_rewards()

    def disable_zero_weight_rewards(self):
        """If the weight of rewards is 0, set rewards to None"""
        for attr in dir(self.rewards):
            if not attr.startswith("__"):
                reward_attr = getattr(self.rewards, attr)
                if not callable(reward_attr) and reward_attr.weight == 0:
                    setattr(self.rewards, attr, None)
