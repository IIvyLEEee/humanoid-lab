# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from era_okcc_humanoid_lab.robots.era_l7_29dof import (
    L7_29DOF_CYLINDER_CFG,
    L7_29DOF_NECK_FIXED_ACTION_SCALE,
)

from ...locomotion_env_cfg import LocomotionEnvCfg


@configclass
class L7_29DofLocomotionEnvCfg(LocomotionEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.use_identify_params = True  # flag to use identify params or not
        self.use_high_waist_stiffness = True  # flag to use high waist stiffness or not

        self.scene.robot = L7_29DOF_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = L7_29DOF_NECK_FIXED_ACTION_SCALE
