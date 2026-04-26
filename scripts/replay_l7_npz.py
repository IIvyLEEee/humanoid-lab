"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    python scripts/replay_l7_npz.py --motion_file motions/0-EraWt_0916_0916-012_walk_run_jump_modified_mirror.npz
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")
# parser.add_argument("--registry_name", type=str, required=True, help="The name of the wand registry.")
parser.add_argument("--registry_name", type=str, required=False, help="The name of the wand registry.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to local motion npz (overrides registry)")
parser.add_argument("--video", action="store_true", default=False, help="Record the replay to an mp4 file.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video in frames.")
parser.add_argument("--video_fps", type=int, default=50, help="Frames per second for the recorded video.")
parser.add_argument(
    "--video_path",
    type=str,
    default="videos/preprocess_replay_run1_subject2.mp4",
    help="Output path for the replay video.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

##
# Pre-defined configs
##
from era_okcc_humanoid_lab.robots.era_l7_29dof import (
    L7_29DOF_CYLINDER_CFG,
)
from era_okcc_humanoid_lab.tasks.mimic.mdp import MotionLoader

import imageio.v2 as imageio
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = L7_29DOF_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, camera: Camera | None = None):
    # Extract scene entities
    robot: Articulation = scene["robot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    # Determine motion file source: prefer local --motion_file, otherwise use WandB registry
    if args_cli.motion_file is not None:
        motion_file = args_cli.motion_file
    else:
        if args_cli.registry_name is None:
            raise ValueError("Please provide either --motion_file or --registry_name")
        registry_name = args_cli.registry_name
        if ":" not in registry_name:  # Check if the registry name includes alias, if not, append ":latest"
            registry_name += ":latest"
        import pathlib

        import wandb

        api = wandb.Api()
        artifact = api.artifact(registry_name)
        motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")

    motion = MotionLoader(
        motion_file,
        torch.tensor([0], dtype=torch.long, device=sim.device),
        sim.device,
    )
    camera_eye = None
    camera_target = None
    if camera is not None:
        camera_frame_count = min(args_cli.video_length + 1, int(motion.time_step_total))
        camera_root_pos = motion.body_pos_w[:camera_frame_count, 0, :]
        camera_min = camera_root_pos.min(dim=0).values
        camera_max = camera_root_pos.max(dim=0).values
        camera_center = (camera_min + camera_max) * 0.5
        camera_span = torch.max(camera_max[:2] - camera_min[:2]).clamp(min=1.0)
        camera_target = torch.tensor(
            [[camera_center[0], camera_center[1], max(float(camera_center[2]), 0.9)]],
            dtype=torch.float32,
            device=sim.device,
        )
        camera_eye = camera_target + torch.tensor(
            [[camera_span * 1.5 + 2.5, camera_span * 1.5 + 2.5, camera_span * 0.7 + 1.8]],
            dtype=torch.float32,
            device=sim.device,
        )
        camera.set_world_poses_from_view(camera_eye, camera_target)
    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)

    writer = None
    recorded_frames = 0
    if args_cli.video:
        os.makedirs(os.path.dirname(args_cli.video_path) or ".", exist_ok=True)
        writer = imageio.get_writer(args_cli.video_path, fps=args_cli.video_fps)
        print(f"[INFO]: Recording replay video to: {args_cli.video_path}")

    try:
        # Simulation loop
        while simulation_app.is_running():
            time_steps += 1
            reset_ids = time_steps >= motion.time_step_total
            time_steps[reset_ids] = 0

            root_states = robot.data.default_root_state.clone()
            root_states[:, :3] = motion.body_pos_w[time_steps][:, 0] + scene.env_origins[:, None, :]
            root_states[:, 3:7] = motion.body_quat_w[time_steps][:, 0]
            root_states[:, 7:10] = motion.body_lin_vel_w[time_steps][:, 0]
            root_states[:, 10:] = motion.body_ang_vel_w[time_steps][:, 0]

            robot.write_root_state_to_sim(root_states)
            robot.write_joint_state_to_sim(motion.joint_pos[time_steps], motion.joint_vel[time_steps])
            scene.write_data_to_sim()

            if camera is not None:
                camera.set_world_poses_from_view(camera_eye, camera_target)

            sim.render()  # We don't want physics (sim.step()).
            scene.update(sim_dt)

            if writer is not None and camera is not None:
                camera.update(sim_dt)
                frame = camera.data.output["rgb"][0].cpu().numpy()
                writer.append_data(frame)
                recorded_frames += 1
                if recorded_frames >= args_cli.video_length:
                    break
    finally:
        if writer is not None:
            writer.close()
            print(f"[INFO]: Wrote {recorded_frames} frames to: {args_cli.video_path}")


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    camera = None
    if args_cli.video:
        camera_cfg = CameraCfg(
            height=720,
            width=1280,
            prim_path="/World/ReplayCamera",
            update_period=0,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1.0e5),
            ),
        )
        camera = Camera(camera_cfg)
        camera._initialize_callback(None)
        # Warm up RTX sensor textures before collecting frames.
        for _ in range(5):
            sim.render()
            camera.update(sim_cfg.dt)
    # Run the simulator
    run_simulator(sim, scene, camera)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
