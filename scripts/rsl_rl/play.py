# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_folder", type=str, default=None, help="Folder where recorded videos are stored.")
parser.add_argument(
    "--eval_length",
    type=int,
    default=None,
    help="Number of play steps before exiting when not recording video.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to local motion npz (overrides wandb_path)")
parser.add_argument("--model_path", type=str, default=None, help="Direct path to model checkpoint file (.pt)")
parser.add_argument("--log_torques", action="store_true", default=False, help="Log and plot joint torques during play.")
parser.add_argument(
    "--torque_log_dir",
    type=str,
    default=None,
    help="Directory where torque logs and plots are saved. Defaults to the checkpoint directory.",
)
parser.add_argument(
    "--torque_log_name",
    type=str,
    default="joint_torque_eval",
    help="Base filename for torque logs and plots.",
)
parser.add_argument(
    "--policy_only",
    "--policy-only",
    action="store_true",
    default=False,
    help="Hide tracking reference visualizers during policy evaluation videos.",
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import era_okcc_humanoid_lab.tasks  # noqa: F401
import gymnasium as gym
import numpy as np
import torch
from era_okcc_humanoid_lab.utils.exporter import (
    attach_loco_onnx_metadata,
    attach_onnx_metadata,
    export_locomotion_policy_as_onnx,
    export_motion_policy_as_onnx,
)  # noqa: F401
from era_okcc_humanoid_lab.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner  # noqa: F401
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


def compute_fixed_tracking_camera(motion_file: str | None, video_length: int):
    """Compute a static camera pose that covers the replay root trajectory segment."""
    if motion_file is None:
        print("[WARN]: Fixed tracking camera requested but no motion file is set.")
        return None

    data = np.load(motion_file)
    root_pos = data["body_pos_w"][: video_length + 1, 0, :]
    if root_pos.shape[0] == 0:
        print(f"[WARN]: Fixed tracking camera skipped because motion has no frames: {motion_file}")
        return None

    camera_min = root_pos.min(axis=0)
    camera_max = root_pos.max(axis=0)
    camera_center = (camera_min + camera_max) * 0.5
    camera_span = max(float(np.max(camera_max[:2] - camera_min[:2])), 1.0)
    camera_target = np.array([camera_center[0], camera_center[1], max(float(camera_center[2]), 1.0)])
    camera_eye = camera_target + np.array(
        [camera_span * 1.7 + 4.0, camera_span * 1.7 + 4.0, camera_span * 0.2 + 1.2]
    )

    return camera_eye.tolist(), camera_target.tolist()


def apply_fixed_tracking_camera(env, camera_pose: tuple[list[float], list[float]] | None):
    """Apply a static camera pose and disable the viewer's asset-root tracking."""
    if camera_pose is None:
        return

    camera_eye, camera_target = camera_pose
    viewport_camera_controller = getattr(env.unwrapped, "viewport_camera_controller", None)
    if viewport_camera_controller is not None:
        viewport_camera_controller.update_view_to_world()
        viewport_camera_controller.update_view_location(eye=camera_eye, lookat=camera_target)
    env.unwrapped.sim.set_camera_view(eye=camera_eye, target=camera_target)


def save_torque_logs(torque_samples, joint_names: list[str], dt: float, output_dir: str, output_name: str):
    """Save joint torque traces and summary plots for a single evaluation rollout."""
    if len(torque_samples) == 0:
        print("[WARN]: Torque logging requested, but no torque samples were collected.")
        return

    import csv

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    torques = np.stack(torque_samples, axis=0)
    times = np.arange(torques.shape[0]) * dt
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    npz_path = os.path.join(output_dir, f"{output_name}.npz")
    csv_path = os.path.join(output_dir, f"{output_name}.csv")
    plot_path = os.path.join(output_dir, f"{output_name}.png")

    np.savez(npz_path, time=times, joint_torque=torques, joint_names=np.array(joint_names, dtype=object))

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", *joint_names])
        for time_value, torque_row in zip(times, torques):
            writer.writerow([time_value, *torque_row.tolist()])

    abs_torques = np.abs(torques)
    mean_abs = abs_torques.mean(axis=0)
    rms_total = np.sqrt(np.mean(np.square(torques), axis=1))
    mean_abs_total = abs_torques.mean(axis=1)
    top_count = min(10, len(joint_names))
    top_indices = np.argsort(mean_abs)[-top_count:][::-1]

    plt.rcParams.update({"font.size": 14, "axes.titlesize": 18, "axes.labelsize": 15})
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)

    axes[0].plot(times, rms_total, label="RMS torque", linewidth=2.0)
    axes[0].plot(times, mean_abs_total, label="Mean |torque|", linewidth=2.0)
    axes[0].set_title("Joint Torque Over Evaluation")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Torque [Nm]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    y_pos = np.arange(top_count)
    axes[1].barh(y_pos, mean_abs[top_indices])
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels([joint_names[i] for i in top_indices])
    axes[1].invert_yaxis()
    axes[1].set_title(f"Top {top_count} Joints by Mean Absolute Torque")
    axes[1].set_xlabel("Mean |torque| [Nm]")
    axes[1].grid(True, axis="x", alpha=0.3)

    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    print(f"[INFO]: Torque log saved to: {npz_path}")
    print(f"[INFO]: Torque CSV saved to: {csv_path}")
    print(f"[INFO]: Torque plot saved to: {plot_path}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")
    is_tracking_task = train_task_name.startswith("Tracking")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    # resolve checkpoint path
    if args_cli.wandb_path:
        import pathlib

        import wandb

        run_path = args_cli.wandb_path

        api = wandb.Api()
        if "model" in args_cli.wandb_path:
            run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
        wandb_run = api.run(run_path)
        # loop over files in the run
        files = [file.name for file in wandb_run.files() if "model" in file.name]
        # files are all model_xxx.pt find the largest filename
        if "model" in args_cli.wandb_path:
            file = args_cli.wandb_path.split("/")[-1]
        else:
            file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

        wandb_file = wandb_run.file(str(file))
        wandb_file.download("./logs/rsl_rl/temp", replace=True)

        print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
        resume_path = f"./logs/rsl_rl/temp/{file}"

        if is_tracking_task:
            if args_cli.motion_file is not None:
                print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
                env_cfg.commands.motion.motion_file = args_cli.motion_file

            art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
            if art is None:
                print("[WARN] No model artifact found in the run.")
            else:
                env_cfg.commands.motion.motion_file = str(pathlib.Path(art.download()) / "motion.npz")

    else:
        # Allow local-only run: require --motion_file and a local checkpoint path via --load_run/--checkpoint
        if is_tracking_task and args_cli.motion_file is not None:
            env_cfg.commands.motion.motion_file = args_cli.motion_file

        # Use direct model path if provided, otherwise use the old logic
        if args_cli.model_path is not None:
            resume_path = args_cli.model_path
            print(f"[INFO]: Loading model checkpoint from direct path: {resume_path}")
        else:
            print(f"[INFO] Loading experiment from directory: {log_root_path}")
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
            print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    if args_cli.policy_only:
        if is_tracking_task:
            env_cfg.commands.motion.debug_vis = False
            print("[INFO]: Policy-only view enabled: hiding tracking reference visualizers.")
        else:
            print("[WARN]: --policy_only was requested for a non-tracking task; no reference visualizers were changed.")

    fixed_camera_pose = None
    if args_cli.video and is_tracking_task and not args_cli.policy_only:
        fixed_camera_pose = compute_fixed_tracking_camera(env_cfg.commands.motion.motion_file, args_cli.video_length)
        if fixed_camera_pose is not None:
            env_cfg.viewer.origin_type = "world"
            env_cfg.viewer.eye = tuple(fixed_camera_pose[0])
            env_cfg.viewer.lookat = tuple(fixed_camera_pose[1])
            print(f"[INFO]: Fixed tracking camera eye={fixed_camera_pose[0]} target={fixed_camera_pose[1]}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    apply_fixed_tracking_camera(env, fixed_camera_pose)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_folder = args_cli.video_folder if args_cli.video_folder is not None else os.path.join(log_dir, "videos", "play")
        video_kwargs = {
            "video_folder": video_folder,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # Recent RSL-RL versions expose the actor through get_policy(), while older
    # versions used policy/actor_critic containers.
    if hasattr(runner.alg, "get_policy"):
        policy_nn = runner.alg.get_policy()
    elif hasattr(runner.alg, "policy"):
        policy_nn = runner.alg.policy
    elif hasattr(runner.alg, "actor_critic"):
        policy_nn = runner.alg.actor_critic
    else:
        raise AttributeError("Unable to locate a policy module on the RSL-RL algorithm.")

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    if is_tracking_task:
        export_motion_policy_as_onnx(
            env.unwrapped, policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx"
        )
        attach_onnx_metadata(env.unwrapped, args_cli.wandb_path if args_cli.wandb_path else "none", export_model_dir)
    else:
        export_locomotion_policy_as_onnx(
            env.unwrapped,
            policy_nn,
            normalizer=normalizer,
            path=export_model_dir,
            filename="policy.onnx",
        )
        attach_loco_onnx_metadata(
            env.unwrapped, args_cli.wandb_path if args_cli.wandb_path else "none", export_model_dir
        )

    dt = env.unwrapped.step_dt
    torque_samples = []
    torque_joint_names = None
    if args_cli.log_torques:
        robot = env.unwrapped.scene["robot"]
        torque_joint_names = list(robot.data.joint_names)
        print(f"[INFO]: Torque logging enabled for {len(torque_joint_names)} joints.")

    # reset environment
    obs = env.get_observations()
    timestep = 0
    max_eval_steps = args_cli.video_length if args_cli.video else args_cli.eval_length
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            apply_fixed_tracking_camera(env, fixed_camera_pose)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            apply_fixed_tracking_camera(env, fixed_camera_pose)
            if args_cli.log_torques:
                torque_samples.append(env.unwrapped.scene["robot"].data.applied_torque[0].detach().cpu().numpy().copy())
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
        timestep += 1
        # Exit the play loop after recording one video or after a requested eval length.
        if max_eval_steps is not None and timestep == max_eval_steps:
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if args_cli.log_torques:
        torque_log_dir = args_cli.torque_log_dir if args_cli.torque_log_dir is not None else os.path.dirname(resume_path)
        save_torque_logs(torque_samples, torque_joint_names, dt, torque_log_dir, args_cli.torque_log_name)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
