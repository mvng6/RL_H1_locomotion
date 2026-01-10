# Copyright (c) 2025, RL Project Workspace
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""H1-Walking-v0 환경 테스트 스크립트."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Test H1-Walking-v0 environment.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="H1-Walking-v0", help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

# Isaac Lab 기본 태스크 import
import isaaclab_tasks  # noqa: F401

# H1 Locomotion 확장 패키지 import (환경 등록을 위해 필수!)
import h1_locomotion.tasks  # noqa: F401

from isaaclab_tasks.utils import parse_env_cfg


def main():
    """Zero actions agent with H1-Walking-v0 environment."""
    print(f"[INFO]: Testing environment: {args_cli.task}")
    
    # 환경이 등록되었는지 확인
    # Gymnasium 최신 버전에서는 registry가 직접 dict 형태
    if args_cli.task not in gym.envs.registry:
        print(f"[ERROR]: Environment '{args_cli.task}' is not registered!")
        print(f"[INFO]: Available H1 environments:")
        h1_envs = [env for env in gym.envs.registry.keys() if "H1" in env]
        for env in h1_envs:
            print(f"  - {env}")
        return
    
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    
    # create environment
    print(f"[INFO]: Creating environment with {args_cli.num_envs} parallel environments...")
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    
    # reset environment
    print("[INFO]: Resetting environment...")
    obs, info = env.reset()
    # obs는 dict 형태 (예: {'policy': tensor})
    obs_policy = obs["policy"] if isinstance(obs, dict) else obs
    print(f"[INFO]: Environment reset successful! Observation shape: {obs_policy.shape}")
    
    # simulate environment for a few steps
    print("[INFO]: Running simulation (press Ctrl+C to stop)...")
    step_count = 0
    while simulation_app.is_running() and step_count < 100:
        # run everything in inference mode
        with torch.inference_mode():
            # compute zero actions
            # action space shape: (num_envs, action_dim)
            actions = torch.zeros(args_cli.num_envs, env.action_space.shape[-1], device=env.unwrapped.device)
            # apply actions
            obs, rewards, terminated, truncated, info = env.step(actions)
            step_count += 1
            
            if step_count % 10 == 0:
                print(f"[INFO]: Step {step_count}, Mean reward: {rewards.mean().item():.4f}")

    print(f"[INFO]: Simulation completed after {step_count} steps.")
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

