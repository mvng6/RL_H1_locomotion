# Copyright (c) 2025, RL Project Workspace
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""H1 Walking 환경 학습 스크립트.

커스텀 환경(H1-Walking-v0)을 사용하여 RSL-RL PPO로 학습합니다.

사용법:
    # GUI 모드
    /home/ldj/IsaacLab/isaaclab.sh -p /home/ldj/RL_project_ws/exts/h1_locomotion/scripts/train_walking_ppo.py \
    --task H1-Walking-v0 --num_envs 4096 --max_iterations 3000

    # Headless 모드 (더 빠름)
    /home/ldj/IsaacLab/isaaclab.sh -p /home/ldj/RL_project_ws/exts/h1_locomotion/scripts/train_walking_ppo.py \
    --task H1-Walking-v0 --num_envs 4096 --max_iterations 3000 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# argparse 설정
parser = argparse.ArgumentParser(description="Train H1-Walking-v0 environment with RSL-RL PPO.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="H1-Walking-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")

# AppLauncher cli args 추가
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 비디오 녹화 시 설정
if args_cli.video:
    args_cli.enable_cameras = True

# Isaac Sim 앱 시작
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

# RSL-RL imports
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

# Isaac Lab 기본 태스크 import (필수)
import isaaclab_tasks  # noqa: F401

# ========================================================================
# 핵심: 커스텀 H1 Locomotion 태스크 import (환경 등록을 위해 필수!)
# ========================================================================
import h1_locomotion.tasks  # noqa: F401

from isaaclab_tasks.utils import parse_env_cfg


def main():
    """RSL-RL로 H1 Walking 환경 학습."""
    
    # 환경 설정 로드
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric if hasattr(args_cli, 'disable_fabric') else True,
    )
    
    # 에이전트 설정 로드 (gymnasium registry에서)
    gym_registry = gym.envs.registry.get(args_cli.task)
    agent_cfg_entry_point = gym_registry.kwargs.get("rsl_rl_cfg_entry_point")
    
    if agent_cfg_entry_point is None:
        raise ValueError(f"No RSL-RL config found for task: {args_cli.task}")
    
    # 모듈에서 설정 클래스 로드
    import importlib
    module_path, class_name = agent_cfg_entry_point.rsplit(":", 1)
    module = importlib.import_module(module_path)
    agent_cfg = getattr(module, class_name)()
    
    # 에이전트 설정 오버라이드
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed

    # 로그 디렉토리 설정
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    
    # 타임스탬프로 run 이름 생성
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        run_name = f"{agent_cfg.run_name}_{run_name}"
    log_dir = os.path.join(log_root_path, run_name)

    # 환경 생성
    print(f"[INFO] Creating environment: {args_cli.task}")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # RSL-RL 래퍼 적용
    # 주의: clip_actions 파라미터는 최신 Gymnasium과 호환성 문제가 있어 제거
    env = RslRlVecEnvWrapper(env)

    # 설정 출력
    print(f"[INFO] Environment: {args_cli.task}")
    print(f"[INFO] Number of environments: {env.num_envs}")
    print(f"[INFO] Max iterations: {agent_cfg.max_iterations}")
    print(f"[INFO] Log directory: {log_dir}")

    # PPO Runner 생성
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    
    # 체크포인트에서 재개 (있는 경우)
    if args_cli.resume is not None:
        print(f"[INFO] Loading model checkpoint from: {args_cli.resume}")
        runner.load(args_cli.resume)

    # 학습 시작
    print("[INFO] Starting training...")
    print(f"[INFO] Total training steps: {agent_cfg.max_iterations} iterations × {agent_cfg.num_steps_per_env} steps/env × {env.num_envs} envs")
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # 환경 종료
    env.close()
    print("[INFO] Training complete!")


if __name__ == "__main__":
    # 학습 실행
    main()
    # Isaac Sim 종료
    simulation_app.close()
