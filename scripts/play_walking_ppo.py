# Copyright (c) 2025, RL Project Workspace
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""H1 Walking 학습된 정책 테스트 스크립트.

학습된 모델을 로드하여 시각화 및 평가를 수행합니다.

사용법:
    # 최신 체크포인트 테스트 (GUI 모드)
    /home/ldj/IsaacLab/isaaclab.sh -p /home/ldj/RL_project_ws/exts/h1_locomotion/scripts/play_walking_ppo.py \
        --task H1-Walking-v0 \
        --num_envs 16 \
        --checkpoint /home/ldj/RL_project_ws/exts/h1_locomotion/logs/rsl_rl/h1_walking/<timestamp>/model_3000.pt

    # 특정 체크포인트 테스트
    /home/ldj/IsaacLab/isaaclab.sh -p /home/ldj/RL_project_ws/exts/h1_locomotion/scripts/play_walking_ppo.py \
        --task H1-Walking-v0 \
        --num_envs 16 \
        --checkpoint /path/to/model_XXXX.pt
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# argparse 설정
parser = argparse.ArgumentParser(description="Play trained H1-Walking-v0 policy.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="H1-Walking-v0", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt file).")
parser.add_argument("--num_steps", type=int, default=1000, help="Number of steps to run.")

# AppLauncher cli args 추가
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Isaac Sim 앱 시작
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

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
    """학습된 정책으로 H1 Walking 환경 테스트."""
    
    # 체크포인트 경로 확인
    if not os.path.exists(args_cli.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args_cli.checkpoint}")
    
    print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
    
    # 환경 설정 로드
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric if hasattr(args_cli, 'disable_fabric') else True,
    )
    
    # 에이전트 설정 로드
    gym_registry = gym.envs.registry.get(args_cli.task)
    agent_cfg_entry_point = gym_registry.kwargs.get("rsl_rl_cfg_entry_point")
    
    if agent_cfg_entry_point is None:
        raise ValueError(f"No RSL-RL config found for task: {args_cli.task}")
    
    import importlib
    module_path, class_name = agent_cfg_entry_point.rsplit(":", 1)
    module = importlib.import_module(module_path)
    agent_cfg = getattr(module, class_name)()

    # 환경 생성
    print(f"[INFO] Creating environment: {args_cli.task}")
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # RSL-RL 래퍼 적용
    env = RslRlVecEnvWrapper(env)

    # 로그 디렉토리 (체크포인트가 있는 디렉토리)
    log_dir = os.path.dirname(args_cli.checkpoint)

    # Runner 생성 및 체크포인트 로드
    print(f"[INFO] Creating policy runner...")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    
    # 체크포인트 로드
    print(f"[INFO] Loading trained policy from: {args_cli.checkpoint}")
    runner.load(args_cli.checkpoint)
    
    # 정책 가져오기
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # 환경 리셋
    print(f"[INFO] Resetting environment...")
    obs = env.get_observations()
    # obs가 tuple인 경우 첫 번째 요소 사용
    if isinstance(obs, tuple):
        obs = obs[0]

    # 시뮬레이션 실행
    print(f"[INFO] Running simulation for {args_cli.num_steps} steps...")
    print(f"[INFO] Press Ctrl+C to stop.")
    
    step_count = 0
    total_reward = 0.0
    
    while simulation_app.is_running() and step_count < args_cli.num_steps:
        with torch.inference_mode():
            # 정책으로부터 액션 추론
            actions = policy(obs)
            
            # 환경 스텝 (RSL-RL wrapper는 4개 값 반환: obs, rewards, dones, infos)
            obs, rewards, dones, infos = env.step(actions)
            
            total_reward += rewards.mean().item()
            step_count += 1
            
            # 진행 상황 출력
            if step_count % 100 == 0:
                avg_reward = total_reward / step_count
                print(f"[INFO] Step {step_count}/{args_cli.num_steps}, "
                      f"Mean reward: {rewards.mean().item():.4f}, "
                      f"Avg reward: {avg_reward:.4f}")

    print(f"\n[INFO] Simulation completed!")
    print(f"[INFO] Total steps: {step_count}")
    print(f"[INFO] Average reward: {total_reward / step_count:.4f}")
    
    # 환경 종료
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

