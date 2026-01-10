# Copyright (c) 2025, RL Project Workspace
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""H1 Walking AMP 환경 학습 스크립트.

AMP 알고리즘을 사용하여 H1-Walking-AMP-v0 환경을 학습합니다.

사용법:
    # GUI 모드
    /home/ldj/IsaacLab/isaaclab.sh -p /home/ldj/RL_project_ws/exts/h1_locomotion/scripts/train_walking_amp.py \
    --task H1-Walking-AMP-v0 --num_envs 4096 --max_iterations 5000

    # Headless 모드 (더 빠름)
    /home/ldj/IsaacLab/isaaclab.sh -p /home/ldj/RL_project_ws/exts/h1_locomotion/scripts/train_walking_amp.py \
    --task H1-Walking-AMP-v0 --num_envs 4096 --max_iterations 5000 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import yaml
from pathlib import Path

from isaaclab.app import AppLauncher

# argparse 설정
parser = argparse.ArgumentParser(description="Train H1-Walking-AMP-v0 environment with AMP algorithm.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="H1-Walking-AMP-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")
parser.add_argument("--expert_motion_file", type=str, default="data/processed/amp_motions.npy",
                   help="Path to expert motion file (.npy)")
parser.add_argument("--discriminator_cfg", type=str, default="config/amp/discriminator_cfg.yaml",
                   help="Path to discriminator config file")
parser.add_argument("--curriculum_cfg", type=str, default=None,
                   help="Path to curriculum config file (optional)")

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
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.utils.data import DataLoader

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

# AMP 관련 import
from h1_locomotion.tasks.walking.amp.discriminator import Discriminator, DiscriminatorCfg
from h1_locomotion.tasks.walking.amp.motion_dataset import MotionDataset


class DiscriminatorTrainer:
    """Discriminator 학습을 관리하는 클래스."""
    
    def __init__(self, discriminator: Discriminator, expert_dataset: MotionDataset, 
                 cfg: DiscriminatorCfg, device: str = "cuda:0"):
        """초기화.
        
        Args:
            discriminator: Discriminator 네트워크
            expert_dataset: Expert 모션 데이터셋
            cfg: Discriminator 설정
            device: 디바이스
        """
        self.discriminator = discriminator.to(device)
        self.expert_dataset = expert_dataset
        self.cfg = cfg
        self.device = device
        
        # 옵티마이저 설정
        self.optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay
        )
        
        # Expert 데이터 로더
        self.expert_loader = DataLoader(
            expert_dataset,
            batch_size=256,  # 배치 크기
            shuffle=True,
            num_workers=0,  # Isaac Lab과 호환성을 위해 0
            pin_memory=True
        )
        
        # Expert 데이터 iterator (효율적인 배치 샘플링을 위해 유지)
        self.expert_iterator = None
        
        # 손실 함수 (Binary Cross Entropy)
        self.criterion = nn.BCELoss()
        
    def train_step(self, policy_state_transitions: torch.Tensor) -> dict:
        """Discriminator 학습 스텝.
        
        Args:
            policy_state_transitions: (B, 2*state_dim) Policy의 상태 전이
            
        Returns:
            Dictionary containing loss and other metrics
        """
        self.discriminator.train()
        
        # Expert 데이터셋이 비어있는지 확인
        if len(self.expert_dataset) == 0:
            raise ValueError(
                "Expert dataset is empty. Cannot train discriminator without expert data. "
                f"Please check the expert motion file and ensure it contains valid data."
            )
        
        # Expert 데이터 샘플링 (iterator 재사용으로 효율성 향상)
        try:
            if self.expert_iterator is None:
                self.expert_iterator = iter(self.expert_loader)
            expert_batch = next(self.expert_iterator)
        except StopIteration:
            # Iterator가 소진되면 새로 생성
            self.expert_iterator = iter(self.expert_loader)
            try:
                expert_batch = next(self.expert_iterator)
            except StopIteration:
                # 데이터셋이 비어있는 경우 (이미 위에서 체크했지만 방어적 코딩)
                raise ValueError(
                    "Expert dataset iterator is empty. This should not happen if dataset length check passed. "
                    f"Dataset length: {len(self.expert_dataset)}"
                )
        
        expert_transitions, expert_labels = expert_batch
        expert_transitions = expert_transitions.to(self.device)
        expert_labels = expert_labels.to(self.device)
        
        # Policy 데이터 준비
        policy_labels = torch.zeros(policy_state_transitions.shape[0], 1, device=self.device)
        
        # 배치 크기 맞추기
        batch_size = min(policy_state_transitions.shape[0], expert_transitions.shape[0])
        policy_transitions = policy_state_transitions[:batch_size]
        expert_transitions = expert_transitions[:batch_size]
        policy_labels = policy_labels[:batch_size]
        expert_labels = expert_labels[:batch_size]
        
        # 합치기
        all_transitions = torch.cat([policy_transitions, expert_transitions], dim=0)
        all_labels = torch.cat([policy_labels, expert_labels], dim=0)
        
        # Forward pass
        predictions = self.discriminator(all_transitions)
        
        # 손실 계산
        loss = self.criterion(predictions, all_labels)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 메트릭 계산
        with torch.no_grad():
            accuracy = ((predictions > 0.5).float() == all_labels).float().mean()
        
        return {
            "discriminator_loss": loss.item(),
            "discriminator_accuracy": accuracy.item(),
        }


def load_discriminator_config(config_path: str) -> DiscriminatorCfg:
    """Discriminator 설정 파일 로드.
    
    Args:
        config_path: 설정 파일 경로
        
    Returns:
        DiscriminatorCfg 객체
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Discriminator config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # YAML에서 설정 로드
    disc_cfg_dict = config_dict.get('discriminator', {})
    
    # DiscriminatorCfg 객체 생성
    cfg = DiscriminatorCfg()
    if 'hidden_dims' in disc_cfg_dict:
        cfg.hidden_dims = disc_cfg_dict['hidden_dims']
    if 'activation' in disc_cfg_dict:
        cfg.activation = disc_cfg_dict['activation']
    if 'state_dim' in disc_cfg_dict:
        cfg.state_dim = disc_cfg_dict['state_dim']
    if 'learning_rate' in disc_cfg_dict:
        cfg.learning_rate = disc_cfg_dict['learning_rate']
    if 'weight_decay' in disc_cfg_dict:
        cfg.weight_decay = disc_cfg_dict['weight_decay']
    
    return cfg


def main():
    """AMP 알고리즘으로 H1 Walking 환경 학습."""
    
    print("[INFO] ========================================")
    print("[INFO] AMP Training Script")
    print("[INFO] ========================================")
    
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
    run_name = datetime.now().strftime("AMP_%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        run_name = f"{agent_cfg.run_name}_{run_name}"
    log_dir = os.path.join(log_root_path, run_name)

    # =====================================================================
    # Discriminator 초기화
    # =====================================================================
    print("[INFO] Initializing Discriminator...")
    disc_cfg = load_discriminator_config(args_cli.discriminator_cfg)
    discriminator = Discriminator(disc_cfg)
    discriminator = discriminator.to(args_cli.device)
    print(f"[INFO] Discriminator initialized: {disc_cfg.hidden_dims}, state_dim={disc_cfg.state_dim}")
    
    # =====================================================================
    # Expert 데이터셋 로드
    # =====================================================================
    print(f"[INFO] Loading expert motion dataset: {args_cli.expert_motion_file}")
    expert_motion_path = Path(args_cli.expert_motion_file)
    if not expert_motion_path.exists():
        raise FileNotFoundError(
            f"Expert motion file not found: {expert_motion_path}\n"
            f"Please run data preprocessing first (process_amass.py)"
        )
    
    expert_dataset = MotionDataset(str(expert_motion_path), device=args_cli.device)
    print(f"[INFO] Expert dataset loaded: {len(expert_dataset)} motion clips")
    
    # 데이터셋이 비어있는지 확인
    if len(expert_dataset) == 0:
        raise ValueError(
            f"Expert dataset is empty. Cannot train discriminator without expert data. "
            f"Please check the expert motion file: {expert_motion_path}"
        )
    
    # Discriminator Trainer 초기화
    disc_trainer = DiscriminatorTrainer(discriminator, expert_dataset, disc_cfg, device=args_cli.device)
    
    # =====================================================================
    # 환경 생성
    # =====================================================================
    print(f"[INFO] Creating environment: {args_cli.task}")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # Discriminator를 환경에 주입 (Style reward 계산을 위해)
    # 주의: 환경의 reward manager에 discriminator를 주입해야 함
    # 이는 환경의 내부 구조에 따라 다를 수 있음
    # TODO: 환경의 reward manager에 discriminator 주입 방법 구현 필요
    
    # RSL-RL 래퍼 적용
    env = RslRlVecEnvWrapper(env)

    # 설정 출력
    print(f"[INFO] Environment: {args_cli.task}")
    print(f"[INFO] Number of environments: {env.num_envs}")
    print(f"[INFO] Max iterations: {agent_cfg.max_iterations}")
    print(f"[INFO] Log directory: {log_dir}")
    print(f"[INFO] Expert motion file: {args_cli.expert_motion_file}")

    # PPO Runner 생성
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    
    # 체크포인트에서 재개 (있는 경우)
    if args_cli.resume is not None:
        print(f"[INFO] Loading model checkpoint from: {args_cli.resume}")
        runner.load(args_cli.resume)
        # Discriminator 체크포인트도 로드 (있는 경우)
        disc_checkpoint = args_cli.resume.replace("model_", "discriminator_")
        if os.path.exists(disc_checkpoint):
            print(f"[INFO] Loading discriminator checkpoint from: {disc_checkpoint}")
            discriminator.load_state_dict(torch.load(disc_checkpoint, map_location=args_cli.device))

    # =====================================================================
    # 학습 시작
    # =====================================================================
    print("[INFO] Starting AMP training...")
    print(f"[INFO] Total training steps: {agent_cfg.max_iterations} iterations × "
          f"{agent_cfg.num_steps_per_env} steps/env × {env.num_envs} envs")
    
    # 주의: 현재 RSL-RL의 기본 학습 루프는 Discriminator 학습을 포함하지 않음
    # 커스텀 학습 루프를 구현하거나, RSL-RL의 확장 기능을 사용해야 함
    # 여기서는 기본 학습 루프를 사용하고, Discriminator 학습은 별도로 구현 필요
    
    # TODO: 커스텀 학습 루프 구현
    # - 각 iteration마다:
    #   1. Policy 데이터 수집
    #   2. Discriminator 학습 (policy transitions vs expert transitions)
    #   3. Policy 학습 (RSL-RL runner 사용)
    #   4. Discriminator 체크포인트 저장
    
    # 현재는 기본 학습 루프 사용 (Discriminator 학습은 추후 통합)
    print("[WARNING] Discriminator training loop not yet integrated.")
    print("[WARNING] Using standard PPO training loop.")
    print("[WARNING] Discriminator will be trained separately.")
    
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    
    # Discriminator 체크포인트 저장
    disc_checkpoint_path = os.path.join(log_dir, f"discriminator_{agent_cfg.max_iterations}.pt")
    torch.save(discriminator.state_dict(), disc_checkpoint_path)
    print(f"[INFO] Discriminator checkpoint saved: {disc_checkpoint_path}")

    # 환경 종료
    env.close()
    print("[INFO] Training complete!")


if __name__ == "__main__":
    # 학습 실행
    main()
    # Isaac Sim 종료
    simulation_app.close()

