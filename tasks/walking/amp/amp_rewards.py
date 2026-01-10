# Copyright (c) 2025, RL Project Workspace
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AMP 보상 함수 (Style reward).

Discriminator 기반 Style reward를 계산합니다.
"""

import torch
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from typing import Optional

# Discriminator는 런타임에 주입되므로 여기서는 import하지 않음
# from .discriminator import Discriminator


def compute_style_reward(
    env,
    discriminator,  # Discriminator 객체 (런타임에 주입)
    weight: float = 1.0
) -> torch.Tensor:
    """Style reward 계산 함수.
    
    Args:
        env: RL 환경
        discriminator: Discriminator 네트워크
        weight: Style reward 가중치
        
    Returns:
        rewards: (num_envs,) Style rewards
    """
    # 환경에서 현재 상태와 이전 상태를 추출하여 state_transitions 계산
    robot = env.scene["robot"]
    
    # 현재 상태 추출 (Root position + rotation + joint positions + velocities)
    # 상태 벡터 구성: Root pos (3) + Root rot (4) + Joint pos (102) + Joint vel (102) = 211
    root_pos = robot.data.root_pos_w  # (num_envs, 3)
    root_rot = robot.data.root_quat_w  # (num_envs, 4) - quaternion
    joint_pos = robot.data.joint_pos  # (num_envs, num_joints, 3) -> flatten
    joint_vel = robot.data.joint_vel  # (num_envs, num_joints, 3) -> flatten
    
    # Joint positions와 velocities를 flatten
    num_envs = root_pos.shape[0]
    joint_pos_flat = joint_pos.view(num_envs, -1)  # (num_envs, num_joints*3)
    joint_vel_flat = joint_vel.view(num_envs, -1)  # (num_envs, num_joints*3)
    
    # 현재 상태 벡터 구성
    current_state = torch.cat([
        root_pos,      # (num_envs, 3)
        root_rot,      # (num_envs, 4)
        joint_pos_flat,  # (num_envs, num_joints*3)
        joint_vel_flat,  # (num_envs, num_joints*3)
    ], dim=-1)  # (num_envs, state_dim)
    
    # 이전 상태는 환경의 이전 스텝 버퍼에서 가져오거나, 없으면 현재 상태 사용
    # Isaac Lab에서는 이전 상태를 저장하는 버퍼가 있을 수 있지만,
    # 없을 경우를 대비해 현재 상태를 이전 상태로 사용 (첫 스텝 처리)
    if not hasattr(env, '_prev_state') or env._prev_state is None:
        # 첫 스텝: 이전 상태 = 현재 상태
        prev_state = current_state.clone()
    else:
        prev_state = env._prev_state
    
    # State transition 구성: [s_t, s_{t+1}]
    state_transitions = torch.cat([prev_state, current_state], dim=-1)  # (num_envs, 2*state_dim)
    
    # 현재 상태를 다음 스텝을 위한 이전 상태로 저장
    env._prev_state = current_state.clone()
    
    with torch.no_grad():
        style_rewards = discriminator.compute_reward(state_transitions)
    return weight * style_rewards


@configclass
class AMPRewardsCfg:
    """AMP 보상 함수 설정.
    
    주의: discriminator는 런타임에 주입되어야 함
    """
    
    # Style reward 가중치
    style_reward_weight: float = 1.0
    
    # Expert 모션 데이터 경로
    expert_motion_file: str = "data/processed/amp_motions.npy"
    
    # Style reward 항목 (런타임에 discriminator 주입 필요)
    style_reward = RewTerm(
        func=compute_style_reward,
        weight=1.0,
        params={
            "discriminator": None,  # 런타임에 주입
            "weight": 1.0,
        },
    )

