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
    state_transitions: torch.Tensor,
    discriminator,  # Discriminator 객체 (런타임에 주입)
    weight: float = 1.0
) -> torch.Tensor:
    """Style reward 계산 함수.
    
    Args:
        env: RL 환경
        state_transitions: (num_envs, 2*state_dim) State transitions
        discriminator: Discriminator 네트워크
        weight: Style reward 가중치
        
    Returns:
        rewards: (num_envs,) Style rewards
    """
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

