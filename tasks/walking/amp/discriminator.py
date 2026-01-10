# Copyright (c) 2025, RL Project Workspace
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AMP Discriminator 네트워크 구현.

Discriminator는 Policy의 상태 전이와 Expert 데이터의 상태 전이를 구분합니다.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from isaaclab.utils import configclass


@configclass
class DiscriminatorCfg:
    """Discriminator 네트워크 설정."""
    
    # 네트워크 구조
    hidden_dims: list = [512, 512, 256]  # 히든 레이어 차원
    activation: str = "elu"  # 활성화 함수
    
    # 입력 차원 (State transition: [s_t, s_{t+1}])
    state_dim: int = 211  # H1 상태 차원 (관절 위치 + 속도 + 베이스 상태)
    
    # 학습 설정
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5


class Discriminator(nn.Module):
    """AMP Discriminator 네트워크.
    
    입력: State transition (s_t, s_{t+1})
    출력: Real/Fake 확률 (0~1)
    
    역할:
    - Policy의 상태 전이와 Expert 데이터의 상태 전이를 구분
    - Policy는 Discriminator를 속이려고 학습 (Style reward)
    """
    
    def __init__(self, cfg: DiscriminatorCfg):
        """초기화.
        
        Args:
            cfg: Discriminator 설정
        """
        super().__init__()
        self.cfg = cfg
        
        # 네트워크 레이어 구성
        layers = []
        input_dim = cfg.state_dim * 2  # [s_t, s_{t+1}] concatenate
        
        for hidden_dim in cfg.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(self._get_activation(cfg.activation))
            input_dim = hidden_dim
        
        # 출력 레이어 (확률)
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
    def _get_activation(self, name: str) -> nn.Module:
        """활성화 함수 반환."""
        activations = {
            "relu": nn.ReLU(),
            "elu": nn.ELU(),
            "tanh": nn.Tanh(),
        }
        return activations.get(name, nn.ReLU())
    
    def forward(self, state_transitions: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            state_transitions: (B, 2*state_dim) State transitions [s_t, s_{t+1}]
            
        Returns:
            probabilities: (B, 1) Real/Fake 확률 (0~1)
        """
        return self.network(state_transitions)
    
    def compute_reward(self, state_transitions: torch.Tensor) -> torch.Tensor:
        """Style reward 계산.
        
        Discriminator의 출력을 보상으로 변환:
        R_style = -log(1 - D(s_t, s_{t+1}))
        
        Discriminator는 1을 expert motion에, 0을 policy motion에 출력하도록 학습됨.
        따라서 policy가 expert처럼 보이려면 D가 높아야 하고, -log(1-D)로 보상하면
        D가 높을수록 (expert-like) 보상이 커짐.
        
        Args:
            state_transitions: (B, 2*state_dim) State transitions
            
        Returns:
            rewards: (B,) Style rewards
        """
        probs = self.forward(state_transitions)
        # 안정성을 위해 클리핑
        probs = torch.clamp(probs, min=1e-7, max=1.0 - 1e-7)
        # 올바른 AMP 공식: -log(1-D)
        # D가 높을수록 (expert-like) 보상이 커짐
        rewards = -torch.log(1.0 - probs).squeeze(-1)
        return rewards

