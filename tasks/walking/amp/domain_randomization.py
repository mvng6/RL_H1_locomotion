# Copyright (c) 2025, RL Project Workspace
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""도메인 랜덤화 매니저.

Sim-to-Real 전이를 위한 도메인 랜덤화를 관리합니다.
"""

import torch
import numpy as np
from typing import Dict, Optional
from isaaclab.utils import configclass
import yaml
from pathlib import Path


@configclass
class DomainRandomizationCfg:
    """도메인 랜덤화 설정."""
    
    enabled: bool = True
    frequency: str = "episode"  # "episode" or "step"
    config_path: str = "config/amp/domain_randomization.yaml"


class DomainRandomizationManager:
    """도메인 랜덤화를 관리하는 클래스.
    
    역할:
    - 시뮬레이션 파라미터를 랜덤하게 변경
    - Sim-to-Real 전이 성능 향상
    """
    
    def __init__(self, cfg: DomainRandomizationCfg, num_envs: int, device: str = "cuda:0"):
        """초기화.
        
        Args:
            cfg: 도메인 랜덤화 설정
            num_envs: 환경 개수
            device: 디바이스
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        
        # 설정 파일 로드
        config_path = Path(cfg.config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Domain randomization config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 랜덤화 파라미터 초기화
        self._init_randomization_params()
        
    def _init_randomization_params(self):
        """랜덤화 파라미터 초기화."""
        params = self.config['domain_randomization']['parameters']
        
        # 각 환경마다 랜덤 값 저장
        self.random_params = {}
        
        if params['link_mass']['enabled']:
            self.random_params['link_mass'] = torch.zeros(
                self.num_envs, device=self.device
            )
            
        if params['com_position']['enabled']:
            self.random_params['com_position'] = torch.zeros(
                self.num_envs, 3, device=self.device
            )
            
        # TODO: 나머지 파라미터들 초기화
        
    def randomize(self, env, episode_start: bool = False):
        """도메인 랜덤화 적용.
        
        Args:
            env: RL 환경 객체
            episode_start: 에피소드 시작 여부
        """
        if not self.cfg.enabled:
            return
            
        # 주기 확인
        if self.cfg.frequency == "episode" and not episode_start:
            return
        
        params = self.config['domain_randomization']['parameters']
        
        # TODO: 각 랜덤화 항목별 메서드 호출
        # 링크 질량 랜덤화
        if params['link_mass']['enabled']:
            self._randomize_link_mass(env, params['link_mass'])
            
        # TODO: 나머지 랜덤화 메서드 호출
    
    def _randomize_link_mass(self, env, config: Dict):
        """링크 질량 랜덤화.
        
        Args:
            env: RL 환경 객체
            config: 랜덤화 설정
        """
        range_val = config['range']
        
        # 각 환경마다 랜덤 스케일 생성 (±10%)
        scales = torch.rand(self.num_envs, device=self.device)
        scales = scales * (range_val[1] - range_val[0]) + range_val[0]
        
        # TODO: Isaac Lab API를 사용하여 링크 질량 설정
        # env.scene.robot.set_link_masses(...)
        
    # TODO: 나머지 랜덤화 메서드들 구현

