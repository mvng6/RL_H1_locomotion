# Copyright (c) 2025, RL Project Workspace
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Expert 모션 데이터셋 로더.

AMASS에서 전처리된 Expert 모션 데이터를 로드합니다.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple
from pathlib import Path


class MotionDataset(Dataset):
    """Expert 모션 데이터셋 (AMASS에서 전처리된 데이터).
    
    역할:
    - Discriminator 학습을 위한 Expert 데이터 제공
    - Policy의 상태 전이와 비교할 실제 인간 보행 패턴
    """
    
    def __init__(self, motion_file: str, device: str = "cuda:0"):
        """초기화.
        
        Args:
            motion_file: AMP 형식 모션 파일 경로 (.npy)
            device: 데이터를 로드할 디바이스
        """
        self.device = device
        
        # 모션 데이터 로드
        data_path = Path(motion_file)
        if not data_path.exists():
            raise FileNotFoundError(f"Motion file not found: {motion_file}")
        
        # .npy 파일 로드 (dict 형태로 저장된 경우)
        data = np.load(motion_file, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.dtype == object:
            data = data.item()
        
        # 데이터 전처리
        self.root_positions = torch.from_numpy(data['root_position']).float().to(device)
        self.root_rotations = torch.from_numpy(data['root_rotation']).float().to(device)
        self.joint_positions = torch.from_numpy(data['joint_positions']).float().to(device)
        self.joint_velocities = torch.from_numpy(data['joint_velocities']).float().to(device)
        
        # 상태 벡터로 변환 (Discriminator 입력 형식)
        self.states = self._compute_states()
        
    def _compute_states(self) -> torch.Tensor:
        """모션 데이터를 상태 벡터로 변환합니다.
        
        상태 벡터 구성:
        - Root position (3)
        - Root rotation (4) [quaternion]
        - Joint positions (34 * 3 = 102)
        - Joint velocities (34 * 3 = 102)
        총: 211 차원
        
        Returns:
            states: (N, T, state_dim) 상태 벡터
        """
        num_clips, num_frames = self.root_positions.shape[:2]
        state_dim = 3 + 4 + 34 * 3 + 34 * 3  # 211
        
        states = torch.zeros(num_clips, num_frames, state_dim, device=self.device)
        
        # Root position (3)
        states[:, :, 0:3] = self.root_positions
        
        # Root rotation (4) [quaternion]
        states[:, :, 3:7] = self.root_rotations
        
        # Joint positions (102)
        joint_pos_flat = self.joint_positions.view(num_clips, num_frames, -1)
        states[:, :, 7:109] = joint_pos_flat
        
        # Joint velocities (102)
        joint_vel_flat = self.joint_velocities.view(num_clips, num_frames, -1)
        states[:, :, 109:211] = joint_vel_flat
        
        return states
        
    def __len__(self) -> int:
        """데이터셋 크기."""
        return len(self.states)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """State transition 샘플 반환.
        
        Args:
            idx: 샘플 인덱스
            
        Returns:
            (state_transition, label): State transition pair와 레이블 (1 = Real)
        """
        # 랜덤하게 시간 스텝 선택
        num_frames = self.states.shape[1]
        t = torch.randint(0, num_frames - 1, (1,)).item()
        
        s_t = self.states[idx, t]
        s_t_next = self.states[idx, t + 1]
        
        # Concatenate
        state_transition = torch.cat([s_t, s_t_next], dim=-1)
        
        return state_transition, torch.ones(1, dtype=torch.float32).to(self.device)  # Label: 1 (Real)

