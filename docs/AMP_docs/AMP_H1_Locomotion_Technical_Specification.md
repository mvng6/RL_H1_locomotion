# AMP 기반 H1 휴머노이드 보행 학습 기술 구현 명세서

**프로젝트**: Unitree H1 Humanoid Robot Natural Locomotion using AMP  
**알고리즘**: Adversarial Motion Priors (AMP) / PPO  
**프레임워크**: NVIDIA Isaac Sim (Isaac Lab), PyTorch, RSL-RL  
**데이터셋**: AMASS Dataset (SMPL format)

---

## 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [프로젝트 디렉토리 구조](#프로젝트-디렉토리-구조)
3. [Phase 1: Mocap Data Preprocessing Pipeline](#phase-1-mocap-data-preprocessing-pipeline)
4. [Phase 2: AMP Network Architecture & Environment Setup](#phase-2-amp-network-architecture--environment-setup)
5. [Phase 3: Curriculum Learning Strategy](#phase-3-curriculum-learning-strategy)
6. [Phase 4: Domain Randomization (Sim-to-Real)](#phase-4-domain-randomization-sim-to-real)
7. [구현 체크리스트](#구현-체크리스트)

---

## 프로젝트 개요

### 목표

AMP 알고리즘을 사용하여 Unitree H1 휴머노이드 로봇이 인간과 유사한 자연스러운 보행 패턴을 학습하도록 구현합니다.

### AMP 알고리즘 핵심 개념

**Adversarial Motion Priors (AMP)**는 모방 학습과 강화학습을 결합한 알고리즘입니다:

1. **Policy (Actor)**: 상태를 입력받아 행동을 출력하는 정책 네트워크
2. **Discriminator**: Policy의 상태 전이와 Expert 데이터(AMASS)의 상태 전이를 구분하는 판별 네트워크
3. **Reward Function**: Task reward (속도 추적) + Style reward (Discriminator 출력)

### 기술 스택

- **시뮬레이터**: NVIDIA Isaac Sim (Isaac Lab)
- **RL 프레임워크**: RSL-RL (PPO 기반)
- **딥러닝**: PyTorch
- **로봇**: Unitree H1 (URDF/USD)
- **모션 데이터**: AMASS Dataset (SMPL)

---

## 프로젝트 디렉토리 구조

```
exts/h1_locomotion/
├── data/                                    # 데이터 디렉토리
│   ├── amass/                              # 원본 AMASS 데이터
│   │   ├── CMU_Mocap/                      # CMU 데이터셋
│   │   ├── HumanML3D/                      # HumanML3D 데이터셋
│   │   └── ...
│   ├── processed/                          # 전처리된 데이터
│   │   ├── retargeted_motions/            # 리타겟팅된 모션 클립
│   │   │   ├── walking_clips/             # 걷기 모션 클립
│   │   │   └── running_clips/             # 달리기 모션 클립
│   │   └── amp_motions.npy                # AMP 호환 형식 (최종 출력)
│   └── mapping/                            # 스켈레톤 매핑 정보
│       ├── smpl_to_h1_joint_mapping.json   # SMPL -> H1 관절 매핑
│       └── t_pose_alignment.json          # T-pose 정렬 정보
│
├── tasks/
│   └── walking/
│       ├── amp/                            # AMP 전용 모듈
│       │   ├── __init__.py
│       │   ├── discriminator.py           # Discriminator 네트워크
│       │   ├── amp_rewards.py             # AMP 보상 함수 (Style reward)
│       │   └── motion_dataset.py          # Expert 모션 데이터셋 로더
│       ├── mdp/
│       │   ├── observations.py            # 관측 공간 (기존 + AMP용)
│       │   ├── rewards.py                 # Task rewards
│       │   └── terminations.py
│       ├── walking_env_cfg.py             # 기본 환경 설정
│       └── amp_env_cfg.py                 # AMP 환경 설정
│
├── scripts/
│   ├── data_preprocessing/
│   │   ├── process_amass.py               # AMASS 전처리 메인 스크립트
│   │   ├── retargeting/
│   │   │   ├── __init__.py
│   │   │   ├── smpl_loader.py            # SMPL 데이터 로더
│   │   │   ├── h1_skeleton.py            # H1 스켈레톤 정의
│   │   │   ├── retargeter.py             # 리타겟팅 엔진
│   │   │   └── utils.py                  # 유틸리티 함수
│   │   └── export_motions.py             # AMP 형식으로 내보내기
│   ├── train_walking_amp.py               # AMP 학습 메인 스크립트
│   └── play_walking_amp.py                # AMP 정책 테스트 스크립트
│
├── config/
│   ├── agents/
│   │   ├── walking_amp_ppo_cfg.py        # AMP PPO 에이전트 설정
│   │   └── curriculum_config.yaml        # 커리큘럼 학습 설정
│   └── amp/
│       ├── discriminator_cfg.yaml        # Discriminator 네트워크 설정
│       └── domain_randomization.yaml     # 도메인 랜덤화 설정
│
└── docs/
    └── AMP_H1_Locomotion_Technical_Specification.md  # 이 문서
```

---

## Phase 1: Mocap Data Preprocessing Pipeline

### 목표

AMASS 데이터셋(SMPL 형식)을 H1 로봇에 맞게 리타겟팅하고, AMP 알고리즘이 사용할 수 있는 형식(`.npy`)으로 변환합니다.

### 1.1 워크플로우 개요

```
AMASS Dataset (SMPL)
    ↓
[1] SMPL 데이터 로드 (관절 위치, 회전)
    ↓
[2] Motion Retargeting (SMPL → H1)
    ├── Joint Mapping (LeftUpLeg → L_hip_pitch)
    ├── T-pose Alignment
    └── Coordinate System Conversion
    ↓
[3] Motion Filtering & Segmentation
    ├── Walking clips 추출
    ├── Running clips 추출
    └── Quality check (속도, 안정성)
    ↓
[4] Export to AMP Format
    └── amp_human_motions.npy
        ├── Root position (N, T, 3)
        ├── Root rotation (N, T, 4) [quaternion]
        ├── Joint positions (N, T, J, 3)
        └── Joint velocities (N, T, J, 3)
```

### 1.2 핵심 클래스 정의

#### 1.2.1 `SMPLLoader` (`scripts/data_preprocessing/retargeting/smpl_loader.py`)

```python
"""SMPL 데이터 로더 클래스."""

import numpy as np
import torch
from typing import Dict, List, Optional
from pathlib import Path

class SMPLLoader:
    """AMASS 데이터셋에서 SMPL 형식의 모션 데이터를 로드합니다.
    
    Attributes:
        data_path: AMASS 데이터셋 경로
        fps: 모션 데이터의 프레임레이트 (기본값: 30 fps)
    """
    
    def __init__(self, data_path: str, fps: int = 30):
        """초기화.
        
        Args:
            data_path: AMASS 데이터셋 루트 디렉토리
            fps: 모션 데이터의 프레임레이트
        """
        self.data_path = Path(data_path)
        self.fps = fps
        
    def load_motion(self, file_path: str) -> Dict[str, np.ndarray]:
        """SMPL 모션 파일을 로드합니다.
        
        Args:
            file_path: .npz 또는 .pkl 파일 경로
            
        Returns:
            Dictionary containing:
                - 'root_translation': (T, 3) 루트 위치
                - 'root_orientation': (T, 3) 루트 오일러 각도
                - 'pose': (T, 72) SMPL 관절 회전 (axis-angle)
                - 'betas': (10,) SMPL shape parameters
                - 'fps': 프레임레이트
        """
        # AMASS 데이터 로드 로직
        # .npz 또는 .pkl 파일 파싱
        pass
        
    def get_joint_positions(self, motion_data: Dict) -> np.ndarray:
        """SMPL 모션 데이터에서 관절 위치를 계산합니다.
        
        Args:
            motion_data: load_motion()의 출력
            
        Returns:
            Joint positions: (T, 24, 3) SMPL 24개 관절의 3D 위치
        """
        # SMPL forward kinematics를 사용하여 관절 위치 계산
        pass
```

#### 1.2.2 `H1Skeleton` (`scripts/data_preprocessing/retargeting/h1_skeleton.py`)

```python
"""H1 로봇 스켈레톤 정의 및 유틸리티."""

import numpy as np
from typing import Dict, List, Tuple

class H1Skeleton:
    """Unitree H1 로봇의 스켈레톤 구조를 정의합니다.
    
    H1의 관절 구조:
    - Root (pelvis)
    - Left/Right: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
    - Torso: waist_yaw, waist_pitch, waist_roll
    - Arms: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow
    """
    
    # H1 관절 이름 리스트 (총 34개 DOF)
    JOINT_NAMES = [
        # Left Leg
        "L_hip_pitch", "L_hip_roll", "L_hip_yaw", "L_knee", 
        "L_ankle_pitch", "L_ankle_roll",
        # Right Leg
        "R_hip_pitch", "R_hip_roll", "R_hip_yaw", "R_knee",
        "R_ankle_pitch", "R_ankle_roll",
        # Torso
        "waist_yaw", "waist_pitch", "waist_roll",
        # Left Arm
        "L_shoulder_pitch", "L_shoulder_roll", "L_shoulder_yaw", "L_elbow",
        # Right Arm
        "R_shoulder_pitch", "R_shoulder_roll", "R_shoulder_yaw", "R_elbow",
    ]
    
    # SMPL -> H1 관절 매핑 딕셔너리
    SMPL_TO_H1_MAPPING = {
        # Lower body
        "LeftUpLeg": "L_hip_pitch",
        "LeftLeg": "L_knee",
        "LeftFoot": "L_ankle_pitch",
        "RightUpLeg": "R_hip_pitch",
        "RightLeg": "R_knee",
        "RightFoot": "R_ankle_pitch",
        # Upper body
        "LeftArm": "L_shoulder_pitch",
        "LeftForeArm": "L_elbow",
        "RightArm": "R_shoulder_pitch",
        "RightForeArm": "R_elbow",
        # Torso
        "Spine1": "waist_pitch",
        "Spine2": "waist_roll",
    }
    
    @staticmethod
    def get_t_pose() -> Dict[str, np.ndarray]:
        """H1의 T-pose 관절 각도를 반환합니다.
        
        Returns:
            Dictionary mapping joint names to joint angles (radians)
        """
        return {
            "L_hip_pitch": 0.0,
            "L_hip_roll": 0.0,
            "L_hip_yaw": 0.0,
            "L_knee": 0.0,
            "L_ankle_pitch": 0.0,
            "L_ankle_roll": 0.0,
            # ... (나머지 관절들)
        }
```

#### 1.2.3 `MotionRetargeter` (`scripts/data_preprocessing/retargeting/retargeter.py`)

```python
"""SMPL 모션을 H1 로봇에 리타겟팅하는 클래스."""

import numpy as np
import torch
from typing import Dict, List, Optional
from .smpl_loader import SMPLLoader
from .h1_skeleton import H1Skeleton

class MotionRetargeter:
    """SMPL 인간 모션을 H1 로봇 모션으로 변환합니다.
    
    주요 작업:
    1. 관절 매핑 (SMPL 24개 관절 → H1 34개 DOF)
    2. T-pose 정렬
    3. 좌표계 변환 (SMPL → H1)
    4. 관절 각도 계산 (Forward/Inverse Kinematics)
    """
    
    def __init__(self, mapping_file: Optional[str] = None):
        """초기화.
        
        Args:
            mapping_file: 관절 매핑 JSON 파일 경로 (선택사항)
        """
        self.smpl_loader = SMPLLoader()
        self.h1_skeleton = H1Skeleton()
        self.mapping = self._load_mapping(mapping_file)
        
    def retarget(self, smpl_motion: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """SMPL 모션을 H1 모션으로 리타겟팅합니다.
        
        Args:
            smpl_motion: SMPL 모션 데이터
                - 'root_translation': (T, 3)
                - 'root_orientation': (T, 3) [euler angles]
                - 'joint_positions': (T, 24, 3)
                
        Returns:
            H1 모션 데이터:
                - 'root_position': (T, 3) H1 pelvis 위치
                - 'root_rotation': (T, 4) H1 pelvis 회전 [quaternion]
                - 'joint_positions': (T, 34, 3) H1 관절 위치
                - 'joint_angles': (T, 34) H1 관절 각도 [radians]
                - 'joint_velocities': (T, 34, 3) H1 관절 속도
        """
        # 1. Root 위치/회전 변환
        root_pos, root_rot = self._retarget_root(
            smpl_motion['root_translation'],
            smpl_motion['root_orientation']
        )
        
        # 2. 관절 위치 매핑
        h1_joint_positions = self._map_joint_positions(
            smpl_motion['joint_positions']
        )
        
        # 3. 관절 각도 계산 (Inverse Kinematics)
        h1_joint_angles = self._compute_joint_angles(
            root_pos, root_rot, h1_joint_positions
        )
        
        # 4. 관절 속도 계산
        h1_joint_velocities = self._compute_joint_velocities(
            h1_joint_angles
        )
        
        return {
            'root_position': root_pos,
            'root_rotation': root_rot,
            'joint_positions': h1_joint_positions,
            'joint_angles': h1_joint_angles,
            'joint_velocities': h1_joint_velocities,
        }
    
    def _retarget_root(self, smpl_root_pos: np.ndarray, 
                      smpl_root_rot: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """루트 위치와 회전을 H1 좌표계로 변환합니다.
        
        SMPL과 H1의 좌표계 차이:
        - SMPL: Y-up, Z-forward
        - H1: Z-up, X-forward
        
        Returns:
            (root_position, root_rotation_quat)
        """
        # 좌표계 변환 로직
        pass
        
    def _map_joint_positions(self, smpl_joint_pos: np.ndarray) -> np.ndarray:
        """SMPL 관절 위치를 H1 관절 위치로 매핑합니다.
        
        Args:
            smpl_joint_pos: (T, 24, 3) SMPL 관절 위치
            
        Returns:
            h1_joint_pos: (T, 34, 3) H1 관절 위치
        """
        # 관절 매핑 로직
        pass
        
    def _compute_joint_angles(self, root_pos: np.ndarray, root_rot: np.ndarray,
                             joint_positions: np.ndarray) -> np.ndarray:
        """관절 위치로부터 관절 각도를 계산합니다 (Inverse Kinematics).
        
        Args:
            root_pos: (T, 3) 루트 위치
            root_rot: (T, 4) 루트 회전 [quaternion]
            joint_positions: (T, 34, 3) 관절 위치
            
        Returns:
            joint_angles: (T, 34) 관절 각도 [radians]
        """
        # IK 계산 로직 (Pinocchio 또는 수치적 최적화 사용)
        pass
```

#### 1.2.4 `process_amass.py` (메인 스크립트)

```python
"""AMASS 데이터셋을 처리하여 AMP 호환 모션 파일을 생성합니다."""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from retargeting.retargeter import MotionRetargeter
from retargeting.utils import filter_walking_motions, segment_motions

def main():
    parser = argparse.ArgumentParser(description="Process AMASS dataset for AMP")
    parser.add_argument("--amass_path", type=str, required=True,
                       help="Path to AMASS dataset root directory")
    parser.add_argument("--output_path", type=str, default="./data/processed/amp_motions.npy",
                       help="Output path for processed motions")
    parser.add_argument("--min_duration", type=float, default=2.0,
                       help="Minimum motion clip duration (seconds)")
    parser.add_argument("--max_duration", type=float, default=10.0,
                       help="Maximum motion clip duration (seconds)")
    args = parser.parse_args()
    
    # 리타겟터 초기화
    retargeter = MotionRetargeter()
    
    # AMASS 데이터셋 스캔
    amass_path = Path(args.amass_path)
    all_motions = []
    
    print("[INFO] Processing AMASS dataset...")
    for dataset_dir in amass_path.iterdir():
        if not dataset_dir.is_dir():
            continue
            
        for motion_file in tqdm(dataset_dir.glob("*.npz"), desc=f"Processing {dataset_dir.name}"):
            try:
                # SMPL 모션 로드
                smpl_motion = retargeter.smpl_loader.load_motion(str(motion_file))
                
                # 리타겟팅
                h1_motion = retargeter.retarget(smpl_motion)
                
                # 걷기 모션 필터링 (속도 기반)
                if filter_walking_motions(h1_motion):
                    # 모션 세그멘테이션 (길이 조정)
                    clips = segment_motions(h1_motion, 
                                          min_duration=args.min_duration,
                                          max_duration=args.max_duration)
                    all_motions.extend(clips)
                    
            except Exception as e:
                print(f"[WARNING] Failed to process {motion_file}: {e}")
                continue
    
    # AMP 형식으로 변환 및 저장
    print(f"[INFO] Saving {len(all_motions)} motion clips to {args.output_path}")
    amp_data = convert_to_amp_format(all_motions)
    np.save(args.output_path, amp_data)
    print("[INFO] Done!")

def convert_to_amp_format(motions: List[Dict]) -> np.ndarray:
    """모션 클립 리스트를 AMP 형식으로 변환합니다.
    
    AMP 형식:
    - Root position: (N, T, 3)
    - Root rotation: (N, T, 4) [quaternion]
    - Joint positions: (N, T, J, 3)
    - Joint velocities: (N, T, J, 3)
    
    Returns:
        Dictionary 또는 structured array
    """
    # 변환 로직
    pass

if __name__ == "__main__":
    main()
```

### 1.3 구현 단계

1. **Step 1.1**: SMPL 데이터 로더 구현
   - AMASS `.npz` 파일 파싱
   - 관절 위치 계산 (SMPL forward kinematics)

2. **Step 1.2**: H1 스켈레톤 정의
   - 관절 이름 및 구조 정의
   - T-pose 정의
   - SMPL → H1 매핑 딕셔너리 작성

3. **Step 1.3**: 리타겟팅 엔진 구현
   - 좌표계 변환 (SMPL → H1)
   - 관절 매핑 로직
   - Inverse Kinematics (Pinocchio 사용 권장)

4. **Step 1.4**: 모션 필터링 및 세그멘테이션
   - 걷기 모션 자동 감지 (속도 기반)
   - 모션 클립 길이 정규화

5. **Step 1.5**: AMP 형식으로 내보내기
   - `.npy` 파일로 저장
   - 데이터 형식 검증

---

## Phase 2: AMP Network Architecture & Environment Setup

### 목표

AMP 알고리즘의 핵심 구성요소(Policy, Discriminator)를 구현하고, Isaac Lab 환경에 통합합니다.

### 2.1 AMP 알고리즘 구조

```
┌─────────────────┐
│   Environment   │
│   (H1 Robot)     │
└────────┬────────┘
         │ State (s_t)
         ▼
┌─────────────────┐         ┌──────────────────┐
│   Policy (π)    │────────▶│   Action (a_t)   │
│   (Actor)       │         └────────┬─────────┘
└─────────────────┘                 │
         │                           │
         │ State Transition (s_t, s_{t+1})
         ▼                           │
┌─────────────────┐                 │
│  Discriminator  │                 │
│     (D_φ)       │                 │
└────────┬────────┘                 │
         │                           │
         │ Style Reward (R_style)    │
         ▼                           │
┌─────────────────┐                 │
│  Total Reward   │◀────────────────┘
│ R = R_task +    │
│     λ * R_style │
└─────────────────┘
```

### 2.2 핵심 클래스 정의

#### 2.2.1 `Discriminator` (`tasks/walking/amp/discriminator.py`)

```python
"""AMP Discriminator 네트워크 구현."""

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
    state_dim: int = 48  # H1 상태 차원 (관절 위치 + 속도 + 베이스 상태)
    
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
        R_style = -log(D(s_t, s_{t+1}))
        
        Args:
            state_transitions: (B, 2*state_dim) State transitions
            
        Returns:
            rewards: (B,) Style rewards
        """
        probs = self.forward(state_transitions)
        # 안정성을 위해 클리핑
        probs = torch.clamp(probs, min=1e-7, max=1.0 - 1e-7)
        rewards = -torch.log(probs).squeeze(-1)
        return rewards
```

#### 2.2.2 `MotionDataset` (`tasks/walking/amp/motion_dataset.py`)

```python
"""Expert 모션 데이터셋 로더."""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple

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
        data = np.load(motion_file, allow_pickle=True).item()
        
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
        # 상태 벡터 계산 로직
        pass
        
    def __len__(self) -> int:
        """데이터셋 크기."""
        return len(self.states)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """State transition 샘플 반환.
        
        Args:
            idx: 샘플 인덱스
            
        Returns:
            (s_t, s_{t+1}): State transition pair
        """
        # 랜덤하게 시간 스텝 선택
        t = torch.randint(0, self.states.shape[1] - 1, (1,)).item()
        
        s_t = self.states[idx, t]
        s_t_next = self.states[idx, t + 1]
        
        # Concatenate
        state_transition = torch.cat([s_t, s_t_next], dim=-1)
        
        return state_transition, torch.ones(1).to(self.device)  # Label: 1 (Real)
```

#### 2.2.3 `AMPRewards` (`tasks/walking/amp/amp_rewards.py`)

```python
"""AMP 보상 함수 (Style reward)."""

import torch
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from .discriminator import Discriminator, DiscriminatorCfg

@configclass
class AMPRewardsCfg:
    """AMP 보상 설정."""
    
    # Style reward 가중치
    style_reward_weight: float = 1.0
    
    # Discriminator 설정
    discriminator: DiscriminatorCfg = DiscriminatorCfg()
    
    # Expert 모션 데이터 경로
    expert_motion_file: str = "data/processed/amp_motions.npy"

def compute_style_reward(
    env,
    state_transitions: torch.Tensor,
    discriminator: Discriminator,
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

# Isaac Lab RewardTerm으로 등록
@configclass
class AMPRewardsCfg:
    """AMP 보상 함수 설정."""
    
    style_reward = RewTerm(
        func=compute_style_reward,
        weight=1.0,
        params={
            "discriminator": None,  # 런타임에 주입
            "weight": 1.0,
        },
    )
```

#### 2.2.4 `H1AmpEnv` (`tasks/walking/amp_env_cfg.py`)

```python
"""AMP 환경 설정."""

from isaaclab.utils import configclass
from .walking_env_cfg import WalkingEnvCfg
from .amp.amp_rewards import AMPRewardsCfg
from .mdp import ObservationsCfg, TerminationsCfg

@configclass
class H1AmpEnvCfg(WalkingEnvCfg):
    """H1 AMP 환경 설정.
    
    기존 WalkingEnvCfg를 확장하여 AMP 알고리즘을 지원합니다.
    """
    
    # AMP 보상 추가
    amp_rewards: AMPRewardsCfg = AMPRewardsCfg()
    
    # 관측 공간 (기존 + State transition)
    observations: ObservationsCfg = ObservationsCfg()
    
    # 종료 조건 (기존과 동일)
    terminations: TerminationsCfg = TerminationsCfg()
    
    def __post_init__(self):
        """환경 설정 초기화."""
        super().__post_init__()
        
        # AMP 관련 설정
        # Discriminator는 학습 스크립트에서 초기화되어 주입됨
```

### 2.3 구현 단계

1. **Step 2.1**: Discriminator 네트워크 구현
   - MLP 구조 설계
   - Forward pass 및 Style reward 계산

2. **Step 2.2**: Expert 데이터셋 로더 구현
   - `.npy` 파일 로드
   - State transition 샘플링

3. **Step 2.3**: AMP 보상 함수 통합
   - Style reward를 Isaac Lab RewardTerm으로 등록
   - Task reward와 결합

4. **Step 2.4**: 환경 설정 업데이트
   - `H1AmpEnvCfg` 클래스 작성
   - Gymnasium 환경 등록

---

## Phase 3: Curriculum Learning Strategy

### 목표

정적 균형 → 느린 걷기 → 빠른 걷기로 점진적으로 학습하는 커리큘럼을 설계합니다.

### 3.1 커리큘럼 레벨 정의

| Level | Target Velocity | Focus | Duration (epochs) |
|-------|----------------|-------|-------------------|
| **Level 0** | 0.0 m/s | 정적 균형, 자세 모방 | 0 ~ 500 |
| **Level 1** | 0.3 ~ 0.6 m/s | 느린 걷기, 접촉 스케줄 | 500 ~ 2000 |
| **Level 2** | 1.0 ~ 1.5 m/s | 빠른 걷기, 동적 안정성 | 2000 ~ 5000 |

### 3.2 설정 파일 구조 (`config/curriculum_config.yaml`)

```yaml
# Curriculum Learning Configuration

curriculum:
  # 커리큘럼 활성화 여부
  enabled: true
  
  # 레벨 정의
  levels:
    - name: "static_balance"
      start_epoch: 0
      end_epoch: 500
      target_velocity:
        lin_vel_x: [0.0, 0.0]
        lin_vel_y: [-0.1, 0.1]
        ang_vel_z: [-0.1, 0.1]
      reward_weights:
        track_lin_vel_xy_exp: 0.5  # 낮은 가중치 (속도 추적 덜 중요)
        style_reward: 2.0  # 높은 가중치 (자세 모방 중요)
        flat_orientation_l2: -3.0  # 강한 자세 페널티
        base_height_l2: -1.0
        
    - name: "slow_walk"
      start_epoch: 500
      end_epoch: 2000
      target_velocity:
        lin_vel_x: [0.3, 0.6]
        lin_vel_y: [-0.3, 0.3]
        ang_vel_z: [-0.5, 0.5]
      reward_weights:
        track_lin_vel_xy_exp: 1.0  # 속도 추적 중요도 증가
        style_reward: 1.5
        feet_air_time: 0.5  # 발 공중 시간 보상 추가
        flat_orientation_l2: -2.0
        
    - name: "fast_walk"
      start_epoch: 2000
      end_epoch: 5000
      target_velocity:
        lin_vel_x: [1.0, 1.5]
        lin_vel_y: [-0.5, 0.5]
        ang_vel_z: [-1.0, 1.0]
      reward_weights:
        track_lin_vel_xy_exp: 1.5  # 속도 추적 매우 중요
        style_reward: 1.0
        feet_air_time: 0.75
        flat_orientation_l2: -2.0
        action_rate_l2: -0.02  # 부드러운 움직임 강조

# 커리큘럼 업데이트 설정
update:
  # 업데이트 주기 (epochs)
  frequency: 50
  
  # 업데이트 방식: "linear", "exponential", "step"
  schedule: "step"
  
  # 부드러운 전이를 위한 interpolation
  smooth_transition: true
  transition_duration: 100  # epochs
```

### 3.3 커리큘럼 매니저 구현 (`scripts/train_walking_amp.py` 내부)

```python
"""AMP 학습 스크립트 (커리큘럼 학습 포함)."""

import yaml
import torch
from pathlib import Path
from typing import Dict, Optional

class CurriculumManager:
    """커리큘럼 학습 매니저.
    
    역할:
    - 현재 epoch에 따라 명령 범위 및 보상 가중치 동적 조정
    - 부드러운 레벨 전이 관리
    """
    
    def __init__(self, config_path: str):
        """초기화.
        
        Args:
            config_path: 커리큘럼 설정 YAML 파일 경로
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.levels = self.config['curriculum']['levels']
        self.current_level_idx = 0
        self.current_epoch = 0
        
    def get_current_config(self, epoch: int) -> Dict:
        """현재 epoch에 해당하는 커리큘럼 설정을 반환합니다.
        
        Args:
            epoch: 현재 학습 epoch
            
        Returns:
            Dictionary containing:
                - 'target_velocity': 명령 속도 범위
                - 'reward_weights': 보상 가중치
        """
        self.current_epoch = epoch
        
        # 현재 레벨 찾기
        for i, level in enumerate(self.levels):
            if level['start_epoch'] <= epoch < level['end_epoch']:
                self.current_level_idx = i
                
                # 부드러운 전이 처리
                if self.config['curriculum']['update']['smooth_transition']:
                    return self._interpolate_levels(i, epoch)
                else:
                    return level
        
        # 마지막 레벨
        return self.levels[-1]
    
    def _interpolate_levels(self, level_idx: int, epoch: int) -> Dict:
        """레벨 간 부드러운 전이를 위한 보간.
        
        Args:
            level_idx: 현재 레벨 인덱스
            epoch: 현재 epoch
            
        Returns:
            보간된 설정
        """
        current_level = self.levels[level_idx]
        
        # 전이 구간인지 확인
        transition_start = current_level['start_epoch']
        transition_duration = self.config['curriculum']['update']['transition_duration']
        
        if epoch < transition_start + transition_duration and level_idx > 0:
            # 이전 레벨과 보간
            prev_level = self.levels[level_idx - 1]
            alpha = (epoch - transition_start) / transition_duration
            alpha = max(0.0, min(1.0, alpha))  # 클리핑
            
            # 속도 범위 보간
            target_vel = {}
            for key in ['lin_vel_x', 'lin_vel_y', 'ang_vel_z']:
                prev_range = prev_level['target_velocity'][key]
                curr_range = current_level['target_velocity'][key]
                target_vel[key] = [
                    prev_range[0] * (1 - alpha) + curr_range[0] * alpha,
                    prev_range[1] * (1 - alpha) + curr_range[1] * alpha,
                ]
            
            # 보상 가중치 보간
            reward_weights = {}
            all_keys = set(prev_level['reward_weights'].keys()) | \
                      set(current_level['reward_weights'].keys())
            for key in all_keys:
                prev_w = prev_level['reward_weights'].get(key, 0.0)
                curr_w = current_level['reward_weights'].get(key, 0.0)
                reward_weights[key] = prev_w * (1 - alpha) + curr_w * alpha
            
            return {
                'target_velocity': target_vel,
                'reward_weights': reward_weights,
            }
        else:
            return current_level
    
    def update_env_config(self, env_cfg, epoch: int):
        """환경 설정을 커리큘럼에 따라 업데이트합니다.
        
        Args:
            env_cfg: 환경 설정 객체
            epoch: 현재 epoch
        """
        config = self.get_current_config(epoch)
        
        # 명령 범위 업데이트
        vel_ranges = config['target_velocity']
        env_cfg.commands.base_velocity.ranges.lin_vel_x = tuple(vel_ranges['lin_vel_x'])
        env_cfg.commands.base_velocity.ranges.lin_vel_y = tuple(vel_ranges['lin_vel_y'])
        env_cfg.commands.base_velocity.ranges.ang_vel_z = tuple(vel_ranges['ang_vel_z'])
        
        # 보상 가중치 업데이트
        reward_weights = config['reward_weights']
        for key, weight in reward_weights.items():
            if hasattr(env_cfg.rewards, key):
                setattr(env_cfg.rewards[key], 'weight', weight)
```

### 3.4 구현 단계

1. **Step 3.1**: 커리큘럼 설정 파일 작성
   - YAML 형식으로 레벨 정의
   - 각 레벨의 명령 범위 및 보상 가중치 설정

2. **Step 3.2**: 커리큘럼 매니저 구현
   - 현재 epoch에 따른 설정 동적 조정
   - 부드러운 레벨 전이 로직

3. **Step 3.3**: 학습 스크립트 통합
   - `train_walking_amp.py`에 커리큘럼 매니저 통합
   - 주기적으로 환경 설정 업데이트

---

## Phase 4: Domain Randomization (Sim-to-Real)

### 목표

시뮬레이션과 실제 환경 간 차이를 줄이기 위한 도메인 랜덤화를 구현합니다.

### 4.1 랜덤화 항목

| 항목 | 범위 | 설명 |
|------|------|------|
| **Link Mass** | ±10% | 각 링크의 질량 변동 |
| **Center of Mass (COM)** | ±5cm | 각 링크의 COM 위치 변동 |
| **Joint Friction** | 0.0 ~ 0.5 | 관절 마찰 계수 |
| **Joint Damping** | 0.0 ~ 0.1 | 관절 댐핑 계수 |
| **Ground Friction** | 0.5 ~ 1.5 | 지면 마찰 계수 |
| **Control Latency** | 0 ~ 50ms | 제어 지연 시간 |
| **Gravity** | 9.6 ~ 9.8 m/s² | 중력 가속도 변동 |
| **Payload** | 0 ~ 5kg | 로봇 상체에 추가 질량 |

### 4.2 설정 파일 (`config/amp/domain_randomization.yaml`)

```yaml
# Domain Randomization Configuration

domain_randomization:
  # 활성화 여부
  enabled: true
  
  # 랜덤화 주기 (에피소드마다 또는 스텝마다)
  frequency: "episode"  # "episode" or "step"
  
  # 랜덤화 항목
  parameters:
    # 링크 질량
    link_mass:
      enabled: true
      distribution: "uniform"  # "uniform" or "normal"
      range: [-0.1, 0.1]  # ±10% 변동
      
    # 중심 질량 위치
    com_position:
      enabled: true
      distribution: "uniform"
      range: [-0.05, 0.05]  # ±5cm (meters)
      
    # 관절 마찰
    joint_friction:
      enabled: true
      distribution: "uniform"
      range: [0.0, 0.5]
      
    # 관절 댐핑
    joint_damping:
      enabled: true
      distribution: "uniform"
      range: [0.0, 0.1]
      
    # 지면 마찰
    ground_friction:
      enabled: true
      distribution: "uniform"
      range: [0.5, 1.5]
      
    # 제어 지연
    control_latency:
      enabled: true
      distribution: "uniform"
      range: [0.0, 0.05]  # 0 ~ 50ms (seconds)
      
    # 중력
    gravity:
      enabled: true
      distribution: "uniform"
      range: [9.6, 9.8]  # m/s²
      
    # 페이로드 (상체 추가 질량)
    payload:
      enabled: true
      distribution: "uniform"
      range: [0.0, 5.0]  # kg
      attachment_link: "torso"  # 질량이 추가될 링크
```

### 4.3 도메인 랜덤화 매니저 구현 (`tasks/walking/amp/domain_randomization.py`)

```python
"""도메인 랜덤화 매니저."""

import torch
import numpy as np
from typing import Dict, Optional
from isaaclab.utils import configclass
import yaml

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
        with open(cfg.config_path, 'r') as f:
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
            
        # ... (나머지 파라미터들)
        
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
        
        # 링크 질량 랜덤화
        if params['link_mass']['enabled']:
            self._randomize_link_mass(env, params['link_mass'])
            
        # COM 위치 랜덤화
        if params['com_position']['enabled']:
            self._randomize_com_position(env, params['com_position'])
            
        # 관절 마찰 랜덤화
        if params['joint_friction']['enabled']:
            self._randomize_joint_friction(env, params['joint_friction'])
            
        # 지면 마찰 랜덤화
        if params['ground_friction']['enabled']:
            self._randomize_ground_friction(env, params['ground_friction'])
            
        # 제어 지연 랜덤화
        if params['control_latency']['enabled']:
            self._randomize_control_latency(env, params['control_latency'])
            
        # 중력 랜덤화
        if params['gravity']['enabled']:
            self._randomize_gravity(env, params['gravity'])
            
        # 페이로드 랜덤화
        if params['payload']['enabled']:
            self._randomize_payload(env, params['payload'])
    
    def _randomize_link_mass(self, env, config: Dict):
        """링크 질량 랜덤화."""
        range_val = config['range']
        
        # 각 환경마다 랜덤 스케일 생성 (±10%)
        scales = torch.rand(self.num_envs, device=self.device)
        scales = scales * (range_val[1] - range_val[0]) + range_val[0]
        
        # 로봇의 각 링크 질량에 적용
        # Isaac Lab의 Articulation API 사용
        for link_name in env.scene.robot.link_names:
            link_idx = env.scene.robot.get_link_index(link_name)
            original_mass = env.scene.robot.get_link_masses(link_idx)
            new_mass = original_mass * (1.0 + scales.unsqueeze(-1))
            env.scene.robot.set_link_masses(link_idx, new_mass)
    
    def _randomize_joint_friction(self, env, config: Dict):
        """관절 마찰 랜덤화."""
        range_val = config['range']
        
        # 각 환경마다 랜덤 마찰 계수 생성
        friction_coeffs = torch.rand(self.num_envs, device=self.device)
        friction_coeffs = friction_coeffs * (range_val[1] - range_val[0]) + range_val[0]
        
        # 관절 마찰 설정
        # Isaac Lab의 Articulation API 사용
        env.scene.robot.set_joint_friction_coefficients(friction_coeffs)
    
    def _randomize_ground_friction(self, env, config: Dict):
        """지면 마찰 랜덤화."""
        range_val = config['range']
        
        # 각 환경마다 랜덤 마찰 계수 생성
        friction_coeffs = torch.rand(self.num_envs, device=self.device)
        friction_coeffs = friction_coeffs * (range_val[1] - range_val[0]) + range_val[0]
        
        # 지면 마찰 설정
        # Isaac Lab의 GroundPlane API 사용
        env.scene.ground.set_friction_coefficients(friction_coeffs)
    
    # ... (나머지 랜덤화 메서드들)
```

### 4.4 환경 설정 통합 (`tasks/walking/amp_env_cfg.py` 업데이트)

```python
"""AMP 환경 설정 (도메인 랜덤화 포함)."""

from .amp.domain_randomization import DomainRandomizationCfg, DomainRandomizationManager

@configclass
class H1AmpEnvCfg(WalkingEnvCfg):
    """H1 AMP 환경 설정."""
    
    # 도메인 랜덤화 설정
    domain_randomization: DomainRandomizationCfg = DomainRandomizationCfg()
    
    # ... (기존 설정들)
```

### 4.5 구현 단계

1. **Step 4.1**: 도메인 랜덤화 설정 파일 작성
   - YAML 형식으로 랜덤화 항목 정의
   - 각 항목의 분포 및 범위 설정

2. **Step 4.2**: 도메인 랜덤화 매니저 구현
   - 각 랜덤화 항목별 메서드 작성
   - Isaac Lab API를 사용한 파라미터 변경

3. **Step 4.3**: 환경에 통합
   - 에피소드 시작 시 랜덤화 적용
   - 학습 스크립트에서 매니저 초기화

---

## 구현 체크리스트

### Phase 1: Mocap Data Preprocessing
- [ ] AMASS 데이터셋 다운로드 및 구조 파악
- [ ] `SMPLLoader` 클래스 구현
- [ ] `H1Skeleton` 클래스 구현 (관절 매핑 정의)
- [ ] `MotionRetargeter` 클래스 구현
  - [ ] 좌표계 변환 로직
  - [ ] 관절 매핑 로직
  - [ ] Inverse Kinematics (Pinocchio 사용)
- [ ] 모션 필터링 및 세그멘테이션 로직
- [ ] `process_amass.py` 스크립트 작성
- [ ] AMP 형식으로 데이터 내보내기
- [ ] 데이터 검증 (형식, 품질)

### Phase 2: AMP Network Architecture
- [ ] `Discriminator` 네트워크 구현
- [ ] `MotionDataset` 클래스 구현
- [ ] AMP 보상 함수 (`amp_rewards.py`) 구현
- [ ] `H1AmpEnvCfg` 환경 설정 작성
- [ ] Gymnasium 환경 등록
- [ ] 기본 학습 파이프라인 테스트

### Phase 3: Curriculum Learning
- [ ] `curriculum_config.yaml` 작성
- [ ] `CurriculumManager` 클래스 구현
- [ ] 학습 스크립트에 커리큘럼 통합
- [ ] 레벨 전이 테스트

### Phase 4: Domain Randomization
- [ ] `domain_randomization.yaml` 작성
- [ ] `DomainRandomizationManager` 클래스 구현
- [ ] 각 랜덤화 항목별 메서드 구현
- [ ] 환경에 통합 및 테스트

### 통합 및 테스트
- [ ] 전체 파이프라인 통합 테스트
- [ ] 학습 실행 및 모니터링
- [ ] 성능 평가 및 하이퍼파라미터 튜닝

---

## 참고 자료

### 논문
- **AMP**: Peng et al., "AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control" (SIGGRAPH 2021)
- **Isaac Lab**: Makoviychuk et al., "Isaac Gym: High Performance GPU-Based Physics Simulation For Robot Learning" (NeurIPS 2021)

### 코드베이스
- Isaac Lab Documentation: https://isaac-sim.github.io/IsaacLab/
- AMASS Dataset: https://amass.is.tue.mpg.de/
- SMPL Model: https://smpl.is.tue.mpg.de/

### 유틸리티
- Pinocchio (IK 계산): https://github.com/stack-of-tasks/pinocchio
- PyTorch: https://pytorch.org/
- RSL-RL: https://github.com/leggedrobotics/rsl_rl

---

**문서 버전**: 1.0  
**최종 수정일**: 2025-01-XX  
**작성자**: AI Robotics Engineer

