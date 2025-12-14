# Isaac Lab H1 기본 보상 함수 분석

이 문서는 Isaac Lab에서 공식으로 제공하는 **검증된 H1 휴머노이드 보행 보상 함수**를 분석한 것입니다.

---

## 📁 소스 파일 위치

```
isaaclab_tasks/manager_based/locomotion/velocity/
├── velocity_env_cfg.py         # 기본 보상 (LocomotionVelocityRoughEnvCfg)
└── config/h1/
    └── rough_env_cfg.py        # H1 전용 보상 (H1RoughEnvCfg)
```

---

## 1. 기본 보상 함수 (`RewardsCfg`)

> **파일**: `velocity_env_cfg.py`  
> **클래스**: `RewardsCfg`  
> **용도**: 모든 locomotion 로봇의 공통 기본 보상

```python
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task (목표 추적)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, 
        weight=1.0, 
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, 
        weight=0.5, 
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    
    # -- penalties (페널티)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.125,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    )
    
    # -- optional penalties (선택적 페널티 - 기본 비활성화)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)
```

### 기본 보상 항목 요약

| 항목 | 함수 | 가중치 | 설명 |
|------|------|--------|------|
| `track_lin_vel_xy_exp` | `mdp.track_lin_vel_xy_exp` | **1.0** | 목표 선속도 추적 (핵심 보상) |
| `track_ang_vel_z_exp` | `mdp.track_ang_vel_z_exp` | **0.5** | 목표 각속도 추적 |
| `lin_vel_z_l2` | `mdp.lin_vel_z_l2` | -2.0 | Z축 속도 페널티 (점프 방지) |
| `ang_vel_xy_l2` | `mdp.ang_vel_xy_l2` | -0.05 | X/Y축 회전 페널티 |
| `dof_torques_l2` | `mdp.joint_torques_l2` | -1.0e-5 | 토크 사용량 페널티 |
| `dof_acc_l2` | `mdp.joint_acc_l2` | -2.5e-7 | 관절 가속도 페널티 |
| `action_rate_l2` | `mdp.action_rate_l2` | -0.01 | 액션 변화율 페널티 |
| `feet_air_time` | `mdp.feet_air_time` | 0.125 | 발 공중 시간 보상 |
| `undesired_contacts` | `mdp.undesired_contacts` | -1.0 | 원치 않는 접촉 페널티 |
| `flat_orientation_l2` | `mdp.flat_orientation_l2` | 0.0 | 수평 자세 페널티 (비활성화) |
| `dof_pos_limits` | `mdp.joint_pos_limits` | 0.0 | 관절 한계 페널티 (비활성화) |

---

## 2. H1 전용 보상 함수 (`H1Rewards`)

> **파일**: `config/h1/rough_env_cfg.py`  
> **클래스**: `H1Rewards(RewardsCfg)`  
> **용도**: H1 휴머노이드 전용 보상 (기본 보상 상속 후 수정)

```python
@configclass
class H1Rewards(RewardsCfg):
    """Reward terms for the MDP."""

    # 종료 페널티 (강력한 페널티!)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    
    # Z축 선속도 페널티 비활성화
    lin_vel_z_l2 = None
    
    # 목표 속도 추적 (yaw frame 기준으로 변경!)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,  # ← 다른 함수 사용!
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,  # ← 다른 함수 사용!
        weight=1.0,  # ← 가중치 증가! (0.5 → 1.0)
        params={"command_name": "base_velocity", "std": 0.5}
    )
    
    # Biped 전용 발 공중 시간 보상
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,  # ← biped 전용 함수!
        weight=0.25,  # ← 가중치 증가! (0.125 → 0.25)
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
            "threshold": 0.4,
        },
    )
    
    # 발 미끄러짐 페널티 (추가!)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_link"),
        },
    )
    
    # 발목 관절 한계 페널티 (추가!)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, 
        weight=-1.0, 
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_ankle")}
    )
    
    # 비필수 관절 기본값 유지 페널티들 (추가!)
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw", ".*_hip_roll"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_.*", ".*_elbow"])},
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1, 
        weight=-0.1, 
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="torso")}
    )
```

### H1 전용 보상 항목 요약

| 항목 | 함수 | 가중치 | 설명 |
|------|------|--------|------|
| `termination_penalty` | `mdp.is_terminated` | **-200.0** | 🔴 종료 시 강력한 페널티 |
| `lin_vel_z_l2` | `None` | - | Z축 속도 페널티 비활성화 |
| `track_lin_vel_xy_exp` | `mdp.track_lin_vel_xy_yaw_frame_exp` | 1.0 | Yaw 프레임 기준 속도 추적 |
| `track_ang_vel_z_exp` | `mdp.track_ang_vel_z_world_exp` | **1.0** | 월드 프레임 각속도 추적 |
| `feet_air_time` | `mdp.feet_air_time_positive_biped` | 0.25 | Biped 전용 발 공중 시간 |
| `feet_slide` | `mdp.feet_slide` | -0.25 | 🆕 발 미끄러짐 페널티 |
| `dof_pos_limits` | `mdp.joint_pos_limits` | -1.0 | 🆕 발목 관절 한계 페널티 |
| `joint_deviation_hip` | `mdp.joint_deviation_l1` | -0.2 | 🆕 엉덩이 관절 기본값 유지 |
| `joint_deviation_arms` | `mdp.joint_deviation_l1` | -0.2 | 🆕 팔 관절 기본값 유지 |
| `joint_deviation_torso` | `mdp.joint_deviation_l1` | -0.1 | 🆕 몸통 관절 기본값 유지 |

---

## 3. 최종 적용 보상 (`H1RoughEnvCfg.__post_init__`)

> H1RoughEnvCfg의 `__post_init__` 메서드에서 추가 조정

```python
@configclass
class H1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: H1Rewards = H1Rewards()

    def __post_init__(self):
        super().__post_init__()
        
        # ... Scene, Randomization 설정 ...
        
        # Rewards 조정
        self.rewards.undesired_contacts = None           # 충돌 페널티 비활성화!
        self.rewards.flat_orientation_l2.weight = -1.0   # 수평 자세 페널티 활성화
        self.rewards.dof_torques_l2.weight = 0.0         # 토크 페널티 비활성화
        self.rewards.action_rate_l2.weight = -0.005      # 액션 변화율 가중치 조정
        self.rewards.dof_acc_l2.weight = -1.25e-7        # 가속도 페널티 유지
```

### 최종 보상 설정 (활성화된 항목만)

| 항목 | 최종 가중치 | 목적 |
|------|-------------|------|
| **목표 추적** | | |
| `track_lin_vel_xy_exp` | 1.0 | 선속도 명령 추적 |
| `track_ang_vel_z_exp` | 1.0 | 각속도 명령 추적 |
| **안전/규제** | | |
| `termination_penalty` | -200.0 | 넘어지면 큰 페널티 |
| `flat_orientation_l2` | -1.0 | 수평 자세 유지 |
| `dof_pos_limits` | -1.0 | 발목 관절 한계 |
| `ang_vel_xy_l2` | -0.05 | X/Y축 회전 억제 |
| **자연스러운 움직임** | | |
| `feet_air_time` | 0.25 | 발 공중 시간 |
| `feet_slide` | -0.25 | 발 미끄러짐 방지 |
| `joint_deviation_hip` | -0.2 | 엉덩이 관절 안정화 |
| `joint_deviation_arms` | -0.2 | 팔 관절 안정화 |
| `joint_deviation_torso` | -0.1 | 몸통 관절 안정화 |
| **효율성** | | |
| `action_rate_l2` | -0.005 | 액션 변화율 제한 |
| `dof_acc_l2` | -1.25e-7 | 관절 가속도 제한 |

### ⚠️ 비활성화된 항목

| 항목 | 상태 | 이유 |
|------|------|------|
| `lin_vel_z_l2` | `None` | 휴머노이드는 점프 가능하므로 비활성화 |
| `undesired_contacts` | `None` | H1 특성상 접촉이 자주 발생하므로 비활성화 |
| `dof_torques_l2` | `0.0` | H1의 토크 특성상 불필요 |

---

## 4. 기본 보상 vs 내 커스텀 보상 비교

| 항목 | Isaac Lab 기본 | 내 커스텀 | 비고 |
|------|----------------|-----------|------|
| `termination_penalty` | **-200.0** | -10.0 | ⚠️ 기본이 훨씬 강함! |
| `joint_pos_limits` | -1.0 (발목만) | -5.0 (전체) | 내 설정이 더 강함 |
| `joint_vel_l2` | ❌ 없음 | -0.001 | 내 설정에만 있음 |
| `undesired_contacts` | `None` (비활성화) | -1.0 | 기본은 비활성화됨! |
| `flat_orientation_l2` | -1.0 | -2.0 | 내 설정이 더 강함 |
| `joint_deviation_*` | -0.1~-0.2 | ❌ 없음 | 기본에만 있음 |
| `feet_slide` | -0.25 | ❌ 없음 | 기본에만 있음 |
| `is_alive` | ❌ 없음 | 0.5 | 내 설정에만 있음 |

---

## 5. 핵심 인사이트

### Isaac Lab H1 설정의 특징

1. **종료 페널티가 매우 강함** (`-200.0`)
   - 넘어지는 것을 극도로 억제
   - 학습 초기에 생존을 최우선시

2. **관절 기본값 유지 보상** (`joint_deviation_*`)
   - 보행에 불필요한 관절(팔, 몸통)은 기본 자세 유지
   - 자연스러운 휴머노이드 자세 학습에 중요

3. **발 미끄러짐 페널티** (`feet_slide`)
   - 발이 땅에 닿았을 때 미끄러지지 않도록 학습
   - 안정적인 보행 패턴 유도

4. **일부 페널티 비활성화**
   - `undesired_contacts`: H1은 접촉이 자주 발생하므로 비활성화
   - `lin_vel_z_l2`: 휴머노이드는 자연스러운 상하 움직임이 있음

### 내 커스텀 설정에 추가 권장

```python
# 1. 종료 페널티 강화 (현재 -10 → -200)
is_terminated = RewTerm(func=mdp.is_terminated, weight=-200.0)

# 2. 관절 기본값 유지 보상 추가
joint_deviation_hip = RewTerm(
    func=mdp.joint_deviation_l1,
    weight=-0.2,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw", ".*_hip_roll"])},
)
joint_deviation_arms = RewTerm(
    func=mdp.joint_deviation_l1,
    weight=-0.2,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_.*", ".*_elbow"])},
)

# 3. 발 미끄러짐 페널티 추가
feet_slide = RewTerm(
    func=mdp.feet_slide,
    weight=-0.25,
    params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
        "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_link"),
    },
)

# 4. undesired_contacts 비활성화 고려
undesired_contacts = None  # 또는 weight를 낮춤
```

---

## 6. 원본 코드 전문

### `rough_env_cfg.py` (H1 전용)

```python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg, 
    RewardsCfg
)
from isaaclab_assets import H1_MINIMAL_CFG


@configclass
class H1Rewards(RewardsCfg):
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    lin_vel_z_l2 = None
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, 
        weight=1.0, 
        params={"command_name": "base_velocity", "std": 0.5}
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_link"),
        },
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, 
        weight=-1.0, 
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_ankle")}
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw", ".*_hip_roll"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_.*", ".*_elbow"])},
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1, 
        weight=-0.1, 
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="torso")}
    )


@configclass
class H1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: H1Rewards = H1Rewards()

    def __post_init__(self):
        super().__post_init__()
        
        # Scene
        self.scene.robot = H1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        if self.scene.height_scanner:
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"

        # Randomization
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = [".*torso_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None

        # Rewards
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.dof_torques_l2.weight = 0.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # Terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = ".*torso_link"
```

---

**작성일**: 2025-12-06  
**소스**: Isaac Lab v2.1.0  
**참조 파일**:
- `isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`
- `isaaclab_tasks/manager_based/locomotion/velocity/config/h1/rough_env_cfg.py`

