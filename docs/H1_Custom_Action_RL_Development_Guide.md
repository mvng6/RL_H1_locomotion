# H1 로봇 커스텀 동작 강화학습 개발 가이드

이 문서는 Isaac Lab의 `train.py`를 활용하여 H1 휴머노이드 로봇의 커스텀 동작(걷기 보행 → 달리기 → 점프)을 단계적으로 학습하는 작업 지시서입니다.

## 목차

1. [개요](#개요)
2. [현재 상황 분석](#현재-상황-분석)
3. [전체 개발 계획](#전체-개발-계획)
4. [단계 1: 기본 보행 (Walking) 환경 구축](#단계-1-기본-보행-walking-환경-구축)
5. [단계 2: 달리기 (Running) 환경 구축](#단계-2-달리기-running-환경-구축)
6. [단계 3: 점프 (Jumping) 환경 구축](#단계-3-점프-jumping-환경-구축)
7. [전이학습 (Transfer Learning) 방법](#전이학습-transfer-learning-방법)
8. [학습 실행 및 모니터링](#학습-실행-및-모니터링)
9. [문제 해결](#문제-해결)
10. [체크리스트](#체크리스트)

---

## 개요

### 프로젝트 목표

H1 휴머노이드 로봇의 다양한 동작을 단계적으로 학습:
1. **기본 보행 (Walking)**: 안정적인 걷기 동작 학습
2. **달리기 (Running)**: 빠른 속도의 달리기 동작 학습 (보행 정책 전이)
3. **점프 (Jumping)**: 수직 점프 동작 학습 (보행/달리기 정책 전이)

### 개발 전략

- **단계적 접근**: 각 동작을 독립적인 환경으로 구현
- **전이학습**: 이전 단계의 학습된 정책을 다음 단계의 초기화에 활용
- **Manager-based 워크플로우**: Isaac Lab의 Manager-based 환경 구조 활용
- **외부 의존성 활용**: Isaac Lab의 `train.py` 스크립트 직접 사용

### 기술 스택

- **프레임워크**: Isaac Lab (Manager-based RL 환경)
- **RL 라이브러리**: RSL-RL (PPO 알고리즘)
- **로봇**: Unitree H1 휴머노이드
- **학습 스크립트**: `scripts/reinforcement_learning/rsl_rl/train.py`

---

## 현재 상황 분석

### 현재 프로젝트 구조

```
RL_project_ws/
├── exts/
│   ├── Example/                    # 예제 코드 및 문서
│   │   ├── docs/                   # 문서 폴더
│   │   └── scripts/                # 예제 스크립트
│   │       └── reinforcement_learning/
│   │           └── rsl_rl/
│   │               ├── train.py    # 학습 스크립트 (외부 의존성)
│   │               └── play.py     # 테스트 스크립트
│   └── h1_locomotion/             # 현재 H1 locomotion 확장
│       ├── tasks/
│       │   └── walking/
│       │       ├── mdp/
│       │       │   └── observations.py  # 관측 공간 정의 (작성 중)
│       │       └── walking_env_cfg.py   # 환경 설정 (미작성)
│       └── config/
│           └── agents/            # 에이전트 설정 (미구현)
```

### 현재 상태

✅ **완료된 작업**:
- H1 locomotion 확장 프로젝트 기본 구조 생성
- 기본 환경 설정 파일 (`env_cfg.py`) 작성 (씬, 액추에이터, 이벤트 설정)
- Extension 설정 완료 (`extension.toml`)
- 관측 공간 정의 (`observations.py`) 작성 중

❌ **미완성 작업**:
- 보상 함수 (Rewards) 정의 미완성
- 종료 조건 (Terminations) 정의 미완성
- 환경 설정 파일 (`walking_env_cfg.py`) 미작성
- 에이전트 설정 (Agent Config) 미구현
- 환경 등록 (Gymnasium register) 미완성
- 다중 태스크 구조 (Walking, Running, Jumping) 미구현

### 필요한 작업

1. **기본 보행 환경 완성**: 관측, 보상, 종료 조건 구현
2. **달리기 환경 추가**: 보행 환경을 기반으로 속도/보상 조정
3. **점프 환경 추가**: 수직 점프 동작을 위한 환경 구현
4. **전이학습 설정**: 각 단계 간 체크포인트 전이 방법 구현

---

## 전체 개발 계획

### 단계별 개발 로드맵

```
Phase 1: 기본 보행 환경 구축
├── 1.1 관측 공간 정의 (observations.py) ✅
├── 1.2 보상 함수 정의 (rewards.py)
├── 1.3 종료 조건 정의 (terminations.py)
├── 1.4 환경 설정 파일 작성 (walking_env_cfg.py)
├── 1.5 에이전트 설정 (agents/rsl_rl_ppo_cfg.py)
├── 1.6 환경 등록 및 테스트
└── 1.7 기본 보행 학습 실행

Phase 2: 달리기 환경 구축 (전이학습)
├── 2.1 달리기 환경 설정 파일 생성
├── 2.2 보상 함수 수정 (속도 가중치 증가)
├── 2.3 보행 정책에서 전이학습 시작
└── 2.4 달리기 학습 실행

Phase 3: 점프 환경 구축 (전이학습)
├── 3.1 점프 환경 설정 파일 생성
├── 3.2 점프 보상 함수 구현
├── 3.3 보행/달리기 정책에서 전이학습 시작
└── 3.4 점프 학습 실행
```

### 최종 프로젝트 구조

```
exts/h1_locomotion/
├── tasks/
│   ├── __init__.py                    # 모든 환경 등록
│   ├── walking/                       # 단계 1: 기본 보행
│   │   ├── __init__.py
│   │   ├── walking_env_cfg.py
│   │   └── mdp/
│   │       ├── observations.py
│   │       ├── rewards.py
│   │       └── terminations.py
│   ├── running/                       # 단계 2: 달리기
│   │   ├── __init__.py
│   │   ├── running_env_cfg.py
│   │   └── mdp/
│   │       ├── observations.py
│   │       ├── rewards.py
│   │       └── terminations.py
│   └── jumping/                       # 단계 3: 점프
│       ├── __init__.py
│       ├── jumping_env_cfg.py
│       └── mdp/
│           ├── observations.py
│           ├── rewards.py
│           └── terminations.py
└── config/
    └── agents/
        ├── walking_ppo_cfg.py         # 보행용 에이전트 설정
        ├── running_ppo_cfg.py         # 달리기용 에이전트 설정
        └── jumping_ppo_cfg.py          # 점프용 에이전트 설정
```

---

## 단계 1: 기본 보행 (Walking) 환경 구축

### 목표

H1 로봇이 안정적으로 걷는 동작을 학습할 수 있는 환경을 구축합니다.

### 작업 1.1: 디렉토리 구조 생성

```bash
cd /home/ldj/RL_project_ws/exts/h1_locomotion/tasks

# Walking 태스크 디렉토리 생성
mkdir -p walking/mdp
touch walking/__init__.py
touch walking/walking_env_cfg.py
touch walking/mdp/__init__.py
touch walking/mdp/observations.py
touch walking/mdp/rewards.py
touch walking/mdp/terminations.py
```

### 작업 1.2: 관측 공간 정의 (`walking/mdp/observations.py`)

**참고**: Isaac Lab의 기존 보행 환경 (`Isaac-Velocity-Rough-H1-v0`)의 관측 공간을 참고하여 작성합니다.

```python
# Copyright (c) 2025, RL Project Workspace
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""관측 공간 정의 - 기본 보행 태스크."""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp


@configclass
class ObservationsCfg:
    """기본 보행을 위한 관측 공간 설정."""

    @configclass
    class PolicyCfg(ObsGroup):
        """정책 네트워크용 관측 그룹."""

        # 관절 상태 (상대 위치 및 속도)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        # 베이스 상태 (자세, 속도)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, params={"normalize": True})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, params={"normalize": True})
        base_yaw_roll_pitch = ObsTerm(func=mdp.base_yaw_roll_pitch)

        # 명령 (목표 속도)
        commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        # 발 접촉 상태
        feet_contact_forces = ObsTerm(
            func=mdp.contact_forces,
            params={
                "sensor_cfg": mdp.SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
                "threshold": 1.0,
            },
        )

        # 관절 액션 히스토리 (선택사항)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            """관측 항목들을 연결하여 하나의 벡터로 만듦."""
            self.concatenate_terms = True

    # 정책 네트워크용 관측 그룹
    policy: PolicyCfg = PolicyCfg()
```

### 작업 1.3: 보상 함수 정의 (`walking/mdp/rewards.py`)

```python
# Copyright (c) 2025, RL Project Workspace
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""보상 함수 정의 - 기본 보행 태스크."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp


@configclass
class RewardsCfg:
    """기본 보행을 위한 보상 함수 설정."""

    # 목표 속도 추적 보상 (가장 중요!)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # 목표 각속도 추적 보상
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # 자세 안정성 보상 (수평 유지)
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-1.0,
    )

    # 발 공중 시간 보상 (보행 리듬)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": mdp.SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
            "threshold": 0.4,
        },
    )

    # 발 미끄러짐 페널티
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.5,
        params={
            "sensor_cfg": mdp.SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
            "asset_cfg": mdp.SceneEntityCfg("robot", body_names=".*ankle_link"),
        },
    )

    # 액션 변화율 페널티 (부드러운 동작)
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )

    # 관절 토크 페널티 (에너지 효율)
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-0.0001,
    )

    # 관절 가속도 페널티 (부드러운 동작)
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.25e-7,
    )
```

### 작업 1.4: 종료 조건 정의 (`walking/mdp/terminations.py`)

```python
# Copyright (c) 2025, RL Project Workspace
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""종료 조건 정의 - 기본 보행 태스크."""

from isaaclab.managers import DoneTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp


@configclass
class TerminationsCfg:
    """기본 보행을 위한 종료 조건 설정."""

    # 시간 초과 (에피소드 길이)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # 로봇 넘어짐 (베이스 접촉)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": mdp.SceneEntityCfg("contact_forces", body_names=".*torso.*"),
            "threshold": 1.0,
        },
    )

    # 로봇 떨어짐 (높이 제한)
    base_height = DoneTerm(
        func=mdp.base_height,
        params={"minimum_height": 0.3, "maximum_height": 2.0},
    )
```

### 작업 1.5: 환경 설정 파일 작성 (`walking/walking_env_cfg.py`)

```python
# Copyright (c) 2025, RL Project Workspace
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""기본 보행 환경 설정 파일."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_assets import H1_MINIMAL_CFG  # isort:skip

from .mdp import ObservationsCfg, RewardsCfg, TerminationsCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp


@configclass
class WalkingSceneCfg(InteractiveSceneCfg):
    """기본 보행을 위한 씬 설정."""

    # 지면 생성
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # 조명 설정
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # H1 로봇 설정
    robot: ArticulationCfg = H1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 접촉 센서 설정 (발 접촉 감지용)
    contact_forces = mdp.ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        body_names=".*ankle_link",
        history_length=1,
    )


@configclass
class WalkingEnvCfg(ManagerBasedRLEnvCfg):
    """기본 보행 강화학습 환경 설정."""

    # 씬 설정
    scene: InteractiveSceneCfg = WalkingSceneCfg(num_envs=4096, env_spacing=2.5)

    # 관측 설정
    observations: ObservationsCfg = ObservationsCfg()

    # 액션 설정 (PD 제어기)
    actions: dict[str, IdealPDActuatorCfg] = {
        ".*_joint": IdealPDActuatorCfg(stiffness=80.0, damping=2.0),
    }

    # 보상 설정
    rewards: RewardsCfg = RewardsCfg()

    # 종료 조건 설정
    terminations: TerminationsCfg = TerminationsCfg()

    # 명령 생성 설정 (속도 명령)
    commands: dict[str, mdp.BaseVelocityCommandCfg] = {
        "base_velocity": mdp.BaseVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            rel_standing_envs=0.02,
            rel_heading_envs=1.0,
            heading_command=True,
            heading_control_stiffness=0.5,
            debug_vis=False,
            ranges=mdp.BaseVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 1.0),  # 전진 속도: 0~1 m/s
                lin_vel_y=(-0.5, 0.5),  # 횡방향 속도: -0.5~0.5 m/s
                ang_vel_z=(-1.0, 1.0),  # 회전 속도: -1~1 rad/s
            ),
        )
    }

    # 이벤트 설정 (리셋 시 랜덤화)
    events: dict = {
        "reset_joints_by_scale": {
            "asset_cfg_name": "robot",
            "func": "isaaclab.utils.assets.reset_joints_by_scale",
            "params": {
                "position_range": (0.5, 1.5),
                "velocity_range": (0.0, 0.0),
            },
        },
        "reset_base": {
            "asset_cfg_name": "robot",
            "func": "isaaclab.utils.assets.reset_root_state_uniform",
            "params": {
                "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-0.0, 0.0)},
                "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            },
        },
    }

    # 에피소드 길이 설정
    episode_length_s = 20.0

    # 시뮬레이션 설정
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(dt=0.005, substeps=1)
```

### 작업 1.6: 에이전트 설정 파일 작성 (`config/agents/walking_ppo_cfg.py`)

```python
# Copyright (c) 2025, RL Project Workspace
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""기본 보행용 PPO 에이전트 설정."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class WalkingPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """기본 보행용 PPO Runner 설정."""

    # 환경 설정
    num_steps_per_env = 24  # 각 환경에서 수집할 스텝 수
    max_iterations = 3000  # 최대 학습 반복 횟수
    save_interval = 50  # 체크포인트 저장 간격

    # 실험 이름 및 로그 설정
    experiment_name = "h1_walking"
    run_name = ""
    seed = 42

    # 정책 네트워크 설정
    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,  # 초기 탐험 노이즈
        actor_hidden_dims=[512, 256, 128],  # Actor 네트워크 구조
        critic_hidden_dims=[512, 256, 128],  # Critic 네트워크 구조
        activation="elu",  # 활성화 함수
    )

    # PPO 알고리즘 설정
    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
```

### 작업 1.7: 환경 등록 (`walking/__init__.py`)

```python
# Copyright (c) 2025, RL Project Workspace
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""기본 보행 태스크 환경 등록."""

import gymnasium as gym

from . import walking_env_cfg
from ..config.agents import walking_ppo_cfg

##
# Register Gym environments.
##

gym.register(
    id="H1-Walking-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": walking_env_cfg.WalkingEnvCfg,
        "rsl_rl_cfg_entry_point": walking_ppo_cfg.WalkingPPORunnerCfg,
    },
)
```

### 작업 1.8: 메인 `__init__.py` 업데이트 (`tasks/__init__.py`)

```python
# Copyright (c) 2025, RL Project Workspace
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""H1 Locomotion 태스크 패키지."""

# Walking 태스크 import (환경 등록)
from . import walking

# Running 태스크 import (추후 추가)
# from . import running

# Jumping 태스크 import (추후 추가)
# from . import jumping
```

### 작업 1.9: 기본 보행 학습 실행

```bash
# 프로젝트 재설치 (변경사항 반영)
cd /home/ldj/RL_project_ws
# Isaac Lab의 Python 환경 사용 (경로는 실제 Isaac Lab 경로로 변경)
/path/to/IsaacLab/isaaclab.sh -p -m pip install -e exts/h1_locomotion --force-reinstall

# 환경 등록 확인
/path/to/IsaacLab/isaaclab.sh -p scripts/environments/list_envs.py | grep H1

# Zero Agent 테스트 (환경 동작 확인)
/path/to/IsaacLab/isaaclab.sh -p scripts/environments/zero_agent.py \
    --task H1-Walking-v0 \
    --num_envs 4

# 기본 보행 학습 시작
/path/to/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task H1-Walking-v0 \
    --num_envs 4096 \
    --max_iterations 3000 \
    --experiment_name h1_walking
```

**예상 학습 시간**: GPU에 따라 다르지만, 일반적으로 3000 iterations에 2-4시간 소요

---

## 단계 2: 달리기 (Running) 환경 구축

### 목표

기본 보행 정책을 전이하여 더 빠른 속도의 달리기 동작을 학습합니다.

### 작업 2.1: 디렉토리 구조 생성

```bash
cd /home/ldj/RL_project_ws/exts/h1_locomotion/tasks

# Running 태스크 디렉토리 생성
mkdir -p running/mdp
touch running/__init__.py
touch running/running_env_cfg.py
touch running/mdp/__init__.py
touch running/mdp/observations.py
touch running/mdp/rewards.py
touch running/mdp/terminations.py
```

### 작업 2.2: 달리기 환경 설정 (`running/running_env_cfg.py`)

**중요**: 기본 보행 환경을 상속받아 속도 범위와 보상 가중치만 수정합니다.

```python
# Copyright (c) 2025, RL Project Workspace
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""달리기 환경 설정 파일."""

from isaaclab.utils import configclass
from ..walking.walking_env_cfg import WalkingEnvCfg, WalkingSceneCfg
from .mdp import ObservationsCfg, RewardsCfg, TerminationsCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp


@configclass
class RunningSceneCfg(WalkingSceneCfg):
    """달리기를 위한 씬 설정 (보행 씬 상속)."""
    pass  # 씬은 동일하므로 변경 없음


@configclass
class RunningEnvCfg(WalkingEnvCfg):
    """달리기 강화학습 환경 설정."""

    # 씬 설정 (보행과 동일)
    scene: RunningSceneCfg = RunningSceneCfg()

    # 관측 설정 (보행과 동일)
    observations: ObservationsCfg = ObservationsCfg()

    # 보상 설정 (속도 가중치 증가)
    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self):
        """환경 설정 초기화."""
        super().__post_init__()

        # 명령 설정 수정: 더 빠른 속도 범위
        self.commands.base_velocity.ranges.lin_vel_x = (1.5, 3.0)  # 전진 속도: 1.5~3.0 m/s
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)  # 횡방향 속도: 동일
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)  # 회전 속도: 동일

        # 보상 가중치 조정: 속도 추적 보상 증가
        self.rewards.track_lin_vel_xy_exp.weight = 2.0  # 보행: 1.0 → 달리기: 2.0
        self.rewards.feet_air_time.weight = 0.8  # 발 공중 시간 보상 증가

        # 에피소드 길이 조정 (달리기는 더 긴 에피소드)
        self.episode_length_s = 30.0
```

### 작업 2.3: 달리기 보상 함수 (`running/mdp/rewards.py`)

보행 보상 함수를 상속받아 속도 관련 가중치만 수정:

```python
# Copyright (c) 2025, RL Project Workspace
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""달리기 보상 함수 정의."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from ..walking.mdp.rewards import RewardsCfg as WalkingRewardsCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp


@configclass
class RewardsCfg(WalkingRewardsCfg):
    """달리기를 위한 보상 함수 설정 (보행 보상 상속)."""

    def __post_init__(self):
        """보상 가중치 초기화."""
        super().__post_init__()

        # 속도 추적 보상 가중치 증가
        self.track_lin_vel_xy_exp.weight = 2.0  # 보행: 1.0

        # 발 공중 시간 보상 증가 (달리기 특성)
        self.feet_air_time.weight = 0.8  # 보행: 0.5

        # 발 미끄러짐 페널티 증가 (고속에서 중요)
        self.feet_slide.weight = -1.0  # 보행: -0.5
```

### 작업 2.4: 관측 및 종료 조건 (보행과 동일)

`running/mdp/observations.py`와 `running/mdp/terminations.py`는 보행과 동일하므로 `walking`에서 import:

```python
# running/mdp/observations.py
from ..walking.mdp.observations import ObservationsCfg

# running/mdp/terminations.py
from ..walking.mdp.terminations import TerminationsCfg
```

### 작업 2.5: 달리기 에이전트 설정 (`config/agents/running_ppo_cfg.py`)

```python
# Copyright (c) 2025, RL Project Workspace
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""달리기용 PPO 에이전트 설정."""

from isaaclab.utils import configclass
from .walking_ppo_cfg import WalkingPPORunnerCfg


@configclass
class RunningPPORunnerCfg(WalkingPPORunnerCfg):
    """달리기용 PPO Runner 설정 (보행 설정 상속)."""

    # 실험 이름 변경
    experiment_name = "h1_running"
    run_name = ""

    # 학습률 조정 (전이학습 시 더 낮은 학습률 사용 가능)
    algorithm: dict = {
        "learning_rate": 5.0e-4,  # 보행: 1.0e-3 → 달리기: 5.0e-4 (더 보수적)
    }
```

### 작업 2.6: 환경 등록 (`running/__init__.py`)

```python
# Copyright (c) 2025, RL Project Workspace
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""달리기 태스크 환경 등록."""

import gymnasium as gym

from . import running_env_cfg
from ..config.agents import running_ppo_cfg

gym.register(
    id="H1-Running-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": running_env_cfg.RunningEnvCfg,
        "rsl_rl_cfg_entry_point": running_ppo_cfg.RunningPPORunnerCfg,
    },
)
```

### 작업 2.7: 전이학습으로 달리기 학습 실행

**중요**: 보행 정책의 체크포인트를 로드하여 달리기 학습을 시작합니다.

```bash
# 1. 보행 학습 완료 후 최종 체크포인트 확인
ls -lh logs/rsl_rl/h1_walking/YYYY-MM-DD_HH-MM-SS/model_*.pt

# 2. 보행 정책에서 전이학습하여 달리기 학습 시작
/path/to/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task H1-Running-v0 \
    --num_envs 4096 \
    --max_iterations 2000 \
    --experiment_name h1_running \
    --resume \
    --load_run <보행_학습_타임스탬프> \
    --checkpoint model_3000.pt
```

**전이학습 효과**: 
- 보행 정책에서 시작하면 달리기 학습이 훨씬 빠르게 수렴합니다
- 예상 학습 시간: 2000 iterations에 1-2시간 (보행에서 시작 시)

---

## 단계 3: 점프 (Jumping) 환경 구축

### 목표

보행/달리기 정책을 전이하여 수직 점프 동작을 학습합니다.

### 작업 3.1: 디렉토리 구조 생성

```bash
cd /home/ldj/RL_project_ws/exts/h1_locomotion/tasks

# Jumping 태스크 디렉토리 생성
mkdir -p jumping/mdp
touch jumping/__init__.py
touch jumping/jumping_env_cfg.py
touch jumping/mdp/__init__.py
touch jumping/mdp/observations.py
touch jumping/mdp/rewards.py
touch jumping/mdp/terminations.py
```

### 작업 3.2: 점프 환경 설정 (`jumping/jumping_env_cfg.py`)

```python
# Copyright (c) 2025, RL Project Workspace
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""점프 환경 설정 파일."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_assets import H1_MINIMAL_CFG  # isort:skip

from .mdp import ObservationsCfg, RewardsCfg, TerminationsCfg


@configclass
class JumpingSceneCfg(InteractiveSceneCfg):
    """점프를 위한 씬 설정."""

    # 지면 생성
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # 조명 설정
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # H1 로봇 설정
    robot: ArticulationCfg = H1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class JumpingEnvCfg(ManagerBasedRLEnvCfg):
    """점프 강화학습 환경 설정."""

    # 씬 설정
    scene: InteractiveSceneCfg = JumpingSceneCfg(num_envs=4096, env_spacing=2.5)

    # 관측 설정
    observations: ObservationsCfg = ObservationsCfg()

    # 액션 설정
    actions: dict[str, IdealPDActuatorCfg] = {
        ".*_joint": IdealPDActuatorCfg(stiffness=80.0, damping=2.0),
    }

    # 보상 설정
    rewards: RewardsCfg = RewardsCfg()

    # 종료 조건 설정
    terminations: TerminationsCfg = TerminationsCfg()

    # 명령 생성 설정 (점프는 수직 속도 명령 없음, 정적 명령)
    commands: dict = {}  # 점프는 명령 없음

    # 이벤트 설정
    events: dict = {
        "reset_joints_by_scale": {
            "asset_cfg_name": "robot",
            "func": "isaaclab.utils.assets.reset_joints_by_scale",
            "params": {
                "position_range": (0.5, 1.5),
                "velocity_range": (0.0, 0.0),
            },
        },
        "reset_base": {
            "asset_cfg_name": "robot",
            "func": "isaaclab.utils.assets.reset_root_state_uniform",
            "params": {
                "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-0.0, 0.0)},
                "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            },
        },
    }

    # 에피소드 길이 설정 (점프는 짧은 에피소드)
    episode_length_s = 5.0

    # 시뮬레이션 설정
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(dt=0.005, substeps=1)
```

### 작업 3.3: 점프 관측 공간 (`jumping/mdp/observations.py`)

```python
# Copyright (c) 2025, RL Project Workspace
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""점프 관측 공간 정의."""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp


@configclass
class ObservationsCfg:
    """점프를 위한 관측 공간 설정."""

    @configclass
    class PolicyCfg(ObsGroup):
        """정책 네트워크용 관측 그룹."""

        # 관절 상태
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        # 베이스 상태 (점프 높이 추적용)
        base_height = ObsTerm(func=mdp.base_height)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, params={"normalize": True})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, params={"normalize": True})
        base_yaw_roll_pitch = ObsTerm(func=mdp.base_yaw_roll_pitch)

        # 발 접촉 상태
        feet_contact_forces = ObsTerm(
            func=mdp.contact_forces,
            params={
                "sensor_cfg": mdp.SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
                "threshold": 1.0,
            },
        )

        # 액션 히스토리
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
```

### 작업 3.4: 점프 보상 함수 (`jumping/mdp/rewards.py`)

```python
# Copyright (c) 2025, RL Project Workspace
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""점프 보상 함수 정의."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
import torch
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp


def jump_height_reward(env, asset_cfg, target_height: float = 0.5) -> torch.Tensor:
    """점프 높이에 대한 보상 함수.

    Args:
        env: 환경 인스턴스
        asset_cfg: 자산 설정
        target_height: 목표 점프 높이 (미터)

    Returns:
        보상 텐서
    """
    root_pos_w = env.scene[asset_cfg.asset_name].data.root_pos_w
    current_height = root_pos_w[:, 2]  # Z 좌표 (높이)

    # 목표 높이 이상일 때 보상 (최대 높이 추적)
    height_reward = torch.clamp(current_height - target_height, min=0.0)

    return height_reward


@configclass
class RewardsCfg:
    """점프를 위한 보상 함수 설정."""

    # 점프 높이 보상 (가장 중요!)
    jump_height = RewTerm(
        func=jump_height_reward,
        weight=2.0,
        params={
            "asset_cfg": mdp.SceneEntityCfg("robot"),
            "target_height": 0.5,  # 목표 점프 높이: 0.5m
        },
    )

    # 자세 안정성 보상
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-1.0,
    )

    # 착지 안정성 보상 (발 접촉 시)
    feet_contact_reward = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.3,
        params={
            "command_name": None,  # 명령 없음
            "sensor_cfg": mdp.SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
            "threshold": 0.1,
        },
    )

    # 액션 변화율 페널티
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )

    # 관절 토크 페널티
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-0.0001,
    )
```

### 작업 3.5: 점프 종료 조건 (`jumping/mdp/terminations.py`)

```python
# Copyright (c) 2025, RL Project Workspace
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""점프 종료 조건 정의."""

from isaaclab.managers import DoneTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp


@configclass
class TerminationsCfg:
    """점프를 위한 종료 조건 설정."""

    # 시간 초과
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # 로봇 넘어짐
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": mdp.SceneEntityCfg("contact_forces", body_names=".*torso.*"),
            "threshold": 1.0,
        },
    )

    # 로봇 떨어짐
    base_height = DoneTerm(
        func=mdp.base_height,
        params={"minimum_height": 0.2, "maximum_height": 2.0},
    )
```

### 작업 3.6: 점프 에이전트 설정 및 환경 등록

점프 에이전트 설정과 환경 등록은 보행/달리기와 유사한 패턴으로 작성합니다.

### 작업 3.7: 전이학습으로 점프 학습 실행

```bash
# 보행 또는 달리기 정책에서 전이학습하여 점프 학습 시작
/path/to/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task H1-Jumping-v0 \
    --num_envs 4096 \
    --max_iterations 2000 \
    --experiment_name h1_jumping \
    --resume \
    --load_run <보행_또는_달리기_학습_타임스탬프> \
    --checkpoint model_3000.pt
```

---

## 전이학습 (Transfer Learning) 방법

### 전이학습 전략

1. **보행 → 달리기**: 보행 정책의 체크포인트를 달리기 학습의 초기화에 사용
2. **보행/달리기 → 점프**: 보행 또는 달리기 정책의 체크포인트를 점프 학습의 초기화에 사용

### 전이학습 실행 방법

#### 방법 1: `--resume` 플래그 사용 (권장)

```bash
# 보행 정책에서 달리기 학습 시작
/path/to/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task H1-Running-v0 \
    --resume \
    --load_run 2025-01-15_10-30-45 \
    --checkpoint model_3000.pt \
    --max_iterations 2000
```

#### 방법 2: 학습률 조정

전이학습 시 더 낮은 학습률을 사용하여 기존 정책을 보존하면서 미세 조정:

```python
# running_ppo_cfg.py에서
algorithm: dict = {
    "learning_rate": 5.0e-4,  # 보행: 1.0e-3 → 달리기: 5.0e-4
}
```

### 전이학습 효과

- **학습 속도 향상**: 처음부터 학습하는 것보다 훨씬 빠르게 수렴
- **안정성 향상**: 기존 정책의 안정적인 동작 패턴을 유지하면서 새로운 동작 학습
- **데이터 효율성**: 더 적은 학습 반복으로 목표 달성

---

## 학습 실행 및 모니터링

### 학습 실행 명령어 요약

```bash
# 1. 기본 보행 학습
/path/to/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task H1-Walking-v0 \
    --num_envs 4096 \
    --max_iterations 3000 \
    --experiment_name h1_walking

# 2. 달리기 학습 (보행 정책 전이)
/path/to/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task H1-Running-v0 \
    --num_envs 4096 \
    --max_iterations 2000 \
    --experiment_name h1_running \
    --resume \
    --load_run <보행_타임스탬프> \
    --checkpoint model_3000.pt

# 3. 점프 학습 (보행/달리기 정책 전이)
/path/to/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task H1-Jumping-v0 \
    --num_envs 4096 \
    --max_iterations 2000 \
    --experiment_name h1_jumping \
    --resume \
    --load_run <보행_또는_달리기_타임스탬프> \
    --checkpoint model_3000.pt
```

### 학습 모니터링

#### TensorBoard 사용

```bash
# TensorBoard 실행
tensorboard --logdir logs/rsl_rl

# 브라우저에서 http://localhost:6006 접속
```

**주요 메트릭**:
- `Mean Episode Reward`: 평균 에피소드 보상
- `Mean Episode Length`: 평균 에피소드 길이
- `Policy Loss`: 정책 손실
- `Value Loss`: 가치 함수 손실

#### 학습된 정책 테스트

```bash
# 보행 정책 테스트
/path/to/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task H1-Walking-v0 \
    --checkpoint logs/rsl_rl/h1_walking/YYYY-MM-DD_HH-MM-SS/model_3000.pt \
    --video

# 달리기 정책 테스트
/path/to/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task H1-Running-v0 \
    --checkpoint logs/rsl_rl/h1_running/YYYY-MM-DD_HH-MM-SS/model_2000.pt \
    --video

# 점프 정책 테스트
/path/to/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task H1-Jumping-v0 \
    --checkpoint logs/rsl_rl/h1_jumping/YYYY-MM-DD_HH-MM-SS/model_2000.pt \
    --video
```

---

## 문제 해결

### 문제 1: 환경이 등록되지 않음

**증상**: `list_envs.py`에서 환경이 보이지 않음

**해결 방법**:
```bash
# 프로젝트 재설치
/path/to/IsaacLab/isaaclab.sh -p -m pip install -e exts/h1_locomotion --force-reinstall

# Python 경로 확인
/path/to/IsaacLab/isaaclab.sh -p -c "import sys; print(sys.path)"
```

### 문제 2: Import 오류

**증상**: `ModuleNotFoundError` 또는 `ImportError`

**해결 방법**:
- `tasks/__init__.py`에서 모든 태스크를 올바르게 import했는지 확인
- 상대 import 경로 확인 (`..`, `.` 사용)
- 프로젝트가 올바르게 설치되었는지 확인

### 문제 3: 전이학습 시 관측 공간 불일치

**증상**: 체크포인트 로드 시 관측 공간 차이로 인한 오류

**해결 방법**:
- 보행/달리기/점프 환경의 관측 공간을 가능한 한 유사하게 유지
- 관측 항목의 순서와 차원이 일치하는지 확인

### 문제 4: 학습이 수렴하지 않음

**해결 방법**:
1. 보상 함수 가중치 조정
2. 학습률 조정 (전이학습 시 더 낮은 학습률 사용)
3. 환경 난이도 조정 (명령 범위, 에피소드 길이 등)
4. 더 많은 학습 반복 횟수 설정

### 문제 5: GPU 메모리 부족

**해결 방법**:
```bash
# 환경 수 감소
--num_envs 2048  # 또는 1024, 512

# 배치 크기 조정 (에이전트 설정 파일에서)
num_steps_per_env = 16  # 기본: 24
```

---

## 체크리스트

### Phase 1: 기본 보행 환경 구축

- [ ] 디렉토리 구조 생성 (`walking/`, `walking/mdp/`)
- [x] 관측 공간 정의 (`observations.py`)
- [ ] 보상 함수 정의 (`rewards.py`)
- [ ] 종료 조건 정의 (`terminations.py`)
- [ ] 환경 설정 파일 작성 (`walking_env_cfg.py`)
- [ ] 에이전트 설정 파일 작성 (`walking_ppo_cfg.py`)
- [ ] 환경 등록 (`walking/__init__.py`)
- [ ] 메인 `__init__.py` 업데이트
- [ ] 프로젝트 재설치
- [ ] 환경 등록 확인 (`list_envs.py`)
- [ ] Zero Agent 테스트 통과
- [ ] 기본 보행 학습 실행
- [ ] 학습 완료 및 체크포인트 저장 확인

### Phase 2: 달리기 환경 구축

- [ ] 디렉토리 구조 생성 (`running/`, `running/mdp/`)
- [ ] 달리기 환경 설정 파일 작성 (`running_env_cfg.py`)
- [ ] 달리기 보상 함수 작성 (`rewards.py`)
- [ ] 달리기 에이전트 설정 작성 (`running_ppo_cfg.py`)
- [ ] 환경 등록 (`running/__init__.py`)
- [ ] 메인 `__init__.py` 업데이트
- [ ] 프로젝트 재설치
- [ ] 보행 정책 체크포인트 확인
- [ ] 전이학습으로 달리기 학습 실행
- [ ] 학습 완료 및 체크포인트 저장 확인

### Phase 3: 점프 환경 구축

- [ ] 디렉토리 구조 생성 (`jumping/`, `jumping/mdp/`)
- [ ] 점프 환경 설정 파일 작성 (`jumping_env_cfg.py`)
- [ ] 점프 관측 공간 정의 (`observations.py`)
- [ ] 점프 보상 함수 작성 (`rewards.py`)
- [ ] 점프 종료 조건 작성 (`terminations.py`)
- [ ] 점프 에이전트 설정 작성 (`jumping_ppo_cfg.py`)
- [ ] 환경 등록 (`jumping/__init__.py`)
- [ ] 메인 `__init__.py` 업데이트
- [ ] 프로젝트 재설치
- [ ] 보행/달리기 정책 체크포인트 확인
- [ ] 전이학습으로 점프 학습 실행
- [ ] 학습 완료 및 체크포인트 저장 확인

### 최종 검증

- [ ] 모든 환경이 올바르게 등록됨
- [ ] 각 태스크의 학습된 정책이 올바르게 동작함
- [ ] 전이학습이 효과적으로 작동함
- [ ] 문서화 완료

---

## 참고 자료

### 관련 문서

- [H1_Locomotion_RL_Guide.md](./H1_Locomotion_RL_Guide.md) - H1 보행 학습 가이드
- [IsaacLab_Codebase_Structure.md](./IsaacLab_Codebase_Structure.md) - Isaac Lab 구조 이해
- [RL_Project_Workspace_Setup_Guide.md](./RL_Project_Workspace_Setup_Guide.md) - 워크스페이스 설정 가이드
- [Scripts_Directory_Structure.md](./Scripts_Directory_Structure.md) - 스크립트 구조 가이드

### Isaac Lab 공식 문서

- [Isaac Lab 공식 문서](https://isaac-sim.github.io/IsaacLab/)
- [Manager-based 워크플로우 가이드](https://isaac-sim.github.io/IsaacLab/main/source/overview/workflows/manager-based.html)
- [환경 생성 가이드](https://isaac-sim.github.io/IsaacLab/main/source/overview/own-project/template.html)

### 예제 코드 위치

- 학습 스크립트: `scripts/reinforcement_learning/rsl_rl/train.py`
- 테스트 스크립트: `scripts/reinforcement_learning/rsl_rl/play.py`
- 환경 예제: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/`

---

**작성일**: 2025-01-15  
**작성자**: AI Assistant  
**버전**: 1.0

