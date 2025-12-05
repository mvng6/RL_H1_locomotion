# H1 커스텀 동작 강화학습 작업 프로세스 체크리스트

이 문서는 `H1_Custom_Action_RL_Development_Guide.md`를 기반으로 한 상세 작업 프로세스 체크리스트입니다. 각 단계를 순차적으로 완료하며 진행 상황을 체크하세요.

## 📊 현재 진행 상황 요약

**전체 진행률**: Phase 1 진행 중 (약 75% 완료 - 코드 작성 완료, 검증 단계)

### ✅ 완료된 작업
- Phase 1.1: 디렉토리 구조 생성 완료
- Phase 1.2: 관측 공간 정의 완료 (`observations.py`)
- Phase 1.3: 보상 함수 정의 완료 (`rewards.py`) - 위상 기반 보상 포함
- Phase 1.4: 종료 조건 정의 완료 (`terminations.py`)
- Phase 1.5: MDP 모듈 초기화 완료 (`mdp/__init__.py`)
- Phase 1.6: 환경 설정 파일 작성 완료 (`walking_env_cfg.py`)
- Phase 1.7: 에이전트 설정 파일 작성 완료 (`config/agents/walking_ppo_cfg.py`)
- Phase 1.8: 환경 등록 완료 (`walking/__init__.py`)
- Phase 1.9: 메인 `__init__.py` 업데이트 완료 (`tasks/__init__.py`)

### ⏳ 진행 중인 작업
- Phase 1.10: 프로젝트 재설치 및 검증 (다음 단계)

### 📝 다음 단계
1. **프로젝트 재설치 및 검증** - 가장 우선순위
2. **Zero Agent 테스트**
3. **기본 보행 학습 실행**
4. **학습 완료 및 체크포인트 확인**

---

## 목차

1. [Phase 1: 기본 보행 (Walking) 환경 구축](#phase-1-기본-보행-walking-환경-구축)
2. [Phase 2: 달리기 (Running) 환경 구축](#phase-2-달리기-running-환경-구축)
3. [Phase 3: 점프 (Jumping) 환경 구축](#phase-3-점프-jumping-환경-구축)
4. [최종 검증 및 테스트](#최종-검증-및-테스트)

---

## Phase 1: 기본 보행 (Walking) 환경 구축

### 1.1 디렉토리 구조 생성

- [ ] 작업 디렉토리로 이동
  ```bash
  cd /home/ldj/RL_project_ws/exts/h1_locomotion/tasks
  ```

- [ ] Walking 태스크 디렉토리 생성 확인
  ```bash
  # 이미 생성되어 있어야 함:
  # walking/
  # walking/mdp/
  ```

- [x] 필요한 파일들이 모두 존재하는지 확인
  ```bash
  ls -la walking/
  ls -la walking/mdp/
  ```
  - [x] `walking/__init__.py` 존재 ✅
  - [x] `walking/walking_env_cfg.py` 존재 ✅
  - [x] `walking/mdp/__init__.py` 존재 ✅
  - [x] `walking/mdp/observations.py` 존재 ✅ (완료됨)
  - [x] `walking/mdp/rewards.py` 존재 ✅ (파일만 생성됨, 내용 작성 필요)
  - [x] `walking/mdp/terminations.py` 존재 ✅ (파일만 생성됨, 내용 작성 필요)

### 1.2 관측 공간 정의 (`walking/mdp/observations.py`)

**상태**: ✅ 완료됨

- [x] 파일 생성 완료
- [x] `ObservationsCfg` 클래스 정의
- [x] `PolicyCfg` 내부 클래스 정의
- [x] 관절 상태 관측 항목 추가 (`joint_pos_rel`, `joint_vel_rel`)
- [x] 베이스 상태 관측 항목 추가 (`base_lin_vel`, `base_ang_vel`, `base_yaw_roll_pitch`)
- [x] 명령 관측 항목 추가 (`commands`)
- [x] 발 접촉 상태 관측 항목 추가 (`feet_contact_forces`)
- [x] 액션 히스토리 관측 항목 추가 (`actions`)
- [x] `concatenate_terms = True` 설정

**검증 사항**:
- [x] 코드에 문법 오류 없음 (IDE에서 확인) ✅
- [x] 모든 import 문이 올바름 ✅
- [x] `@configclass` 데코레이터 사용 ✅
- [x] `__post_init__` 메서드에서 `concatenate_terms = True` 설정 ✅

### 1.3 보상 함수 정의 (`walking/mdp/rewards.py`)

**상태**: ✅ 완료됨

- [x] 파일 생성 완료 ✅
- [x] 기본 구조 작성 ✅
  ```python
  # Copyright (c) 2025, RL Project Workspace
  # All rights reserved.
  #
  # SPDX-License-Identifier: BSD-3-Clause
  
  """보상 함수 정의 - 기본 보행 태스크."""
  
  from isaaclab.managers import RewardTermCfg as RewTerm
  from isaaclab.utils import configclass
  
  import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
  ```

- [x] `RewardsCfg` 클래스 정의 ✅
  - [x] `@configclass` 데코레이터 추가 ✅

- [x] 목표 속도 추적 보상 추가 ✅
  - [x] `track_lin_vel_xy_exp` 보상 항목 ✅
  - [x] 가중치: `1.5` ✅ (상향 조정됨)
  - [x] 파라미터: `command_name="base_velocity"`, `std=0.5` ✅

- [x] 목표 각속도 추적 보상 추가 ✅
  - [x] `track_ang_vel_z_exp` 보상 항목 ✅
  - [x] 가중치: `0.5` ✅
  - [x] 파라미터: `command_name="base_velocity"`, `std=0.5` ✅

- [x] 자세 안정성 보상 추가 ✅
  - [x] `flat_orientation_l2` 보상 항목 ✅
  - [x] 가중치: `-1.0` (페널티) ✅
  
- [x] 베이스 높이 보상 추가 ✅
  - [x] `base_height` 보상 항목 ✅
  - [x] 함수: `mdp.base_height_l2` ✅
  - [x] 가중치: `-1.0` (페널티) ✅
  - [x] 파라미터: `target_height=0.98` ✅

- [x] 위상 기반 보행 보상 추가 ✅ (개선됨)
  - [x] `gait_phase_tracking` 보상 항목 ✅
  - [x] 함수: 커스텀 `gait_phase_tracking` 함수 (접촉 힘 기반) ✅
  - [x] 가중치: `1.0` ✅
  - [x] 파라미터: `command_name="base_velocity"`, `threshold=1.0` ✅
  - [x] 교대 보행 패턴 학습을 위한 접촉 힘 기반 위상 추정 구현 ✅

- [x] 발 공중 시간 보상 추가 ✅
  - [x] `feet_air_time` 보상 항목 ✅
  - [x] 함수: `mdp.feet_air_time_positive_biped` ✅
  - [x] 가중치: `0.5` ✅
  - [x] 파라미터 설정 확인 ✅ (`command_name`, `sensor_cfg`, `threshold=0.4`)

- [x] 발 미끄러짐 페널티 추가 ✅
  - [x] `feet_slide` 보상 항목 ✅
  - [x] 가중치: `-0.5` ✅
  - [x] 파라미터: `sensor_cfg`, `asset_cfg` ✅

- [x] 액션 변화율 페널티 추가 ✅
  - [x] `action_rate_l2` 보상 항목 ✅
  - [x] 가중치: `-0.01` ✅

- [x] 관절 토크 페널티 추가 ✅
  - [x] `dof_torques_l2` 보상 항목 ✅
  - [x] 가중치: `-0.0001` ✅

- [x] 관절 가속도 페널티 추가 ✅
  - [x] `dof_acc_l2` 보상 항목 ✅
  - [x] 가중치: `-2.5e-7` ✅ (떨림 방지를 위해 강화됨)

- [x] 원치 않는 충돌 방지 페널티 추가 ✅
  - [x] `undesired_contacts` 보상 항목 ✅
  - [x] 함수: `mdp.undesired_contacts` ✅
  - [x] 가중치: `-1.0` (페널티) ✅
  - [x] 파라미터: `sensor_cfg` (허벅지, 종아리, 손, 엉덩이) ✅

**검증 사항**:
- [x] 모든 보상 항목이 올바르게 정의됨 ✅
- [x] 가중치 값이 적절함 ✅
- [x] 파라미터 설정이 올바름 ✅
- [x] 코드에 문법 오류 없음 ✅ (Linter 경고는 Isaac Lab 미설치로 인한 것으로 정상)

### 1.4 종료 조건 정의 (`walking/mdp/terminations.py`)

**상태**: ✅ 완료됨

- [x] 파일 생성 완료 ✅
- [x] 기본 구조 작성 ✅
  ```python
  # Copyright (c) 2025, RL Project Workspace
  # All rights reserved.
  #
  # SPDX-License-Identifier: BSD-3-Clause
  
  """종료 조건 정의 - 기본 보행 태스크."""
  
  from isaaclab.managers import DoneTermCfg as DoneTerm
  from isaaclab.utils import configclass
  from isaaclab.managers import SceneEntityCfg
  
  import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
  ```

- [x] `TerminationsCfg` 클래스 정의 ✅
  - [x] `@configclass` 데코레이터 추가 ✅

- [x] 시간 초과 종료 조건 추가 ✅
  - [x] `time_out` 항목 ✅
  - [x] 함수: `mdp.time_out` ✅
  - [x] `time_out=True` 설정 ✅

- [x] 로봇 넘어짐 종료 조건 추가 ✅
  - [x] `base_contact` 항목 ✅
  - [x] 함수: `mdp.illegal_contact` ✅
  - [x] 파라미터: `sensor_cfg` (베이스/토르소 링크), `threshold=1.0` ✅

- [x] 로봇 떨어짐 종료 조건 추가 ✅
  - [x] `base_height` 항목 ✅
  - [x] 함수: `mdp.base_height` ✅
  - [x] 파라미터: `minimum_height=0.3`, `maximum_height=2.0` ✅

**검증 사항**:
- [x] 모든 종료 조건이 올바르게 정의됨 ✅
- [x] 파라미터 값이 적절함 ✅
- [x] 코드에 문법 오류 없음 ✅ (Linter 경고는 Isaac Lab 미설치로 인한 것으로 정상)

### 1.5 MDP 모듈 초기화 (`walking/mdp/__init__.py`)

**상태**: ✅ 완료됨

- [x] 파일 생성 완료 ✅
- [x] MDP 모듈에서 필요한 클래스들을 export ✅
  ```python
  from .observations import ObservationsCfg
  from .rewards import RewardsCfg
  from .terminations import TerminationsCfg
  
  __all__ = ["ObservationsCfg", "RewardsCfg", "TerminationsCfg"]
  ```

**검증 사항**:
- [x] 모든 클래스가 올바르게 import됨 ✅
- [x] `__all__` 리스트에 모든 클래스 포함 ✅
- [x] 코드에 문법 오류 없음 ✅

### 1.6 환경 설정 파일 작성 (`walking/walking_env_cfg.py`)

**상태**: ✅ 완료됨

- [x] 파일이 존재하는지 확인 ✅
- [x] 기본 구조 작성 ✅
  - [x] Copyright 헤더 ✅
  - [x] 필요한 import 문들 ✅

- [x] `WalkingSceneCfg` 클래스 작성 ✅
  - [x] `InteractiveSceneCfg` 상속 ✅
  - [x] `@configclass` 데코레이터 ✅
  - [x] 지면 생성 설정 (`ground`) ✅
  - [x] 조명 설정 (`dome_light`) ✅
  - [x] H1 로봇 설정 (`robot`) ✅
  - [x] 접촉 센서 설정 (`contact_forces`) ✅

- [x] `WalkingEnvCfg` 클래스 작성 ✅
  - [x] `ManagerBasedRLEnvCfg` 상속 ✅
  - [x] `@configclass` 데코레이터 ✅
  - [x] 씬 설정 (`scene`) ✅
  - [x] 관측 설정 (`observations`) ✅
  - [x] 액션 설정 (`actions`) ✅
  - [x] 보상 설정 (`rewards`) ✅
  - [x] 종료 조건 설정 (`terminations`) ✅
  - [x] 명령 생성 설정 (`commands`) ✅ ← **중요**: `"base_velocity"` 이름으로 정의됨
  - [x] 이벤트 설정 (`events`) ✅
  - [x] 에피소드 길이 설정 (`episode_length_s=20.0`) ✅
  - [x] 시뮬레이션 설정 (`sim`) ✅

**검증 사항**:
- [x] 모든 설정이 올바르게 정의됨 ✅
- [x] 명령 범위가 적절함 (`lin_vel_x=(0.0, 1.0)`) ✅
- [x] 에피소드 길이가 적절함 (`20.0` 초) ✅
- [x] `commands` 딕셔너리에 `"base_velocity"` 키가 있음 (observations.py와 일치) ✅
- [x] 코드에 문법 오류 없음 ✅

### 1.7 에이전트 설정 파일 작성 (`config/agents/walking_ppo_cfg.py`)

**상태**: ✅ 완료됨

- [x] 디렉토리 확인 ✅
  ```bash
  ls -la config/agents/
  ```
  - [x] `config/agents/__init__.py` 존재 ✅

- [x] 파일 생성 및 기본 구조 작성 ✅
  ```python
  # Copyright (c) 2025, RL Project Workspace
  # All rights reserved.
  #
  # SPDX-License-Identifier: BSD-3-Clause
  
  """기본 보행용 PPO 에이전트 설정."""
  
  from isaaclab.utils import configclass
  from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
  ```

- [x] `WalkingPPORunnerCfg` 클래스 정의 ✅
  - [x] `RslRlOnPolicyRunnerCfg` 상속 ✅
  - [x] `@configclass` 데코레이터 ✅

- [x] 환경 설정 파라미터 추가 ✅
  - [x] `num_steps_per_env = 24` ✅
  - [x] `max_iterations = 3000` ✅
  - [x] `save_interval = 50` ✅

- [x] 실험 설정 파라미터 추가 ✅
  - [x] `experiment_name = "h1_walking"` ✅
  - [x] `run_name = ""` ✅
  - [x] `seed = 42` ✅

- [x] 정책 네트워크 설정 추가 ✅
  - [x] `policy: RslRlPpoActorCriticCfg` 설정 ✅
  - [x] `init_noise_std=1.0` ✅
  - [x] `actor_hidden_dims=[512, 256, 128]` ✅
  - [x] `critic_hidden_dims=[512, 256, 128]` ✅
  - [x] `activation="elu"` ✅

- [x] PPO 알고리즘 설정 추가 ✅
  - [x] `algorithm: RslRlPpoAlgorithmCfg` 설정 ✅
  - [x] 모든 하이퍼파라미터 설정 확인 ✅
    - [x] `value_loss_coef=1.0` ✅
    - [x] `use_clipped_value_loss=True` ✅
    - [x] `clip_param=0.2` ✅
    - [x] `entropy_coef=0.01` ✅
    - [x] `num_learning_epochs=5` ✅
    - [x] `num_mini_batches=4` ✅
    - [x] `learning_rate=1.0e-3` ✅
    - [x] `schedule="adaptive"` ✅
    - [x] `gamma=0.99` ✅
    - [x] `lam=0.95` ✅
    - [x] `desired_kl=0.01` ✅
    - [x] `max_grad_norm=1.0` ✅

- [x] `config/agents/__init__.py`에 export 추가 ✅

**검증 사항**:
- [x] 모든 설정이 올바르게 정의됨 ✅
- [x] 하이퍼파라미터 값이 적절함 ✅
- [x] 코드에 문법 오류 없음 ✅

### 1.8 환경 등록 (`walking/__init__.py`)

**상태**: ✅ 완료됨

- [x] 파일이 존재하는지 확인 ✅
- [x] 기본 구조 작성 ✅
  - [x] Copyright 헤더 ✅
  - [x] `gymnasium as gym` import ✅

- [x] 환경 설정 및 에이전트 설정 import ✅
  ```python
  from . import walking_env_cfg
  from ..config.agents import walking_ppo_cfg
  ```

- [x] Gymnasium 환경 등록 ✅
  ```python
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

**검증 사항**:
- [x] 환경 ID가 올바름 (`H1-Walking-v0`) ✅
- [x] Entry point가 올바름 ✅
- [x] Config entry point가 올바름 ✅
- [x] 코드에 문법 오류 없음 ✅

### 1.9 메인 `__init__.py` 업데이트 (`tasks/__init__.py`)

**상태**: ✅ 완료됨

- [x] 파일 확인 ✅
  ```bash
  cat tasks/__init__.py
  ```
  - [x] 기본 구조 존재 (Copyright 헤더, 주석) ✅

- [x] Walking 태스크 import 추가 ✅
  ```python
  from . import walking
  ```

- [x] 다른 태스크 import는 주석 처리 (아직 구현 전) ✅
  ```python
  # from . import running
  # from . import jumping
  ```

**검증 사항**:
- [x] Walking 태스크가 올바르게 import됨 ✅
- [x] 다른 태스크는 주석 처리됨 ✅
- [x] 코드에 문법 오류 없음 ✅

### 1.10 프로젝트 재설치 및 검증

**상태**: ⏳ 다음 단계

**중요**: Isaac Lab이 외부 의존성으로 사용되는 경우, Isaac Lab의 Python 환경을 사용하여 설치해야 합니다.

#### 방법 1: Isaac Lab의 `isaaclab.sh` 사용 (권장)

- [ ] Isaac Lab 경로 확인
  ```bash
  # Isaac Lab이 설치된 경로 확인
  # 예: /home/ldj/IsaacLab 또는 /path/to/IsaacLab
  echo $ISAAC_LAB_PATH  # 환경 변수가 설정되어 있는지 확인
  ```

- [ ] 프로젝트 디렉토리로 이동
  ```bash
  cd /home/ldj/RL_project_ws
  ```

- [ ] 프로젝트 재설치
  ```bash
  # 방법 1-A: Isaac Lab 경로를 직접 지정
  /path/to/IsaacLab/isaaclab.sh -p -m pip install -e exts/h1_locomotion --force-reinstall
  
  # 방법 1-B: 환경 변수 사용 (ISAAC_LAB_PATH가 설정된 경우)
  $ISAAC_LAB_PATH/isaaclab.sh -p -m pip install -e exts/h1_locomotion --force-reinstall
  
  # 방법 1-C: 상대 경로 사용 (현재 디렉토리가 Isaac Lab인 경우)
  # cd /path/to/IsaacLab
  # ./isaaclab.sh -p -m pip install -e /home/ldj/RL_project_ws/exts/h1_locomotion --force-reinstall
  ```

#### 방법 2: PYTHONPATH 설정 후 일반 pip 사용

- [ ] Isaac Lab의 Python 경로 확인
  ```bash
  # Isaac Lab의 Python 경로 확인
  /path/to/IsaacLab/isaaclab.sh -p -c "import sys; print(sys.executable)"
  ```

- [ ] PYTHONPATH 설정
  ```bash
  # Isaac Lab의 Python 경로를 PYTHONPATH에 추가
  export PYTHONPATH="/path/to/IsaacLab/source:$PYTHONPATH"
  ```

- [ ] 프로젝트 설치
  ```bash
  cd /home/ldj/RL_project_ws/exts/h1_locomotion
  /path/to/IsaacLab/isaaclab.sh -p -m pip install -e . --force-reinstall
  ```

#### 설치 성공 확인

- [x] 설치 성공 확인 ✅
  ```bash
  # 설치 후 출력 확인
  # "Successfully installed h1-locomotion" 메시지가 나타남 ✅
  # 패키지가 설치되었지만, conda 환경에서 테스트 필요
  ```

**⚠️ 중요**: 일반 Python 환경이 아닌 **conda 환경(`env_isaaclab`) 또는 Isaac Lab의 Python 환경**에서 테스트해야 합니다.

- [ ] 환경 등록 확인
  ```bash
  # 방법 1: Isaac Lab의 list_envs.py 사용
  /path/to/IsaacLab/isaaclab.sh -p scripts/environments/list_envs.py | grep H1
  
  # 방법 2: Python에서 직접 확인
  /path/to/IsaacLab/isaaclab.sh -p -c "import gymnasium as gym; print([env for env in gym.envs.registry.env_specs.keys() if 'H1' in env])"
  ```
  - [ ] `H1-Walking-v0` 환경이 목록에 나타남

- [ ] Import 테스트
  ```bash
  # Python에서 직접 import 테스트
  /path/to/IsaacLab/isaaclab.sh -p -c "
  from h1_locomotion.tasks.walking import walking_env_cfg
  from h1_locomotion.config.agents import walking_ppo_cfg
  import gymnasium as gym
  print('Import 성공!')
  print('환경 등록 확인:', 'H1-Walking-v0' in gym.envs.registry.env_specs)
  "
  ```

**검증 사항**:
- [ ] 프로젝트가 올바르게 설치됨
- [ ] 환경이 올바르게 등록됨 (`H1-Walking-v0` 확인)
- [ ] Import 오류 없음
- [ ] 모든 모듈이 올바르게 로드됨

### 1.11 Zero Agent 테스트

- [ ] Zero Agent 테스트 실행
  ```bash
  /path/to/IsaacLab/isaaclab.sh -p scripts/environments/zero_agent.py \
      --task H1-Walking-v0 \
      --num_envs 4
  ```

- [ ] 테스트 결과 확인
  - [ ] 환경이 정상적으로 로드됨
  - [ ] 시뮬레이션이 실행됨
  - [ ] 에러 없이 종료됨

**검증 사항**:
- [ ] 환경이 올바르게 작동함
- [ ] 씬이 올바르게 생성됨
- [ ] 로봇이 올바르게 스폰됨

### 1.12 기본 보행 학습 실행

- [ ] 학습 디렉토리 확인
  ```bash
  mkdir -p logs/rsl_rl
  ```

- [ ] 학습 명령어 준비
  ```bash
  /path/to/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
      --task H1-Walking-v0 \
      --num_envs 4096 \
      --max_iterations 3000 \
      --experiment_name h1_walking
  ```

- [ ] GPU 메모리 확인
  ```bash
  nvidia-smi
  ```
  - [ ] 충분한 GPU 메모리 확인 (최소 8GB 권장)

- [ ] 학습 시작
  - [ ] 학습이 정상적으로 시작됨
  - [ ] 로그가 올바르게 출력됨
  - [ ] 체크포인트가 저장됨

- [ ] 학습 모니터링
  - [ ] TensorBoard 실행 (선택사항)
  - [ ] 학습 진행 상황 확인
  - [ ] 보상이 증가하는지 확인

**검증 사항**:
- [ ] 학습이 정상적으로 진행됨
- [ ] 체크포인트가 주기적으로 저장됨
- [ ] 학습이 완료됨 (3000 iterations)

### 1.13 학습 완료 및 체크포인트 확인

- [ ] 학습 완료 확인
  - [ ] 최종 iteration까지 학습 완료
  - [ ] 에러 없이 종료됨

- [ ] 체크포인트 파일 확인
  ```bash
  ls -lh logs/rsl_rl/h1_walking/YYYY-MM-DD_HH-MM-SS/model_*.pt
  ```
  - [ ] `model_0.pt` 존재
  - [ ] `model_50.pt` 존재 (50 iteration마다 저장)
  - [ ] `model_3000.pt` 존재 (최종 모델)

- [ ] 학습 로그 확인
  ```bash
  ls -lh logs/rsl_rl/h1_walking/YYYY-MM-DD_HH-MM-SS/
  ```
  - [ ] `progress.csv` 존재
  - [ ] `params/env.yaml` 존재
  - [ ] `params/agent.yaml` 존재

**검증 사항**:
- [ ] 모든 체크포인트가 올바르게 저장됨
- [ ] 학습 로그가 올바르게 기록됨
- [ ] 최종 모델이 존재함

### 1.14 학습된 정책 테스트

- [ ] Play 스크립트로 테스트
  ```bash
  /path/to/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
      --task H1-Walking-v0 \
      --checkpoint logs/rsl_rl/h1_walking/YYYY-MM-DD_HH-MM-SS/model_3000.pt \
      --video
  ```

- [ ] 테스트 결과 확인
  - [ ] 로봇이 안정적으로 걷는지 확인
  - [ ] 비디오가 생성됨 (선택사항)
  - [ ] 에러 없이 실행됨

**검증 사항**:
- [ ] 학습된 정책이 올바르게 동작함
- [ ] 로봇이 목표 속도를 추적함
- [ ] 안정적인 보행 패턴 확인

---

## Phase 2: 달리기 (Running) 환경 구축

### 2.1 디렉토리 구조 생성

- [ ] Running 태스크 디렉토리 생성
  ```bash
  cd /home/ldj/RL_project_ws/exts/h1_locomotion/tasks
  mkdir -p running/mdp
  touch running/__init__.py
  touch running/running_env_cfg.py
  touch running/mdp/__init__.py
  touch running/mdp/observations.py
  touch running/mdp/rewards.py
  touch running/mdp/terminations.py
  ```

### 2.2 달리기 환경 설정 파일 작성

- [ ] `running/running_env_cfg.py` 작성
  - [ ] `WalkingEnvCfg` 상속
  - [ ] 속도 범위 수정 (`lin_vel_x=(1.5, 3.0)`)
  - [ ] 보상 가중치 조정
  - [ ] 에피소드 길이 조정 (`30.0` 초)

### 2.3 달리기 보상 함수 작성

- [ ] `running/mdp/rewards.py` 작성
  - [ ] `WalkingRewardsCfg` 상속
  - [ ] `__post_init__`에서 가중치 조정
  - [ ] 속도 추적 보상 가중치 증가 (`2.0`)
  - [ ] 발 공중 시간 보상 증가 (`0.8`)

### 2.4 관측 및 종료 조건 (보행과 동일)

- [ ] `running/mdp/observations.py` 작성
  - [ ] `walking.mdp.observations`에서 import

- [ ] `running/mdp/terminations.py` 작성
  - [ ] `walking.mdp.terminations`에서 import

### 2.5 달리기 에이전트 설정 작성

- [ ] `config/agents/running_ppo_cfg.py` 작성
  - [ ] `WalkingPPORunnerCfg` 상속
  - [ ] `experiment_name = "h1_running"`
  - [ ] 학습률 조정 (`5.0e-4`)

### 2.6 환경 등록

- [ ] `running/__init__.py` 작성
  - [ ] Gymnasium 환경 등록 (`H1-Running-v0`)

### 2.7 메인 `__init__.py` 업데이트

- [ ] `tasks/__init__.py` 업데이트
  - [ ] Running 태스크 import 추가

### 2.8 프로젝트 재설치 및 검증

- [ ] 프로젝트 재설치
- [ ] 환경 등록 확인 (`H1-Running-v0`)
- [ ] Zero Agent 테스트

### 2.9 전이학습으로 달리기 학습 실행

- [ ] 보행 정책 체크포인트 경로 확인
- [ ] 전이학습 명령어 실행
  ```bash
  --resume \
  --load_run <보행_타임스탬프> \
  --checkpoint model_3000.pt
  ```
- [ ] 학습 완료 및 체크포인트 확인

---

## Phase 3: 점프 (Jumping) 환경 구축

### 3.1 디렉토리 구조 생성

- [ ] Jumping 태스크 디렉토리 생성
  ```bash
  mkdir -p jumping/mdp
  ```

### 3.2 점프 환경 설정 파일 작성

- [ ] `jumping/jumping_env_cfg.py` 작성
  - [ ] 명령 없음 (`commands: dict = {}`)
  - [ ] 짧은 에피소드 길이 (`5.0` 초)

### 3.3 점프 관측 공간 정의

- [ ] `jumping/mdp/observations.py` 작성
  - [ ] 베이스 높이 관측 추가
  - [ ] 명령 관측 제거

### 3.4 점프 보상 함수 작성

- [ ] `jumping/mdp/rewards.py` 작성
  - [ ] 점프 높이 보상 함수 구현
  - [ ] 목표 높이: `0.5m`

### 3.5 점프 종료 조건 작성

- [ ] `jumping/mdp/terminations.py` 작성
  - [ ] 보행과 유사한 종료 조건

### 3.6 점프 에이전트 설정 및 환경 등록

- [ ] `config/agents/jumping_ppo_cfg.py` 작성
- [ ] `jumping/__init__.py` 작성
- [ ] `tasks/__init__.py` 업데이트

### 3.7 전이학습으로 점프 학습 실행

- [ ] 보행/달리기 정책 체크포인트 확인
- [ ] 전이학습 명령어 실행
- [ ] 학습 완료 및 체크포인트 확인

---

## 최종 검증 및 테스트

### 모든 환경 등록 확인

- [ ] 모든 환경이 올바르게 등록됨
  ```bash
  /path/to/IsaacLab/isaaclab.sh -p scripts/environments/list_envs.py | grep H1
  ```
  - [ ] `H1-Walking-v0` 존재
  - [ ] `H1-Running-v0` 존재
  - [ ] `H1-Jumping-v0` 존재

### 각 태스크 정책 테스트

- [ ] 보행 정책 테스트
- [ ] 달리기 정책 테스트
- [ ] 점프 정책 테스트

### 전이학습 효과 검증

- [ ] 달리기 학습이 보행에서 시작했을 때 더 빠르게 수렴하는지 확인
- [ ] 점프 학습이 보행/달리기에서 시작했을 때 더 빠르게 수렴하는지 확인

### 문서화 완료

- [ ] 코드 주석 작성 완료
- [ ] README 파일 업데이트 (선택사항)
- [ ] 학습 결과 정리 (선택사항)

---

## 진행 상황 추적

### 현재 진행 단계

- **Phase 1**: 기본 보행 환경 구축
  - [x] 1.1 디렉토리 구조 생성 ✅
  - [x] 1.2 관측 공간 정의 ✅ (완료됨)
  - [x] 1.3 보상 함수 정의 ✅ (완료됨 - 위상 기반 보상 포함)
  - [x] 1.4 종료 조건 정의 ✅ (완료됨)
  - [x] 1.5 MDP 모듈 초기화 ✅ (완료됨)
  - [x] 1.6 환경 설정 파일 작성 ✅ (완료됨)
  - [x] 1.7 에이전트 설정 파일 작성 ✅ (완료됨)
  - [x] 1.8 환경 등록 ✅ (완료됨)
  - [x] 1.9 메인 `__init__.py` 업데이트 ✅ (완료됨)
  - [ ] 1.10 프로젝트 재설치 및 검증
  - [ ] 1.11 Zero Agent 테스트
  - [ ] 1.12 기본 보행 학습 실행
  - [ ] 1.13 학습 완료 및 체크포인트 확인
  - [ ] 1.14 학습된 정책 테스트

- **Phase 2**: 달리기 환경 구축
  - [ ] 아직 시작하지 않음

- **Phase 3**: 점프 환경 구축
  - [ ] 아직 시작하지 않음

---

## 참고 사항

### 각 단계 완료 후 확인할 사항

1. **코드 작성 후**:
   - [ ] 문법 오류 없음
   - [ ] Import 경로 올바름
   - [ ] 설정 값이 적절함

2. **파일 생성 후**:
   - [ ] 파일이 올바른 위치에 있음
   - [ ] 파일 권한이 올바름

3. **환경 등록 후**:
   - [ ] 환경이 목록에 나타남
   - [ ] Import 오류 없음

4. **학습 시작 전**:
   - [ ] GPU 메모리 충분함
   - [ ] 디스크 공간 충분함
   - [ ] 학습 명령어가 올바름

5. **학습 완료 후**:
   - [ ] 체크포인트 파일 존재
   - [ ] 학습 로그 존재
   - [ ] 정책이 올바르게 동작함

### 문제 발생 시 확인 사항

- [ ] 에러 메시지 확인
- [ ] 로그 파일 확인
- [ ] 관련 문서 참조
- [ ] 예제 코드와 비교

---

**작성일**: 2025-01-15  
**작성자**: AI Assistant  
**버전**: 1.0  
**기반 문서**: `H1_Custom_Action_RL_Development_Guide.md`

