# MDP 모듈 함수 확인을 위한 파일 경로 가이드

이 문서는 MDP 모듈의 다른 함수들을 확인하기 위해 어떤 파일을 열어야 하는지 안내합니다.

## 목차

1. [MDP 모듈 파일 구조](#mdp-모듈-파일-구조)
2. [함수별 확인 파일](#함수별-확인-파일)
3. [파일 경로 찾기 방법](#파일-경로-찾기-방법)
4. [실제 확인 방법](#실제-확인-방법)

---

## MDP 모듈 파일 구조

MDP 모듈은 Isaac Lab 소스 코드 내부에 있으며, 다음과 같은 파일 구조를 가집니다:

```
IsaacLab/
└── source/
    └── isaaclab_tasks/
        └── isaaclab_tasks/
            └── manager_based/
                └── locomotion/
                    └── velocity/
                        └── mdp/                    # MDP 모듈 디렉토리
                            ├── __init__.py         # 모듈 초기화 파일 (모든 함수 export)
                            ├── observations.py     # 관측 함수들
                            ├── rewards.py          # 보상 함수들
                            ├── terminations.py     # 종료 조건 함수들
                            ├── actions.py          # 액션 함수들
                            └── commands.py         # 명령 생성 함수들
```

---

## 함수별 확인 파일

### 1. 관측 관련 추가 함수들

**확인할 파일**: `observations.py`

**파일 경로**:
```
/path/to/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/observations.py
```

**이 파일에서 확인할 수 있는 함수들**:
- `mdp.base_height`: 베이스 높이
- `mdp.projected_gravity`: 중력 벡터의 투영
- `mdp.height_scan`: 지형 높이 스캔 (레이 캐스터 사용)
- `mdp.joint_pos`: 절대 관절 위치
- `mdp.joint_vel`: 절대 관절 속도
- `mdp.joint_pos_rel`: 관절 상대 위치 (이미 사용 중)
- `mdp.joint_vel_rel`: 관절 상대 속도 (이미 사용 중)
- `mdp.base_lin_vel`: 베이스 선속도 (이미 사용 중)
- `mdp.base_ang_vel`: 베이스 각속도 (이미 사용 중)
- `mdp.base_yaw_roll_pitch`: 베이스 자세 (이미 사용 중)
- `mdp.contact_forces`: 접촉 힘 (이미 사용 중)
- `mdp.generated_commands`: 생성된 명령 (이미 사용 중)
- `mdp.last_action`: 마지막 액션 (이미 사용 중)

---

### 2. 보상 관련 함수들

**확인할 파일**: `rewards.py`

**파일 경로**:
```
/path/to/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py
```

**이 파일에서 확인할 수 있는 함수들**:
- `mdp.track_lin_vel_xy_exp`: 선속도 추적 보상 (지수 함수)
- `mdp.track_ang_vel_z_exp`: 각속도 추적 보상 (지수 함수)
- `mdp.flat_orientation_l2`: 수평 자세 유지 보상
- `mdp.feet_air_time_positive_biped`: 발 공중 시간 보상
- `mdp.feet_slide`: 발 미끄러짐 페널티
- `mdp.action_rate_l2`: 액션 변화율 페널티
- `mdp.joint_torques_l2`: 관절 토크 페널티
- `mdp.energy_expenditure`: 에너지 소비 페널티
- `mdp.track_lin_vel_xy_l2`: 선속도 추적 보상 (L2 norm)
- `mdp.track_ang_vel_z_l2`: 각속도 추적 보상 (L2 norm)

**참고**: 이 파일은 `walking/mdp/rewards.py`를 작성할 때 참조해야 합니다.

---

### 3. 종료 조건 관련 함수들

**확인할 파일**: `terminations.py`

**파일 경로**:
```
/path/to/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/terminations.py
```

**이 파일에서 확인할 수 있는 함수들**:
- `mdp.time_out`: 시간 초과
- `mdp.illegal_contact`: 불법 접촉 (로봇 넘어짐)
- `mdp.base_height`: 베이스 높이 제한
- `mdp.robot_heading`: 로봇 방향 제한
- `mdp.undesired_contacts`: 원하지 않는 접촉 감지

**참고**: 이 파일은 `walking/mdp/terminations.py`를 작성할 때 참조해야 합니다.

---

### 4. 액션 관련 함수들

**확인할 파일**: `actions.py`

**파일 경로**:
```
/path/to/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/actions.py
```

**이 파일에서 확인할 수 있는 함수들**:
- `mdp.joint_pos_target`: 관절 위치 타겟 설정
- `mdp.joint_vel_target`: 관절 속도 타겟 설정
- `mdp.delayed_joint_pos`: 지연된 관절 위치

**참고**: 액션은 환경 설정 파일(`walking_env_cfg.py`)에서 정의하므로 직접 사용하지 않을 수 있습니다.

---

### 5. 명령 생성 관련 함수들

**확인할 파일**: `commands.py`

**파일 경로**:
```
/path/to/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/commands.py
```

**이 파일에서 확인할 수 있는 함수들**:
- `mdp.base_velocity_command`: 베이스 속도 명령 생성
- `mdp.terrain_velocity_command`: 지형 기반 속도 명령 생성

**참고**: 명령 생성은 환경 설정 파일(`walking_env_cfg.py`)에서 정의합니다.

---

### 6. 모든 함수 확인 (모듈 초기화 파일)

**확인할 파일**: `__init__.py`

**파일 경로**:
```
/path/to/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/__init__.py
```

**이 파일의 역할**:
- 모든 MDP 함수들을 모듈 레벨로 export
- 어떤 함수들이 사용 가능한지 전체 목록 확인 가능
- 각 함수가 어느 파일에서 import되는지 확인 가능

---

## 파일 경로 찾기 방법

### 방법 1: Python에서 모듈 파일 경로 확인

Isaac Lab Python 환경에서 실행:

```bash
# Isaac Lab Python 환경 실행
/path/to/IsaacLab/isaaclab.sh -p

# Python 인터프리터에서
>>> import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
>>> import inspect
>>> 
>>> # MDP 모듈의 파일 경로 확인
>>> print(inspect.getfile(mdp))
/path/to/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/__init__.py
>>> 
>>> # 특정 함수의 파일 경로 확인
>>> print(inspect.getfile(mdp.track_lin_vel_xy_exp))
/path/to/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py
```

### 방법 2: Isaac Lab 소스 코드 디렉토리에서 직접 찾기

```bash
# Isaac Lab 디렉토리로 이동
cd /path/to/IsaacLab

# MDP 모듈 디렉토리 확인
ls -la source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/

# 특정 파일 내용 확인
cat source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py
```

### 방법 3: IDE에서 확인

IDE(예: VS Code, PyCharm)를 사용하는 경우:

1. **Go to Definition**: 
   - `mdp.track_lin_vel_xy_exp`에서 `Cmd/Ctrl + Click` 또는 `F12`
   - 자동으로 해당 함수가 정의된 파일로 이동

2. **Find in Files**:
   - `track_lin_vel_xy_exp` 검색
   - 모든 사용처와 정의 위치 확인

3. **File Explorer**:
   - Isaac Lab 소스 코드 디렉토리 열기
   - 위 경로 구조대로 파일 탐색

---

## 실제 확인 방법

### 단계별 확인 절차

#### 1단계: 관측 함수 확인

```bash
# observations.py 파일 열기
code /path/to/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/observations.py

# 또는
cat /path/to/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/observations.py
```

**확인할 내용**:
- `base_height` 함수의 구현 방식
- `projected_gravity` 함수의 계산 방법
- `height_scan` 함수의 레이 캐스터 사용법
- 각 함수의 파라미터와 반환값

#### 2단계: 보상 함수 확인

```bash
# rewards.py 파일 열기
code /path/to/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py

# 또는
cat /path/to/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py
```

**확인할 내용**:
- `track_lin_vel_xy_exp` 함수의 지수 함수 사용법
- `flat_orientation_l2` 함수의 수평 자세 계산 방법
- `feet_air_time_positive_biped` 함수의 발 공중 시간 계산
- 각 보상 함수의 가중치 설정 방법

#### 3단계: 종료 조건 함수 확인

```bash
# terminations.py 파일 열기
code /path/to/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/terminations.py

# 또는
cat /path/to/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/terminations.py
```

**확인할 내용**:
- `time_out` 함수의 시간 초과 로직
- `illegal_contact` 함수의 접촉 감지 방법
- `base_height` 함수의 높이 제한 로직

---

## 요약: 확인할 파일 목록

| 함수 카테고리 | 파일명 | 전체 경로 (상대) |
|-------------|--------|-----------------|
| **관측 함수** | `observations.py` | `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/observations.py` |
| **보상 함수** | `rewards.py` | `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py` |
| **종료 조건 함수** | `terminations.py` | `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/terminations.py` |
| **액션 함수** | `actions.py` | `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/actions.py` |
| **명령 함수** | `commands.py` | `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/commands.py` |
| **모듈 초기화** | `__init__.py` | `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/__init__.py` |

**참고**: 전체 경로는 Isaac Lab이 설치된 디렉토리를 `/path/to/IsaacLab`으로 가정한 것입니다. 실제 경로는 Isaac Lab 설치 위치에 따라 다릅니다.

---

## 다음 단계

1. **보상 함수 작성 전**: `rewards.py` 파일을 열어서 사용 가능한 보상 함수들을 확인하세요.
2. **종료 조건 작성 전**: `terminations.py` 파일을 열어서 사용 가능한 종료 조건 함수들을 확인하세요.
3. **함수 파라미터 확인**: 각 함수의 시그니처와 파라미터를 확인하여 올바르게 사용하세요.

---

## 관련 문서

- [01_observations_mdp_module_study.md](./01_observations_mdp_module_study.md) - MDP 모듈 기본 이해
- [H1_Custom_Action_RL_Development_Guide.md](../../../docs/H1_Custom_Action_RL_Development_Guide.md) - 전체 개발 가이드

---

**작성일**: 2025-01-15  
**작성자**: AI Assistant  
**버전**: 1.0  
**관련 파일**: `tasks/walking/mdp/observations.py`, `tasks/walking/mdp/rewards.py`, `tasks/walking/mdp/terminations.py`

