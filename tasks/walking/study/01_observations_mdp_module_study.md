# Observations.py 코드 스터디: MDP 모듈 이해하기

이 문서는 `walking/mdp/observations.py` 파일에서 사용되는 `mdp` 모듈에 대한 상세한 스터디입니다.

## 목차

1. [MDP 모듈이란?](#mdp-모듈이란)
2. [MDP 모듈의 위치](#mdp-모듈의-위치)
3. [MDP 모듈 구조 확인 방법](#mdp-모듈-구조-확인-방법)
4. [코드에서 사용된 MDP 함수들](#코드에서-사용된-mdp-함수들)
5. [각 함수 상세 설명](#각-함수-상세-설명)
6. [추가 학습 자료](#추가-학습-자료)

---

## MDP 모듈이란?

### 정의

**MDP (Markov Decision Process)** 모듈은 Isaac Lab에서 제공하는 **사전 정의된 함수들의 집합**입니다. 이 모듈은 강화학습 환경의 핵심 구성 요소인 관측(Observations), 보상(Rewards), 종료 조건(Terminations), 액션(Actions) 등을 계산하기 위한 유틸리티 함수들을 제공합니다.

### 역할

`mdp` 모듈은 다음과 같은 역할을 합니다:

1. **관측 함수 제공**: 로봇의 상태를 관측 벡터로 변환하는 함수들
2. **보상 함수 제공**: 에이전트의 행동에 대한 보상을 계산하는 함수들
3. **종료 조건 함수 제공**: 에피소드 종료 조건을 확인하는 함수들
4. **액션 함수 제공**: 액션을 시뮬레이션 명령으로 변환하는 함수들

### Import 방식

```python
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
```

이 코드는 Isaac Lab의 `isaaclab_tasks` 패키지 내부에 있는 `manager_based.locomotion.velocity.mdp` 모듈을 `mdp`라는 별칭으로 import합니다.

**왜 `as mdp`로 import하는가?**
- 모듈 이름이 길어서 간단한 별칭 사용
- 코드 가독성 향상 (`mdp.joint_pos_rel` vs `isaaclab_tasks.manager_based.locomotion.velocity.mdp.joint_pos_rel`)

---

## MDP 모듈의 위치

### 실제 파일 위치

`mdp` 모듈은 **Isaac Lab의 소스 코드** 내부에 위치합니다. Isaac Lab이 설치된 디렉토리에서 다음 경로를 확인할 수 있습니다:

```
IsaacLab/
└── source/
    └── isaaclab_tasks/
        └── isaaclab_tasks/
            └── manager_based/
                └── locomotion/
                    └── velocity/
                        └── mdp/                    # MDP 모듈 디렉토리
                            ├── __init__.py         # 모듈 초기화 파일
                            ├── observations.py     # 관측 함수들
                            ├── rewards.py          # 보상 함수들
                            ├── terminations.py     # 종료 조건 함수들
                            ├── actions.py          # 액션 함수들
                            └── commands.py         # 명령 생성 함수들
```

### Python 패키지 경로

Python에서 import할 때의 경로:
```python
isaaclab_tasks.manager_based.locomotion.velocity.mdp
```

이는 다음을 의미합니다:
- `isaaclab_tasks`: 최상위 패키지
- `manager_based`: Manager-based 워크플로우 관련 코드
- `locomotion`: 보행(Locomotion) 관련 코드
- `velocity`: 속도 기반 보행 관련 코드
- `mdp`: MDP 구성 요소 함수들

---

## MDP 모듈 구조 확인 방법

### 방법 1: Python 인터프리터에서 확인

Isaac Lab의 Python 환경에서 직접 확인:

```bash
# Isaac Lab Python 환경 실행
/path/to/IsaacLab/isaaclab.sh -p

# Python 인터프리터에서
>>> import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
>>> dir(mdp)  # mdp 모듈의 모든 속성 확인
```

**출력 예시**:
```
['SceneEntityCfg', 'ContactSensorCfg', 'BaseVelocityCommandCfg', 
 'joint_pos_rel', 'joint_vel_rel', 'base_lin_vel', 'base_ang_vel', 
 'base_height', 'contact_forces', 'generated_commands', 'last_action', 
 ...]
```

### 방법 2: 소스 코드 직접 확인

Isaac Lab 소스 코드가 있는 경우:

```bash
# Isaac Lab 디렉토리로 이동
cd /path/to/IsaacLab

# MDP 모듈 디렉토리 확인
ls -la source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/

# 특정 파일 내용 확인 (예: observations.py)
cat source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/observations.py
```

### 방법 3: IDE에서 확인

IDE(예: VS Code, PyCharm)를 사용하는 경우:

1. **Go to Definition**: `mdp.joint_pos_rel`에서 `Cmd/Ctrl + Click` 또는 `F12`
2. **Find References**: 특정 함수의 사용처 찾기
3. **Auto-completion**: `mdp.` 입력 후 자동완성으로 사용 가능한 함수 확인

### 방법 4: 문서 확인

Isaac Lab 공식 문서에서 확인:
- [Isaac Lab 공식 문서](https://isaac-sim.github.io/IsaacLab/)
- MDP 함수들의 API 문서 참조

---

## 코드에서 사용된 MDP 함수들

`walking/mdp/observations.py`에서 사용된 함수들을 분석해보겠습니다:

```python
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # 1. 관절 상태
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        
        # 2. 베이스 상태
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, params={"normalize": True})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, params={"normalize": True})
        base_yaw_roll_pitch = ObsTerm(func=mdp.base_yaw_roll_pitch)
        
        # 3. 명령
        commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        
        # 4. 발 접촉 상태
        feet_contact_forces = ObsTerm(
            func=mdp.contact_forces,
            params={
                "sensor_cfg": mdp.SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
                "threshold": 1.0,
            },
        )
        
        # 5. 액션 히스토리
        actions = ObsTerm(func=mdp.last_action)
```

### 사용된 함수 목록

1. **`mdp.joint_pos_rel`**: 관절 상대 위치
2. **`mdp.joint_vel_rel`**: 관절 상대 속도
3. **`mdp.base_lin_vel`**: 베이스 선속도
4. **`mdp.base_ang_vel`**: 베이스 각속도
5. **`mdp.base_yaw_roll_pitch`**: 베이스 자세 (Yaw, Roll, Pitch)
6. **`mdp.generated_commands`**: 생성된 명령 (목표 속도)
7. **`mdp.contact_forces`**: 접촉 힘
8. **`mdp.last_action`**: 마지막 액션
9. **`mdp.SceneEntityCfg`**: 씬 엔티티 설정 클래스

---

## 각 함수 상세 설명

### 1. `mdp.joint_pos_rel`

**역할**: 관절의 상대 위치를 계산합니다.

**설명**:
- 로봇의 각 관절의 현재 위치를 기본 위치(neutral position)에 대한 상대값으로 반환
- 정규화된 관절 위치 값 (일반적으로 -1 ~ 1 범위)

**사용 예시**:
```python
joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
```

**반환값**: `torch.Tensor` 형태의 관절 상대 위치 벡터
- Shape: `(num_envs, num_joints)`
- 각 관절의 상대 위치 값

**왜 상대 위치를 사용하는가?**
- 절대 위치보다 상대 위치가 학습에 더 안정적
- 로봇의 초기 자세와 무관하게 동작 패턴 학습 가능

---

### 2. `mdp.joint_vel_rel`

**역할**: 관절의 상대 속도를 계산합니다.

**설명**:
- 로봇의 각 관절의 현재 속도를 정규화된 값으로 반환
- 관절 속도의 크기를 정규화하여 학습 안정성 향상

**사용 예시**:
```python
joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
```

**반환값**: `torch.Tensor` 형태의 관절 상대 속도 벡터
- Shape: `(num_envs, num_joints)`
- 각 관절의 정규화된 속도 값

---

### 3. `mdp.base_lin_vel`

**역할**: 로봇 베이스의 선속도(Linear Velocity)를 계산합니다.

**파라미터**:
- `normalize`: 정규화 여부 (기본값: `False`)

**사용 예시**:
```python
base_lin_vel = ObsTerm(func=mdp.base_lin_vel, params={"normalize": True})
```

**설명**:
- 로봇 베이스의 X, Y, Z 방향 선속도를 반환
- `normalize=True`일 경우 속도를 정규화하여 학습 안정성 향상

**반환값**: `torch.Tensor` 형태의 선속도 벡터
- Shape: `(num_envs, 3)` - [vx, vy, vz]

**보행에서의 중요성**:
- 목표 속도 추적에 필수적인 정보
- 로봇이 얼마나 빠르게 움직이는지 알 수 있음

---

### 4. `mdp.base_ang_vel`

**역할**: 로봇 베이스의 각속도(Angular Velocity)를 계산합니다.

**파라미터**:
- `normalize`: 정규화 여부 (기본값: `False`)

**사용 예시**:
```python
base_ang_vel = ObsTerm(func=mdp.base_ang_vel, params={"normalize": True})
```

**설명**:
- 로봇 베이스의 X, Y, Z 축 회전 속도를 반환
- `normalize=True`일 경우 각속도를 정규화

**반환값**: `torch.Tensor` 형태의 각속도 벡터
- Shape: `(num_envs, 3)` - [ωx, ωy, ωz]

**보행에서의 중요성**:
- 로봇의 회전 속도 추적에 필요
- 방향 전환 동작 학습에 중요

---

### 5. `mdp.base_yaw_roll_pitch`

**역할**: 로봇 베이스의 자세를 Yaw, Roll, Pitch 각도로 반환합니다.

**사용 예시**:
```python
base_yaw_roll_pitch = ObsTerm(func=mdp.base_yaw_roll_pitch)
```

**설명**:
- 로봇 베이스의 방향을 오일러 각도로 표현
- Yaw: 수직축 회전 (방향)
- Roll: 전진축 회전 (좌우 기울기)
- Pitch: 횡축 회전 (앞뒤 기울기)

**반환값**: `torch.Tensor` 형태의 자세 각도 벡터
- Shape: `(num_envs, 3)` - [yaw, roll, pitch]

**보행에서의 중요성**:
- 로봇이 넘어지지 않도록 자세 안정성 유지에 필요
- 수평 자세 유지 학습에 중요

---

### 6. `mdp.generated_commands`

**역할**: 환경에서 생성된 명령(Command)을 반환합니다.

**파라미터**:
- `command_name`: 명령의 이름 (예: `"base_velocity"`)

**사용 예시**:
```python
commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
```

**설명**:
- 환경에서 랜덤하게 생성된 목표 속도 명령을 반환
- 보행 태스크에서는 목표 전진 속도, 횡방향 속도, 회전 속도 등을 포함

**반환값**: `torch.Tensor` 형태의 명령 벡터
- Shape: `(num_envs, command_dim)` - 명령의 차원에 따라 다름
- 예: `(num_envs, 3)` - [목표 전진 속도, 목표 횡방향 속도, 목표 회전 속도]

**보행에서의 중요성**:
- 에이전트가 추적해야 할 목표 속도를 알려줌
- 다양한 속도에서의 보행 학습 가능

---

### 7. `mdp.contact_forces`

**역할**: 로봇의 특정 부위에 가해지는 접촉 힘을 계산합니다.

**파라미터**:
- `sensor_cfg`: 접촉 센서 설정 (`SceneEntityCfg` 객체)
- `threshold`: 접촉 판단 임계값

**사용 예시**:
```python
feet_contact_forces = ObsTerm(
    func=mdp.contact_forces,
    params={
        "sensor_cfg": mdp.SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
        "threshold": 1.0,
    },
)
```

**설명**:
- 로봇의 특정 부위(예: 발목)가 지면과 접촉하는지 확인
- 접촉 힘이 임계값 이상이면 접촉 중으로 판단
- 보행 리듬 파악에 중요

**`mdp.SceneEntityCfg` 설명**:
- 씬 내의 특정 엔티티(로봇 부위, 센서 등)를 참조하는 설정 클래스
- `"contact_forces"`: 접촉 센서의 이름
- `body_names=".*ankle_link"`: 발목 링크를 정규표현식으로 매칭

**반환값**: `torch.Tensor` 형태의 접촉 힘 벡터 또는 접촉 여부
- Shape: 접촉 센서 설정에 따라 다름
- 예: `(num_envs, num_feet, 3)` - 각 발의 접촉 힘 [fx, fy, fz]

**보행에서의 중요성**:
- 발이 지면에 닿았는지 확인하여 보행 리듬 학습
- 착지 타이밍 파악에 필수

---

### 8. `mdp.last_action`

**역할**: 이전 스텝에서 실행된 액션을 반환합니다.

**사용 예시**:
```python
actions = ObsTerm(func=mdp.last_action)
```

**설명**:
- 한 스텝 전에 에이전트가 선택한 액션을 반환
- 액션 히스토리를 관측에 포함하여 학습 안정성 향상

**반환값**: `torch.Tensor` 형태의 액션 벡터
- Shape: `(num_envs, action_dim)` - 액션 공간의 차원에 따라 다름

**보행에서의 중요성**:
- 부드러운 동작 학습에 도움
- 액션 변화율을 제어하여 급격한 움직임 방지

---

## MDP 모듈의 다른 함수들

`observations.py`에서 사용하지 않았지만, MDP 모듈에는 더 많은 함수들이 있습니다:

### 관측 관련 추가 함수들

- `mdp.base_height`: 베이스 높이
- `mdp.projected_gravity`: 중력 벡터의 투영
- `mdp.height_scan`: 지형 높이 스캔 (레이 캐스터 사용)
- `mdp.joint_pos`: 절대 관절 위치
- `mdp.joint_vel`: 절대 관절 속도

### 보상 관련 함수들 (rewards.py에서 사용)

- `mdp.track_lin_vel_xy_exp`: 선속도 추적 보상 (지수 함수)
- `mdp.track_ang_vel_z_exp`: 각속도 추적 보상 (지수 함수)
- `mdp.flat_orientation_l2`: 수평 자세 유지 보상
- `mdp.feet_air_time_positive_biped`: 발 공중 시간 보상
- `mdp.feet_slide`: 발 미끄러짐 페널티
- `mdp.action_rate_l2`: 액션 변화율 페널티
- `mdp.joint_torques_l2`: 관절 토크 페널티

### 종료 조건 관련 함수들 (terminations.py에서 사용)

- `mdp.time_out`: 시간 초과
- `mdp.illegal_contact`: 불법 접촉 (로봇 넘어짐)
- `mdp.base_height`: 베이스 높이 제한

---

## 함수 사용 패턴 이해하기

### 패턴 1: 기본 함수 사용

```python
# 파라미터 없이 사용
joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
```

### 패턴 2: 파라미터와 함께 사용

```python
# params 딕셔너리로 파라미터 전달
base_lin_vel = ObsTerm(
    func=mdp.base_lin_vel, 
    params={"normalize": True}
)
```

### 패턴 3: 복잡한 파라미터 사용

```python
# SceneEntityCfg 객체를 파라미터로 사용
feet_contact_forces = ObsTerm(
    func=mdp.contact_forces,
    params={
        "sensor_cfg": mdp.SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
        "threshold": 1.0,
    },
)
```

### 패턴 4: 명령 이름 지정

```python
# command_name으로 특정 명령 참조
commands = ObsTerm(
    func=mdp.generated_commands, 
    params={"command_name": "base_velocity"}
)
```

---

## MDP 함수 확인 실습

### 실습 1: Python에서 함수 목록 확인

```bash
# Isaac Lab Python 환경 실행
/path/to/IsaacLab/isaaclab.sh -p

# Python 인터프리터에서
>>> import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
>>> 
>>> # 관측 관련 함수들 확인
>>> [attr for attr in dir(mdp) if 'joint' in attr.lower() or 'base' in attr.lower()]
>>> 
>>> # 함수의 도움말 확인
>>> help(mdp.joint_pos_rel)
```

### 실습 2: 함수 시그니처 확인

```python
>>> import inspect
>>> import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
>>> 
>>> # 함수의 시그니처 확인
>>> inspect.signature(mdp.base_lin_vel)
```

### 실습 3: 소스 코드 직접 읽기

Isaac Lab 소스 코드가 있는 경우:

```bash
# 관측 함수들 확인
cat /path/to/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/observations.py

# 보상 함수들 확인
cat /path/to/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py
```

---

## 요약

### 핵심 개념

1. **`mdp`는 무엇인가?**
   - Isaac Lab에서 제공하는 사전 정의된 함수들의 모듈
   - 관측, 보상, 종료 조건 등을 계산하는 유틸리티 함수들

2. **어디에 있는가?**
   - `isaaclab_tasks.manager_based.locomotion.velocity.mdp`
   - Isaac Lab 소스 코드의 `source/isaaclab_tasks/.../mdp/` 디렉토리

3. **어떻게 확인하는가?**
   - Python `dir()` 함수 사용
   - IDE의 Go to Definition 기능
   - 소스 코드 직접 확인
   - 공식 문서 참조

4. **어떻게 사용하는가?**
   - `ObsTerm(func=mdp.function_name, params={...})` 형태로 사용
   - 함수마다 필요한 파라미터가 다름

### 다음 단계

- `rewards.py` 작성 시 보상 관련 MDP 함수들 학습
- `terminations.py` 작성 시 종료 조건 관련 MDP 함수들 학습
- 커스텀 함수 작성 방법 학습

---

## 추가 학습 자료

### 관련 문서

- [H1_Custom_Action_RL_Development_Guide.md](../../../docs/H1_Custom_Action_RL_Development_Guide.md) - 전체 개발 가이드
- [IsaacLab_Codebase_Structure.md](../../../../Example/docs/IsaacLab_Codebase_Structure.md) - Isaac Lab 구조 이해

### Isaac Lab 공식 문서

- [Isaac Lab 공식 문서](https://isaac-sim.github.io/IsaacLab/)
- [Manager-based 워크플로우](https://isaac-sim.github.io/IsaacLab/main/source/overview/workflows/manager-based.html)

### 예제 코드

- Isaac Lab의 기존 보행 환경 코드 참조
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/` 디렉토리

---

**작성일**: 2025-01-15  
**작성자**: AI Assistant  
**버전**: 1.0  
**관련 파일**: `tasks/walking/mdp/observations.py`

