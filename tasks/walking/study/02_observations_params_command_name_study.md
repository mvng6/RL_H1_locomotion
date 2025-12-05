# Observations.py 코드 스터디: params와 command_name 이해하기

이 문서는 `walking/mdp/observations.py`의 27번째 줄에서 사용되는 `params`와 `command_name`에 대한 상세한 설명입니다.

## 목차

1. [params의 의미와 역할](#params의-의미와-역할)
2. [command_name의 의미와 역할](#command_name의-의미와-역할)
3. [command_name 커스터마이징 가능 여부](#command_name-커스터마이징-가능-여부)
4. [실제 동작 원리](#실제-동작-원리)
5. [다른 params 사용 예시](#다른-params-사용-예시)
6. [주의사항 및 베스트 프랙티스](#주의사항-및-베스트-프랙티스)

---

## params의 의미와 역할

### 기본 개념

**`params`**는 `ObsTerm` (Observation Term)에 전달되는 **추가 파라미터 딕셔너리**입니다. 이 파라미터들은 MDP 함수(`func`)가 관측 값을 계산할 때 필요한 추가 정보를 제공합니다.

### 코드 구조

```python
commands = ObsTerm(
    func=mdp.generated_commands,           # 관측을 계산할 함수
    params={"command_name": "base_velocity"}  # 함수에 전달할 파라미터
)
```

### params의 역할

1. **함수에 추가 정보 전달**: MDP 함수가 관측을 계산하는 데 필요한 설정값 전달
2. **함수 동작 커스터마이징**: 같은 함수를 다른 파라미터로 사용하여 다양한 관측 생성
3. **환경 설정과의 연결**: 환경 설정에서 정의한 요소들을 참조

### params 전달 방식

`params`는 딕셔너리 형태로 전달되며, MDP 함수 내부에서 다음과 같이 사용됩니다:

```python
# MDP 함수 내부 (의사 코드)
def generated_commands(env, command_name: str):
    # command_name을 사용하여 환경에서 해당 명령을 찾음
    command = env.command_manager[command_name]
    return command.current_value
```

---

## command_name의 의미와 역할

### 기본 개념

**`command_name`**은 환경 설정 파일(`walking_env_cfg.py`)에서 정의한 **명령(Command)의 이름**을 지정하는 문자열입니다.

### 명령(Command)이란?

강화학습 환경에서 **명령(Command)**은 에이전트가 추적해야 할 목표값을 의미합니다:

- **보행 태스크**: 목표 속도 (전진 속도, 횡방향 속도, 회전 속도)
- **조작 태스크**: 목표 위치, 목표 자세
- **내비게이션 태스크**: 목표 위치

### 환경 설정에서의 명령 정의

`walking_env_cfg.py` 파일에서 명령이 다음과 같이 정의됩니다:

```python
@configclass
class WalkingEnvCfg(ManagerBasedRLEnvCfg):
    # 명령 생성 설정 (속도 명령)
    commands: dict[str, mdp.BaseVelocityCommandCfg] = {
        "base_velocity": mdp.BaseVelocityCommandCfg(  # ← 이 이름이 중요!
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            ranges=mdp.BaseVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 1.0),  # 전진 속도: 0~1 m/s
                lin_vel_y=(-0.5, 0.5),  # 횡방향 속도: -0.5~0.5 m/s
                ang_vel_z=(-1.0, 1.0),  # 회전 속도: -1~1 rad/s
            ),
        )
    }
```

**핵심 포인트**:
- `commands` 딕셔너리의 **키(key)**가 명령의 이름입니다
- 위 예시에서 `"base_velocity"`가 명령의 이름입니다
- 이 이름은 관측에서 `command_name`으로 참조됩니다

### command_name의 역할

1. **명령 참조**: 환경 설정에서 정의한 특정 명령을 찾아서 참조
2. **명령 값 반환**: 해당 명령의 현재 값을 관측으로 반환
3. **다중 명령 지원**: 여러 명령이 있을 때 특정 명령만 선택

---

## command_name 커스터마이징 가능 여부

### ✅ 네, 커스터마이징 가능합니다!

`command_name`은 **자유롭게 변경할 수 있습니다**. 다만, 다음 조건을 만족해야 합니다:

### 필수 조건

1. **환경 설정 파일에서 같은 이름으로 명령 정의**
   - `walking_env_cfg.py`의 `commands` 딕셔너리에 같은 이름의 키가 있어야 함

2. **관측과 환경 설정의 이름 일치**
   - `observations.py`의 `command_name`과 `env_cfg.py`의 `commands` 키가 일치해야 함

### 커스터마이징 예시

#### 예시 1: 다른 이름으로 변경

**환경 설정 파일 (`walking_env_cfg.py`)**:
```python
@configclass
class WalkingEnvCfg(ManagerBasedRLEnvCfg):
    commands: dict[str, mdp.BaseVelocityCommandCfg] = {
        "target_speed": mdp.BaseVelocityCommandCfg(  # ← 이름 변경
            asset_name="robot",
            ranges=mdp.BaseVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 1.0),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-1.0, 1.0),
            ),
        )
    }
```

**관측 파일 (`observations.py`)**:
```python
# command_name도 같은 이름으로 변경
commands = ObsTerm(
    func=mdp.generated_commands, 
    params={"command_name": "target_speed"}  # ← 일치해야 함!
)
```

#### 예시 2: 여러 명령 사용

**환경 설정 파일**:
```python
@configclass
class WalkingEnvCfg(ManagerBasedRLEnvCfg):
    commands: dict[str, mdp.BaseVelocityCommandCfg] = {
        "base_velocity": mdp.BaseVelocityCommandCfg(...),  # 기본 속도 명령
        "emergency_stop": mdp.BaseVelocityCommandCfg(...),  # 비상 정지 명령
    }
```

**관측 파일**:
```python
# 기본 속도 명령 관측
base_velocity_cmd = ObsTerm(
    func=mdp.generated_commands, 
    params={"command_name": "base_velocity"}
)

# 비상 정지 명령 관측 (선택사항)
emergency_stop_cmd = ObsTerm(
    func=mdp.generated_commands, 
    params={"command_name": "emergency_stop"}
)
```

### ⚠️ 주의사항

1. **이름 불일치 시 오류 발생**
   ```python
   # ❌ 잘못된 예시
   # env_cfg.py에서 "base_velocity"로 정의했는데
   # observations.py에서 "target_velocity"로 참조하면 오류!
   commands = ObsTerm(
       func=mdp.generated_commands, 
       params={"command_name": "target_velocity"}  # ← 오류!
   )
   ```

2. **기존 이름 사용 권장**
   - Isaac Lab의 기존 환경들은 `"base_velocity"`를 사용
   - 일관성을 위해 같은 이름 사용 권장
   - 커스터마이징이 필요한 경우에만 변경

---

## 실제 동작 원리

### 동작 흐름

```
1. 환경 초기화
   └── walking_env_cfg.py의 commands 딕셔너리 읽기
       └── "base_velocity" 명령 등록

2. 관측 계산 시
   └── mdp.generated_commands 함수 호출
       └── params["command_name"] = "base_velocity" 사용
           └── env.command_manager["base_velocity"] 찾기
               └── 해당 명령의 현재 값 반환
```

### 코드 레벨 동작

```python
# MDP 함수 내부 (의사 코드)
def generated_commands(env, command_name: str):
    """
    Args:
        env: 환경 인스턴스
        command_name: 명령의 이름 (예: "base_velocity")
    
    Returns:
        명령의 현재 값 (torch.Tensor)
    """
    # 환경의 명령 매니저에서 해당 이름의 명령 찾기
    command = env.command_manager[command_name]
    
    # 명령의 현재 값 반환
    return command.current_value  # Shape: (num_envs, command_dim)
```

### 실제 반환값

`mdp.generated_commands` 함수는 다음과 같은 값을 반환합니다:

```python
# 예시: base_velocity 명령의 반환값
# Shape: (num_envs, 3)
# [목표 전진 속도, 목표 횡방향 속도, 목표 회전 속도]
tensor([
    [0.5, 0.0, 0.0],  # 환경 0: 전진 0.5 m/s
    [0.8, 0.2, 0.5],  # 환경 1: 전진 0.8 m/s, 횡방향 0.2 m/s, 회전 0.5 rad/s
    [0.3, -0.1, -0.3], # 환경 2: ...
    ...
])
```

---

## 다른 params 사용 예시

### 예시 1: normalize 파라미터

```python
# base_lin_vel 관측에 normalize 파라미터 사용
base_lin_vel = ObsTerm(
    func=mdp.base_lin_vel, 
    params={"normalize": True}  # ← 속도를 정규화
)
```

**의미**: 선속도를 정규화하여 학습 안정성 향상

### 예시 2: sensor_cfg와 threshold 파라미터

```python
# contact_forces 관측에 여러 파라미터 사용
feet_contact_forces = ObsTerm(
    func=mdp.contact_forces,
    params={
        "sensor_cfg": mdp.SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
        "threshold": 1.0,  # ← 접촉 판단 임계값
    },
)
```

**의미**: 
- `sensor_cfg`: 어떤 센서를 사용할지 지정
- `threshold`: 접촉 힘이 이 값 이상이면 접촉으로 판단

### 예시 3: 파라미터 없는 함수

```python
# 파라미터가 필요 없는 함수
joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)  # params 없음
```

**의미**: 함수가 기본 설정으로 동작

---

## 주의사항 및 베스트 프랙티스

### 1. 이름 일치 확인

**✅ 올바른 방법**:
```python
# env_cfg.py
commands: dict = {
    "base_velocity": mdp.BaseVelocityCommandCfg(...)
}

# observations.py
commands = ObsTerm(
    func=mdp.generated_commands, 
    params={"command_name": "base_velocity"}  # ← 일치!
)
```

**❌ 잘못된 방법**:
```python
# env_cfg.py
commands: dict = {
    "base_velocity": mdp.BaseVelocityCommandCfg(...)
}

# observations.py
commands = ObsTerm(
    func=mdp.generated_commands, 
    params={"command_name": "target_velocity"}  # ← 불일치! 오류 발생
)
```

### 2. 명령 이름 네이밍 규칙

**권장 사항**:
- **소문자와 언더스코어 사용**: `base_velocity`, `target_speed`
- **의미 있는 이름 사용**: `walking_speed`, `running_speed`
- **일관성 유지**: 프로젝트 전체에서 동일한 네이밍 규칙 사용

**예시**:
```python
# ✅ 좋은 예시
commands: dict = {
    "walking_velocity": mdp.BaseVelocityCommandCfg(...),
    "running_velocity": mdp.BaseVelocityCommandCfg(...),
}

# ❌ 나쁜 예시
commands: dict = {
    "cmd1": mdp.BaseVelocityCommandCfg(...),  # 의미 없음
    "CMD2": mdp.BaseVelocityCommandCfg(...),  # 대문자 사용
}
```

### 3. 다중 명령 사용 시

여러 명령을 사용할 때는 각각 다른 이름을 사용:

```python
# env_cfg.py
commands: dict = {
    "base_velocity": mdp.BaseVelocityCommandCfg(...),
    "target_height": mdp.BaseHeightCommandCfg(...),  # 다른 타입의 명령
}

# observations.py
base_vel_cmd = ObsTerm(
    func=mdp.generated_commands, 
    params={"command_name": "base_velocity"}
)

target_height_cmd = ObsTerm(
    func=mdp.generated_commands, 
    params={"command_name": "target_height"}
)
```

### 4. 디버깅 팁

명령이 제대로 참조되는지 확인하는 방법:

```python
# 환경 생성 후 확인
env = gym.make("H1-Walking-v0")
print(env.unwrapped.command_manager.keys())  # 등록된 명령 이름들 출력
# 출력: dict_keys(['base_velocity'])
```

---

## 요약

### 핵심 개념

1. **`params`란?**
   - `ObsTerm`에 전달되는 추가 파라미터 딕셔너리
   - MDP 함수가 관측을 계산할 때 필요한 설정값 전달

2. **`command_name`이란?**
   - 환경 설정에서 정의한 명령의 이름
   - `env_cfg.py`의 `commands` 딕셔너리 키와 일치해야 함

3. **커스터마이징 가능한가?**
   - ✅ 네, 가능합니다
   - 다만 환경 설정과 관측 설정의 이름이 일치해야 함

### 체크리스트

커스터마이징 시 확인할 사항:

- [ ] 환경 설정 파일(`env_cfg.py`)에서 명령 이름 정의
- [ ] 관측 파일(`observations.py`)에서 같은 이름으로 참조
- [ ] 이름이 정확히 일치하는지 확인 (대소문자 구분)
- [ ] 의미 있는 이름 사용
- [ ] 프로젝트 전체에서 일관성 유지

### 다음 단계

- `walking_env_cfg.py` 파일에서 `commands` 설정 확인
- 다른 MDP 함수들의 `params` 사용법 학습
- 커스텀 명령 생성 방법 학습

---

## 추가 학습 자료

### 관련 문서

- [01_observations_mdp_module_study.md](./01_observations_mdp_module_study.md) - MDP 모듈 전체 이해
- [H1_Custom_Action_RL_Development_Guide.md](../../../docs/H1_Custom_Action_RL_Development_Guide.md) - 전체 개발 가이드

### 관련 코드

- `tasks/walking/walking_env_cfg.py` - 환경 설정 파일 (명령 정의)
- `tasks/walking/mdp/observations.py` - 관측 파일 (명령 참조)

---

**작성일**: 2025-01-15  
**작성자**: AI Assistant  
**버전**: 1.0  
**관련 코드**: `tasks/walking/mdp/observations.py:27`

