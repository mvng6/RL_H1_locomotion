# 개발 순서와 Command 참조 관계 이해하기

이 문서는 `observations.py`에서 `command_name: "base_velocity"`를 참조하는 것과 `walking_env_cfg.py`에서 명령을 정의하는 것의 관계를 설명합니다.

## 목차

1. [현재 상황 분석](#현재-상황-분석)
2. [개발 순서와 의존성](#개발-순서와-의존성)
3. [Command 참조의 의미](#command-참조의-의미)
4. [올바른 개발 순서](#올바른-개발-순서)
5. [실제 동작 확인](#실제-동작-확인)
6. [주의사항](#주의사항)

---

## 현재 상황 분석

### 현재 파일 상태

```
walking/
├── mdp/
│   └── observations.py          ✅ 작성 완료 (command_name: "base_velocity" 참조)
└── walking_env_cfg.py            ❌ 아직 작성 안 됨 (명령 정의 필요)
```

### 코드 상태

**`observations.py` (27번째 줄)**:
```python
commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
```

**`walking_env_cfg.py`**:
```python
# 아직 비어있음
```

### 질문

> `observations.py`에서 `command_name: "base_velocity"`를 참조하고 있는데, 이것은 나중 단계인 `walking_env_cfg.py` 구현 단계에서 추가할 내용이었던 것인가?

**답변**: ✅ **네, 맞습니다!** 하지만 더 정확히는 **"약속"**입니다.

---

## 개발 순서와 의존성

### 개발 가이드의 작업 순서

개발 가이드(`H1_Custom_Action_RL_Development_Guide.md`)에 따르면:

1. **작업 1.2**: 관측 공간 정의 (`observations.py`) ← **먼저**
2. **작업 1.5**: 환경 설정 파일 작성 (`walking_env_cfg.py`) ← **나중**

### 의존성 관계

```
observations.py
    ↓ (참조)
command_name: "base_velocity"
    ↓ (필요)
walking_env_cfg.py
    ↓ (정의)
commands: {"base_velocity": ...}
```

**의존성 방향**: `observations.py` → `walking_env_cfg.py`

### 왜 이 순서로 작성했는가?

1. **Isaac Lab의 표준 이름 사용**
   - `"base_velocity"`는 Isaac Lab의 기존 보행 환경에서 표준적으로 사용하는 이름
   - 예제 코드를 참고하여 작성했기 때문에 이 이름을 사용

2. **"약속"으로서의 참조**
   - `observations.py`에서 `command_name: "base_velocity"`를 참조하는 것은
   - "나중에 `walking_env_cfg.py`에서 `base_velocity`라는 이름의 명령을 정의할 것"이라는 **약속**

3. **개발 편의성**
   - 관측 공간을 먼저 설계하고
   - 나중에 환경 설정에서 필요한 명령을 정의하는 것이 자연스러움

---

## Command 참조의 의미

### `observations.py`에서의 참조

```python
commands = ObsTerm(
    func=mdp.generated_commands, 
    params={"command_name": "base_velocity"}  # ← 이것은 "약속"
)
```

**의미**:
- "나중에 환경이 초기화될 때 `base_velocity`라는 이름의 명령을 찾아서 그 값을 관측으로 사용하겠다"
- **현재 시점에서는 명령이 정의되지 않았지만, 런타임에 정의될 것이라고 가정**

### `walking_env_cfg.py`에서의 정의 (나중에 작성할 내용)

```python
@configclass
class WalkingEnvCfg(ManagerBasedRLEnvCfg):
    # 명령 생성 설정 (속도 명령)
    commands: dict[str, mdp.BaseVelocityCommandCfg] = {
        "base_velocity": mdp.BaseVelocityCommandCfg(  # ← 여기서 정의!
            asset_name="robot",
            ranges=mdp.BaseVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 1.0),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-1.0, 1.0),
            ),
        )
    }
```

**의미**:
- "`base_velocity`라는 이름으로 속도 명령을 생성하겠다"
- 이 명령은 환경 초기화 시 등록됨

---

## 올바른 개발 순서

### 방법 1: 현재 순서 유지 (권장)

**장점**:
- 관측 공간을 먼저 설계하여 전체 구조 파악 용이
- Isaac Lab의 표준 이름 사용으로 일관성 유지
- 예제 코드 참고가 쉬움

**주의사항**:
- `walking_env_cfg.py` 작성 시 반드시 `"base_velocity"` 이름으로 명령 정의 필요
- 이름 불일치 시 런타임 오류 발생

**체크리스트**:
- [x] `observations.py`에서 `command_name: "base_velocity"` 참조
- [ ] `walking_env_cfg.py`에서 `commands: {"base_velocity": ...}` 정의 ← **다음 단계**

### 방법 2: 환경 설정 먼저 작성

**순서**:
1. `walking_env_cfg.py`에서 명령 정의
2. `observations.py`에서 정의한 명령 참조

**장점**:
- 명령이 먼저 정의되어 있어 참조가 확실함
- 이름 불일치 오류 방지

**단점**:
- 전체 구조를 파악하기 전에 환경 설정을 작성해야 함
- 개발 흐름이 덜 자연스러울 수 있음

---

## 실제 동작 확인

### 런타임 동작 순서

```
1. 환경 초기화 시작
   └── WalkingEnvCfg 인스턴스 생성
       └── commands 딕셔너리 읽기
           └── "base_velocity" 명령 등록

2. 관측 매니저 초기화
   └── ObservationsCfg 인스턴스 생성
       └── commands = ObsTerm(...) 읽기
           └── params["command_name"] = "base_velocity" 확인

3. 관측 계산 시
   └── mdp.generated_commands 함수 호출
       └── env.command_manager["base_velocity"] 찾기
           ├── ✅ 찾으면: 명령 값 반환
           └── ❌ 못 찾으면: KeyError 발생!
```

### 오류 발생 시나리오

**시나리오 1: 명령 이름 불일치**

```python
# walking_env_cfg.py
commands: dict = {
    "target_velocity": mdp.BaseVelocityCommandCfg(...)  # 다른 이름!
}

# observations.py
commands = ObsTerm(
    func=mdp.generated_commands, 
    params={"command_name": "base_velocity"}  # ← 불일치!
)
```

**결과**: 런타임 오류 발생
```
KeyError: 'base_velocity'
```

**시나리오 2: 명령이 정의되지 않음**

```python
# walking_env_cfg.py
commands: dict = {}  # 명령 없음!

# observations.py
commands = ObsTerm(
    func=mdp.generated_commands, 
    params={"command_name": "base_velocity"}
)
```

**결과**: 런타임 오류 발생
```
KeyError: 'base_velocity'
```

---

## 주의사항

### 1. 이름 일치 확인

**✅ 올바른 예시**:
```python
# observations.py
commands = ObsTerm(
    func=mdp.generated_commands, 
    params={"command_name": "base_velocity"}
)

# walking_env_cfg.py
commands: dict = {
    "base_velocity": mdp.BaseVelocityCommandCfg(...)  # ← 일치!
}
```

**❌ 잘못된 예시**:
```python
# observations.py
commands = ObsTerm(
    func=mdp.generated_commands, 
    params={"command_name": "base_velocity"}
)

# walking_env_cfg.py
commands: dict = {
    "target_velocity": mdp.BaseVelocityCommandCfg(...)  # ← 불일치!
}
```

### 2. 개발 단계별 체크리스트

**Phase 1: 관측 공간 정의 (`observations.py`)**
- [x] `command_name: "base_velocity"` 참조 작성
- [ ] **메모**: 나중에 `walking_env_cfg.py`에서 `"base_velocity"` 명령 정의 필요

**Phase 2: 환경 설정 작성 (`walking_env_cfg.py`)**
- [ ] `commands` 딕셔너리 작성
- [ ] `"base_velocity"` 키로 명령 정의
- [ ] `observations.py`의 `command_name`과 일치하는지 확인

**Phase 3: 검증**
- [ ] 환경 생성 테스트
- [ ] 명령이 올바르게 참조되는지 확인
- [ ] 런타임 오류 없음 확인

### 3. 다른 참조들도 확인

`observations.py`에는 `command_name` 외에도 다른 참조가 있습니다:

```python
# 접촉 센서 참조
feet_contact_forces = ObsTerm(
    func=mdp.contact_forces,
    params={
        "sensor_cfg": mdp.SceneEntityCfg("contact_forces", ...),  # ← 이것도!
        "threshold": 1.0,
    },
)
```

**의미**: `walking_env_cfg.py`의 `WalkingSceneCfg`에서 `contact_forces` 센서를 정의해야 함

**체크리스트**:
- [x] `observations.py`에서 `"contact_forces"` 센서 참조
- [ ] `walking_env_cfg.py`의 `WalkingSceneCfg`에서 `contact_forces` 센서 정의 필요

---

## 요약

### 핵심 답변

**Q**: `observations.py`에서 `command_name: "base_velocity"`를 참조하는 것은 나중 단계인 `walking_env_cfg.py` 구현 단계에서 추가할 내용이었나?

**A**: ✅ **네, 맞습니다!**

더 정확히는:
1. **"약속"**: `observations.py`에서 `command_name: "base_velocity"`를 참조하는 것은 "나중에 이 이름의 명령을 정의할 것"이라는 약속
2. **의존성**: `walking_env_cfg.py`에서 반드시 `"base_velocity"` 이름으로 명령을 정의해야 함
3. **표준 이름**: Isaac Lab의 표준 이름을 사용하여 일관성 유지

### 개발 순서

1. ✅ **현재**: `observations.py` 작성 완료 (`command_name: "base_velocity"` 참조)
2. ⏳ **다음**: `walking_env_cfg.py` 작성 시 `commands: {"base_velocity": ...}` 정의
3. ✅ **검증**: 이름이 일치하는지 확인

### 체크리스트

`walking_env_cfg.py` 작성 시 확인할 사항:

- [ ] `commands` 딕셔너리에 `"base_velocity"` 키가 있는가?
- [ ] `observations.py`의 `command_name`과 이름이 일치하는가?
- [ ] `contact_forces` 센서도 `WalkingSceneCfg`에 정의되어 있는가?
- [ ] 환경 생성 시 오류가 발생하지 않는가?

---

## 다음 단계

### `walking_env_cfg.py` 작성 시 참고할 코드

개발 가이드의 "작업 1.5" 섹션을 참고하여 다음 내용을 작성하세요:

```python
@configclass
class WalkingEnvCfg(ManagerBasedRLEnvCfg):
    # ... 다른 설정들 ...
    
    # 명령 생성 설정 (속도 명령) ← 이 부분!
    commands: dict[str, mdp.BaseVelocityCommandCfg] = {
        "base_velocity": mdp.BaseVelocityCommandCfg(  # ← observations.py와 일치!
            asset_name="robot",
            ranges=mdp.BaseVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 1.0),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-1.0, 1.0),
            ),
        )
    }
```

### 관련 문서

- [H1_Custom_Action_RL_Development_Guide.md](../../../docs/H1_Custom_Action_RL_Development_Guide.md) - 작업 1.5 섹션 참고
- [02_observations_params_command_name_study.md](./02_observations_params_command_name_study.md) - params와 command_name 상세 설명

---

**작성일**: 2025-01-15  
**작성자**: AI Assistant  
**버전**: 1.0  
**관련 파일**: 
- `tasks/walking/mdp/observations.py:27`
- `tasks/walking/walking_env_cfg.py` (작성 예정)

