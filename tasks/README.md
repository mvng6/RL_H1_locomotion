# Tasks 폴더

이 폴더는 강화학습 환경의 MDP (Markov Decision Process) 정의 및 태스크 구현을 포함합니다.

## 📁 폴더 구조

```
tasks/
├── locomotion/          # 보행 태스크 구현
│   ├── env_cfg.py      # 환경 설정 클래스
│   └── __init__.py
└── __init__.py         # 패키지 초기화 파일
```

## 🎯 목적

이 폴더는 강화학습 환경의 핵심 구성 요소들을 구현합니다:

- **환경 설정 (Environment Configuration)**: 시뮬레이션 씬, 로봇, 액추에이터 설정
- **관측 공간 (Observations)**: 에이전트가 관찰하는 상태 정보
- **보상 함수 (Rewards)**: 에이전트의 행동에 대한 보상 계산
- **종료 조건 (Terminations)**: 에피소드 종료 조건 정의

## 📋 현재 구현 상태

### ✅ 완료된 항목

- [x] **로코모션 환경 설정** (`locomotion/env_cfg.py`)
  - 씬 설정 클래스 (`H1LocomotionSceneCfg`)
  - 환경 설정 클래스 (`H1LocomotionEnvCfg`)

### 🚧 진행 예정 항목

- [ ] **관측 공간 정의** (`locomotion/observations.py`)
  - 로봇 상태 관측 (관절 위치, 속도 등)
  - 목표 속도 관측
  - 기타 환경 정보 관측

- [ ] **보상 함수 정의** (`locomotion/rewards.py`)
  - 보행 보상 (속도 추적, 안정성 등)
  - 에너지 효율 보상
  - 페널티 (넘어짐, 관절 제한 위반 등)

- [ ] **종료 조건 정의** (`locomotion/terminations.py`)
  - 로봇 넘어짐 감지
  - 관절 제한 위반 감지
  - 최대 에피소드 길이

## 📄 파일 설명

### `locomotion/` 폴더

보행 태스크의 구체적인 구현을 포함합니다. 자세한 내용은 [`locomotion/README.md`](./locomotion/README.md)를 참조하세요.

**주요 파일:**
- `env_cfg.py`: 환경 설정 클래스 정의
- `observations.py`: 관측 공간 정의 (향후 구현)
- `rewards.py`: 보상 함수 정의 (향후 구현)
- `terminations.py`: 종료 조건 정의 (향후 구현)

## 🔧 사용 방법

### 환경 설정 Import

```python
from h1_locomotion.tasks.locomotion import H1LocomotionEnvCfg, H1LocomotionSceneCfg

# 환경 설정 인스턴스 생성
env_cfg = H1LocomotionEnvCfg()

# 환경 생성 (향후 구현 예정)
# env = ManagerBasedRLEnv(env_cfg)
```

## 🛠️ 개발 가이드

### 새로운 태스크 추가하기

1. **태스크 디렉토리 생성**
   ```bash
   mkdir -p tasks/{task_name}
   ```

2. **환경 설정 파일 작성** (`env_cfg.py`)
   - `ManagerBasedRLEnvCfg`를 상속받는 환경 설정 클래스 작성
   - 씬 설정, 액추에이터, 이벤트 설정 포함

3. **MDP 구성 요소 구현**
   - `observations.py`: 관측 공간 정의
   - `rewards.py`: 보상 함수 정의
   - `terminations.py`: 종료 조건 정의

4. **패키지 초기화** (`__init__.py`)
   - 필요한 클래스들을 export

5. **환경 등록** (향후 구현)
   - Gymnasium 환경으로 등록

### 코드 구조 예시

```python
# tasks/{task_name}/env_cfg.py
@configclass
class TaskEnvCfg(ManagerBasedRLEnvCfg):
    scene: InteractiveSceneCfg = TaskSceneCfg()
    actions: dict[str, ActuatorCfg] = {...}
    events: dict = {...}

# tasks/{task_name}/observations.py
class TaskObservations:
    def __init__(self, env):
        self.env = env
    
    def compute(self) -> torch.Tensor:
        # 관측 계산 로직
        pass

# tasks/{task_name}/rewards.py
class TaskRewards:
    def __init__(self, env):
        self.env = env
    
    def compute(self) -> torch.Tensor:
        # 보상 계산 로직
        pass
```

## 📚 MDP 구성 요소 설명

### Observations (관측 공간)

에이전트가 관찰할 수 있는 상태 정보를 정의합니다. 일반적으로 포함되는 정보:

- **로봇 상태**: 관절 위치, 속도, 토크
- **루트 상태**: 위치, 방향, 선속도, 각속도
- **목표 정보**: 목표 속도, 목표 방향 등
- **환경 정보**: 지형 정보, 장애물 정보 등

### Rewards (보상 함수)

에이전트의 행동에 대한 보상을 계산합니다. 일반적인 보상 구성:

- **작업 보상**: 목표 달성 정도 (속도 추적, 방향 추적 등)
- **안정성 보상**: 로봇의 안정성 유지
- **에너지 효율**: 전력 소비 최소화
- **페널티**: 넘어짐, 관절 제한 위반 등

### Terminations (종료 조건)

에피소드가 종료되는 조건을 정의합니다:

- **실패 조건**: 로봇 넘어짐, 관절 제한 위반
- **성공 조건**: 목표 달성
- **시간 제한**: 최대 에피소드 길이 도달

## 🔗 관련 문서

- [`locomotion/README.md`](./locomotion/README.md): 보행 태스크 상세 설명
- [`../README.md`](../README.md): 프로젝트 메인 README

---

**마지막 업데이트**: 2025년 11월 28일

