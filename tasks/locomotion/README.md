# Locomotion 태스크

이 폴더는 H1 휴머노이드 로봇의 보행 강화학습 태스크를 구현합니다.

## 📁 폴더 구조

```
locomotion/
├── env_cfg.py          # 환경 설정 클래스 (완료)
├── observations.py    # 관측 공간 정의 (예정)
├── rewards.py          # 보상 함수 정의 (예정)
├── terminations.py     # 종료 조건 정의 (예정)
└── __init__.py        # 패키지 초기화 파일
```

## 🎯 태스크 목표

H1 휴머노이드 로봇이 안정적으로 보행할 수 있도록 강화학습을 통해 제어 정책을 학습합니다.

**주요 목표:**
- 목표 속도에 맞춰 보행
- 로봇의 안정성 유지
- 에너지 효율적인 보행 패턴 학습

## 📄 파일 상세 설명

### `env_cfg.py` ✅ (완료)

강화학습 환경의 핵심 설정을 정의하는 파일입니다.

#### 주요 클래스

**1. `H1LocomotionSceneCfg`**
- `InteractiveSceneCfg`를 상속받는 씬 설정 클래스
- 시뮬레이션 환경의 물리적 요소들을 정의

**구성 요소:**
```python
@configclass
class H1LocomotionSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(...)           # 지면 생성
    dome_light = AssetBaseCfg(...)       # 조명 설정
    robot: ArticulationCfg = ...         # H1 로봇 설정
```

**설정 내용:**
- **Ground**: 기본 지면 생성 (`/World/defaultGroundPlane`)
- **DomeLight**: 조명 설정 (강도: 3000.0, 색상: 회색)
- **Robot**: H1 로봇 에셋 설정
  - `H1_MINIMAL_CFG` 사용
  - `prim_path`: `"{ENV_REGEX_NS}/Robot"` (각 환경별 네임스페이스)

**2. `H1LocomotionEnvCfg`**
- `ManagerBasedRLEnvCfg`를 상속받는 강화학습 환경 설정 클래스
- 환경의 전체 구성을 정의

**구성 요소:**
```python
@configclass
class H1LocomotionEnvCfg(ManagerBasedRLEnvCfg):
    scene: InteractiveSceneCfg = H1LocomotionSceneCfg()
    actions: dict[str, IdealPDActuatorCfg] = {...}
    events: dict = {...}
```

**설정 내용:**

1. **Scene (씬 설정)**
   - `H1LocomotionSceneCfg` 인스턴스 사용
   - 시뮬레이션 환경의 물리적 구성

2. **Actions (액추에이터 설정)**
   - `IdealPDActuator` 사용 (이상적인 PD 제어기)
   - 관절 이름 패턴: `".*_joint"` (모든 관절)
   - **Stiffness (강성)**: 80.0 N⋅m/rad
   - **Damping (감쇠)**: 2.0 N⋅m⋅s/rad
   
   **설명:**
   - PD 제어기는 관절 위치와 속도를 제어하는 제어기입니다
   - Stiffness는 관절이 목표 위치로 돌아가려는 강성을 의미합니다
   - Damping은 관절의 진동을 억제하는 감쇠 계수입니다

3. **Events (이벤트 설정)**
   - 에피소드 시작 시 관절 상태를 랜덤하게 리셋
   - 함수: `isaaclab.utils.assets.reset_joints_by_scale`
   - **Position Range**: (0.5, 1.5) - 기본 관절 위치의 0.5~1.5배 범위로 랜덤 리셋
   - **Velocity Range**: (0.0, 0.0) - 관절 속도를 0으로 리셋
   
   **설명:**
   - 각 에피소드 시작 시 관절 위치를 랜덤하게 설정하여 다양한 초기 상태에서 학습
   - 이는 강화학습 정책의 일반화 성능을 향상시킵니다

#### 사용 예시

```python
from h1_locomotion.tasks.locomotion import H1LocomotionEnvCfg

# 환경 설정 인스턴스 생성
env_cfg = H1LocomotionEnvCfg()

# 환경 생성 (향후 구현 예정)
# env = ManagerBasedRLEnv(env_cfg)
```

### `observations.py` 🚧 (예정)

에이전트가 관찰할 수 있는 상태 정보를 정의합니다.

**예상 구성:**
- 로봇 관절 상태 (위치, 속도)
- 루트 상태 (위치, 방향, 속도)
- 목표 속도
- 이전 액션

### `rewards.py` 🚧 (예정)

에이전트의 행동에 대한 보상을 계산합니다.

**예상 보상 구성:**
- 속도 추적 보상
- 안정성 보상 (자세 유지)
- 에너지 효율 보상
- 페널티 (넘어짐, 관절 제한 위반)

### `terminations.py` 🚧 (예정)

에피소드 종료 조건을 정의합니다.

**예상 종료 조건:**
- 로봇 넘어짐 감지
- 관절 제한 위반
- 최대 에피소드 길이 도달

## 🔧 개발 가이드

### 환경 설정 수정하기

**액추에이터 파라미터 조정:**
```python
# env_cfg.py에서 수정
actions: dict[str, IdealPDActuatorCfg] = {
    ".*_joint": IdealPDActuatorCfg(
        stiffness=100.0,  # 강성 증가
        damping=3.0,      # 감쇠 증가
    ),
}
```

**이벤트 파라미터 조정:**
```python
# 관절 리셋 범위 조정
events: dict = {
    "reset_joints_by_scale": {
        "params": {
            "position_range": (0.8, 1.2),  # 더 좁은 범위
            "velocity_range": (0.0, 0.0),
        },
    }
}
```

### 새로운 관측 추가하기

`observations.py` 파일에 새로운 관측 클래스를 추가합니다:

```python
class LocomotionObservations:
    def __init__(self, env):
        self.env = env
    
    def compute(self) -> torch.Tensor:
        # 관측 계산 로직
        robot = self.env.scene["robot"]
        obs = torch.cat([
            robot.data.joint_pos,
            robot.data.joint_vel,
            robot.data.root_lin_vel_b,
            robot.data.root_ang_vel_b,
        ], dim=-1)
        return obs
```

### 새로운 보상 추가하기

`rewards.py` 파일에 보상 클래스를 추가합니다:

```python
class LocomotionRewards:
    def __init__(self, env):
        self.env = env
    
    def compute(self) -> torch.Tensor:
        # 보상 계산 로직
        # 속도 추적 보상, 안정성 보상 등
        pass
```

## 📚 참고 자료

### Isaac Lab 관련
- [ManagerBasedRLEnvCfg 문서](https://isaac-sim.github.io/IsaacLab/)
- [IdealPDActuator 문서](https://isaac-sim.github.io/IsaacLab/)

### 강화학습 보행 관련
- [Learning to Walk in the Wild Using Sim-to-Real Reinforcement Learning](https://arxiv.org/abs/2108.03276)
- [Sim-to-Real Transfer for Humanoid Locomotion](https://arxiv.org/abs/1903.01387)

## 🔗 관련 문서

- [`../README.md`](../README.md): Tasks 폴더 상세 설명
- [`../../README.md`](../../README.md): 프로젝트 메인 README

---

**마지막 업데이트**: 2025년 11월 28일

