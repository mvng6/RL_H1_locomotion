# H1 Locomotion 프로젝트 구조 설계: PPO + AMP 통합

**목적**: 기존 PPO 기반 보행 학습 환경과 새로운 AMP 알고리즘 기반 학습 환경을 효율적으로 통합하기 위한 프로젝트 구조 설계 문서

**작성일**: 2025-01-XX  
**버전**: 1.0

---

## 목차

1. [설계 원칙](#설계-원칙)
2. [전체 프로젝트 구조](#전체-프로젝트-구조)
3. [디렉토리별 상세 설명](#디렉토리별-상세-설명)
4. [공통 모듈 재사용 전략](#공통-모듈-재사용-전략)
5. [환경 등록 및 네이밍 규칙](#환경-등록-및-네이밍-규칙)
6. [마이그레이션 가이드](#마이그레이션-가이드)

---

## 설계 원칙

### 1. **기존 코드 보존**
- 기존 PPO 기반 학습 환경은 그대로 유지
- 기존 학습 결과 및 체크포인트 호환성 보장

### 2. **모듈화 및 재사용**
- 공통 MDP 모듈 (observations, rewards, terminations)은 두 알고리즘에서 공유
- 환경 설정은 상속 구조로 확장 가능하게 설계

### 3. **명확한 분리**
- PPO와 AMP 관련 코드는 명확히 구분
- 각 알고리즘별 설정 파일 분리

### 4. **확장성**
- 향후 다른 알고리즘 추가 시에도 동일한 패턴 적용 가능
- 데이터 전처리 파이프라인 독립적으로 관리

---

## 전체 프로젝트 구조

```
exts/h1_locomotion/
├── data/                                    # [NEW] 데이터 디렉토리
│   ├── amass/                              # 원본 AMASS 데이터셋
│   │   ├── CMU_Mocap/
│   │   ├── HumanML3D/
│   │   └── ...
│   ├── processed/                          # 전처리된 데이터
│   │   ├── retargeted_motions/            # 리타겟팅된 모션 클립
│   │   │   ├── walking_clips/
│   │   │   └── running_clips/
│   │   └── amp_motions.npy                # AMP 호환 형식 (최종)
│   └── mapping/                            # 스켈레톤 매핑 정보
│       ├── smpl_to_h1_joint_mapping.json
│       └── t_pose_alignment.json
│
├── tasks/
│   ├── __init__.py                         # 모든 환경 등록
│   ├── walking/                            # 보행 태스크 (공통)
│   │   ├── __init__.py                     # PPO 환경 등록
│   │   ├── walking_env_cfg.py             # [기존] PPO 환경 설정
│   │   ├── amp_env_cfg.py                  # [NEW] AMP 환경 설정
│   │   ├── mdp/                            # 공통 MDP 모듈
│   │   │   ├── __init__.py
│   │   │   ├── observations.py            # 공통 관측 공간
│   │   │   ├── rewards.py                 # 공통 보상 함수 (Task rewards)
│   │   │   └── terminations.py            # 공통 종료 조건
│   │   └── amp/                            # [NEW] AMP 전용 모듈
│   │       ├── __init__.py
│   │       ├── discriminator.py           # Discriminator 네트워크
│   │       ├── amp_rewards.py             # AMP Style reward
│   │       ├── motion_dataset.py          # Expert 모션 데이터셋 로더
│   │       └── domain_randomization.py    # 도메인 랜덤화 매니저
│   │
│   ├── locomotion/                         # [기존] 기본 locomotion 환경
│   │   └── ...
│   │
│   └── running/                            # [미래] 달리기 태스크
│       └── ...
│
├── scripts/
│   ├── train_walking_ppo.py               # [기존] PPO 학습 스크립트
│   ├── play_walking_ppo.py                # [기존] PPO 테스트 스크립트
│   ├── train_walking_amp.py               # [NEW] AMP 학습 스크립트
│   ├── play_walking_amp.py                # [NEW] AMP 테스트 스크립트
│   ├── test_walking_env.py                 # [기존] 환경 테스트
│   │
│   └── data_preprocessing/                 # [NEW] 데이터 전처리 파이프라인
│       ├── __init__.py
│       ├── process_amass.py               # AMASS 전처리 메인 스크립트
│       ├── export_motions.py              # AMP 형식으로 내보내기
│       └── retargeting/                    # 리타겟팅 모듈
│           ├── __init__.py
│           ├── smpl_loader.py            # SMPL 데이터 로더
│           ├── h1_skeleton.py            # H1 스켈레톤 정의
│           ├── retargeter.py             # 리타겟팅 엔진
│           └── utils.py                  # 유틸리티 함수
│
├── config/
│   ├── __init__.py
│   ├── agents/                            # 에이전트 설정
│   │   ├── __init__.py
│   │   ├── walking_ppo_cfg.py            # [기존] PPO 에이전트 설정
│   │   └── walking_amp_ppo_cfg.py        # [NEW] AMP PPO 에이전트 설정
│   │
│   └── amp/                               # [NEW] AMP 전용 설정
│       ├── __init__.py
│       ├── discriminator_cfg.yaml         # Discriminator 네트워크 설정
│       ├── curriculum_config.yaml         # 커리큘럼 학습 설정
│       └── domain_randomization.yaml      # 도메인 랜덤화 설정
│
├── docs/                                   # 문서
│   ├── H1_Custom_Action_RL_Development_Guide.md
│   ├── Work_Process_Checklist.md
│   ├── AMP_H1_Locomotion_Technical_Specification.md
│   └── Project_Structure_Design_PPO_AMP.md  # 이 문서
│
├── logs/                                   # 학습 로그 및 체크포인트
│   └── rsl_rl/
│       ├── h1_walking/                    # [기존] PPO 학습 결과
│       └── h1_walking_amp/                 # [NEW] AMP 학습 결과
│
├── __init__.py
├── setup.py
├── pyproject.toml
└── README.md
```

---

## 디렉토리별 상세 설명

### 1. `data/` 디렉토리 (NEW)

**목적**: AMP 알고리즘에 필요한 모션 데이터 저장 및 관리

**구조**:
```
data/
├── amass/                      # 원본 AMASS 데이터셋 (외부에서 다운로드)
├── processed/                  # 전처리된 데이터
│   ├── retargeted_motions/    # 리타겟팅된 모션 클립 (중간 결과)
│   └── amp_motions.npy        # 최종 AMP 호환 형식
└── mapping/                   # 스켈레톤 매핑 정보 (JSON)
```

**특징**:
- `.gitignore`에 `data/amass/` 추가 (용량이 크므로 버전 관리 제외)
- `data/processed/`와 `data/mapping/`은 버전 관리 포함 (작은 파일)

### 2. `tasks/walking/` 디렉토리 구조

#### 2.1 공통 구조

```
tasks/walking/
├── __init__.py                 # PPO 환경 등록 (기존 유지)
├── walking_env_cfg.py          # [기존] PPO 환경 설정
├── amp_env_cfg.py              # [NEW] AMP 환경 설정
├── mdp/                        # 공통 MDP 모듈 (두 알고리즘 공유)
│   ├── observations.py
│   ├── rewards.py              # Task rewards (속도 추적 등)
│   └── terminations.py
└── amp/                        # [NEW] AMP 전용 모듈
    ├── discriminator.py
    ├── amp_rewards.py          # Style reward (Discriminator 기반)
    ├── motion_dataset.py
    └── domain_randomization.py
```

#### 2.2 파일별 역할

**`walking_env_cfg.py`** (기존 유지)
- PPO 알고리즘용 환경 설정
- `WalkingEnvCfg` 클래스 정의
- 공통 MDP 모듈 (`mdp/`) 사용

**`amp_env_cfg.py`** (NEW)
- AMP 알고리즘용 환경 설정
- `H1AmpEnvCfg` 클래스 정의
- `WalkingEnvCfg`를 상속하여 확장
- AMP 전용 모듈 (`amp/`) 추가 사용

**`mdp/`** (공통 모듈)
- 두 알고리즘 모두에서 사용하는 MDP 구성요소
- `observations.py`: 관측 공간 정의
- `rewards.py`: Task rewards (속도 추적, 자세 유지 등)
- `terminations.py`: 종료 조건

**`amp/`** (AMP 전용)
- AMP 알고리즘에만 필요한 모듈
- `discriminator.py`: Discriminator 네트워크
- `amp_rewards.py`: Style reward 계산
- `motion_dataset.py`: Expert 모션 데이터셋 로더
- `domain_randomization.py`: 도메인 랜덤화 매니저

### 3. `scripts/` 디렉토리 구조

#### 3.1 학습/테스트 스크립트

```
scripts/
├── train_walking_ppo.py       # [기존] PPO 학습
├── play_walking_ppo.py        # [기존] PPO 테스트
├── train_walking_amp.py       # [NEW] AMP 학습
├── play_walking_amp.py        # [NEW] AMP 테스트
└── test_walking_env.py        # [기존] 환경 테스트
```

**네이밍 규칙**:
- PPO: `train_walking_ppo.py`, `play_walking_ppo.py`
- AMP: `train_walking_amp.py`, `play_walking_amp.py`

#### 3.2 데이터 전처리 스크립트 (NEW)

```
scripts/data_preprocessing/
├── __init__.py
├── process_amass.py           # 메인 전처리 스크립트
├── export_motions.py          # AMP 형식으로 내보내기
└── retargeting/               # 리타겟팅 모듈
    ├── smpl_loader.py
    ├── h1_skeleton.py
    ├── retargeter.py
    └── utils.py
```

### 4. `config/` 디렉토리 구조

```
config/
├── agents/                    # 에이전트 설정
│   ├── walking_ppo_cfg.py    # [기존] PPO 에이전트
│   └── walking_amp_ppo_cfg.py # [NEW] AMP PPO 에이전트
└── amp/                       # [NEW] AMP 전용 설정
    ├── discriminator_cfg.yaml
    ├── curriculum_config.yaml
    └── domain_randomization.yaml
```

**설정 파일 분리 이유**:
- PPO와 AMP는 서로 다른 하이퍼파라미터 필요
- AMP는 추가 설정 파일 (YAML) 필요 (Discriminator, 커리큘럼 등)

---

## 공통 모듈 재사용 전략

### 1. MDP 모듈 공유

**전략**: `tasks/walking/mdp/` 모듈을 두 알고리즘에서 공통으로 사용

**장점**:
- 코드 중복 방지
- 일관된 관측/보상/종료 조건 유지
- 유지보수 용이

**구현 예시**:

```python
# tasks/walking/walking_env_cfg.py (PPO)
from .mdp import ObservationsCfg, RewardsCfg, TerminationsCfg

@configclass
class WalkingEnvCfg(H1RoughEnvCfg):
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

# tasks/walking/amp_env_cfg.py (AMP)
from ..walking.mdp import ObservationsCfg, TerminationsCfg
from .amp.amp_rewards import AMPRewardsCfg

@configclass
class H1AmpEnvCfg(WalkingEnvCfg):
    # 공통 모듈 재사용
    observations: ObservationsCfg = ObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    
    # AMP 전용 보상 추가
    amp_rewards: AMPRewardsCfg = AMPRewardsCfg()
```

### 2. 환경 설정 상속 구조

```
H1RoughEnvCfg (Isaac Lab 기본)
    ↓
WalkingEnvCfg (PPO용, 기존)
    ↓
H1AmpEnvCfg (AMP용, NEW)
```

**상속 관계**:
- `H1AmpEnvCfg`는 `WalkingEnvCfg`를 상속
- 공통 설정은 부모 클래스에서 상속
- AMP 전용 설정만 추가

### 3. 보상 함수 통합 전략

**PPO 보상 구조**:
```python
# tasks/walking/mdp/rewards.py
@configclass
class RewardsCfg:
    track_lin_vel_xy_exp = RewTerm(...)  # Task reward
    flat_orientation_l2 = RewTerm(...)   # Task reward
    # ... 기타 Task rewards
```

**AMP 보상 구조**:
```python
# tasks/walking/amp/amp_rewards.py
@configclass
class AMPRewardsCfg:
    style_reward = RewTerm(...)  # Style reward (Discriminator 기반)

# tasks/walking/amp_env_cfg.py
@configclass
class H1AmpEnvCfg(WalkingEnvCfg):
    # 부모 클래스의 Task rewards 상속
    rewards: RewardsCfg = RewardsCfg()  # Task rewards
    
    # AMP 전용 Style reward 추가
    amp_rewards: AMPRewardsCfg = AMPRewardsCfg()
    
    def __post_init__(self):
        super().__post_init__()
        # 최종 보상 = Task rewards + Style reward
        # (런타임에 통합)
```

---

## 환경 등록 및 네이밍 규칙

### 1. 환경 ID 네이밍 규칙

**PPO 환경** (기존 유지):
- `H1-Walking-v0`: PPO 학습용
- `H1-Walking-Play-v0`: PPO 테스트용

**AMP 환경** (NEW):
- `H1-Walking-AMP-v0`: AMP 학습용
- `H1-Walking-AMP-Play-v0`: AMP 테스트용

### 2. 환경 등록 구조

```python
# tasks/walking/__init__.py (기존 PPO 환경 등록 유지)
import gymnasium as gym
from . import walking_env_cfg

gym.register(
    id="H1-Walking-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walking_env_cfg:WalkingEnvCfg",
        "rsl_rl_cfg_entry_point": "h1_locomotion.config.agents.walking_ppo_cfg:WalkingPPORunnerCfg",
    },
)

# tasks/walking/amp/__init__.py (NEW - AMP 환경 등록)
import gymnasium as gym
from .. import amp_env_cfg

gym.register(
    id="H1-Walking-AMP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_env_cfg:H1AmpEnvCfg",
        "rsl_rl_cfg_entry_point": "h1_locomotion.config.agents.walking_amp_ppo_cfg:WalkingAMPPPORunnerCfg",
    },
)
```

### 3. 메인 `__init__.py` 업데이트

```python
# tasks/__init__.py
"""H1 Locomotion 태스크 패키지."""

# PPO 환경 등록
from . import walking  # H1-Walking-v0 등록

# AMP 환경 등록
from .walking import amp  # H1-Walking-AMP-v0 등록

# 향후 추가 가능
# from . import running
# from . import jumping
```

---

## 마이그레이션 가이드

### Phase 1: 디렉토리 구조 생성

```bash
cd /home/ldj/RL_project_ws/exts/h1_locomotion

# 1. 데이터 디렉토리 생성
mkdir -p data/amass data/processed/retargeted_motions data/mapping

# 2. AMP 모듈 디렉토리 생성
mkdir -p tasks/walking/amp
mkdir -p scripts/data_preprocessing/retargeting
mkdir -p config/amp

# 3. 초기화 파일 생성
touch tasks/walking/amp/__init__.py
touch scripts/data_preprocessing/__init__.py
touch scripts/data_preprocessing/retargeting/__init__.py
touch config/amp/__init__.py
```

### Phase 2: 공통 모듈 확인

**확인 사항**:
- [ ] `tasks/walking/mdp/` 모듈이 두 알고리즘에서 사용 가능한지 확인
- [ ] 필요한 경우 공통 모듈 확장 (AMP용 관측 항목 추가 등)

### Phase 3: AMP 환경 설정 파일 작성

**작성 순서**:
1. `tasks/walking/amp/discriminator.py` 작성
2. `tasks/walking/amp/motion_dataset.py` 작성
3. `tasks/walking/amp/amp_rewards.py` 작성
4. `tasks/walking/amp_env_cfg.py` 작성
5. `tasks/walking/amp/__init__.py`에 환경 등록

### Phase 4: 설정 파일 작성

**작성 순서**:
1. `config/amp/discriminator_cfg.yaml` 작성
2. `config/amp/curriculum_config.yaml` 작성
3. `config/amp/domain_randomization.yaml` 작성
4. `config/agents/walking_amp_ppo_cfg.py` 작성

### Phase 5: 학습 스크립트 작성

**작성 순서**:
1. `scripts/train_walking_amp.py` 작성
2. `scripts/play_walking_amp.py` 작성
3. 데이터 전처리 스크립트 작성 (`scripts/data_preprocessing/`)

### Phase 6: 통합 테스트

**테스트 순서**:
1. 환경 등록 확인: `list_envs.py`에서 `H1-Walking-AMP-v0` 확인
2. 환경 생성 테스트: `test_walking_env.py` 수정하여 AMP 환경 테스트
3. 학습 파이프라인 테스트: 소규모 학습 실행

---

## 파일 생성 체크리스트

### 필수 파일 (NEW)

#### AMP 모듈
- [ ] `tasks/walking/amp/__init__.py`
- [ ] `tasks/walking/amp/discriminator.py`
- [ ] `tasks/walking/amp/amp_rewards.py`
- [ ] `tasks/walking/amp/motion_dataset.py`
- [ ] `tasks/walking/amp/domain_randomization.py`

#### 환경 설정
- [ ] `tasks/walking/amp_env_cfg.py`

#### 설정 파일
- [ ] `config/agents/walking_amp_ppo_cfg.py`
- [ ] `config/amp/discriminator_cfg.yaml`
- [ ] `config/amp/curriculum_config.yaml`
- [ ] `config/amp/domain_randomization.yaml`

#### 학습 스크립트
- [ ] `scripts/train_walking_amp.py`
- [ ] `scripts/play_walking_amp.py`

#### 데이터 전처리
- [ ] `scripts/data_preprocessing/__init__.py`
- [ ] `scripts/data_preprocessing/process_amass.py`
- [ ] `scripts/data_preprocessing/export_motions.py`
- [ ] `scripts/data_preprocessing/retargeting/__init__.py`
- [ ] `scripts/data_preprocessing/retargeting/smpl_loader.py`
- [ ] `scripts/data_preprocessing/retargeting/h1_skeleton.py`
- [ ] `scripts/data_preprocessing/retargeting/retargeter.py`
- [ ] `scripts/data_preprocessing/retargeting/utils.py`

### 수정 파일

- [ ] `tasks/__init__.py` - AMP 환경 import 추가
- [ ] `.gitignore` - `data/amass/` 추가

---

## 주의사항

### 1. 기존 코드 보존
- **절대 기존 PPO 환경 코드를 수정하지 않음**
- AMP는 별도 파일로 구현
- 공통 모듈 수정 시 두 알고리즘에 영향 확인

### 2. Import 경로 관리
- 상대 import 사용 시 경로 주의
- `from ..walking.mdp import ...` (AMP에서 공통 모듈 import)
- `from .amp import ...` (같은 디렉토리 내 AMP 모듈)

### 3. 환경 ID 충돌 방지
- PPO: `H1-Walking-v0`
- AMP: `H1-Walking-AMP-v0`
- 명확히 구분하여 혼동 방지

### 4. 데이터 관리
- `data/amass/`는 `.gitignore`에 추가 (용량 큼)
- `data/processed/`와 `data/mapping/`은 버전 관리 포함
- 데이터 경로는 상대 경로 또는 환경 변수 사용

---

## 향후 확장 가능성

### 1. 다른 알고리즘 추가
이 구조는 다른 알고리즘 추가 시에도 동일한 패턴 적용 가능:
- `tasks/walking/sac_env_cfg.py` (SAC 알고리즘)
- `tasks/walking/td3_env_cfg.py` (TD3 알고리즘)

### 2. 다른 태스크 추가
- `tasks/running/` - 달리기 태스크
- `tasks/jumping/` - 점프 태스크
- 각 태스크도 PPO와 AMP 버전 분리 가능

### 3. 하이브리드 학습
- PPO로 초기 학습 → AMP로 미세 조정
- 전이학습 파이프라인 구축 가능

---

## 참고 자료

- [AMP_H1_Locomotion_Technical_Specification.md](./AMP_H1_Locomotion_Technical_Specification.md) - AMP 구현 명세서
- [H1_Custom_Action_RL_Development_Guide.md](./H1_Custom_Action_RL_Development_Guide.md) - 기존 PPO 개발 가이드
- [Work_Process_Checklist.md](./Work_Process_Checklist.md) - 작업 체크리스트

---

**문서 버전**: 1.0  
**최종 수정일**: 2025-01-XX  
**작성자**: AI Robotics Engineer

