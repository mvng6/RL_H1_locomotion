# H1 Locomotion Project

H1 휴머노이드 로봇의 보행 강화학습 프로젝트입니다. 이 프로젝트는 Isaac Lab을 기반으로 Unitree H1 로봇의 보행 제어를 위한 강화학습 환경을 구축합니다.

## 📋 목차

- [프로젝트 개요](#프로젝트-개요)
- [프로젝트 구조](#프로젝트-구조)
- [진행 사항](#진행-사항)
- [설치 및 사용 방법](#설치-및-사용-방법)
- [폴더별 상세 설명](#폴더별-상세-설명)
- [개발 가이드](#개발-가이드)

## 🎯 프로젝트 개요

이 프로젝트는 Isaac Lab의 확장 패키지로 구현되며, 다음과 같은 목표를 가집니다:

- **H1 휴머노이드 로봇의 보행 제어**: Unitree H1 로봇을 사용한 보행 강화학습 환경 구축
- **모듈화된 구조**: 환경 설정, 태스크 정의, 알고리즘 설정을 분리하여 유지보수성 향상
- **확장 가능한 아키텍처**: 다양한 보행 태스크와 알고리즘을 쉽게 추가할 수 있는 구조

## 📁 프로젝트 구조

```
h1_locomotion/
├── config/                    # 환경 및 알고리즘 설정 파일들
│   ├── agents/               # 강화학습 알고리즘 설정
│   ├── extension.toml        # Isaac Sim Extension 설정
│   └── __init__.py
├── tasks/                    # MDP 정의 (태스크 구현)
│   ├── locomotion/          # 보행 태스크 구현
│   │   ├── env_cfg.py       # 환경 설정 클래스
│   │   └── __init__.py
│   └── __init__.py
├── pyproject.toml            # Python 패키지 설정
├── __init__.py               # 패키지 초기화
└── README.md                 # 이 파일
```

## ✅ 진행 사항

### Phase 1: 프로젝트 초기 설정 (완료)

- [x] **프로젝트 구조 생성**
  - Isaac Lab 확장 패키지 구조 생성
  - 기본 디렉토리 및 `__init__.py` 파일 생성

- [x] **패키지 설정 파일 작성**
  - `pyproject.toml`: Python 패키지 메타데이터 및 의존성 정의
    - 프로젝트 이름: `h1_locomotion`
    - Isaac Lab 의존성 추가
  - `config/extension.toml`: Isaac Sim Extension 설정
    - Extension 제목 및 설명 설정
    - Python 모듈 경로 등록

- [x] **패키지 초기화 설정**
  - `__init__.py`: 패키지 import 시 `tasks` 모듈 자동 로드

### Phase 2: 환경 설정 구현 (완료)

- [x] **로코모션 환경 설정 클래스 구현**
  - `tasks/locomotion/env_cfg.py` 파일 생성
  - `H1LocomotionSceneCfg`: 씬 설정 클래스
    - GroundPlane (지면) 설정
    - DomeLight (조명) 설정
    - H1 로봇 에셋 설정 (`H1_MINIMAL_CFG` 사용)
  - `H1LocomotionEnvCfg`: 강화학습 환경 설정 클래스
    - `ManagerBasedRLEnvCfg` 상속
    - 액추에이터 설정: IdealPDActuator (stiffness=80.0, damping=2.0)
    - 이벤트 설정: 에피소드 시작 시 관절 상태 리셋

### Phase 3: 향후 계획

- [ ] **MDP 구성 요소 구현**
  - [ ] Observations (관측 공간) 정의
  - [ ] Rewards (보상 함수) 정의
  - [ ] Terminations (종료 조건) 정의

- [ ] **환경 등록 및 테스트**
  - [ ] Gymnasium 환경 등록
  - [ ] 환경 실행 테스트

- [ ] **강화학습 알고리즘 설정**
  - [ ] PPO 알고리즘 설정 파일 작성
  - [ ] 학습 스크립트 작성

## 🚀 설치 및 사용 방법

### 환경 요구사항

- Ubuntu 22.04 LTS
- Python 3.10+
- Isaac Lab (설치 완료)
- Isaac Sim

### 설치

1. **패키지 설치**
```bash
cd ~/RL_project_ws/exts/h1_locomotion
pip install -e .
```

2. **환경 확인**
```python
from h1_locomotion.tasks.locomotion import H1LocomotionEnvCfg
print("환경 설정 로드 성공!")
```

### 사용 예시 (향후 구현 예정)

```python
import gymnasium as gym

# 환경 생성
env = gym.make("Isaac-Velocity-Flat-H1-v0")

# 환경 실행
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

## 📚 폴더별 상세 설명

각 폴더의 상세한 설명은 해당 폴더의 README.md 파일을 참조하세요:

- [`config/`](./config/README.md): 환경 및 알고리즘 설정 파일들
- [`config/agents/`](./config/agents/README.md): 강화학습 알고리즘 설정
- [`tasks/`](./tasks/README.md): MDP 정의 및 태스크 구현
- [`tasks/locomotion/`](./tasks/locomotion/README.md): 보행 태스크 상세 구현

## 🛠️ 개발 가이드

### 새로운 태스크 추가하기

1. `tasks/` 폴더에 새로운 태스크 디렉토리 생성
2. `env_cfg.py` 파일에 환경 설정 클래스 작성
3. MDP 구성 요소 (observations, rewards, terminations) 구현
4. 환경 등록 및 테스트

### 새로운 알고리즘 설정 추가하기

1. `config/agents/` 폴더에 알고리즘 설정 파일 추가
2. 해당 알고리즘의 하이퍼파라미터 설정
3. 학습 스크립트에서 설정 파일 사용

### 코드 스타일 가이드

- **Python**: PEP 8 스타일 가이드 준수
- **타입 힌팅**: 함수 및 클래스에 타입 힌팅 사용
- **주석**: 한글로 명확한 주석 작성
- **변수명**: 명확하고 이해하기 쉬운 변수명 사용

## 📝 참고 자료

- [Isaac Lab 공식 문서](https://isaac-sim.github.io/IsaacLab/)
- [Isaac Lab Assets](https://github.com/isaac-sim/IsaacLab-Assets)
- [Unitree H1 로봇 문서](https://www.unitree.com/products/h1)

## 👥 기여자

- 프로젝트 초기 설정 및 환경 구현

---

**마지막 업데이트**: 2025년 11월 28일
