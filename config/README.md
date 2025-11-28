# Config 폴더

이 폴더는 H1 Locomotion 프로젝트의 환경 설정 및 알고리즘 설정 파일들을 포함합니다.

## 📁 폴더 구조

```
config/
├── agents/              # 강화학습 알고리즘 설정 파일들
├── extension.toml       # Isaac Sim Extension 설정 파일
└── __init__.py         # 패키지 초기화 파일
```

## 📄 파일 설명

### `extension.toml`

Isaac Sim Extension 설정 파일입니다. 이 파일은 Isaac Sim에서 이 확장 패키지를 인식하고 로드하기 위한 메타데이터를 포함합니다.

**주요 설정 항목:**
- `title`: Extension 제목 ("H1 Locomotion")
- `description`: Extension 설명 ("RL environment for Unitree H1")
- `version`: Extension 버전 (0.1.0)
- `dependencies`: 필요한 의존성 패키지 (isaaclab, isaaclab_assets)
- `python.module`: Python 모듈 경로 (`rl_project_ws.exts.h1_locomotion`)

**참고**: 이 파일은 Isaac Sim이 자동으로 인식하므로 수동으로 수정할 필요는 없습니다.

### `agents/` 폴더

강화학습 알고리즘의 하이퍼파라미터 설정 파일들을 저장하는 폴더입니다.

**예상 파일 구조:**
- `rsl_rl_ppo_cfg.py`: RSL-RL 라이브러리의 PPO 알고리즘 설정
- `skrl_ppo_cfg.yaml`: SKRL 라이브러리의 PPO 알고리즘 설정

**향후 추가 예정:**
- PPO (Proximal Policy Optimization) 설정
- SAC (Soft Actor-Critic) 설정
- 기타 알고리즘 설정

## 🔧 사용 방법

### Extension 설정 확인

Extension이 제대로 등록되었는지 확인하려면 Isaac Sim의 Extension Manager에서 확인할 수 있습니다.

### 알고리즘 설정 파일 사용

학습 스크립트에서 알고리즘 설정을 import하여 사용합니다:

```python
# 예시 (향후 구현 예정)
from h1_locomotion.config.agents.rsl_rl_ppo_cfg import PPO_CFG

# 학습 시 설정 사용
trainer = PPOTrainer(PPO_CFG)
```

## 📝 개발 가이드

### 새로운 알고리즘 설정 추가하기

1. `agents/` 폴더에 새로운 설정 파일 생성
2. 해당 알고리즘의 하이퍼파라미터를 설정 클래스로 정의
3. 학습 스크립트에서 import하여 사용

### 설정 파일 네이밍 규칙

- 파일명: `{library}_{algorithm}_cfg.{ext}`
  - 예: `rsl_rl_ppo_cfg.py`, `skrl_ppo_cfg.yaml`
- 클래스명: `{Algorithm}CFG` 또는 `{Algorithm}_CFG`
  - 예: `PPOCFG`, `PPO_CFG`

## 🔗 관련 문서

- [`agents/README.md`](./agents/README.md): 알고리즘 설정 폴더 상세 설명
- [`../README.md`](../README.md): 프로젝트 메인 README

---

**마지막 업데이트**: 2025년 11월 28일

