# Agents 폴더

이 폴더는 강화학습 알고리즘의 하이퍼파라미터 설정 파일들을 저장합니다.

## 📁 폴더 구조

```
agents/
└── __init__.py         # 패키지 초기화 파일
```

## 🎯 목적

강화학습 알고리즘의 하이퍼파라미터를 별도의 설정 파일로 관리하여:
- **재현성 향상**: 설정 파일로 실험 재현 가능
- **유연성**: 다양한 알고리즘 및 하이퍼파라미터 조합 쉽게 테스트
- **관리 용이성**: 설정 파일만 수정하여 실험 변경

## 📋 예상 파일 구조

향후 다음과 같은 파일들이 추가될 예정입니다:

```
agents/
├── rsl_rl_ppo_cfg.py      # RSL-RL 라이브러리의 PPO 설정
├── skrl_ppo_cfg.yaml      # SKRL 라이브러리의 PPO 설정
├── skrl_sac_cfg.yaml      # SKRL 라이브러리의 SAC 설정
└── __init__.py
```

## 🔧 설정 파일 예시

### Python 설정 파일 예시 (`rsl_rl_ppo_cfg.py`)

```python
from dataclasses import dataclass

@dataclass
class PPO_CFG:
    """PPO 알고리즘 하이퍼파라미터 설정"""
    
    # 학습률
    learning_rate: float = 3.0e-4
    
    # 배치 크기
    batch_size: int = 4096
    
    # 에포크 수
    num_epochs: int = 5
    
    # 클리핑 범위
    clip_range: float = 0.2
    
    # 가치 함수 손실 계수
    value_loss_coef: float = 0.5
    
    # 엔트로피 계수
    entropy_coef: float = 0.0
```

### YAML 설정 파일 예시 (`skrl_ppo_cfg.yaml`)

```yaml
# PPO 알고리즘 설정
algorithm:
  learning_rate: 3.0e-4
  batch_size: 4096
  num_epochs: 5
  clip_range: 0.2
  value_loss_coef: 0.5
  entropy_coef: 0.0
```

## 📝 사용 방법

### 설정 파일 Import 및 사용

```python
# Python 설정 파일 사용
from h1_locomotion.config.agents.rsl_rl_ppo_cfg import PPO_CFG

# 설정 인스턴스 생성
cfg = PPO_CFG()

# 학습에 사용
trainer = PPOTrainer(cfg)
```

```python
# YAML 설정 파일 사용
import yaml
from pathlib import Path

# 설정 파일 로드
cfg_path = Path(__file__).parent / "skrl_ppo_cfg.yaml"
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

# 학습에 사용
trainer = PPOTrainer(**cfg['algorithm'])
```

## 🛠️ 개발 가이드

### 새로운 알고리즘 설정 추가하기

1. **파일 생성**
   - Python 설정: `{library}_{algorithm}_cfg.py`
   - YAML 설정: `{library}_{algorithm}_cfg.yaml`

2. **설정 클래스/구조 정의**
   - Python: `@dataclass` 또는 `@configclass` 사용
   - YAML: 계층적 구조로 정의

3. **하이퍼파라미터 설정**
   - 알고리즘별 필수 하이퍼파라미터 포함
   - 기본값 설정
   - 주석으로 각 파라미터 설명 추가

4. **문서화**
   - 각 파라미터의 의미와 권장 범위 명시
   - 사용 예시 추가

### 네이밍 규칙

- **파일명**: `{library}_{algorithm}_cfg.{ext}`
  - 소문자와 언더스코어 사용
  - 예: `rsl_rl_ppo_cfg.py`, `skrl_sac_cfg.yaml`

- **클래스명**: `{Algorithm}CFG` 또는 `{Algorithm}_CFG`
  - 파스칼 케이스 사용
  - 예: `PPOCFG`, `SAC_CFG`

## 📚 참고 자료

- [RSL-RL 문서](https://github.com/leggedrobotics/rsl_rl)
- [SKRL 문서](https://skrl.readthedocs.io/)
- [PPO 알고리즘 논문](https://arxiv.org/abs/1707.06347)

## 🔗 관련 문서

- [`../README.md`](../README.md): Config 폴더 상세 설명
- [`../../README.md`](../../README.md): 프로젝트 메인 README

---

**마지막 업데이트**: 2025년 11월 28일

