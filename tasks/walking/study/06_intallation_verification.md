# 설치 검증 가이드

## 설치 상태 확인

터미널 출력을 보면 `h1_locomotion-0.1.0`이 성공적으로 설치되었습니다. 하지만 의존성 충돌 경고가 있습니다.

## 설치 검증 방법

### 1. Conda 환경 활성화

```bash
# conda 환경 활성화
conda activate env_isaaclab

# 또는 Isaac Lab의 Python 환경 사용
# /path/to/IsaacLab/isaaclab.sh -p
```

### 2. 패키지 설치 확인

```bash
# conda 환경에서 실행
python -m pip show h1_locomotion

# 또는 Isaac Lab의 Python 사용
/path/to/IsaacLab/isaaclab.sh -p -m pip show h1_locomotion
```

**예상 출력:**
```
Name: h1_locomotion
Version: 0.1.0
Location: /home/ldj/miniconda3/envs/env_isaaclab/lib/python3.11/site-packages
Editable project location: /home/ldj/RL_project_ws/exts/h1_locomotion
```

### 3. Import 테스트

```bash
# conda 환경에서 실행
python -c "
from h1_locomotion.tasks.walking import walking_env_cfg
from h1_locomotion.config.agents import walking_ppo_cfg
print('✅ Import 성공!')
"

# 또는 Isaac Lab의 Python 사용
/path/to/IsaacLab/isaaclab.sh -p -c "
from h1_locomotion.tasks.walking import walking_env_cfg
from h1_locomotion.config.agents import walking_ppo_cfg
print('✅ Import 성공!')
"
```

### 4. 환경 등록 확인

```bash
# conda 환경에서 실행
python -c "
import gymnasium as gym
from h1_locomotion.tasks import walking
envs = [env for env in gym.envs.registry.env_specs.keys() if 'H1' in env]
print('등록된 H1 환경:', envs)
if 'H1-Walking-v0' in envs:
    print('✅ 환경 등록 성공!')
else:
    print('❌ 환경 등록 실패')
"

# 또는 Isaac Lab의 list_envs.py 사용
/path/to/IsaacLab/isaaclab.sh -p scripts/environments/list_envs.py | grep H1
```

## 의존성 충돌 해결 방법

### 경고 메시지 분석

터미널 출력에서 다음 경고가 나타났습니다:

1. **torch 버전 충돌**:
   ```
   torchaudio 2.7.0+cu128 requires torch==2.7.0, but you have torch 2.9.1
   ```
   - **영향**: torchaudio가 제대로 작동하지 않을 수 있음
   - **해결**: Isaac Lab이 사용하는 torch 버전에 맞춰야 함

2. **lxml 버전 충돌**:
   ```
   nvidia-srl-usd-to-urdf requires lxml<5.0.0,>=4.9.2, but you have lxml 5.4.0
   ```
   - **영향**: USD to URDF 변환 기능에 문제가 있을 수 있음
   - **해결**: lxml 버전 다운그레이드 필요

3. **누락된 패키지**:
   - `docstring-parser==0.16`
   - `usd-core`
   - `PyJWT[crypto]`

### 해결 방법

#### 방법 1: Isaac Lab의 Python 환경 사용 (권장)

Isaac Lab의 Python 환경을 사용하면 의존성 충돌이 자동으로 해결됩니다:

```bash
# Isaac Lab의 Python 환경에서 설치
/home/ldj/IsaacLab/isaaclab.sh -p -m pip install -e /home/ldj/RL_project_ws/exts/h1_locomotion --force-reinstall

# Isaac Lab의 Python 환경에서 테스트
/home/ldj/IsaacLab/isaaclab.sh -p -c "
from h1_locomotion.tasks.walking import walking_env_cfg
import gymnasium as gym
from h1_locomotion.tasks import walking
print('환경 등록 확인:', 'H1-Walking-v0' in gym.envs.registry.env_specs)
"
```

#### 방법 2: 누락된 패키지 설치

```bash
# conda 환경에서 실행
conda activate env_isaaclab
pip install docstring-parser==0.16 PyJWT[crypto]

# usd-core는 Isaac Sim과 함께 설치되어야 함
```

#### 방법 3: lxml 버전 다운그레이드

```bash
# conda 환경에서 실행
conda activate env_isaaclab
pip install "lxml<5.0.0,>=4.9.2"
```

## 설치 성공 기준

다음 조건을 모두 만족하면 설치가 성공한 것입니다:

- [x] `h1_locomotion-0.1.0` 패키지가 설치됨 ✅
- [ ] 환경 설정 import 성공 (`walking_env_cfg`, `walking_ppo_cfg`)
- [ ] MDP 모듈 import 성공 (`ObservationsCfg`, `RewardsCfg`, `TerminationsCfg`)
- [ ] `H1-Walking-v0` 환경이 등록됨
- [ ] Import 오류 없음

## 다음 단계

설치가 확인되면 다음 단계로 진행하세요:

1. **Zero Agent 테스트**: 환경이 올바르게 작동하는지 확인
2. **기본 보행 학습 실행**: 실제 학습 시작

