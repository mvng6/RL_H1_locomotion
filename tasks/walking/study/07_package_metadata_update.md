# 패키지 메타데이터 수정 및 재설치 가이드

## 1. 메타데이터 수정 방법

### `pyproject.toml` 파일 수정

`exts/h1_locomotion/pyproject.toml` 파일의 `[project]` 섹션에 `authors` 필드를 추가하거나 수정합니다.

**현재 상태:**
```toml
[project]
name = "h1_locomotion"
version = "0.1.0"
description = "RL environment for Unitree H1 humanoid robot locomotion"
requires-python = ">=3.10"
authors = [
    {name = "Your Name", email = "your.email@example.com"},
]
dependencies = [
    "isaaclab",
]
```

**수정 예시:**
```toml
[project]
name = "h1_locomotion"
version = "0.1.0"
description = "RL environment for Unitree H1 humanoid robot locomotion"
requires-python = ">=3.10"
authors = [
    {name = "홍길동", email = "hong@katech.re.kr"},
]
# 또는 여러 명의 경우:
# authors = [
#     {name = "홍길동", email = "hong@katech.re.kr"},
#     {name = "김철수", email = "kim@katech.re.kr"},
# ]
dependencies = [
    "isaaclab",
]
```

### 추가 메타데이터 필드 (선택사항)

필요한 경우 다음 필드들도 추가할 수 있습니다:

```toml
[project]
name = "h1_locomotion"
version = "0.1.0"
description = "RL environment for Unitree H1 humanoid robot locomotion"
readme = "README.md"  # README 파일 경로
license = {text = "BSD-3-Clause"}  # 또는 {file = "LICENSE"}
authors = [
    {name = "홍길동", email = "hong@katech.re.kr"},
]
keywords = ["robotics", "reinforcement-learning", "humanoid", "h1"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
requires-python = ">=3.10"
dependencies = [
    "isaaclab",
]
```

## 2. 재설치 방법

### 방법 1: Isaac Lab의 Python 환경 사용 (권장)

터미널 출력을 보면 Isaac Lab의 Python 환경(`/home/ldj/IsaacLab/isaaclab.sh`)을 사용하는 것이 가장 안전합니다.

#### 단계별 재설치 과정

**1단계: 프로젝트 디렉토리로 이동**
```bash
cd /home/ldj/RL_project_ws
```

**2단계: 기존 패키지 제거 (선택사항)**
```bash
# Isaac Lab의 Python 환경에서 제거
/home/ldj/IsaacLab/isaaclab.sh -p -m pip uninstall h1_locomotion -y
```

**3단계: 재설치 (Editable 모드)**
```bash
# Editable 모드로 재설치 (코드 변경이 즉시 반영됨)
/home/ldj/IsaacLab/isaaclab.sh -p -m pip install -e exts/h1_locomotion --force-reinstall
```

**4단계: 설치 확인**
```bash
# 패키지 정보 확인
/home/ldj/IsaacLab/isaaclab.sh -p -m pip show h1_locomotion

# 예상 출력:
# Name: h1_locomotion
# Version: 0.1.0
# Summary: RL environment for Unitree H1 humanoid robot locomotion
# Author: 홍길동
# Author-email: hong@katech.re.kr
# ...
```

**5단계: Import 테스트**
```bash
# 환경 설정 import 테스트
/home/ldj/IsaacLab/isaaclab.sh -p -c "
from h1_locomotion.tasks.walking import walking_env_cfg
from h1_locomotion.config.agents import walking_ppo_cfg
print('✅ Import 성공!')
"

# 환경 등록 확인
/home/ldj/IsaacLab/isaaclab.sh -p -c "
import gymnasium as gym
from h1_locomotion.tasks import walking
envs = [env for env in gym.envs.registry.env_specs.keys() if 'H1' in env]
print('등록된 H1 환경:', envs)
if 'H1-Walking-v0' in envs:
    print('✅ 환경 등록 성공!')
"
```

### 방법 2: Conda 환경 사용

conda 환경(`env_isaaclab`)을 직접 사용하는 경우:

**1단계: Conda 환경 활성화**
```bash
conda activate env_isaaclab
```

**2단계: 기존 패키지 제거 (선택사항)**
```bash
pip uninstall h1_locomotion -y
```

**3단계: 재설치**
```bash
cd /home/ldj/RL_project_ws
pip install -e exts/h1_locomotion --force-reinstall
```

**4단계: 설치 확인**
```bash
pip show h1_locomotion
python -c "from h1_locomotion.tasks.walking import walking_env_cfg; print('✅ Import 성공!')"
```

## 3. 의존성 충돌 해결

재설치 시 의존성 충돌 경고가 나타날 수 있습니다. 이는 일반적으로 문제가 되지 않지만, Isaac Lab의 Python 환경을 사용하면 대부분 자동으로 해결됩니다.

### 의존성 충돌이 발생하는 경우

**경고 메시지 예시:**
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
torchaudio 2.7.0+cu128 requires torch==2.7.0, but you have torch 2.9.1
```

**해결 방법:**
1. Isaac Lab의 Python 환경 사용 (권장) - 의존성이 자동으로 관리됨
2. 경고 무시 - 실제 실행에는 문제가 없을 수 있음
3. 필요시 특정 패키지 버전 고정

## 4. 빠른 재설치 스크립트

다음 스크립트를 사용하면 한 번에 재설치 및 검증을 수행할 수 있습니다:

```bash
#!/bin/bash
# reinstall_h1_locomotion.sh

ISAAC_LAB_PATH="/home/ldj/IsaacLab"
PROJECT_DIR="/home/ldj/RL_project_ws"

echo "=== H1 Locomotion 패키지 재설치 ==="
echo ""

# 1. 기존 패키지 제거
echo "[1/4] 기존 패키지 제거 중..."
$ISAAC_LAB_PATH/isaaclab.sh -p -m pip uninstall h1_locomotion -y

# 2. 재설치
echo "[2/4] 패키지 재설치 중..."
cd $PROJECT_DIR
$ISAAC_LAB_PATH/isaaclab.sh -p -m pip install -e exts/h1_locomotion --force-reinstall

# 3. 설치 확인
echo "[3/4] 설치 확인 중..."
$ISAAC_LAB_PATH/isaaclab.sh -p -m pip show h1_locomotion

# 4. Import 테스트
echo "[4/4] Import 테스트 중..."
$ISAAC_LAB_PATH/isaaclab.sh -p -c "
from h1_locomotion.tasks.walking import walking_env_cfg
from h1_locomotion.config.agents import walking_ppo_cfg
import gymnasium as gym
from h1_locomotion.tasks import walking
envs = [env for env in gym.envs.registry.env_specs.keys() if 'H1' in env]
print('✅ 모든 테스트 통과!')
print('등록된 환경:', envs)
"

echo ""
echo "=== 재설치 완료 ==="
```

## 5. 체크리스트

메타데이터 수정 및 재설치 완료 후 확인사항:

- [ ] `pyproject.toml`의 `authors` 필드가 올바르게 수정됨
- [ ] 재설치 명령어 실행 완료
- [ ] `pip show h1_locomotion`에서 Author와 Author-email이 올바르게 표시됨
- [ ] Import 테스트 성공
- [ ] 환경 등록 확인 성공 (`H1-Walking-v0` 등록됨)

## 6. 문제 해결

### 문제 1: 메타데이터가 업데이트되지 않음

**증상**: `pip show`에서 여전히 Author가 비어있음

**해결 방법**:
```bash
# 완전히 제거 후 재설치
/home/ldj/IsaacLab/isaaclab.sh -p -m pip uninstall h1_locomotion -y
/home/ldj/IsaacLab/isaaclab.sh -p -m pip install -e exts/h1_locomotion --force-reinstall --no-cache-dir
```

### 문제 2: TOML 구문 오류

**증상**: 설치 시 TOML 파싱 오류 발생

**해결 방법**:
- `pyproject.toml` 파일의 구문 확인
- 특히 `authors` 필드는 리스트 형태여야 함: `[{name = "...", email = "..."}]`
- 따옴표와 쉼표 확인

### 문제 3: 의존성 충돌

**증상**: 재설치 시 의존성 충돌 경고

**해결 방법**:
- Isaac Lab의 Python 환경 사용 (권장)
- 경고가 나타나도 실제 실행에는 문제가 없을 수 있으므로 테스트 진행
- 필요시 특정 패키지 버전 고정

