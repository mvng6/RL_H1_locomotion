# 패키지 메타데이터 수정 및 재설치 가이드

## 빠른 참조

### 메타데이터 수정 후 재설치 (권장 순서)

```bash
# 1. 프로젝트 디렉토리로 이동
cd /home/ldj/RL_project_ws

# 2. 재설치
/home/ldj/IsaacLab/isaaclab.sh -p -m pip install -e exts/h1_locomotion --force-reinstall

# 3. 의존성 충돌 해결 (누락된 필수 패키지 설치)
/home/ldj/IsaacLab/isaaclab.sh -p -m pip install docstring-parser==0.16 "PyJWT[crypto]<3,>=1.0.0"

# 4. 설치 확인
/home/ldj/IsaacLab/isaaclab.sh -p -m pip show h1_locomotion

# 5. Import 테스트
/home/ldj/IsaacLab/isaaclab.sh -p -c "
from h1_locomotion.tasks.walking import walking_env_cfg
from h1_locomotion.config.agents import walking_ppo_cfg
import gymnasium as gym
from h1_locomotion.tasks import walking
envs = [env for env in gym.envs.registry.env_specs.keys() if 'H1' in env]
print('등록된 환경:', envs)
"
```

**의존성 충돌 경고 해석:**
- ✅ **무시 가능**: torch/torchaudio 버전 충돌 (Import 테스트 성공 시)
- ⚠️ **해결 필요**: docstring-parser, PyJWT[crypto] 누락 (위 명령어로 설치)
- ⚠️ **선택사항**: lxml 버전 충돌 (실제 오류 발생 시에만 해결)

---

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
nvidia-srl-base 1.3.0 requires docstring-parser==0.16, which is not installed.
nvidia-srl-usd 2.0.0 requires usd-core<26.0,>=25.2.post1; python_version >= "3.11", which is not installed.
msal 1.27.0 requires PyJWT[crypto]<3,>=1.0.0, which is not installed.
nvidia-srl-usd-to-urdf 1.0.2 requires lxml<5.0.0,>=4.9.2, but you have lxml 5.4.0 which is incompatible.
torchaudio 2.7.0+cu128 requires torch==2.7.0, but you have torch 2.9.1 which is incompatible.
```

### 단계별 의존성 충돌 해결 방법

#### 단계 1: 누락된 필수 패키지 설치

다음 명령어로 누락된 패키지를 설치합니다:

```bash
# Isaac Lab의 Python 환경에서 실행
/home/ldj/IsaacLab/isaaclab.sh -p -m pip install docstring-parser==0.16 "PyJWT[crypto]<3,>=1.0.0"
```

**설명:**
- `docstring-parser==0.16`: nvidia-srl-base가 요구하는 버전
- `PyJWT[crypto]<3,>=1.0.0`: msal이 요구하는 패키지 (crypto 확장 포함)

#### 단계 2: lxml 버전 다운그레이드 (선택사항)

lxml 버전 충돌이 실제 문제를 일으키는 경우에만 수행:

```bash
# Isaac Lab의 Python 환경에서 실행
/home/ldj/IsaacLab/isaaclab.sh -p -m pip install "lxml<5.0.0,>=4.9.2"
```

**주의**: lxml 다운그레이드는 다른 패키지에 영향을 줄 수 있으므로, 실제로 문제가 발생하는 경우에만 수행하세요.

#### 단계 3: torch/torchaudio 버전 충돌 처리

**중요**: torch 버전 충돌은 일반적으로 무시해도 됩니다. Isaac Lab이 사용하는 torch 버전(2.9.1)이 더 최신 버전이며, 실제 실행에는 문제가 없습니다.

만약 torchaudio 관련 오류가 실제로 발생하는 경우:

```bash
# Isaac Lab의 Python 환경에서 실행
# 주의: 이 명령어는 Isaac Lab의 다른 기능에 영향을 줄 수 있으므로 신중하게 사용하세요
/home/ldj/IsaacLab/isaaclab.sh -p -m pip install torchaudio --upgrade
```

#### 단계 4: usd-core 패키지 처리

`usd-core`는 Isaac Sim과 함께 설치되어야 하는 패키지입니다. 일반적으로 Isaac Lab 환경에는 이미 설치되어 있어야 합니다.

확인 방법:
```bash
/home/ldj/IsaacLab/isaaclab.sh -p -c "import usd.core; print('USD Core 설치됨')"
```

만약 설치되지 않은 경우, Isaac Sim 재설치가 필요할 수 있습니다.

#### 단계 5: Import 테스트로 실제 동작 확인

의존성 충돌 경고가 나타나도 실제로는 문제가 없을 수 있습니다. 다음 명령어로 실제 동작을 확인하세요:

```bash
# Isaac Lab의 Python 환경에서 실행
/home/ldj/IsaacLab/isaaclab.sh -p -c "
from h1_locomotion.tasks.walking import walking_env_cfg
from h1_locomotion.config.agents import walking_ppo_cfg
import gymnasium as gym
from h1_locomotion.tasks import walking
envs = [env for env in gym.envs.registry.env_specs.keys() if 'H1' in env]
print('✅ Import 성공!')
print('등록된 환경:', envs)
if 'H1-Walking-v0' in envs:
    print('✅ 환경 등록 성공!')
else:
    print('❌ 환경 등록 실패')
"
```

**성공 기준**: Import 오류 없이 환경이 등록되면 정상입니다.

### 의존성 충돌 해결 우선순위

1. **우선순위 1: 누락된 필수 패키지 설치** (필수)
   ```bash
   /home/ldj/IsaacLab/isaaclab.sh -p -m pip install docstring-parser==0.16 "PyJWT[crypto]<3,>=1.0.0"
   ```

2. **우선순위 2: Import 테스트로 실제 동작 확인** (필수)
   - Import가 성공하면 경고는 무시해도 됨

3. **우선순위 3: lxml 버전 다운그레이드** (선택사항)
   - 실제로 lxml 관련 오류가 발생하는 경우에만 수행

4. **우선순위 4: torch/torchaudio 버전 조정** (비권장)
   - Isaac Lab의 다른 기능에 영향을 줄 수 있으므로 신중하게 결정

### 완전한 의존성 충돌 해결 스크립트

다음 스크립트를 사용하면 의존성 충돌을 한 번에 해결할 수 있습니다:

```bash
#!/bin/bash
# fix_dependency_conflicts.sh

ISAAC_LAB_PATH="/home/ldj/IsaacLab"

echo "=== 의존성 충돌 해결 ==="
echo ""

# 1. 누락된 필수 패키지 설치
echo "[1/3] 누락된 필수 패키지 설치 중..."
$ISAAC_LAB_PATH/isaaclab.sh -p -m pip install docstring-parser==0.16 "PyJWT[crypto]<3,>=1.0.0"

# 2. Import 테스트
echo "[2/3] Import 테스트 중..."
$ISAAC_LAB_PATH/isaaclab.sh -p -c "
try:
    from h1_locomotion.tasks.walking import walking_env_cfg
    from h1_locomotion.config.agents import walking_ppo_cfg
    import gymnasium as gym
    from h1_locomotion.tasks import walking
    envs = [env for env in gym.envs.registry.env_specs.keys() if 'H1' in env]
    print('✅ Import 성공!')
    print('등록된 환경:', envs)
    if 'H1-Walking-v0' in envs:
        print('✅ 환경 등록 성공!')
    else:
        print('❌ 환경 등록 실패')
except Exception as e:
    print('❌ Import 실패:', e)
    exit(1)
"

# 3. lxml 버전 확인 (선택사항)
echo "[3/3] lxml 버전 확인 중..."
$ISAAC_LAB_PATH/isaaclab.sh -p -c "
import lxml
print('현재 lxml 버전:', lxml.__version__)
if lxml.__version__.startswith('5.'):
    print('⚠️  lxml 5.x 버전이 설치되어 있습니다.')
    print('   문제가 발생하면 다음 명령어로 다운그레이드하세요:')
    print('   pip install \"lxml<5.0.0,>=4.9.2\"')
else:
    print('✅ lxml 버전이 호환됩니다.')
"

echo ""
echo "=== 의존성 충돌 해결 완료 ==="
```

### 의존성 충돌 경고 해석

**경고가 나타나도 괜찮은 경우:**
- `torchaudio`와 `torch` 버전 불일치: Isaac Lab이 더 최신 torch를 사용하므로 일반적으로 문제 없음
- `lxml` 버전 불일치: 실제로 lxml 관련 오류가 발생하지 않으면 무시 가능

**반드시 해결해야 하는 경우:**
- `docstring-parser` 누락: nvidia-srl-base가 필요로 하므로 설치 필요
- `PyJWT[crypto]` 누락: msal이 필요로 하므로 설치 필요
- Import 오류 발생: 의존성 충돌이 실제 문제를 일으키는 경우

### 요약

1. **필수**: 누락된 패키지 설치 (`docstring-parser`, `PyJWT[crypto]`)
2. **확인**: Import 테스트로 실제 동작 확인
3. **선택**: 실제 문제가 발생하는 경우에만 lxml 다운그레이드
4. **무시**: torch/torchaudio 버전 충돌은 일반적으로 무시 가능

## 4. 빠른 재설치 스크립트

다음 스크립트를 사용하면 한 번에 재설치, 의존성 충돌 해결 및 검증을 수행할 수 있습니다:

```bash
#!/bin/bash
# reinstall_h1_locomotion.sh

ISAAC_LAB_PATH="/home/ldj/IsaacLab"
PROJECT_DIR="/home/ldj/RL_project_ws"

echo "=== H1 Locomotion 패키지 재설치 및 의존성 충돌 해결 ==="
echo ""

# 1. 기존 패키지 제거
echo "[1/5] 기존 패키지 제거 중..."
$ISAAC_LAB_PATH/isaaclab.sh -p -m pip uninstall h1_locomotion -y

# 2. 재설치
echo "[2/5] 패키지 재설치 중..."
cd $PROJECT_DIR
$ISAAC_LAB_PATH/isaaclab.sh -p -m pip install -e exts/h1_locomotion --force-reinstall

# 3. 의존성 충돌 해결 (누락된 필수 패키지 설치)
echo "[3/5] 의존성 충돌 해결 중 (누락된 필수 패키지 설치)..."
$ISAAC_LAB_PATH/isaaclab.sh -p -m pip install docstring-parser==0.16 "PyJWT[crypto]<3,>=1.0.0" 2>&1 | grep -v "already satisfied" || true

# 4. 설치 확인
echo "[4/5] 설치 확인 중..."
$ISAAC_LAB_PATH/isaaclab.sh -p -m pip show h1_locomotion

# 5. Import 테스트
echo "[5/5] Import 테스트 중..."
$ISAAC_LAB_PATH/isaaclab.sh -p -c "
try:
from h1_locomotion.tasks.walking import walking_env_cfg
from h1_locomotion.config.agents import walking_ppo_cfg
import gymnasium as gym
from h1_locomotion.tasks import walking
envs = [env for env in gym.envs.registry.env_specs.keys() if 'H1' in env]
print('✅ 모든 테스트 통과!')
print('등록된 환경:', envs)
    if 'H1-Walking-v0' in envs:
        print('✅ 환경 등록 성공!')
    else:
        print('⚠️  환경이 등록되지 않았습니다.')
except Exception as e:
    print('❌ Import 실패:', e)
    exit(1)
"

echo ""
echo "=== 재설치 완료 ==="
echo ""
echo "의존성 충돌 경고가 나타날 수 있지만, Import 테스트가 성공하면 정상입니다."
echo "torch/torchaudio 버전 충돌은 일반적으로 무시해도 됩니다."
```

## 5. 체크리스트

메타데이터 수정 및 재설치 완료 후 확인사항:

- [ ] `pyproject.toml`의 `authors` 필드가 올바르게 수정됨 (문자열은 따옴표로 감싸야 함)
- [ ] 재설치 명령어 실행 완료
- [ ] `pip show h1_locomotion`에서 Author와 Author-email이 올바르게 표시됨
- [ ] 의존성 충돌 해결: 누락된 필수 패키지 설치 (`docstring-parser`, `PyJWT[crypto]`)
- [ ] Import 테스트 성공 (오류 없이 import 및 환경 등록 확인)
- [ ] 환경 등록 확인 성공 (`H1-Walking-v0` 등록됨)
- [ ] 의존성 충돌 경고 해석 완료 (torch/torchaudio 경고는 무시 가능)

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

### 문제 4: Import 오류 (ModuleNotFoundError)

**증상**: `pip show`는 성공하지만 `import h1_locomotion`이 실패함

**원인**: 패키지 구조가 올바르게 인식되지 않음

**해결 방법**:

**1단계: `pyproject.toml`에 패키지 설정 추가**
```toml
[tool.setuptools]
packages = {find = {}}
```

**2단계: 완전히 제거 후 재설치**
```bash
# 기존 패키지 제거
/home/ldj/IsaacLab/isaaclab.sh -p -m pip uninstall h1_locomotion -y

# .egg-info 디렉토리 제거 (선택사항)
rm -rf /home/ldj/RL_project_ws/exts/h1_locomotion/h1_locomotion.egg-info

# 재설치
cd /home/ldj/RL_project_ws
/home/ldj/IsaacLab/isaaclab.sh -p -m pip install -e exts/h1_locomotion --force-reinstall --no-cache-dir
```

**3단계: Import 테스트**
```bash
/home/ldj/IsaacLab/isaaclab.sh -p -c "
from h1_locomotion.tasks.walking import walking_env_cfg
from h1_locomotion.config.agents import walking_ppo_cfg
print('✅ Import 성공!')
"
```

**4단계: 여전히 실패하는 경우**
- `__init__.py` 파일이 모든 패키지 디렉토리에 있는지 확인
- 패키지 구조가 올바른지 확인:
  ```
  h1_locomotion/
    __init__.py
    config/
      __init__.py
      agents/
        __init__.py
    tasks/
      __init__.py
      walking/
        __init__.py
  ```

### 문제 3: 의존성 충돌

**증상**: 재설치 시 의존성 충돌 경고

**해결 방법**:

**1단계: 누락된 필수 패키지 설치**
```bash
/home/ldj/IsaacLab/isaaclab.sh -p -m pip install docstring-parser==0.16 "PyJWT[crypto]<3,>=1.0.0"
```

**2단계: Import 테스트로 실제 동작 확인**
```bash
/home/ldj/IsaacLab/isaaclab.sh -p -c "
from h1_locomotion.tasks.walking import walking_env_cfg
from h1_locomotion.config.agents import walking_ppo_cfg
import gymnasium as gym
from h1_locomotion.tasks import walking
envs = [env for env in gym.envs.registry.env_specs.keys() if 'H1' in env]
print('등록된 환경:', envs)
"
```

**3단계: Import가 성공하면 경고는 무시 가능**
- torch/torchaudio 버전 충돌은 일반적으로 무시해도 됨
- lxml 버전 충돌은 실제 오류가 발생하는 경우에만 해결

**4단계: 실제 오류가 발생하는 경우에만 추가 조치**
```bash
# lxml 다운그레이드 (선택사항)
/home/ldj/IsaacLab/isaaclab.sh -p -m pip install "lxml<5.0.0,>=4.9.2"
```

자세한 내용은 위의 "3. 의존성 충돌 해결" 섹션을 참고하세요.

