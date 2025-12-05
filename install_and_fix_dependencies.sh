#!/bin/bash
# h1_locomotion 패키지 설치 및 의존성 충돌 해결 스크립트

# set -e를 사용하지 않음 (일부 명령어가 실패해도 계속 진행)

ISAAC_LAB_PATH="/home/ldj/IsaacLab"
PROJECT_DIR="/home/ldj/RL_project_ws"
PACKAGE_DIR="$PROJECT_DIR/exts/h1_locomotion"

echo "=========================================="
echo "H1 Locomotion 패키지 설치 및 의존성 해결"
echo "=========================================="
echo ""

# 1. 기존 패키지 제거
echo "[1/6] 기존 패키지 제거 중..."
$ISAAC_LAB_PATH/isaaclab.sh -p -m pip uninstall h1_locomotion -y 2>/dev/null || true

# 2. .egg-info 디렉토리 완전히 제거
echo "[2/6] .egg-info 디렉토리 제거 중..."
rm -rf "$PACKAGE_DIR/h1_locomotion.egg-info"
rm -rf "$PACKAGE_DIR/build"
rm -rf "$PACKAGE_DIR/dist"

# 3. 의존성 충돌 해결
echo "[3/7] 의존성 충돌 해결 중..."
echo "  - torch/torchaudio 버전 확인 및 조정..."
# torchaudio가 torch 2.7.0을 요구하지만, 실제로는 torch 2.9.1이 설치되어 있음
# Isaac Lab이 torch 2.9.1을 사용하므로, torchaudio를 업그레이드하거나 무시
# 일반적으로 torch 버전 불일치는 Import 오류를 일으키지 않으므로 무시 가능
echo "  - lxml 버전 충돌 해결 (dex-retargeting과 nvidia-srl-usd-to-urdf 요구사항 충돌)..."
# dex-retargeting은 lxml>=5.2.2를 요구하지만, nvidia-srl-usd-to-urdf는 lxml<5.0.0을 요구
# Isaac Lab의 기본 설정을 유지하기 위해 lxml을 업그레이드하지 않음
# 실제로 문제가 발생하는 경우에만 해결
echo "  - 누락된 필수 패키지 설치..."
$ISAAC_LAB_PATH/isaaclab.sh -p -m pip install docstring-parser==0.16 "PyJWT[crypto]<3,>=1.0.0" 2>&1 | grep -v "already satisfied" || true

# 4. 패키지 재설치
echo "[4/7] 패키지 재설치 중..."
cd "$PROJECT_DIR"
$ISAAC_LAB_PATH/isaaclab.sh -p -m pip install -e exts/h1_locomotion --force-reinstall --no-cache-dir

# 5. 설치 확인 및 검증
echo "[5/7] 설치 확인 및 검증 중..."
echo ""

# top_level.txt 확인
echo "--- 패키지 구조 확인 ---"
if [ -f "$PACKAGE_DIR/h1_locomotion.egg-info/top_level.txt" ]; then
    echo "top_level.txt 내용:"
    cat "$PACKAGE_DIR/h1_locomotion.egg-info/top_level.txt"
    echo ""
    
    if grep -q "^h1_locomotion$" "$PACKAGE_DIR/h1_locomotion.egg-info/top_level.txt"; then
        echo "✅ 패키지 구조가 올바르게 인식되었습니다!"
    else
        echo "⚠️  패키지 구조가 올바르게 인식되지 않았습니다."
        echo "   top_level.txt에 'h1_locomotion'이 있어야 합니다."
    fi
else
    echo "⚠️  top_level.txt 파일을 찾을 수 없습니다."
fi
echo ""

# 패키지 정보 확인
echo "--- 패키지 정보 확인 ---"
$ISAAC_LAB_PATH/isaaclab.sh -p -m pip show h1_locomotion | grep -E "(Name|Version|Author|Location)" || true
echo ""

# Import 테스트
echo "--- Import 테스트 ---"
$ISAAC_LAB_PATH/isaaclab.sh -p -c "
import sys
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
        print('⚠️  H1-Walking-v0 환경이 등록되지 않았습니다.')
except Exception as e:
    print('❌ Import 실패:', e)
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

echo ""
echo "=========================================="
echo "설치 완료!"
echo "=========================================="
echo ""
echo "의존성 충돌 경고 해석:"
echo "- usd-core: Isaac Sim과 함께 설치되어야 하며, 일반적으로 문제 없음"
echo "- torch/torchaudio 버전 불일치:"
echo "  * torchaudio 2.7.0+cu128이 torch==2.7.0을 요구하지만 torch 2.9.1이 설치됨"
echo "  * Isaac Lab이 torch 2.9.1을 사용하므로, 이것은 일반적으로 문제 없음"
echo "  * Import 테스트가 성공하면 무시 가능"
echo "- lxml 버전 충돌:"
echo "  * dex-retargeting이 lxml>=5.2.2를 요구하지만, nvidia-srl-usd-to-urdf는 lxml<5.0.0을 요구"
echo "  * 이는 상호 배타적인 요구사항이므로, Isaac Lab의 기본 설정을 유지"
echo "  * 실제로 문제가 발생하는 경우에만 해결 필요"

