# H1 커스텀 동작 강화학습 작업 프로세스 체크리스트

이 문서는 `H1_Custom_Action_RL_Development_Guide.md`를 기반으로 한 상세 작업 프로세스 체크리스트입니다. 각 단계를 순차적으로 완료하며 진행 상황을 체크하세요.

## 📊 현재 진행 상황 요약

**전체 진행률**: Phase 1 진행 중 (약 90% 완료 - 학습 진행 중)

### ✅ 완료된 작업
- Phase 1.1: 디렉토리 구조 생성 완료
- Phase 1.2: 관측 공간 정의 완료 (`observations.py`)
- Phase 1.3: 보상 함수 정의 완료 (`rewards.py`) - 안전성 강화 버전
- Phase 1.4: 종료 조건 정의 완료 (`terminations.py`) - 안전성 강화 버전
- Phase 1.5: MDP 모듈 초기화 완료 (`mdp/__init__.py`)
- Phase 1.6: 환경 설정 파일 작성 완료 (`walking_env_cfg.py`)
- Phase 1.7: 에이전트 설정 파일 작성 완료 (`config/agents/walking_ppo_cfg.py`)
- Phase 1.8: 환경 등록 완료 (`walking/__init__.py`)
- Phase 1.9: 메인 `__init__.py` 업데이트 완료 (`tasks/__init__.py`)
- Phase 1.10: 프로젝트 재설치 및 검증 완료
- Phase 1.11: Zero Agent 테스트 완료 (커스텀 스크립트 사용)
- Phase 1.12: 기본 보행 학습 실행 완료 (1차 학습 완료, 2차 안전성 강화 학습 진행 중)

### ⏳ 진행 중인 작업
- Phase 1.12: 안전성 강화 보상 함수로 재학습 진행 중

### 📝 다음 단계
1. **학습 완료 후 체크포인트 확인**
2. **학습된 정책 테스트 (`play_walking_ppo.py`)**
3. **결과 분석 및 보상 함수 튜닝**

---

## 목차

1. [Phase 1: 기본 보행 (Walking) 환경 구축](#phase-1-기본-보행-walking-환경-구축)
2. [Phase 2: 달리기 (Running) 환경 구축](#phase-2-달리기-running-환경-구축)
3. [Phase 3: 점프 (Jumping) 환경 구축](#phase-3-점프-jumping-환경-구축)
4. [최종 검증 및 테스트](#최종-검증-및-테스트)
5. [발생한 오류 및 해결 방법](#발생한-오류-및-해결-방법)

---

## Phase 1: 기본 보행 (Walking) 환경 구축

### 1.1 디렉토리 구조 생성

- [x] 작업 디렉토리로 이동 ✅
- [x] Walking 태스크 디렉토리 생성 확인 ✅
- [x] 필요한 파일들이 모두 존재하는지 확인 ✅

### 1.2 관측 공간 정의 (`walking/mdp/observations.py`)

**상태**: ✅ 완료됨

- [x] 파일 생성 완료
- [x] `ObservationsCfg` 클래스 정의
- [x] 관절 상태, 베이스 상태, 명령, 발 접촉 상태, 액션 히스토리 관측 항목 추가
- [x] `concatenate_terms = True` 설정

### 1.3 보상 함수 정의 (`walking/mdp/rewards.py`)

**상태**: ✅ 완료됨 (안전성 강화 버전)

**주요 보상 항목**:
| 항목 | 가중치 | 설명 |
|------|--------|------|
| `track_lin_vel_xy_exp` | 1.0 | 목표 속도 추적 |
| `track_ang_vel_z_exp` | 0.5 | 목표 각속도 추적 |
| `joint_pos_limits` | -5.0 | ⚠️ 관절 한계 페널티 (핵심!) |
| `joint_vel_l2` | -0.001 | 관절 속도 페널티 |
| `undesired_contacts` | -1.0 | 충돌 방지 (torso, pelvis 포함) |
| `flat_orientation_l2` | -2.0 | 수평 자세 유지 |
| `base_height_l2` | -0.5 | 기본 높이 유지 |
| `feet_air_time` | 0.25 | 발 공중 시간 보상 |
| `action_rate_l2` | -0.01 | 액션 변화율 제한 |
| `dof_torques_l2` | -0.0001 | 토크 사용량 제한 |
| `dof_acc_l2` | -2.5e-7 | 가속도 제한 |
| `is_alive` | 0.5 | 생존 보상 |
| `is_terminated` | -10.0 | 종료 페널티 |

### 1.4 종료 조건 정의 (`walking/mdp/terminations.py`)

**상태**: ✅ 완료됨 (안전성 강화 버전)

**주요 종료 조건**:
| 항목 | 설명 |
|------|------|
| `time_out` | 에피소드 시간 초과 |
| `base_contact` | 몸통/골반 접촉 (넘어짐) |
| `base_height` | 높이 범위 벗어남 (0.5m~1.5m) |
| `bad_orientation` | 기울기 40도 초과 |

### 1.5 MDP 모듈 초기화 (`walking/mdp/__init__.py`)

**상태**: ✅ 완료됨

### 1.6 환경 설정 파일 작성 (`walking/walking_env_cfg.py`)

**상태**: ✅ 완료됨

- [x] `H1RoughEnvCfg` 상속
- [x] 커스텀 보상 설정 적용
- [x] 커스텀 종료 조건 적용
- [x] 에피소드 길이: 10초
- [x] 속도 범위: 0~0.5 m/s (안정화용)

### 1.7 에이전트 설정 파일 작성 (`config/agents/walking_ppo_cfg.py`)

**상태**: ✅ 완료됨

### 1.8 환경 등록 (`walking/__init__.py`)

**상태**: ✅ 완료됨

- [x] `H1-Walking-v0` 환경 등록

### 1.9 메인 `__init__.py` 업데이트 (`tasks/__init__.py`)

**상태**: ✅ 완료됨

### 1.10 프로젝트 재설치 및 검증

**상태**: ✅ 완료됨

- [x] `pip install -e exts/h1_locomotion` 성공
- [x] Import 테스트 성공
- [x] 환경 등록 확인

### 1.11 Zero Agent 테스트

**상태**: ✅ 완료됨 (커스텀 스크립트 사용)

- [x] `test_walking_env.py` 스크립트 생성
- [x] 환경 생성 및 실행 확인
- [x] 100 스텝 테스트 성공

### 1.12 기본 보행 학습 실행

**상태**: ⏳ 진행 중 (2차 학습)

**1차 학습 결과** (문제 발생):
- 평균 보상: 0.023 (매우 낮음)
- 문제: 비정상적 자세, 신체 관통
- 원인: 안전성 보상 부족

**2차 학습** (안전성 강화):
- [x] 보상 함수 수정 완료
- [x] 종료 조건 강화 완료
- [ ] 학습 진행 중

**학습 명령어**:
```bash
/home/ldj/IsaacLab/isaaclab.sh -p /home/ldj/RL_project_ws/exts/h1_locomotion/scripts/train_walking_ppo.py \
    --task H1-Walking-v0 --num_envs 4096 --max_iterations 3000 --headless
```

### 1.13 학습 완료 및 체크포인트 확인

**상태**: ⏳ 대기 중

- [ ] 체크포인트 파일 확인
  ```bash
  ls -lh logs/rsl_rl/h1_walking/*/model_*.pt
  ```

### 1.14 학습된 정책 테스트

**상태**: ⏳ 대기 중

**테스트 명령어**:
```bash
/home/ldj/IsaacLab/isaaclab.sh -p /home/ldj/RL_project_ws/exts/h1_locomotion/scripts/play_walking_ppo.py \
    --task H1-Walking-v0 --num_envs 16 \
    --checkpoint /path/to/model_3000.pt
```

---

## Phase 2: 달리기 (Running) 환경 구축

**상태**: ⏳ 아직 시작하지 않음

---

## Phase 3: 점프 (Jumping) 환경 구축

**상태**: ⏳ 아직 시작하지 않음

---

## 최종 검증 및 테스트

**상태**: ⏳ 아직 시작하지 않음

---

## 발생한 오류 및 해결 방법

이 섹션은 개발 과정에서 발생한 주요 오류들과 해결 방법을 정리한 것입니다. 동일한 실수를 반복하지 않도록 참고하세요.

### 1. RslRlVecEnvWrapper의 `clip_actions` 파라미터 오류

**오류 메시지**:
```
ValueError: Box high must be a np.ndarray, integer, or float, actual type=<class 'bool'>
```

**원인**:
- `RslRlVecEnvWrapper(env, clip_actions=True)` 호출 시 발생
- 최신 Gymnasium 버전에서 `Box` 공간의 `high` 파라미터에 boolean 값이 전달됨

**해결 방법**:
```python
# ❌ 잘못된 코드
env = RslRlVecEnvWrapper(env, clip_actions=True)

# ✅ 올바른 코드
env = RslRlVecEnvWrapper(env)  # clip_actions 파라미터 제거
```

---

### 2. Hydra 데코레이터와 argparse 충돌

**오류 메시지**:
```
error: unrecognized arguments: --task --num_envs 4096 --max_iterations 3000
```

**원인**:
- `@hydra_task_config` 데코레이터가 Hydra의 argument parser를 사용
- 기존 argparse와 충돌 발생

**해결 방법**:
- Hydra 데코레이터 제거하고 직접 설정 로드
```python
# ❌ Hydra 데코레이터 사용 (충돌 발생)
@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    ...

# ✅ 직접 설정 로드
def main():
    env_cfg = parse_env_cfg(args_cli.task, ...)
    gym_registry = gym.envs.registry.get(args_cli.task)
    agent_cfg_entry_point = gym_registry.kwargs.get("rsl_rl_cfg_entry_point")
    # ... 동적 import
```

---

### 3. `get_observations()` 반환값 언패킹 오류

**오류 메시지**:
```
ValueError: too many values to unpack (expected 2)
```

**원인**:
- `RslRlVecEnvWrapper.get_observations()` 메서드의 반환값 개수가 예상과 다름

**해결 방법**:
```python
# ❌ 잘못된 코드
obs, _ = env.get_observations()

# ✅ 올바른 코드
obs = env.get_observations()
if isinstance(obs, tuple):
    obs = obs[0]
```

---

### 4. 부모 클래스가 참조하는 보상 이름 불일치

**오류 메시지**:
```
AttributeError: 'RewardsCfg' object has no attribute 'dof_torques_l2'. Did you mean: 'joint_torques_l2'?
```

**원인**:
- `H1RoughEnvCfg.__post_init__`에서 특정 보상 이름을 참조
- 커스텀 `RewardsCfg`에서 다른 이름 사용

**부모 클래스가 참조하는 이름**:
```python
# H1RoughEnvCfg.__post_init__에서 참조하는 이름들
self.rewards.undesired_contacts = None
self.rewards.flat_orientation_l2.weight = -1.0
self.rewards.dof_torques_l2.weight = 0.0      # ← 이 이름 필수!
self.rewards.action_rate_l2.weight = -0.005
self.rewards.dof_acc_l2.weight = -1.25e-7     # ← 이 이름 필수!
```

**해결 방법**:
```python
# ❌ 잘못된 이름
joint_torques_l2 = RewTerm(...)
joint_acc_l2 = RewTerm(...)

# ✅ 부모 클래스와 일치하는 이름
dof_torques_l2 = RewTerm(...)  # 이름 변경!
dof_acc_l2 = RewTerm(...)      # 이름 변경!
```

---

### 5. SceneEntityCfg 속성 이름 오류

**오류 메시지**:
```
AttributeError: 'SceneEntityCfg' object has no attribute 'asset_name'
```

**원인**:
- `SceneEntityCfg`는 `asset_name`이 아닌 `name` 속성을 사용

**해결 방법**:
```python
# ❌ 잘못된 코드
root_pos_w = env.scene[asset_cfg.asset_name].data.root_pos_w

# ✅ 올바른 코드
root_pos_w = env.scene[asset_cfg.name].data.root_pos_w
```

---

### 6. 학습 결과 비정상 (신체 관통, 비정상 자세)

**증상**:
- 평균 보상이 매우 낮음 (0.023)
- 로봇 신체 부위가 서로 관통
- 관절이 비정상적으로 꺾임

**원인 분석**:
| 문제점 | 설명 |
|--------|------|
| 관절 한계 페널티 없음 | 관절이 물리적 한계를 초과해도 페널티 없음 |
| Self-collision 체크 부족 | `undesired_contacts`가 torso, pelvis 미포함 |
| 관절 속도 페널티 없음 | 급격한 움직임에 제한 없음 |
| 부모 클래스 보상 덮어쓰기 | Isaac Lab의 검증된 보상이 손실됨 |

**해결 방법**:
```python
@configclass
class RewardsCfg:
    # 1. 관절 한계 페널티 추가 (핵심!)
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-5.0,  # 강한 페널티
    )
    
    # 2. 관절 속도 페널티 추가
    joint_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.001,
    )
    
    # 3. 충돌 감지 범위 확대
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[".*torso.*", ".*pelvis.*", ".*thigh.*", ".*calf.*", ".*hip.*"]
            ),
        },
    )
    
    # 4. 생존/종료 보상 추가
    is_alive = RewTerm(func=mdp.is_alive, weight=0.5)
    is_terminated = RewTerm(func=mdp.is_terminated, weight=-10.0)
```

---

### 7. 커스텀 환경이 Isaac Lab 스크립트에서 인식되지 않음

**증상**:
- `zero_agent.py` 실행 시 `H1-Walking-v0` 환경을 찾을 수 없음
- `train.py` 실행 시 환경 등록 오류

**원인**:
- Isaac Lab의 기본 스크립트는 커스텀 확장 패키지를 자동으로 import하지 않음

**해결 방법**:
- 커스텀 스크립트 작성하여 명시적으로 import
```python
# 필수! 환경 등록을 위해 명시적 import
import h1_locomotion.tasks  # noqa: F401

# 이후 환경 사용
env = gym.make("H1-Walking-v0", cfg=env_cfg)
```

---

### 요약: 주의해야 할 핵심 사항

1. **부모 클래스 상속 시**: 부모 클래스의 `__post_init__`에서 참조하는 속성 이름을 반드시 확인
2. **SceneEntityCfg 사용 시**: `name` 속성 사용 (`asset_name` 아님)
3. **RSL-RL Wrapper 사용 시**: `clip_actions` 파라미터 사용 주의
4. **커스텀 환경 사용 시**: 반드시 `import h1_locomotion.tasks` 명시
5. **보상 함수 설계 시**: 안전성 관련 페널티 (관절 한계, 충돌 방지) 반드시 포함

---

**최종 업데이트**: 2025-12-06  
**작성자**: AI Assistant  
**버전**: 2.0
