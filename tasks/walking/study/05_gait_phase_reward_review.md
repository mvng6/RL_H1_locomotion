# 위상 기반 보상 함수 검토 보고서

## 📋 개요

`test_rewards.py`에 작성된 `gait_phase_tracking` 함수가 실제로 사람처럼 자연스럽게 보행하기 위한 보상 함수인지 검토합니다.

---

## 🔍 현재 구현 분석

### 함수 구조

```python
def gait_phase_tracking(env, command_name: str, cycle_time: float = 0.8):
    # 1. 시간 기반 위상 계산
    current_time = env.episode_length_buf * env.step_dt
    phase = (current_time % cycle_time) / cycle_time
    
    # 2. 사인파 기반 목표 접촉 상태 생성
    desired_contact_left = torch.sin(2 * pi * phase)
    desired_contact_right = torch.sin(2 * pi * phase + pi)
    
    # 3. 발 높이 기반 보상 계산
    left_foot_pos = env.scene["robot"].data.body_pos_w[:, ...]
    right_foot_pos = env.scene["robot"].data.body_pos_w[:, ...]
    
    # 4. Swing/Stance 보상 계산
    reward_swing_left = torch.clamp(desired_contact_left, min=0) * left_foot_pos
    reward_stance_left = torch.clamp(-desired_contact_left, min=0) * torch.exp(-torch.abs(left_foot_pos) * 10)
```

---

## ⚠️ 발견된 문제점

### 1. **링크 이름 불확실성** (Critical)

**문제:**
```python
left_foot_pos = env.scene["robot"].data.body_pos_w[:, env.scene["robot"].find_bodies("left_ankle_link")[0], 2]
right_foot_pos = env.scene["robot"].data.body_pos_w[:, env.scene["robot"].find_bodies("right_ankle_link")[0], 2]
```

- `left_ankle_link`, `right_ankle_link`가 실제 H1 로봇의 링크 이름인지 확인되지 않았습니다.
- `find_bodies()`가 빈 리스트를 반환하면 `IndexError`가 발생합니다.
- Isaac Lab의 표준 접근 방식은 정규표현식(`.*ankle_link`)을 사용합니다.

**영향:** 코드 실행 시 오류 발생 가능성이 높습니다.

---

### 2. **접촉 힘 미사용** (Major)

**문제:**
- 주석으로만 남겨두고 실제로는 발 높이만 사용합니다.
- 실제 보행에서는 **접촉 힘(Contact Force)**이 Swing/Stance 위상을 판단하는 가장 정확한 지표입니다.

**현재 코드:**
```python
# (주의) 아래 코드는 예시이며, 실제 환경의 센서 텐서 구조에 맞춰야 합니다.
contact_forces = env.scene["contact_forces"].data.net_forces_w_history[:, 0, :, 2]
# ... 하지만 실제로는 사용하지 않음
```

**영향:** 
- 발 높이만으로는 접촉 상태를 정확히 판단할 수 없습니다.
- 예: 발이 땅에 닿았지만 높이가 낮은 경우, 발이 공중에 있지만 높이가 낮은 경우를 구분할 수 없습니다.

---

### 3. **위상 계산 방식의 문제** (Major)

**문제:**
```python
current_time = env.episode_length_buf * env.step_dt
phase = (current_time % cycle_time) / cycle_time
```

- **고정된 주기(`cycle_time`)**: 실제 보행 속도와 무관하게 고정된 주기(0.8초)를 사용합니다.
- **에피소드 리셋 시 위상 초기화**: 에피소드가 리셋되면 위상이 0으로 돌아가지만, 로봇의 실제 보행 상태와 동기화되지 않습니다.

**실제 인간 보행:**
- 보행 속도에 따라 주기가 달라집니다 (빠르게 걷으면 주기가 짧아짐).
- 각 발의 접촉 상태를 기반으로 위상을 추정해야 합니다.

**영향:** 로봇이 실제 보행 속도와 무관하게 고정된 리듬을 따르려고 시도하여, 자연스럽지 않은 보행 패턴이 학습될 수 있습니다.

---

### 4. **보상 함수 로직의 문제** (Moderate)

**문제:**

**Swing 보상:**
```python
reward_swing_left = torch.clamp(desired_contact_left, min=0) * left_foot_pos
```
- 발 높이(`left_foot_pos`)가 절대값이므로, 발이 땅 아래로 가면 음수가 되어 보상이 음수가 됩니다.
- 하지만 발 높이만으로는 충분한 클리어런스(clearance)를 보장할 수 없습니다.

**Stance 보상:**
```python
reward_stance_left = torch.clamp(-desired_contact_left, min=0) * torch.exp(-torch.abs(left_foot_pos) * 10)
```
- 발 높이가 0에 가까울수록 보상이 커지지만, 실제 접촉 힘을 확인하지 않습니다.
- 발이 땅에 닿지 않았는데 높이만 낮으면 잘못된 보상을 받을 수 있습니다.

**영향:** 보상 신호가 모호하여 학습이 불안정할 수 있습니다.

---

### 5. **사인파 기반 위상의 한계** (Moderate)

**문제:**
```python
desired_contact_left = torch.sin(2 * pi * phase)
desired_contact_right = torch.sin(2 * pi * phase + pi)
```

- 사인파는 부드러운 전이를 제공하지만, 실제 인간 보행 패턴과 다를 수 있습니다.
- 실제 보행에서는 Swing과 Stance의 비율이 보행 속도에 따라 달라집니다 (빠르게 걷으면 Swing 비율이 증가).

**영향:** 자연스러운 보행 패턴을 학습하기 어려울 수 있습니다.

---

### 6. **Isaac Lab 표준 방식과의 차이** (Moderate)

**Isaac Lab의 표준 접근:**
- `mdp.feet_air_time_positive_biped`: 실제 접촉 힘을 기반으로 발 공중 시간을 계산합니다.
- 접촉 힘의 임계값(`threshold`)을 사용하여 Swing/Stance를 판단합니다.
- 보행 속도에 따라 자동으로 적응합니다.

**현재 구현:**
- 시간 기반 고정 주기를 사용합니다.
- 접촉 힘을 사용하지 않습니다.

**영향:** Isaac Lab의 검증된 방식과 다르므로, 예상치 못한 동작이 발생할 수 있습니다.

---

## ✅ 개선 방안

### 1. **접촉 힘 기반 위상 추정** (권장)

```python
def gait_phase_tracking(env, command_name: str, threshold: float = 1.0):
    """
    접촉 힘 기반 위상 추정 보상 함수.
    
    실제 접촉 힘을 사용하여 각 발의 Swing/Stance 위상을 추정하고,
    목표 위상과의 일치도를 보상으로 제공합니다.
    """
    # 1. 접촉 힘 가져오기 (Z축 수직 힘)
    contact_sensor = env.scene["contact_forces"]
    contact_forces_z = contact_sensor.data.net_forces_w_history[:, 0, :, 2]  # (num_envs, num_feet)
    
    # 2. 접촉 상태 판단 (임계값 기반)
    in_contact = torch.abs(contact_forces_z) > threshold  # (num_envs, num_feet)
    
    # 3. 각 발의 위상 추정 (접촉 힘 기반)
    # 접촉 중이면 Stance, 비접촉이면 Swing
    left_contact = in_contact[:, 0]  # 첫 번째 발 (왼발)
    right_contact = in_contact[:, 1]  # 두 번째 발 (오른발)
    
    # 4. 목표: 왼발과 오른발이 교대로 접촉
    # 왼발이 접촉 중일 때 오른발은 비접촉이어야 함 (반대도 마찬가지)
    reward_alternating = (
        (left_contact.float() * (1 - right_contact.float())) +
        (right_contact.float() * (1 - left_contact.float()))
    )
    
    # 5. 동시 접촉 페널티 (양발이 동시에 땅에 닿거나 떨어지면 페널티)
    penalty_double_contact = (left_contact & right_contact).float()
    penalty_double_air = (~left_contact & ~right_contact).float()
    
    total_reward = reward_alternating - 0.5 * (penalty_double_contact + penalty_double_air)
    
    return total_reward
```

**장점:**
- 실제 접촉 상태를 기반으로 하므로 정확합니다.
- 보행 속도에 자동으로 적응합니다.
- Isaac Lab의 표준 방식과 일치합니다.

---

### 2. **발 높이 + 접촉 힘 조합** (대안)

```python
def gait_phase_tracking_hybrid(env, command_name: str, threshold: float = 1.0):
    """
    접촉 힘과 발 높이를 조합한 위상 보상 함수.
    """
    # 1. 접촉 힘 가져오기
    contact_sensor = env.scene["contact_forces"]
    contact_forces_z = contact_sensor.data.net_forces_w_history[:, 0, :, 2]
    in_contact = torch.abs(contact_forces_z) > threshold
    
    # 2. 발 높이 가져오기 (정규표현식 사용)
    robot = env.scene["robot"]
    ankle_body_ids = robot.find_bodies(".*ankle_link")
    
    if len(ankle_body_ids) < 2:
        return torch.zeros(env.num_envs, device=env.device)
    
    left_foot_height = robot.data.body_pos_w[:, ankle_body_ids[0], 2]
    right_foot_height = robot.data.body_pos_w[:, ankle_body_ids[1], 2]
    
    # 3. Swing Phase: 접촉하지 않으면서 발 높이가 충분히 높아야 함
    left_swing = (~in_contact[:, 0]) & (left_foot_height > 0.05)  # 최소 클리어런스 5cm
    right_swing = (~in_contact[:, 1]) & (right_foot_height > 0.05)
    
    # 4. Stance Phase: 접촉하면서 발 높이가 낮아야 함
    left_stance = in_contact[:, 0] & (left_foot_height < 0.02)
    right_stance = in_contact[:, 1] & (right_foot_height < 0.02)
    
    # 5. 보상 계산
    reward_swing = (left_swing.float() + right_swing.float()) * 0.5
    reward_stance = (left_stance.float() + right_stance.float()) * 0.5
    
    # 6. 교대 보행 보상 (한 발이 Swing일 때 다른 발은 Stance)
    reward_alternating = (
        (left_swing & right_stance).float() +
        (right_swing & left_stance).float()
    )
    
    total_reward = reward_swing + reward_stance + reward_alternating
    
    return total_reward
```

**장점:**
- 접촉 힘과 발 높이를 모두 활용하여 더 정확합니다.
- 클리어런스를 명시적으로 보장합니다.

---

### 3. **보행 속도 적응형 주기** (고급)

```python
def gait_phase_tracking_adaptive(env, command_name: str):
    """
    보행 속도에 따라 주기가 자동으로 조정되는 위상 보상 함수.
    """
    # 1. 현재 보행 속도 가져오기
    commands = env.command_manager.get_command(command_name)
    target_vel = commands[:, 0]  # 전진 속도 (x축)
    
    # 2. 속도에 따른 주기 계산 (빠를수록 주기가 짧아짐)
    # 기본 주기: 0.8초, 최소 주기: 0.4초
    cycle_time = 0.8 - 0.4 * torch.clamp(target_vel / 2.0, 0, 1)
    
    # 3. 접촉 힘 기반 위상 추정
    contact_sensor = env.scene["contact_forces"]
    contact_forces_z = contact_sensor.data.net_forces_w_history[:, 0, :, 2]
    in_contact = torch.abs(contact_forces_z) > 1.0
    
    # 4. 위상 기반 보상 (속도 적응형)
    # ... (접촉 힘 기반 위상 추정 로직)
    
    return total_reward
```

**장점:**
- 보행 속도에 따라 자연스럽게 적응합니다.
- 더 현실적인 보행 패턴을 학습할 수 있습니다.

---

## 📊 비교 분석

| 항목 | 현재 구현 | 개선 방안 1 (접촉 힘 기반) | 개선 방안 2 (하이브리드) |
|------|----------|---------------------------|------------------------|
| **접촉 상태 판단** | 발 높이만 사용 | 접촉 힘 사용 | 접촉 힘 + 발 높이 |
| **위상 추정** | 고정 주기 (0.8초) | 접촉 힘 기반 | 접촉 힘 기반 |
| **보행 속도 적응** | 없음 | 자동 적응 | 자동 적응 |
| **정확도** | 낮음 | 높음 | 매우 높음 |
| **구현 복잡도** | 중간 | 낮음 | 중간 |
| **Isaac Lab 호환성** | 낮음 | 높음 | 높음 |

---

## 🎯 결론 및 권장사항

### 현재 구현의 문제점 요약

1. ❌ **링크 이름 불확실성**: `left_ankle_link`, `right_ankle_link` 사용 시 오류 발생 가능
2. ❌ **접촉 힘 미사용**: 가장 정확한 접촉 상태 지표를 사용하지 않음
3. ❌ **고정 주기**: 보행 속도와 무관하게 고정된 주기 사용
4. ⚠️ **보상 로직 모호**: 발 높이만으로는 정확한 위상 판단 어려움
5. ⚠️ **Isaac Lab 표준과 불일치**: 검증된 방식과 다름

### 권장사항

1. **즉시 수정 필요:**
   - 링크 이름을 정규표현식(`.*ankle_link`)으로 변경
   - 접촉 힘을 실제로 사용하도록 수정

2. **개선 방안 1 (접촉 힘 기반) 권장:**
   - 구현이 간단하고 정확합니다.
   - Isaac Lab의 표준 방식과 일치합니다.
   - 보행 속도에 자동으로 적응합니다.

3. **대안:**
   - 현재 `mdp.feet_air_time_positive_biped`를 사용하는 것이 더 안정적일 수 있습니다.
   - 위상 기반 보상을 추가하려면 접촉 힘 기반 구현을 권장합니다.

### 최종 평가

**현재 구현:** ⚠️ **부분적으로 작동하지만 자연스러운 보행을 보장하지 않음**

**주요 이유:**
- 접촉 힘을 사용하지 않아 정확한 위상 판단이 어렵습니다.
- 고정 주기로 인해 보행 속도에 적응하지 못합니다.
- 링크 이름 문제로 실행 시 오류가 발생할 수 있습니다.

**개선 후 예상:** ✅ **자연스러운 보행 패턴 학습 가능**

---

## 📚 참고 자료

1. **Isaac Lab 표준 방식:**
   - `mdp.feet_air_time_positive_biped`: 접촉 힘 기반 발 공중 시간 계산
   - `mdp.contact_forces`: 접촉 힘 데이터 접근

2. **인간 보행 생체역학:**
   - Swing Phase: 약 40% (보행 속도에 따라 변동)
   - Stance Phase: 약 60% (보행 속도에 따라 변동)
   - 보행 주기: 속도에 따라 0.5초 ~ 1.2초

3. **관련 논문:**
   - "Learning to Walk in the Wild: A Lightweight, Hierarchical Approach" (2021)
   - "Learning Agile and Dynamic Motor Skills for Legged Robots" (2019)

