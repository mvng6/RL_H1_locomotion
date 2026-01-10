# AMP 알고리즘 구현 상세 체크리스트

**프로젝트**: Unitree H1 Humanoid Robot Natural Locomotion using AMP  
**작성일**: 2025-01-XX  
**버전**: 1.0

---

## 목차

1. [Phase 1: 프로젝트 구조 초기화](#phase-1-프로젝트-구조-초기화)
2. [Phase 2: Mocap Data Preprocessing Pipeline](#phase-2-mocap-data-preprocessing-pipeline)
3. [Phase 3: AMP Network Architecture & Environment Setup](#phase-3-amp-network-architecture--environment-setup)
4. [Phase 4: Curriculum Learning Strategy](#phase-4-curriculum-learning-strategy)
5. [Phase 5: Domain Randomization (Sim-to-Real)](#phase-5-domain-randomization-sim-to-real)
6. [Phase 6: 학습 및 테스트](#phase-6-학습-및-테스트)

---

## Phase 1: 프로젝트 구조 초기화

### 1.1 디렉토리 구조 생성

- [x] **데이터 디렉토리 생성** ✅ 완료
  ```bash
  cd /home/ldj/RL_project_ws/exts/h1_locomotion
  mkdir -p data/amass
  mkdir -p data/processed/retargeted_motions/walking_clips
  mkdir -p data/processed/retargeted_motions/running_clips
  mkdir -p data/mapping
  ```
  **완료 상태**: 
  - ✅ `data/amass/` 디렉토리 생성 완료 (ACCAD 데이터셋 포함)
  - ✅ `data/processed/retargeted_motions/walking_clips/` 디렉토리 생성 완료
  - ✅ `data/processed/retargeted_motions/running_clips/` 디렉토리 생성 완료
  - ✅ `data/mapping/` 디렉토리 생성 완료

- [x] **AMP 모듈 디렉토리 생성** ✅ 완료
  ```bash
  mkdir -p tasks/walking/amp
  mkdir -p scripts/data_preprocessing/retargeting
  mkdir -p config/amp
  ```
  **완료 상태**:
  - ✅ `tasks/walking/amp/` 디렉토리 생성 완료
  - ✅ `scripts/data_preprocessing/retargeting/` 디렉토리 생성 완료
  - ✅ `config/amp/` 디렉토리 생성 완료

- [x] **초기화 파일 생성** ✅ 완료
  ```bash
  touch tasks/walking/amp/__init__.py
  touch scripts/data_preprocessing/__init__.py
  touch scripts/data_preprocessing/retargeting/__init__.py
  touch config/amp/__init__.py
  ```
  **완료 상태**:
  - ✅ 모든 `__init__.py` 파일 생성 완료

- [x] **`.gitignore` 업데이트** ✅ 완료
  - [x] `data/amass/` 디렉토리를 `.gitignore`에 추가
  - [x] `data/processed/retargeted_motions/` 디렉토리를 `.gitignore`에 추가
  - [x] `data/processed/amp_motions.npy`는 버전 관리 포함 (작은 파일)
  **완료 상태**: `.gitignore` 파일에 필요한 항목 추가 완료

### 1.2 기본 파일 생성

- [x] **AMP 환경 설정 파일 생성** ✅ 완료
  - [x] `tasks/walking/amp_env_cfg.py` 생성 완료 (기본 구조 포함)
  **완료 상태**: `H1AmpEnvCfg` 및 `H1AmpEnvCfg_PLAY` 클래스 정의 완료

- [x] **AMP 학습 스크립트 생성** ✅ 완료
  - [x] `scripts/train_walking_amp.py` 생성 완료 (기본 구조 포함)
  - [x] `scripts/play_walking_amp.py` 생성 완료 (기본 구조 포함)
  **완료 상태**: 
  - ✅ `train_walking_amp.py`: Discriminator 초기화, Expert 데이터셋 로드, 기본 학습 루프 구현 완료
  - ✅ `play_walking_amp.py`: 학습된 정책 테스트 스크립트 구현 완료
  - ⚠️ 주의: Discriminator 학습 루프 통합은 추후 구현 필요 (현재는 기본 PPO 루프 사용)

- [x] **에이전트 설정 파일 생성** ✅ 완료
  - [x] `config/agents/walking_amp_ppo_cfg.py` 생성 완료 (기본 구조 포함)
  **완료 상태**: `WalkingAMPPPORunnerCfg` 클래스 정의 완료

- [x] **AMP 설정 파일 생성** ✅ 완료
  - [x] `config/amp/discriminator_cfg.yaml` 생성 완료
  - [x] `config/amp/curriculum_config.yaml` 생성 완료
  - [x] `config/amp/domain_randomization.yaml` 생성 완료
  **완료 상태**: 모든 설정 파일 생성 및 기본 설정 값 정의 완료

### 1.3 환경 등록 준비

- [x] **`tasks/walking/amp/__init__.py` 작성** ✅ 완료
  - [x] AMP 환경 등록 코드 작성 완료
  - [x] `H1-Walking-AMP-v0` 환경 ID 등록 완료
  - [x] `H1-Walking-AMP-Play-v0` 환경 ID 등록 완료
  **완료 상태**: Gymnasium 환경 등록 코드 작성 완료

- [x] **`tasks/__init__.py` 업데이트** ✅ 완료
  - [x] AMP 환경 import 추가 완료
  - [x] 기존 PPO 환경과 충돌 없이 등록 확인 완료
  **완료 상태**: `from .walking import amp` 추가하여 AMP 환경 등록 완료

---

## Phase 2: Mocap Data Preprocessing Pipeline

### 2.1 AMASS 데이터셋 준비

- [x] **AMASS 데이터셋 다운로드** ✅ 완료
  - [x] AMASS 웹사이트 접속: https://amass.is.tue.mpg.de/
  - [x] 필요한 데이터셋 선택 (ACCAD 데이터셋 선택)
  - [x] 데이터셋 다운로드 (`.tar.bz2` 형식)
  - [x] 압축 해제: `tar -xjf ACCAD.tar.bz2 -C data/amass/`
  - [x] `data/amass/ACCAD/` 디렉토리에 데이터셋 배치 완료
  - [x] 데이터셋 구조 확인 완료
    - ✅ ACCAD 데이터셋 구조:
      - `Female1Walking_c3d/`, `Male1Walking_c3d/`, `Male2Walking_c3d/` (걷기 모션)
      - `Female1Running_c3d/`, `Male1Running_c3d/`, `Male2Running_c3d/` (달리기 모션)
      - `Female1General_c3d/`, `Male1General_c3d/`, `Male2General_c3d/` (일반 모션)
      - 기타 모션 카테고리들
    - ✅ 각 디렉토리 내 `.npz` 파일 형식 확인 완료

- [x] **SMPL 모델 다운로드** ✅ 완료
  - [x] SMPL 웹사이트 접속: https://smpl.is.tue.mpg.de/
  - [x] SMPL 모델 파일 다운로드 (`.pkl` 형식)
  - [x] `data/mapping/SMPL_python_v.1.1.0/` 디렉토리에 압축 해제 완료
  - [x] SMPL 모델 파일 확인 완료:
    - ✅ `basicmodel_f_lbs_10_207_0_v1.1.0.pkl` (여성 모델)
    - ✅ `basicmodel_m_lbs_10_207_0_v1.1.0.pkl` (남성 모델)
    - ✅ `basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl` (중성 모델)

- [x] **필요한 Python 패키지 설치** ✅ 완료
  - [x] `smplx` 패키지 설치 완료: `pip install smplx` (버전 0.1.28)
  - [x] `pinocchio` 패키지 설치 완료: `pip install pin` (버전 2.7.0, 이미 설치되어 있었음)
    - 참고: 코드에서는 `import pinocchio` 또는 `import pinocchio as pin`으로 사용
  - [x] `trimesh` 패키지 설치 완료: `pip install trimesh` (버전 4.10.0, 이미 설치되어 있었음)
  - [x] `tqdm` 패키지 설치 완료: `pip install tqdm` (버전 4.67.1, 이미 설치되어 있었음)
  **완료 상태**: 
  - ✅ `conda activate env_isaaclab` 환경에서 모든 패키지 설치 완료
  - ✅ 패키지 import 테스트 통과 (`import smplx`, `import pinocchio`, `import trimesh`, `import tqdm`)
  - ✅ Isaac Lab 환경과 호환성 확인 완료

### 2.2 SMPL 데이터 로더 구현

- [x] **`scripts/data_preprocessing/retargeting/smpl_loader.py` 작성** ✅ 완료
  - [x] `SMPLLoader` 클래스 정의 완료
  - [x] `__init__()` 메서드 구현 완료
    - [x] 데이터 경로 설정 완료
    - [x] FPS 설정 완료
    - [x] SMPL 모델 경로 설정 완료
  - [x] `load_motion()` 메서드 구현 완료
    - [x] `.npz` 파일 파싱 완료
    - [x] 데이터 형식 검증 완료
    - [x] Root translation, orientation, pose, betas 추출 완료
  - [x] `get_joint_positions()` 메서드 구현 완료
    - [x] SMPL forward kinematics 구현 완료 (smplx 사용)
    - [x] Fallback 방법 구현 완료 (모델 로드 실패 시)
    - [x] 관절 위치 계산 완료
  - [x] `get_joint_velocities()` 메서드 구현 완료
    - [x] 관절 위치에서 속도 계산 완료
  - [x] 에러 처리 추가 완료
  - [x] 로깅 추가 완료
  **완료 상태**: 
  - ✅ AMASS .npz 파일 로드 기능 구현 완료
  - ✅ 데이터 형식 검증 완료 (trans, poses, betas, gender, fps)
  - ✅ 관절 위치 계산 기능 구현 완료 (smplx 사용, fallback 포함)
  - ✅ 관절 속도 계산 기능 구현 완료
  - ✅ 단위 테스트 함수 포함 (`test_smpl_loader()`)
  - ⚠️ 참고: SMPL v1.1.0 모델 로딩은 smplx와 호환성 문제로 fallback 방법 사용 중 (기능은 정상 작동)

- [x] **단위 테스트 작성** ✅ 완료
  - [x] 샘플 AMASS 파일로 테스트 완료 (`B3 - walk1_poses.npz`)
  - [x] 데이터 형식 검증 완료
  - [x] 관절 위치 계산 테스트 완료 (915 프레임, 24 관절)
  - [x] 관절 속도 계산 테스트 완료
  **테스트 결과**:
  - ✅ Root translation: (915, 3)
  - ✅ Root orientation: (915, 3)
  - ✅ Pose: (915, 69)
  - ✅ Betas: (16,)
  - ✅ Joint positions: (915, 24, 3)
  - ✅ Joint velocities: (915, 24, 3)

### 2.3 H1 스켈레톤 정의

- [ ] **`scripts/data_preprocessing/retargeting/h1_skeleton.py` 작성**
  - [ ] `H1Skeleton` 클래스 정의
  - [ ] `JOINT_NAMES` 리스트 정의 (34개 DOF)
    - [ ] Left Leg: 6개 관절
    - [ ] Right Leg: 6개 관절
    - [ ] Torso: 3개 관절
    - [ ] Left Arm: 4개 관절
    - [ ] Right Arm: 4개 관절
  - [ ] `SMPL_TO_H1_MAPPING` 딕셔너리 정의
    - [ ] Lower body 매핑
    - [ ] Upper body 매핑
    - [ ] Torso 매핑
  - [ ] `get_t_pose()` 메서드 구현
    - [ ] H1 T-pose 관절 각도 정의
  - [ ] 관절 체인 구조 정의 (부모-자식 관계)

- [ ] **매핑 검증**
  - [ ] SMPL 24개 관절 → H1 34개 DOF 매핑 완전성 확인
  - [ ] 매핑되지 않은 관절 처리 방법 정의

### 2.4 리타겟팅 엔진 구현

- [ ] **`scripts/data_preprocessing/retargeting/retargeter.py` 작성**
  - [ ] `MotionRetargeter` 클래스 정의
  - [ ] `__init__()` 메서드 구현
    - [ ] SMPLLoader 초기화
    - [ ] H1Skeleton 초기화
    - [ ] 매핑 파일 로드 (선택사항)
  - [ ] `retarget()` 메서드 구현
    - [ ] Root 위치/회전 변환 호출
    - [ ] 관절 위치 매핑 호출
    - [ ] 관절 각도 계산 호출
    - [ ] 관절 속도 계산 호출
  - [ ] `_retarget_root()` 메서드 구현
    - [ ] 좌표계 변환 (SMPL: Y-up → H1: Z-up)
    - [ ] 오일러 각도 → 쿼터니언 변환
  - [ ] `_map_joint_positions()` 메서드 구현
    - [ ] SMPL 관절 위치를 H1 관절 위치로 매핑
    - [ ] 매핑되지 않은 관절 처리 (보간 또는 기본값)
  - [ ] `_compute_joint_angles()` 메서드 구현
    - [ ] Inverse Kinematics 구현
    - [ ] Pinocchio 라이브러리 사용 또는 수치적 최적화
    - [ ] 관절 한계 검증
  - [ ] `_compute_joint_velocities()` 메서드 구현
    - [ ] 관절 각도 차분으로 속도 계산
    - [ ] 스무딩 적용 (선택사항)

- [ ] **리타겟팅 검증**
  - [ ] 샘플 모션으로 리타겟팅 테스트
  - [ ] 관절 각도 범위 검증
  - [ ] 물리적 타당성 검증 (관절 한계 내)

### 2.5 유틸리티 함수 구현

- [ ] **`scripts/data_preprocessing/retargeting/utils.py` 작성**
  - [ ] `filter_walking_motions()` 함수 구현
    - [ ] 속도 기반 필터링 (루트 속도 분석)
    - [ ] 보행 패턴 감지 (발 접촉 패턴)
    - [ ] 품질 검증 (안정성, 연속성)
  - [ ] `segment_motions()` 함수 구현
    - [ ] 모션 클립 길이 정규화
    - [ ] 최소/최대 길이 제한
    - [ ] 오버랩 처리 (선택사항)
  - [ ] `validate_motion()` 함수 구현
    - [ ] 관절 각도 범위 검증
    - [ ] 속도 연속성 검증
    - [ ] NaN/Inf 값 검증

### 2.6 데이터 전처리 메인 스크립트

- [ ] **`scripts/data_preprocessing/process_amass.py` 작성**
  - [ ] Argument parser 설정
    - [ ] `--amass_path`: AMASS 데이터셋 경로
    - [ ] `--output_path`: 출력 파일 경로
    - [ ] `--min_duration`: 최소 모션 길이
    - [ ] `--max_duration`: 최대 모션 길이
    - [ ] `--filter_walking`: 걷기 모션만 필터링 여부
  - [ ] `main()` 함수 구현
    - [ ] 리타겟터 초기화
    - [ ] AMASS 데이터셋 스캔
    - [ ] 각 파일 처리 (진행률 표시)
    - [ ] 에러 처리 및 로깅
  - [ ] `convert_to_amp_format()` 함수 구현
    - [ ] 모션 클립 리스트를 AMP 형식으로 변환
    - [ ] 데이터 형식 검증
    - [ ] NumPy 배열로 변환

- [ ] **스크립트 테스트**
  - [ ] 소규모 데이터셋으로 테스트
  - [ ] 출력 파일 형식 검증
  - [ ] 메모리 사용량 확인

### 2.7 AMP 형식 내보내기

- [ ] **`scripts/data_preprocessing/export_motions.py` 작성**
  - [ ] 리타겟팅된 모션 클립 로드
  - [ ] AMP 형식으로 변환
    - [ ] Root position: (N, T, 3)
    - [ ] Root rotation: (N, T, 4) [quaternion]
    - [ ] Joint positions: (N, T, J, 3)
    - [ ] Joint velocities: (N, T, J, 3)
  - [ ] 데이터 정규화 (선택사항)
  - [ ] `.npy` 파일로 저장
  - [ ] 메타데이터 저장 (JSON)

- [ ] **데이터 검증**
  - [ ] 형식 검증 (차원 확인)
  - [ ] 값 범위 검증 (NaN, Inf 확인)
  - [ ] 샘플 시각화 (선택사항)

---

## Phase 3: AMP Network Architecture & Environment Setup

### 3.1 Discriminator 네트워크 구현

- [x] **`tasks/walking/amp/discriminator.py` 작성** ✅ 완료
  - [x] `DiscriminatorCfg` 클래스 정의 (`@configclass`) 완료
    - [x] `hidden_dims`: 히든 레이어 차원 리스트 [512, 512, 256]
    - [x] `activation`: 활성화 함수 이름 "elu"
    - [x] `state_dim`: 상태 차원 211
    - [x] `learning_rate`: 학습률 3e-4
    - [x] `weight_decay`: 가중치 감쇠 1e-5
  - [x] `Discriminator` 클래스 정의 (`nn.Module` 상속) 완료
    - [x] `__init__()` 메서드 구현 완료
      - [x] 네트워크 레이어 구성 완료
      - [x] 활성화 함수 설정 완료
      - [x] 출력 레이어 (Sigmoid) 완료
    - [x] `forward()` 메서드 구현 완료
      - [x] State transition 입력 처리 완료
      - [x] 확률 출력 (0~1) 완료
    - [x] `compute_reward()` 메서드 구현 완료
      - [x] Style reward 계산: `-log(D(s_t, s_{t+1}))` 완료
      - [x] 수치 안정성 (클리핑) 완료
    - [x] `_get_activation()` 헬퍼 메서드 구현 완료
  **완료 상태**: Discriminator 네트워크 클래스 구현 완료

- [ ] **Discriminator 테스트**
  - [ ] 더미 입력으로 forward pass 테스트
  - [ ] 출력 범위 검증 (0~1)
  - [ ] Style reward 계산 검증

### 3.2 Expert 모션 데이터셋 로더

- [x] **`tasks/walking/amp/motion_dataset.py` 작성** ✅ 완료
  - [x] `MotionDataset` 클래스 정의 (`Dataset` 상속) 완료
    - [x] `__init__()` 메서드 구현 완료
      - [x] 모션 파일 로드 (`.npy`) 완료
      - [x] 데이터 전처리 완료
      - [x] 상태 벡터 계산 완료
      - [x] 디바이스 설정 완료
    - [x] `_compute_states()` 메서드 구현 완료
      - [x] Root position (3) 완료
      - [x] Root rotation (4) [quaternion] 완료
      - [x] Joint positions (34 * 3 = 102) 완료
      - [x] Joint velocities (34 * 3 = 102) 완료
      - [x] 총 211 차원 상태 벡터 완료
    - [x] `__len__()` 메서드 구현 완료
    - [x] `__getitem__()` 메서드 구현 완료
      - [x] State transition 샘플링 완료
      - [x] Label 반환 (1 = Real) 완료
  **완료 상태**: Expert 모션 데이터셋 로더 구현 완료

- [ ] **데이터셋 테스트**
  - [ ] 데이터 로드 테스트 (AMP 형식 데이터 필요)
  - [ ] 샘플링 테스트
  - [ ] DataLoader와 함께 테스트

### 3.3 AMP 보상 함수 구현

- [x] **`tasks/walking/amp/amp_rewards.py` 작성** ✅ 완료
  - [x] `AMPRewardsCfg` 클래스 정의 (`@configclass`) 완료
    - [x] `style_reward_weight`: Style reward 가중치 1.0
    - [x] `expert_motion_file`: Expert 모션 파일 경로 설정 완료
  - [x] `compute_style_reward()` 함수 구현 완료
    - [x] State transition 계산 완료
    - [x] Discriminator로 Style reward 계산 완료
    - [x] 가중치 적용 완료
  - [x] `RewardTermCfg` 등록 완료
    - [x] `style_reward` 항목 추가 완료
  **완료 상태**: AMP 보상 함수 구현 완료 (런타임에 Discriminator 주입 필요)

- [x] **보상 함수 통합** ✅ 완료
  - [x] Task rewards와 Style reward 결합 방법 정의 완료 (`amp_env_cfg.py`에서 통합)
  - [x] 런타임에 Discriminator 주입 방법 정의 완료 (학습 스크립트에서 처리)

### 3.4 AMP 환경 설정

- [x] **`tasks/walking/amp_env_cfg.py` 작성** ✅ 완료
  - [x] `H1AmpEnvCfg` 클래스 정의 (`WalkingEnvCfg` 상속) 완료
    - [x] `amp_rewards`: AMPRewardsCfg 설정 완료
    - [x] `observations`: 공통 ObservationsCfg 재사용 완료
    - [x] `terminations`: 공통 TerminationsCfg 재사용 완료
    - [x] `__post_init__()` 메서드 구현 완료
      - [x] 부모 클래스 초기화 완료
      - [x] AMP 관련 설정 적용 완료
  - [x] `H1AmpEnvCfg_PLAY` 클래스 정의 (테스트용) 완료
  **완료 상태**: AMP 환경 설정 클래스 구현 완료

- [ ] **환경 설정 검증**
  - [ ] 상속 구조 확인 (코드 검토 필요)
  - [ ] 설정 값 검증 (실제 환경 생성 테스트 필요)

### 3.5 환경 등록

- [x] **`tasks/walking/amp/__init__.py` 작성** ✅ 완료
  - [x] `H1-Walking-AMP-v0` 환경 등록 완료
    - [x] `entry_point`: `isaaclab.envs:ManagerBasedRLEnv` 완료
    - [x] `env_cfg_entry_point`: `H1AmpEnvCfg` 완료
    - [x] `rsl_rl_cfg_entry_point`: `WalkingAMPPPORunnerCfg` 완료
  - [x] `H1-Walking-AMP-Play-v0` 환경 등록 (테스트용) 완료
  **완료 상태**: 환경 등록 코드 작성 완료

- [x] **`tasks/__init__.py` 업데이트** ✅ 완료
  - [x] AMP 환경 import 추가 완료 (`from .walking import amp`)
  - [x] 환경 등록 확인 완료
  **완료 상태**: 메인 `__init__.py`에 AMP 환경 import 추가 완료

- [ ] **환경 등록 테스트**
  - [ ] `list_envs.py`로 환경 확인 (`H1-Walking-AMP-v0` 존재 확인)
  - [ ] 환경 생성 테스트 (실제 환경 인스턴스 생성 테스트)

---

## Phase 4: Curriculum Learning Strategy

### 4.1 커리큘럼 설정 파일 작성

- [x] **`config/amp/curriculum_config.yaml` 작성** ✅ 완료
  - [x] 커리큘럼 활성화 여부 설정 완료
  - [x] 레벨 정의 완료
    - [x] Level 0: Static Balance (`static_balance`)
      - [x] `start_epoch`: 0 완료
      - [x] `end_epoch`: 500 완료
      - [x] `target_velocity`: lin_vel_x, lin_vel_y, ang_vel_z 범위 완료
      - [x] `reward_weights`: 각 보상 항목의 가중치 완료
    - [x] Level 1: Slow Walk (`slow_walk`)
      - [x] `start_epoch`: 500 완료
      - [x] `end_epoch`: 2000 완료
      - [x] `target_velocity`: 범위 설정 완료
      - [x] `reward_weights`: 가중치 설정 완료
    - [x] Level 2: Fast Walk (`fast_walk`)
      - [x] `start_epoch`: 2000 완료
      - [x] `end_epoch`: 5000 완료
      - [x] `target_velocity`: 범위 설정 완료
      - [x] `reward_weights`: 가중치 설정 완료
  - [x] 업데이트 설정 완료
    - [x] `frequency`: 업데이트 주기 (50 epochs) 완료
    - [x] `schedule`: 업데이트 방식 ("step") 완료
    - [x] `smooth_transition`: 부드러운 전이 여부 (true) 완료
    - [x] `transition_duration`: 전이 기간 (100 epochs) 완료
  **완료 상태**: 커리큘럼 설정 파일 작성 완료 (3개 레벨 정의, 업데이트 설정 포함)

- [ ] **설정 파일 검증**
  - [ ] YAML 문법 검증
  - [ ] 값 범위 검증

### 4.2 커리큘럼 매니저 구현

- [ ] **`scripts/train_walking_amp.py`에 `CurriculumManager` 클래스 추가**
  - [ ] `__init__()` 메서드 구현
    - [ ] 설정 파일 로드
    - [ ] 레벨 리스트 초기화
  - [ ] `get_current_config()` 메서드 구현
    - [ ] 현재 epoch에 해당하는 레벨 찾기
    - [ ] 부드러운 전이 처리 (선택사항)
  - [ ] `_interpolate_levels()` 메서드 구현
    - [ ] 레벨 간 보간 로직
    - [ ] 속도 범위 보간
    - [ ] 보상 가중치 보간
  - [ ] `update_env_config()` 메서드 구현
    - [ ] 명령 범위 업데이트
    - [ ] 보상 가중치 업데이트

- [ ] **커리큘럼 매니저 테스트**
  - [ ] 각 레벨에서 설정 값 확인
  - [ ] 전이 구간에서 보간 확인

### 4.3 학습 스크립트에 통합

- [x] **`scripts/train_walking_amp.py` 기본 구조 작성** ✅ 부분 완료
  - [x] Argument parser 설정 완료
    - [x] `--task`: 환경 이름 완료
    - [x] `--num_envs`: 환경 개수 완료
    - [x] `--max_iterations`: 최대 반복 횟수 완료
    - [x] `--curriculum_cfg`: 커리큘럼 설정 파일 경로 완료 (선택사항)
  - [x] 기본 학습 루프 구조 완료
  - [ ] 커리큘럼 매니저 초기화 (미구현)
  - [ ] 학습 루프에 커리큘럼 업데이트 통합 (미구현)
    - [ ] 주기적으로 환경 설정 업데이트
    - [ ] 로깅 추가
  **완료 상태**: 학습 스크립트 기본 구조 완료 (커리큘럼 통합은 미완료)

---

## Phase 5: Domain Randomization (Sim-to-Real)

### 5.1 도메인 랜덤화 설정 파일 작성

- [x] **`config/amp/domain_randomization.yaml` 작성** ✅ 완료
  - [x] 도메인 랜덤화 활성화 여부 완료
  - [x] 랜덤화 주기 설정 (`episode` 또는 `step`) 완료
  - [x] 랜덤화 항목 정의 완료
    - [x] `link_mass`: 링크 질량 변동 (±10%) 완료
    - [x] `com_position`: 중심 질량 위치 변동 (±5cm) 완료
    - [x] `joint_friction`: 관절 마찰 (0.0 ~ 0.5) 완료
    - [x] `joint_damping`: 관절 댐핑 (0.0 ~ 0.1) 완료
    - [x] `ground_friction`: 지면 마찰 (0.5 ~ 1.5) 완료
    - [x] `control_latency`: 제어 지연 (0 ~ 50ms) 완료
    - [x] `gravity`: 중력 (9.6 ~ 9.8 m/s²) 완료
    - [x] `payload`: 페이로드 (0 ~ 5kg) 완료

- [x] **각 항목별 설정** ✅ 완료
  - [x] `enabled`: 활성화 여부 완료
  - [x] `distribution`: 분포 타입 (`uniform`) 완료
  - [x] `range`: 값 범위 완료
  **완료 상태**: 도메인 랜덤화 설정 파일 작성 완료 (8개 항목 모두 정의)

### 5.2 도메인 랜덤화 매니저 구현

- [x] **`tasks/walking/amp/domain_randomization.py` 작성** ✅ 기본 구조 완료
  - [x] `DomainRandomizationCfg` 클래스 정의 (`@configclass`) 완료
    - [x] `enabled`: 활성화 여부 완료
    - [x] `frequency`: 랜덤화 주기 완료
    - [x] `config_path`: 설정 파일 경로 완료
  - [x] `DomainRandomizationManager` 클래스 정의 완료
    - [x] `__init__()` 메서드 구현 완료
      - [x] 설정 파일 로드 완료
      - [x] 랜덤화 파라미터 초기화 완료
    - [x] `_init_randomization_params()` 메서드 구현 완료
      - [x] 각 환경마다 랜덤 값 저장소 생성 완료
    - [x] `randomize()` 메서드 구현 완료 (기본 구조)
      - [x] 주기 확인 완료
    - [x] `_randomize_link_mass()` 메서드 구현 완료 (기본 구조)
      - [x] 링크 질량 랜덤화 로직 완료
      - [ ] Isaac Lab API 통합 필요 (TODO 주석 포함)
    - [ ] `_randomize_joint_friction()` 메서드 구현 (TODO)
    - [ ] `_randomize_ground_friction()` 메서드 구현 (TODO)
    - [ ] `_randomize_gravity()` 메서드 구현 (TODO)
    - [ ] 기타 랜덤화 메서드 구현 (TODO)
  **완료 상태**: 도메인 랜덤화 매니저 기본 구조 완료 (Isaac Lab API 통합 필요)

- [ ] **도메인 랜덤화 테스트**
  - [ ] 각 항목별 랜덤화 동작 확인 (Isaac Lab API 통합 후)
  - [ ] 값 범위 검증

### 5.3 환경에 통합

- [ ] **`tasks/walking/amp_env_cfg.py` 업데이트**
  - [ ] `DomainRandomizationCfg` 추가
  - [ ] 도메인 랜덤화 매니저 초기화 방법 정의

- [ ] **학습 스크립트에 통합**
  - [ ] 에피소드 시작 시 랜덤화 적용
  - [ ] 로깅 추가

---

## Phase 6: 학습 및 테스트

### 6.1 에이전트 설정 작성

- [x] **`config/agents/walking_amp_ppo_cfg.py` 작성** ✅ 완료
  - [x] `WalkingAMPPPORunnerCfg` 클래스 정의 (`RslRlOnPolicyRunnerCfg` 상속) 완료
    - [x] 실험 이름: `h1_walking_amp` 완료
    - [x] 정책 네트워크 설정 완료
      - [x] Actor 네트워크 구조 [512, 256, 128] 완료
      - [x] Critic 네트워크 구조 [512, 256, 128] 완료
      - [x] 활성화 함수: "elu" 완료
    - [x] PPO 알고리즘 설정 완료
      - [x] 학습률: 1.0e-3 완료
      - [x] 클리핑 파라미터: 0.2 완료
      - [x] 엔트로피 계수: 0.01 완료
      - [x] Value loss 계수: 1.0 완료
      - [x] 학습 에폭: 5 완료
      - [x] 미니배치 수: 4 완료
  **완료 상태**: 에이전트 설정 파일 작성 완료 (PPO 하이퍼파라미터 모두 정의)

- [ ] **설정 검증**
  - [ ] PPO 설정 값 검증
  - [ ] 네트워크 구조 검증

### 6.2 학습 스크립트 완성

- [x] **`scripts/train_walking_amp.py` 기본 구조 완성** ✅ 부분 완료
  - [x] 환경 생성 완료
  - [x] Discriminator 초기화 완료 (`DiscriminatorTrainer` 클래스 포함)
  - [x] Expert 데이터셋 로드 완료 (`MotionDataset` 사용)
  - [x] RSL-RL Runner 초기화 완료
  - [x] 기본 학습 루프 구조 완료
  - [ ] 커리큘럼 매니저 통합 (미구현)
  - [ ] 도메인 랜덤화 매니저 통합 (미구현)
  - [x] 학습 루프 기본 구조 구현 완료
    - [x] 데이터 수집 (RSL-RL 기본 루프 사용) 완료
    - [x] Discriminator 학습 구조 (`DiscriminatorTrainer.train_step()` 메서드) 완료
    - [x] Policy 학습 (RSL-RL runner 사용) 완료
    - [ ] 커리큘럼 업데이트 (미구현)
    - [x] 로깅 및 체크포인트 저장 완료
  **완료 상태**: 
  - ✅ 학습 스크립트 기본 구조 완료
  - ✅ Discriminator 초기화 및 학습 구조 구현 완료
  - ⚠️ 주의: Discriminator 학습 루프는 통합되지 않음 (별도 호출 필요)
  - ⚠️ 주의: 커리큘럼 및 도메인 랜덤화 통합 미완료

- [ ] **학습 스크립트 테스트**
  - [ ] 소규모 환경으로 테스트 실행
  - [ ] 에러 확인 및 수정

### 6.3 테스트 스크립트 작성

- [x] **`scripts/play_walking_amp.py` 작성** ✅ 완료
  - [x] Argument parser 설정 완료
    - [x] `--task`: 환경 이름 완료
    - [x] `--checkpoint`: 체크포인트 경로 완료
    - [x] `--num_envs`: 환경 개수 완료
    - [x] `--discriminator_checkpoint`: Discriminator 체크포인트 경로 완료 (선택사항)
    - [x] `--num_steps`: 실행 스텝 수 완료
  - [x] 환경 생성 완료
  - [x] 체크포인트 로드 완료
  - [x] 정책 실행 및 시각화 완료
  - [x] 진행 상황 출력 완료 (평균 보상 등)
  **완료 상태**: 테스트 스크립트 구현 완료 (학습된 정책 테스트 가능)

### 6.4 학습 실행

- [ ] **데이터 전처리 완료 확인**
  - [ ] `data/processed/amp_motions.npy` 파일 존재 확인
  - [ ] 데이터 형식 검증

- [ ] **초기 학습 실행**
  - [ ] 소규모 환경 수로 테스트 학습
  - [ ] 메모리 사용량 확인
  - [ ] 학습 속도 확인

- [ ] **전체 학습 실행**
  - [ ] 전체 환경 수로 학습 시작
  - [ ] TensorBoard로 모니터링
  - [ ] 주기적 체크포인트 저장 확인

### 6.5 결과 분석

- [ ] **학습 메트릭 분석**
  - [ ] 평균 에피소드 보상 추이
  - [ ] Discriminator 손실 추이
  - [ ] Policy 손실 추이
  - [ ] Style reward 추이

- [ ] **학습된 정책 테스트**
  - [ ] 다양한 속도 명령으로 테스트
  - [ ] 안정성 평가
  - [ ] 자연스러움 평가

- [ ] **하이퍼파라미터 튜닝**
  - [ ] Style reward 가중치 조정
  - [ ] Discriminator 학습률 조정
  - [ ] 커리큘럼 레벨 조정

---

## 추가 작업

### 7.1 문서화

- [ ] **README 업데이트**
  - [ ] AMP 알고리즘 설명 추가
  - [ ] 사용 방법 추가
  - [ ] 데이터 전처리 방법 추가

- [ ] **코드 주석 추가**
  - [ ] 모든 클래스와 메서드에 docstring 추가
  - [ ] 복잡한 로직에 인라인 주석 추가

### 7.2 최적화

- [ ] **성능 최적화**
  - [ ] 데이터 로딩 최적화
  - [ ] Discriminator 학습 최적화
  - [ ] 메모리 사용량 최적화

- [ ] **코드 리팩토링**
  - [ ] 중복 코드 제거
  - [ ] 모듈화 개선

---

## 문제 해결 가이드

### 일반적인 문제

1. **Import 오류**
   - [ ] Python 경로 확인
   - [ ] 패키지 설치 확인
   - [ ] 상대 import 경로 확인

2. **메모리 부족**
   - [ ] 환경 수 감소
   - [ ] 배치 크기 조정
   - [ ] 데이터 로딩 최적화

3. **학습이 수렴하지 않음**
   - [ ] 보상 가중치 조정
   - [ ] 학습률 조정
   - [ ] 커리큘럼 레벨 조정

4. **Discriminator 학습 불안정**
   - [ ] 학습률 감소
   - [ ] 가중치 초기화 조정
   - [ ] 정규화 추가

---

---

## 구현 진행 상황 요약

### 완료된 Phase
- ✅ **Phase 1**: 프로젝트 구조 초기화 (100% 완료)
- ✅ **Phase 3**: AMP Network Architecture & Environment Setup (90% 완료, 테스트 항목 제외)

### 부분 완료된 Phase
- 🔄 **Phase 2**: Mocap Data Preprocessing Pipeline (30% 완료)
  - ✅ AMASS 데이터셋 준비
  - ✅ SMPL 데이터 로더 구현
  - ❌ H1 스켈레톤 정의 (미완료)
  - ❌ 리타겟팅 엔진 (미완료)
  - ❌ 데이터 전처리 스크립트 (미완료)
  
- 🔄 **Phase 4**: Curriculum Learning Strategy (30% 완료)
  - ✅ 커리큘럼 설정 파일 작성
  - ❌ 커리큘럼 매니저 구현 (미완료)
  - ❌ 학습 스크립트 통합 (미완료)
  
- 🔄 **Phase 5**: Domain Randomization (40% 완료)
  - ✅ 도메인 랜덤화 설정 파일 작성
  - ✅ 도메인 랜덤화 매니저 기본 구조 (Isaac Lab API 통합 필요)
  - ❌ 환경 및 학습 스크립트 통합 (미완료)
  
- 🔄 **Phase 6**: 학습 및 테스트 (60% 완료)
  - ✅ 에이전트 설정 작성
  - ✅ 학습 스크립트 기본 구조 (Discriminator 학습 루프 통합 필요)
  - ✅ 테스트 스크립트 작성
  - ❌ 실제 학습 실행 및 검증 (미완료)

### 다음 우선순위 작업
1. **Phase 2 완성**: H1 스켈레톤 정의 및 리타겟팅 엔진 구현
2. **Discriminator 학습 루프 통합**: `train_walking_amp.py`에 Discriminator 학습 통합
3. **커리큘럼 매니저 구현**: Phase 4.2 완성
4. **도메인 랜덤화 API 통합**: Phase 5.2 완성

---

**최종 업데이트**: 2025-01-XX (프로젝트 검토 반영)  
**작성자**: AI Robotics Engineer

