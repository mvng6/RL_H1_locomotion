# Copyright (c) 2025, RL Project Workspace
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""기본 보행용 PPO 에이전트 설정."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class WalkingPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """기본 보행용 PPO Runner 설정."""

    # 환경 설정
    # 각 환경에서 수집할 스텝 수 (rollout 길이)
    num_steps_per_env = 24

    # 최대 학습 반복 횟수
    max_iterations = 3000

    # 체크포인트 저장 간격 (iteration 단위)
    save_interval = 50

    # 실험 이름 및 로그 설정
    experiment_name = "h1_walking"
    run_name = ""  # 빈 문자열이면 타임스탬프가 자동으로 추가됨
    seed = 42  # 랜덤 시드

    # 정책 네트워크 설정 (Actor-Critic 구조)
    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,  # 초기 탐험 노이즈 표준편차
        actor_hidden_dims=[512, 256, 128],  # Actor 네트워크 은닉층 크기
        critic_hidden_dims=[512, 256, 128],  # Critic 네트워크 은닉층 크기
        activation="elu",  # 활성화 함수 (ELU 사용)
    )

    # PPO 알고리즘 설정
    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        # 가치 함수 손실 계수
        value_loss_coef=1.0,
        # 클리핑된 가치 손실 사용 여부
        use_clipped_value_loss=True,
        # PPO 클리핑 파라미터 (정책 업데이트 범위)
        clip_param=0.2,
        # 엔트로피 계수 (탐험을 위한 정규화)
        entropy_coef=0.01,
        # 학습 에포크 수 (각 업데이트마다 데이터를 몇 번 사용할지)
        num_learning_epochs=5,
        # 미니 배치 수 (학습 데이터를 나눌 배치 개수)
        num_mini_batches=4,
        # 학습률
        learning_rate=1.0e-3,
        # 학습률 스케줄링 방식 ("adaptive"는 KL divergence 기반 적응형)
        schedule="adaptive",
        # 할인 계수 (미래 보상의 현재 가치)
        gamma=0.99,
        # GAE (Generalized Advantage Estimation) 람다 파라미터
        lam=0.95,
        # 목표 KL divergence (적응형 학습률 조정 기준)
        desired_kl=0.01,
        # 그래디언트 클리핑 최대 노름
        max_grad_norm=1.0,
    )

