# Copyright (c) 2025, RL Project Workspace
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AMP 기반 보행용 PPO 에이전트 설정."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class WalkingAMPPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """AMP 기반 보행용 PPO Runner 설정."""
    
    # 환경 설정
    num_steps_per_env = 24  # 각 환경에서 수집할 스텝 수
    max_iterations = 5000  # 최대 학습 반복 횟수 (커리큘럼 학습 고려)
    save_interval = 50  # 체크포인트 저장 간격
    
    # 실험 이름 및 로그 설정
    experiment_name = "h1_walking_amp"
    run_name = ""
    seed = 42
    
    # 정책 네트워크 설정
    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,  # 초기 탐험 노이즈
        actor_hidden_dims=[512, 256, 128],  # Actor 네트워크 구조
        critic_hidden_dims=[512, 256, 128],  # Critic 네트워크 구조
        activation="elu",  # 활성화 함수
    )
    
    # PPO 알고리즘 설정
    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

