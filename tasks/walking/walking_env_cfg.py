# Copyright (c) 2025, RL Project Workspace
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""기본 보행 환경 설정 파일.

Isaac Lab의 기존 H1RoughEnvCfg를 상속하여 H1 로봇용 보행 환경을 구성합니다.
H1RoughEnvCfg는 H1 휴머노이드 로봇에 맞게 이미 설정된 환경입니다.

커스텀 보상 함수를 적용하여 자연스러운 보행 패턴을 학습합니다.
"""

from isaaclab.utils import configclass

# Isaac Lab의 H1 전용 locomotion 환경 설정을 상속
from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.rough_env_cfg import (
    H1RoughEnvCfg,
    H1RoughEnvCfg_PLAY,
)

# 커스텀 MDP 설정 import
from .mdp import RewardsCfg


@configclass
class WalkingEnvCfg(H1RoughEnvCfg):
    """H1 로봇을 위한 기본 보행 환경 설정.
    
    Isaac Lab의 H1RoughEnvCfg를 상속하여 필요한 부분만 수정합니다.
    H1RoughEnvCfg는 이미 H1 로봇의 링크 이름(pelvis 등)에 맞게 설정되어 있습니다.
    
    커스텀 보상 함수(RewardsCfg)를 적용하여:
    - gait_phase_tracking: 교대 보행 패턴 학습
    - feet_air_time: 발 공중 시간 보상
    - 기타 안정성 및 규제 보상
    """

    # 커스텀 보상 설정 적용
    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self):
        """환경 설정 초기화.
        
        부모 클래스의 __post_init__을 먼저 호출하여 기본 설정을 적용한 후,
        기본 보행에 맞게 필요한 부분만 커스터마이징합니다.
        """
        # 부모 클래스의 기본 설정 적용
        super().__post_init__()

        # =====================================================================
        # 1. 에피소드 설정
        # =====================================================================
        # 보행 학습을 위한 에피소드 길이 (초 단위)
        self.episode_length_s = 20.0

        # =====================================================================
        # 2. 명령 범위 설정
        # =====================================================================
        # 기본 보행을 위한 속도 명령 범위 (천천히 걷기)
        # - lin_vel_x: 전진 속도 (m/s)
        # - lin_vel_y: 횡방향 속도 (m/s)
        # - ang_vel_z: 회전 속도 (rad/s)
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)   # 전진: 0~1 m/s
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)  # 횡방향: -0.5~0.5 m/s
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)  # 회전: -1~1 rad/s


@configclass
class WalkingEnvCfg_PLAY(H1RoughEnvCfg_PLAY):
    """테스트/플레이용 환경 설정.
    
    학습된 정책을 테스트하거나 시연할 때 사용합니다.
    """

    # 테스트 시에도 커스텀 보상 설정 적용 (일관성 유지)
    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self):
        """테스트용 환경 설정 초기화."""
        super().__post_init__()

        # 테스트 시에는 더 적은 환경 사용 (시각화 용이)
        self.scene.num_envs = 50

        # 디버그 시각화 활성화
        self.commands.base_velocity.debug_vis = True

        # 명령 범위 설정 (기본 보행)
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
