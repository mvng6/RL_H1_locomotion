# Copyright (c) 2025, RL Project Workspace
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""기본 보행 환경 설정 파일 (안전성 강화 버전).

수정 사항:
1. 커스텀 보상 함수 적용 (관절 한계, 충돌 방지 강화)
2. 커스텀 종료 조건 적용 (높이, 기울기 체크 강화)
3. 에피소드 길이 단축 (초기 학습 안정화)
"""

from isaaclab.utils import configclass

# Isaac Lab의 H1 전용 locomotion 환경 설정을 상속
from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.rough_env_cfg import (
    H1RoughEnvCfg,
    H1RoughEnvCfg_PLAY,
)

# 커스텀 MDP 설정 import
from .mdp import RewardsCfg, TerminationsCfg


@configclass
class WalkingEnvCfg(H1RoughEnvCfg):
    """H1 로봇을 위한 기본 보행 환경 설정 (안전성 강화).
    
    변경 사항:
    - 커스텀 보상: 관절 한계, 충돌 방지 페널티 강화
    - 커스텀 종료: 높이/기울기 체크 강화
    - 짧은 에피소드: 초기 학습 안정화
    """

    # 커스텀 보상 설정 적용
    rewards: RewardsCfg = RewardsCfg()
    
    # 커스텀 종료 조건 적용
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        """환경 설정 초기화."""
        # 부모 클래스의 기본 설정 적용
        super().__post_init__()

        # =====================================================================
        # 1. 에피소드 설정 - 초기 학습 시 짧게 유지
        # =====================================================================
        self.episode_length_s = 10.0  # 10초 (안정화 후 20초로 증가 가능)

        # =====================================================================
        # 2. 명령 범위 설정 - 처음에는 느린 속도로 시작
        # =====================================================================
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.5)   # 전진: 0~0.5 m/s (느리게)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)  # 횡방향: 작게
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)  # 회전: 작게

        # =====================================================================
        # 3. 시뮬레이션 설정
        # =====================================================================
        # 더 안정적인 물리 시뮬레이션
        self.sim.dt = 0.005  # 5ms (기본값 유지)
        self.decimation = 4   # 환경 스텝 = 20ms


@configclass
class WalkingEnvCfg_PLAY(H1RoughEnvCfg_PLAY):
    """테스트/플레이용 환경 설정."""

    # 테스트 시에도 동일한 설정 적용
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        """테스트용 환경 설정 초기화."""
        super().__post_init__()

        # 테스트 시에는 더 적은 환경 사용
        self.scene.num_envs = 50

        # 디버그 시각화 활성화
        self.commands.base_velocity.debug_vis = True

        # 테스트용 명령 범위
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
