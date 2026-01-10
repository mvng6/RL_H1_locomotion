# Copyright (c) 2025, RL Project Workspace
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AMP 알고리즘 기반 보행 환경 설정.

기존 WalkingEnvCfg를 상속하여 AMP 알고리즘에 필요한 기능을 추가합니다.
"""

from isaaclab.utils import configclass

from .walking_env_cfg import WalkingEnvCfg, WalkingEnvCfg_PLAY
from .amp.amp_rewards import AMPRewardsCfg
from .mdp import ObservationsCfg, TerminationsCfg


@configclass
class H1AmpEnvCfg(WalkingEnvCfg):
    """H1 AMP 환경 설정.
    
    기존 WalkingEnvCfg를 확장하여 AMP 알고리즘을 지원합니다.
    - Task rewards: 부모 클래스에서 상속
    - Style reward: AMP 전용 모듈에서 추가
    """
    
    # AMP 보상 추가
    amp_rewards: AMPRewardsCfg = AMPRewardsCfg()
    
    # 관측 공간 (기존과 동일, 공통 모듈 재사용)
    observations: ObservationsCfg = ObservationsCfg()
    
    # 종료 조건 (기존과 동일, 공통 모듈 재사용)
    terminations: TerminationsCfg = TerminationsCfg()
    
    def __post_init__(self):
        """환경 설정 초기화."""
        super().__post_init__()
        
        # AMP 관련 설정
        # Discriminator는 학습 스크립트에서 초기화되어 주입됨


@configclass
class H1AmpEnvCfg_PLAY(WalkingEnvCfg_PLAY):
    """H1 AMP 테스트/플레이용 환경 설정."""
    
    # AMP 보상 추가
    amp_rewards: AMPRewardsCfg = AMPRewardsCfg()
    
    # 관측 공간
    observations: ObservationsCfg = ObservationsCfg()
    
    # 종료 조건
    terminations: TerminationsCfg = TerminationsCfg()
    
    def __post_init__(self):
        """테스트용 환경 설정 초기화."""
        super().__post_init__()

