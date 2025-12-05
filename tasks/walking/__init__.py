# Copyright (c) 2025, RL Project Workspace
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""기본 보행 태스크 환경 등록.

H1 로봇을 위한 기본 보행 환경을 Gymnasium에 등록합니다.
Isaac Lab의 표준 패턴을 따르며, LocomotionVelocityRoughEnvCfg를 상속합니다.
"""

import gymnasium as gym

from . import walking_env_cfg

##
# Register Gym environments.
##

# 학습용 환경 (H1-Walking-v0)
gym.register(
    id="H1-Walking-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walking_env_cfg:WalkingEnvCfg",
        "rsl_rl_cfg_entry_point": "h1_locomotion.config.agents.walking_ppo_cfg:WalkingPPORunnerCfg",
    },
)

# 테스트/플레이용 환경 (H1-Walking-Play-v0)
gym.register(
    id="H1-Walking-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walking_env_cfg:WalkingEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "h1_locomotion.config.agents.walking_ppo_cfg:WalkingPPORunnerCfg",
    },
)
