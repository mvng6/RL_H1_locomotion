# Copyright (c) 2025, RL Project Workspace
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AMP 알고리즘 기반 보행 태스크 환경 등록.

H1 로봇을 위한 AMP 기반 보행 환경을 Gymnasium에 등록합니다.
"""

import gymnasium as gym

from . import amp_env_cfg

##
# Register Gym environments.
##

# 학습용 환경 (H1-Walking-AMP-v0)
gym.register(
    id="H1-Walking-AMP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_env_cfg:H1AmpEnvCfg",
        "rsl_rl_cfg_entry_point": "h1_locomotion.config.agents.walking_amp_ppo_cfg:WalkingAMPPPORunnerCfg",
    },
)

# 테스트/플레이용 환경 (H1-Walking-AMP-Play-v0)
gym.register(
    id="H1-Walking-AMP-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_env_cfg:H1AmpEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "h1_locomotion.config.agents.walking_amp_ppo_cfg:WalkingAMPPPORunnerCfg",
    },
)

