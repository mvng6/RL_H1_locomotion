# Copyright (c) 2025, RL Project Workspace
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""기본 보행 태스크 환경 등록."""

import gymnasium as gym

from . import walking_env_cfg
from ..config.agents import walking_ppo_cfg

##
# Register Gym environments.
##

gym.register(
    id="H1-Walking-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": walking_env_cfg.WalkingEnvCfg,
        "rsl_rl_cfg_entry_point": walking_ppo_cfg.WalkingPPORunnerCfg,
    },
)

