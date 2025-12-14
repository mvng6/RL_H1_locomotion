# Copyright (c) 2025, RL Project Workspace
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""H1 Locomotion 태스크 패키지 초기화."""

##
# Register Gym environments.
##

# PPO 기반 Walking 태스크 import (환경 등록을 위해 필요)
from . import walking  # H1-Walking-v0 등록

# AMP 기반 Walking 태스크 import (환경 등록을 위해 필요)
from .walking import amp  # H1-Walking-AMP-v0 등록

# Running 태스크 (아직 구현 전)
# from . import running

# Jumping 태스크 (아직 구현 전)
# from . import jumping
