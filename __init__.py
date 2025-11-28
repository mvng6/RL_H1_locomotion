# Copyright (c) 2025, RL Project Workspace
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""H1 Locomotion Project for Isaac Lab."""

# H1 보행 프로젝트를 Isaac Lab 확장 패키지로 등록
# 패키지 import 시 tasks 모듈을 자동으로 import합니다
from . import tasks

__all__ = ["tasks"]

