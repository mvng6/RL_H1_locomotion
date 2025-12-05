# Copyright (c) 2025, RL Project Workspace
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""종료 조건 정의 - 기본 보행 태스크."""

from isaaclab.managers import DoneTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp


@configclass
class TerminationsCfg:
    """기본 보행을 위한 종료 조건 설정."""

    # 시간 초과 (에피소드 길이 초과 시 종료)
    # 에피소드가 최대 길이에 도달하면 종료됩니다.
    # 이는 정상적인 종료 조건으로, 학습 중 에피소드가 너무 길어지는 것을 방지합니다.
    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True,
    )

    # 로봇 넘어짐 (베이스 접촉 감지)
    # 베이스(몸통) 링크가 땅에 닿으면 로봇이 넘어졌다고 판단하여 종료합니다.
    # 이는 실패 조건으로, 로봇이 안정적으로 서 있지 못할 때 학습을 중단합니다.
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*torso.*|.*base.*"),
            "threshold": 1.0,  # 접촉 힘 임계값: 1.0N 이상이면 접촉으로 판단
        },
    )

    # 로봇 떨어짐 (베이스 높이 제한)
    # 베이스 높이가 허용 범위를 벗어나면 종료합니다.
    # - minimum_height: 최소 높이 (너무 낮으면 땅에 떨어짐)
    # - maximum_height: 최대 높이 (너무 높으면 비현실적)
    base_height = DoneTerm(
        func=mdp.base_height,
        params={
            "minimum_height": 0.3,  # 최소 높이: 0.3m (땅에 떨어짐 방지)
            "maximum_height": 2.0,  # 최대 높이: 2.0m (비현실적인 점프 방지)
        },
    )

