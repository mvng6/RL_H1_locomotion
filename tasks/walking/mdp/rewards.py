# Copyright (c) 2025, RL Project Workspace
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""보상 함수 정의 - 기본 보행 태스크 (With Phase-based Reward)."""

import torch
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp


@configclass
class RewardsCfg:
    """기본 보행을 위한 보상 함수 설정."""

    # 목표 속도 추적 보상 (가장 중요!)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # 목표 각속도 추적 보상
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # 자세 안정성 보상 (수평 유지)
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-1.0,
    )

    # 발 공중 시간 보상 (보행 리듬)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": mdp.SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
            "threshold": 0.4,
        },
    )

    # 발 미끄러짐 페널티
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.5,
        params={
            "sensor_cfg": mdp.SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
            "asset_cfg": mdp.SceneEntityCfg("robot", body_names=".*ankle_link"),
        },
    )

    # 액션 변화율 페널티 (부드러운 동작)
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )

    # 관절 토크 페널티 (에너지 효율)
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-0.0001,
    )

    # 관절 가속도 페널티 (부드러운 동작)
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.25e-7,
    )

    # 로봇 높이 페널티 (주저 앉음 방지)
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-1.0,
        params={"target_height": 0.98},
    )

    # 발 이외의 부위가 땅에 닿으면 페널티
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        # 허벅지, 종아리, 손, 엉덩이
        params={sensor_cfg: mdp.SceneEntityCfg("contact_forces", body_names=".*thigh|.*calf|.*hand|.*hip")},
    )
