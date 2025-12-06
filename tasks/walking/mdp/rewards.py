# Copyright (c) 2025, RL Project Workspace
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""보상 함수 정의 - 기본 보행 태스크 (안전성 강화 버전).

문제점 분석 후 수정된 버전:
1. 관절 한계 페널티 추가 (joint_pos_limits)
2. 관절 속도 페널티 추가 (joint_vel_l2)
3. Self-collision 체크 범위 확대
4. 보상 가중치 재조정
"""

import torch
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp


@configclass
class RewardsCfg:
    """안전하고 자연스러운 보행을 위한 보상 함수 설정.
    
    우선순위:
    1. 안전성 (관절 한계, 충돌 방지) - 가장 중요
    2. 안정성 (자세 유지, 높이 유지)
    3. 목표 추적 (속도 명령 따르기)
    4. 자연스러움 (보행 패턴, 부드러운 움직임)
    """

    # =========================================================================
    # 1. [핵심] 목표 속도 추적 - 이것이 학습의 주요 목표
    # =========================================================================
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.25},
    )

    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": 0.25},
    )

    # =========================================================================
    # 2. [안전성] 관절 한계 - 신체 관통 방지의 핵심!
    # =========================================================================
    # 관절이 한계에 가까워지면 페널티
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-5.0,  # 강한 페널티로 극단적 자세 방지
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # 관절 속도 제한 - 급격한 움직임 방지
    joint_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # =========================================================================
    # 3. [안전성] 충돌 방지 - 범위 확대
    # =========================================================================
    # 원치 않는 신체 부위 접촉 페널티 (torso, pelvis 추가!)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    ".*torso.*",      # 몸통 (추가!)
                    ".*pelvis.*",     # 골반 (추가!)
                    ".*thigh.*",      # 허벅지
                    ".*calf.*",       # 종아리
                    ".*hip.*",        # 엉덩이
                ]
            ),
            "threshold": 1.0,
        },
    )

    # =========================================================================
    # 4. [안정성] 자세 유지
    # =========================================================================
    # 수평 자세 유지
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-2.0,  # 가중치 증가
    )

    # 기본 높이 유지 (H1의 정상 서있는 높이)
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_height": 1.05,  # H1의 정상 서있는 높이 (약 1.05m)
        },
    )

    # =========================================================================
    # 5. [자연스러움] 보행 패턴
    # =========================================================================
    # 발 공중 시간 보상 (자연스러운 걸음걸이)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
            "threshold": 0.4,
        },
    )

    # =========================================================================
    # 6. [규제] 부드러운 움직임 및 에너지 효율
    # =========================================================================
    # 액션 변화율 제한 (떨림 방지)
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )

    # 토크 사용량 제한 (에너지 효율)
    # 주의: 부모 클래스(H1RoughEnvCfg)가 이 이름을 참조하므로 dof_torques_l2로 유지
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # 관절 가속도 제한 (부드러운 움직임)
    # 주의: 부모 클래스(H1RoughEnvCfg)가 이 이름을 참조하므로 dof_acc_l2로 유지
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # =========================================================================
    # 7. [생존] 살아있기 보상 (선택적)
    # =========================================================================
    # 넘어지지 않고 살아있으면 보상
    is_alive = RewTerm(
        func=mdp.is_alive,
        weight=0.5,
    )

    # =========================================================================
    # 8. [페널티] 종료 조건 페널티
    # =========================================================================
    # 종료되면 큰 페널티 (넘어짐 등)
    is_terminated = RewTerm(
        func=mdp.is_terminated,
        weight=-10.0,
    )
