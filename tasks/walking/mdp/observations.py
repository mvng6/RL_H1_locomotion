# Copyright (c) 2025, RL Project Workspace
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""관측 공간 정의 - 기본 보행 태스크."""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp


@configclass
class ObservationsCfg:
    """기본 보행을 위한 관측 공간 설정."""

    @configclass
    class PolicyCfg(ObsGroup):
        """정책 네트워크용 관측 그룹."""

        # 관절 상태 (상대 위치 및 속도)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        # 베이스 상태 (자세, 속도)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, params={"normalize": True})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, params={"normalize": True})
        # projected_gravity: 중력 벡터를 로봇 로컬 좌표계로 투영 (자세 정보 제공)
        # 수평 자세일 때 [0, 0, -1], 기울어지면 해당 방향으로 벡터가 변함
        projected_gravity = ObsTerm(func=mdp.projected_gravity)

        # 명령 (목표 속도)
        commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        # 발 접촉 상태
        feet_contact_forces = ObsTerm(
            func=mdp.contact_forces,
            params={
                "sensor_cfg": mdp.SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
                "threshold": 1.0,                
            },
        )

        # 관절 액션 히스토리
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            """관측 항목들을 연결하여 하나의 벡터로 만듦."""
            self.concatenate_terms = True

    # 정책 네트워크용 관측 그룹
    policy: PolicyCfg = PolicyCfg()