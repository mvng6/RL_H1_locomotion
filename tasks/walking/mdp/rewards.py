# Copyright (c) 2025, RL Project Workspace
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""보상 함수 정의 - 기본 보행 태스크 (With Improved Phase-based Reward)."""

import torch
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp


# -------------------------------------------------------------------------
# [Improved] Contact Force-based Phase Tracking Reward Function
# -------------------------------------------------------------------------
def gait_phase_tracking(env, command_name: str, threshold: float = 1.0):
    """
    접촉 힘 기반 위상 추정 보상 함수.
    
    실제 접촉 힘을 사용하여 각 발의 Swing/Stance 위상을 추정하고,
    교대 보행 패턴을 보상으로 제공합니다. 이 방식은 보행 속도에 자동으로 적응하며,
    Isaac Lab의 표준 방식과 일치합니다.
    
    Args:
        env: 강화학습 환경 객체
        command_name: 명령 이름 (사용하지 않지만 인터페이스 일관성을 위해 유지)
        threshold: 접촉 힘 임계값 (N). 이 값 이상이면 접촉으로 판단.
    
    Returns:
        torch.Tensor: (num_envs,) 형태의 보상 텐서
    """
    # 1. 접촉 센서에서 접촉 힘 가져오기
    # net_forces_w_history의 형태: (num_envs, history_length, num_bodies, 3)
    # Z축(인덱스 2)은 수직 힘을 의미합니다.
    contact_sensor = env.scene["contact_forces"]
    contact_forces_z = contact_sensor.data.net_forces_w_history[:, 0, :, 2]  # (num_envs, num_feet)
    
    # 2. 접촉 상태 판단 (임계값 기반)
    # 절대값을 사용하여 위/아래 힘 모두 접촉으로 판단
    in_contact = torch.abs(contact_forces_z) > threshold  # (num_envs, num_feet)
    
    # 3. 각 발의 접촉 상태 추출
    # Isaac Lab은 body_names 리스트 순서대로 데이터를 반환하므로,
    # 첫 번째 발이 왼발, 두 번째 발이 오른발이라고 가정합니다.
    # 만약 발이 2개가 아닌 경우를 대비하여 안전하게 처리합니다.
    if contact_forces_z.shape[1] < 2:
        # 발이 2개 미만이면 보상을 0으로 반환
        return torch.zeros(env.num_envs, device=env.device)
    
    left_contact = in_contact[:, 0]   # 첫 번째 발 (왼발)
    right_contact = in_contact[:, 1]  # 두 번째 발 (오른발)
    
    # 4. 교대 보행 보상 계산
    # 목표: 왼발과 오른발이 교대로 접촉
    # - 왼발이 접촉 중일 때 오른발은 비접촉이어야 함 (Stance-Swing)
    # - 오른발이 접촉 중일 때 왼발은 비접촉이어야 함 (Swing-Stance)
    reward_alternating = (
        (left_contact.float() * (1 - right_contact.float())) +   # 왼발 Stance, 오른발 Swing
        (right_contact.float() * (1 - left_contact.float()))    # 오른발 Stance, 왼발 Swing
    )
    
    # 5. 동시 접촉/비접촉 페널티
    # 양발이 동시에 땅에 닿으면 (Double Stance) - 정상적인 보행이지만 교대 보행이 아님
    # 양발이 동시에 공중에 있으면 (Double Swing) - 위험한 상태
    penalty_double_contact = (left_contact & right_contact).float()  # 양발 동시 접촉
    penalty_double_air = (~left_contact & ~right_contact).float()    # 양발 동시 공중
    
    # 6. 최종 보상 계산
    # 교대 보행은 보상, 동시 접촉/비접촉은 페널티
    total_reward = reward_alternating - 0.5 * (penalty_double_contact + penalty_double_air)
    
    return total_reward


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

@configclass
class RewardsCfg:
    """기본 보행 및 자연스러운 모션을 위한 보상 함수 설정."""

    # 1. [필수] 목표 속도 추적 (생존 및 이동)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.5,  # 가중치 상향 (일단 걷는게 중요)
        params={"command_name": "base_velocity", "std": 0.5},
    )

    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # 2. [안정성] 자세 유지
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-1.0,
    )
    
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-1.0,
        params={"target_height": 0.98},  # H1의 적정 허리 높이 확인 필요
    )

    # 3. [자연스러움] Phase-based Gait Reward (개선됨)
    # 접촉 힘 기반 위상 추정을 통해 교대 보행 패턴을 학습합니다.
    # 이 보상은 보행 속도에 자동으로 적응하며, 실제 접촉 상태를 기반으로 합니다.
    gait_phase_tracking = RewTerm(
        func=gait_phase_tracking,  # 위에서 정의한 개선된 함수 사용
        weight=1.0,
        params={"command_name": "base_velocity", "threshold": 1.0},  # 접촉 힘 임계값: 1.0N
    )

    # 4. [자연스러움] 발 클리어런스 및 에어 타임
    # Phase Reward와 함께 사용하여 발 공중 시간을 보장합니다.
    # Phase Reward가 교대 보행 패턴을 학습하고, 이 보상은 충분한 클리어런스를 보장합니다.
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.5,  # Phase Reward와 함께 사용하므로 적절한 가중치 유지
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
            "threshold": 0.4,
        },
    )

    # 5. [규제] 부드러운 움직임 및 에너지 효율
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )

    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-0.0001,
    )

    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,  # 떨림 방지를 위해 약간 강화
    )
    
    # 6. [안전] 원치 않는 충돌 방지
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*thigh|.*calf|.*hand|.*hip")},
    )
