# Copyright (c) 2025, RL Project Workspace
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""종료 조건 정의 - 기본 보행 태스크 (안전성 강화 버전).

수정 사항:
1. 최소 높이 상향 (0.3m → 0.5m) - H1 로봇 키에 맞게
2. pelvis 접촉 체크 추가
3. 극단적 자세(기울어짐) 감지 추가
"""

import torch
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp


def base_height(env, asset_cfg: SceneEntityCfg, minimum_height: float = 0.5, maximum_height: float = 1.5) -> torch.Tensor:
    """베이스 높이가 허용 범위를 벗어나면 종료.
    
    H1 로봇의 정상 서있는 높이는 약 1.0-1.1m입니다.
    - minimum_height: 0.5m 미만이면 심하게 쪼그려 앉거나 넘어진 상태
    - maximum_height: 1.5m 초과면 점프 등 비정상 상태
    
    Args:
        env: 강화학습 환경 객체
        asset_cfg: 에셋 설정 (로봇)
        minimum_height: 최소 높이 (m)
        maximum_height: 최대 높이 (m)
    
    Returns:
        torch.Tensor: (num_envs,) 형태의 종료 플래그 텐서
    """
    # SceneEntityCfg는 'name' 속성 사용 (asset_name이 아님)
    root_pos_w = env.scene[asset_cfg.name].data.root_pos_w
    current_height = root_pos_w[:, 2]  # Z 좌표 (높이)
    
    out_of_bounds = (current_height < minimum_height) | (current_height > maximum_height)
    
    return out_of_bounds


def bad_orientation(env, asset_cfg: SceneEntityCfg, limit_angle: float = 0.5) -> torch.Tensor:
    """로봇이 너무 기울어지면 종료.
    
    로봇의 상체가 수직에서 limit_angle(radian) 이상 기울어지면 종료합니다.
    이는 비정상적인 자세나 넘어지려는 상태를 조기에 감지합니다.
    
    Args:
        env: 강화학습 환경 객체
        asset_cfg: 에셋 설정 (로봇)
        limit_angle: 허용 기울기 (radian). 약 30도 = 0.52 rad
    
    Returns:
        torch.Tensor: (num_envs,) 형태의 종료 플래그 텐서
    """
    # projected_gravity를 사용하여 기울기 계산
    # 수평이면 [0, 0, -1], 기울어지면 x, y 성분이 증가
    root_quat_w = env.scene[asset_cfg.name].data.root_quat_w
    
    # 중력 벡터 (월드 좌표계에서 [0, 0, -1])
    gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=env.device)
    
    # 로봇 로컬 좌표계로 변환 (쿼터니언 회전)
    # quat: [w, x, y, z] 형식
    w, x, y, z = root_quat_w[:, 0], root_quat_w[:, 1], root_quat_w[:, 2], root_quat_w[:, 3]
    
    # 중력 벡터의 z 성분 (로봇 로컬 좌표계)
    # 수직이면 -1, 기울어지면 -1보다 큰 값
    # 간단한 계산: 중력 벡터와 z축의 내적
    gravity_z_local = 1.0 - 2.0 * (x*x + y*y)  # 쿼터니언에서 z축 성분
    
    # 기울기가 limit_angle 이상이면 종료
    # cos(limit_angle) 보다 작으면 기울어진 것
    is_tilted = gravity_z_local < torch.cos(torch.tensor(limit_angle, device=env.device))
    
    return is_tilted


@configclass
class TerminationsCfg:
    """기본 보행을 위한 종료 조건 설정 (안전성 강화)."""

    # 시간 초과
    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True,
    )

    # 로봇 몸통 접촉 (넘어짐 감지)
    # pelvis도 추가하여 더 정확한 감지
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", 
                body_names=[".*torso.*", ".*pelvis.*"]
            ),
            "threshold": 1.0,
        },
    )

    # 베이스 높이 제한 (더 엄격하게)
    base_height = DoneTerm(
        func=base_height,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "minimum_height": 0.5,   # H1 높이 고려 (약 1.0m 정상)
            "maximum_height": 1.5,   # 점프 방지
        },
    )

    # 극단적 기울어짐 감지 (추가!)
    bad_orientation = DoneTerm(
        func=bad_orientation,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "limit_angle": 0.7,  # 약 40도 이상 기울어지면 종료
        },
    )
