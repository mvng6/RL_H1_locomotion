# Copyright (c) 2025, RL Project Workspace
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""H1 Locomotion 환경 설정 파일."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_assets import H1_MINIMAL_CFG  # isort:skip


@configclass
class H1LocomotionSceneCfg(InteractiveSceneCfg):
    """H1 휴머노이드 로봇을 포함한 씬 설정"""

    # 지면 생성
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # 조명 설정
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # H1 휴머노이드 로봇 설정
    # {ENV_REGEX_NS}는 각 환경의 네임스페이스를 자동으로 생성합니다
    robot: ArticulationCfg = H1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class H1LocomotionEnvCfg(ManagerBasedRLEnvCfg):
    """H1 Locomotion 강화학습 환경 설정"""

    # 씬 설정
    scene: InteractiveSceneCfg = H1LocomotionSceneCfg()

    # 액추에이터 설정
    # IdealPDActuator를 사용하여 관절 제어기 정의
    # H1 로봇의 관절(.*_joint)에 대해 stiffness=80.0, damping=2.0으로 설정
    actions: dict[str, IdealPDActuatorCfg] = {
        ".*_joint": IdealPDActuatorCfg(stiffness=80.0, damping=2.0),
    }

    # 이벤트 설정
    # 에피소드 시작 시 로봇의 관절 상태를 리셋하는 reset_joints_by_scale 함수 사용
    # Isaac Lab의 내장 함수를 사용하여 관절 위치를 랜덤하게 리셋
    events: dict = {
        "reset_joints_by_scale": {
            "asset_cfg_name": "robot",  # 리셋할 로봇 에셋 이름
            "func": "isaaclab.utils.assets.reset_joints_by_scale",
            "params": {
                "position_range": (0.5, 1.5),  # 기본 관절 위치의 0.5~1.5배 범위로 랜덤 리셋
                "velocity_range": (0.0, 0.0),  # 관절 속도는 0으로 리셋
            },
        }
    }

