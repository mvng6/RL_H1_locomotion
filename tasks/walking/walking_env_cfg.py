# Copyright (c) 2025, RL Project Workspace
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""기본 보행 환경 설정 파일."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_assets import H1_MINIMAL_CFG  # isort:skip

from .mdp import ObservationsCfg, RewardsCfg, TerminationsCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp


@configclass
class WalkingSceneCfg(InteractiveSceneCfg):
    """기본 보행을 위한 씬 설정."""

    # 지면 생성
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # 조명 설정
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # H1 로봇 설정
    # {ENV_REGEX_NS}는 각 환경의 네임스페이스를 자동으로 생성합니다
    robot: ArticulationCfg = H1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 접촉 센서 설정 (발 접촉 감지용)
    # observations.py와 rewards.py에서 사용하는 접촉 힘 데이터를 제공합니다
    contact_forces = mdp.ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        body_names=".*ankle_link",  # 발목 링크에 접촉 센서 설정
        history_length=1,  # 히스토리 길이: 1 (현재 프레임만 사용)
    )


@configclass
class WalkingEnvCfg(ManagerBasedRLEnvCfg):
    """기본 보행 강화학습 환경 설정."""

    # 씬 설정
    # num_envs: 병렬 환경 개수 (4096개 권장)
    # env_spacing: 환경 간 간격 (미터 단위)
    scene: InteractiveSceneCfg = WalkingSceneCfg(num_envs=4096, env_spacing=2.5)

    # 관측 설정
    # observations.py에서 정의한 ObservationsCfg 사용
    observations: ObservationsCfg = ObservationsCfg()

    # 액션 설정 (PD 제어기)
    # 모든 관절에 Ideal PD Actuator 적용
    # stiffness: 강성 계수 (80.0)
    # damping: 감쇠 계수 (2.0)
    actions: dict[str, IdealPDActuatorCfg] = {
        ".*_joint": IdealPDActuatorCfg(stiffness=80.0, damping=2.0),
    }

    # 보상 설정
    # rewards.py에서 정의한 RewardsCfg 사용
    rewards: RewardsCfg = RewardsCfg()

    # 종료 조건 설정
    # terminations.py에서 정의한 TerminationsCfg 사용
    terminations: TerminationsCfg = TerminationsCfg()

    # 명령 생성 설정 (속도 명령)
    # 중요: "base_velocity" 이름으로 정의해야 observations.py와 일치합니다
    commands: dict[str, mdp.BaseVelocityCommandCfg] = {
        "base_velocity": mdp.BaseVelocityCommandCfg(
            asset_name="robot",  # 로봇 에셋 이름
            resampling_time_range=(10.0, 10.0),  # 명령 재샘플링 시간 범위 (초)
            rel_standing_envs=0.02,  # 서 있는 환경 비율 (2%)
            rel_heading_envs=1.0,  # 방향 명령 환경 비율 (100%)
            heading_command=True,  # 방향 명령 활성화
            heading_control_stiffness=0.5,  # 방향 제어 강성
            debug_vis=False,  # 디버그 시각화 비활성화
            ranges=mdp.BaseVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 1.0),  # 전진 속도: 0~1 m/s
                lin_vel_y=(-0.5, 0.5),  # 횡방향 속도: -0.5~0.5 m/s
                ang_vel_z=(-1.0, 1.0),  # 회전 속도: -1~1 rad/s
            ),
        )
    }

    # 이벤트 설정 (리셋 시 랜덤화)
    # 환경이 리셋될 때마다 실행되는 이벤트들
    events: dict = {
        # 관절 상태 랜덤화 (스케일 기반)
        "reset_joints_by_scale": {
            "asset_cfg_name": "robot",
            "func": "isaaclab.utils.assets.reset_joints_by_scale",
            "params": {
                "position_range": (0.5, 1.5),  # 관절 위치 범위 (기본값의 0.5~1.5배)
                "velocity_range": (0.0, 0.0),  # 관절 속도 범위 (정지 상태)
            },
        },
        # 베이스 상태 랜덤화 (균등 분포)
        "reset_base": {
            "asset_cfg_name": "robot",
            "func": "isaaclab.utils.assets.reset_root_state_uniform",
            "params": {
                "pose_range": {
                    "x": (-0.5, 0.5),  # X 위치 범위 (미터)
                    "y": (-0.5, 0.5),  # Y 위치 범위 (미터)
                    "yaw": (-0.0, 0.0),  # Yaw 각도 범위 (라디안)
                },
                "velocity_range": {
                    "x": (0.0, 0.0),  # X 선속도 범위 (m/s)
                    "y": (0.0, 0.0),  # Y 선속도 범위 (m/s)
                    "z": (0.0, 0.0),  # Z 선속도 범위 (m/s)
                },
            },
        },
    }

    # 에피소드 길이 설정 (초 단위)
    # 각 에피소드의 최대 길이를 20초로 설정
    episode_length_s = 20.0

    # 시뮬레이션 설정
    # dt: 시뮬레이션 시간 스텝 (초)
    # substeps: 물리 서브스텝 수
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(dt=0.005, substeps=1)

