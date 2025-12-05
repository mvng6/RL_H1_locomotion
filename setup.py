# Copyright (c) 2025, RL Project Workspace
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Setup script for h1_locomotion package."""

from setuptools import setup

setup(
    name="h1_locomotion",
    version="0.1.0",
    description="RL environment for Unitree H1 humanoid robot locomotion",
    author="ldj",
    author_email="zinith7@naver.com",
    # 현재 디렉토리가 패키지 루트이므로 package_dir을 사용하여 매핑
    # setuptools는 절대 경로를 허용하지 않으므로 상대 경로 "."를 사용
    # "h1_locomotion" 패키지가 현재 디렉토리(".")에 있다는 것을 명시
    package_dir={"h1_locomotion": "."},
    # 패키지를 명시적으로 지정
    # 현재 디렉토리의 __init__.py가 h1_locomotion 패키지의 루트
    packages=[
        "h1_locomotion",
        "h1_locomotion.config",
        "h1_locomotion.config.agents",
        "h1_locomotion.tasks",
        "h1_locomotion.tasks.walking",
        "h1_locomotion.tasks.walking.mdp",
        "h1_locomotion.tasks.locomotion",
    ],
    # 패키지 데이터 포함 (필요한 경우)
    include_package_data=True,
    install_requires=[
        "isaaclab",
    ],
    python_requires=">=3.10",
)
