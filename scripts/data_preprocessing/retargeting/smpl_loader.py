# Copyright (c) 2025, RL Project Workspace
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SMPL 데이터 로더 클래스.

AMASS 데이터셋에서 SMPL 형식의 모션 데이터를 로드하고 관절 위치를 계산합니다.
"""

import numpy as np
import torch
import smplx
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
import sys

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SMPLLoader:
    """AMASS 데이터셋에서 SMPL 형식의 모션 데이터를 로드합니다.
    
    AMASS 데이터 형식:
    - trans: (T, 3) 루트 위치
    - poses: (T, 156) SMPL pose 파라미터 (루트 3개 + 51개 관절 * 3 = 156)
    - betas: (16,) SMPL shape parameters
    - gender: 성별 ('male', 'female', 'neutral')
    - mocap_framerate: 프레임레이트
    
    Attributes:
        data_path: AMASS 데이터셋 루트 디렉토리
        fps: 모션 데이터의 프레임레이트 (기본값: 30 fps)
        smpl_model_path: SMPL 모델 파일 경로
        smpl_model: SMPL 모델 인스턴스
    """
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        fps: int = 30,
        smpl_model_path: Optional[str] = None,
        gender: str = "neutral"
    ):
        """초기화.
        
        Args:
            data_path: AMASS 데이터셋 루트 디렉토리 (선택사항)
            fps: 모션 데이터의 프레임레이트 (기본값: 30 fps)
            smpl_model_path: SMPL 모델 파일 경로 (선택사항)
            gender: 기본 성별 ('male', 'female', 'neutral')
        """
        if data_path is not None:
            self.data_path = Path(data_path)
            if not self.data_path.exists():
                logger.warning(f"Data path does not exist: {data_path}")
        else:
            self.data_path = None
        
        self.fps = fps
        self.gender = gender
        
        # SMPL 모델 경로 설정
        if smpl_model_path is None:
            # 기본 경로: data/mapping/SMPL_python_v.1.1.0/smpl/models/
            default_path = Path(__file__).parent.parent.parent.parent / "data" / "mapping" / "SMPL_python_v.1.1.0" / "smpl" / "models"
            if default_path.exists():
                smpl_model_path = str(default_path)
            else:
                logger.warning(f"Default SMPL model path not found: {default_path}")
                smpl_model_path = None
        
        self.smpl_model_path = smpl_model_path
        self.smpl_model = None
        self.smpl_model_original = None
        self.use_original_smpl = False
        
        # SMPL 모델 로드 (필요시)
        if smpl_model_path is not None:
            self._load_smpl_model(gender)
    
    def _load_smpl_model(self, gender: str = "neutral"):
        """SMPL 모델을 로드합니다.
        
        Args:
            gender: 성별 ('male', 'female', 'neutral')
        """
        if self.smpl_model_path is None:
            logger.warning("SMPL model path not set. Cannot load model.")
            return
        
        model_path = Path(self.smpl_model_path)
        
        # 모델 파일 선택
        if gender == "male":
            model_file = model_path / "basicmodel_m_lbs_10_207_0_v1.1.0.pkl"
        elif gender == "female":
            model_file = model_path / "basicmodel_f_lbs_10_207_0_v1.1.0.pkl"
        else:
            model_file = model_path / "basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"
        
        if not model_file.exists():
            logger.warning(f"SMPL model file not found: {model_file}")
            return
        
        try:
            # smplx는 특정 디렉토리 구조를 요구함:
            # model_path/smpl/SMPL_FEMALE.pkl 또는
            # model_path/SMPL_FEMALE.pkl 형식
            
            # 방법 1: 상위 디렉토리에서 smpl 서브디렉토리 찾기
            parent_dir = model_path.parent
            smpl_subdir = parent_dir / "smpl"
            
            if smpl_subdir.exists():
                # smplx가 기대하는 구조: parent_dir/smpl/SMPL_FEMALE.pkl
                # 하지만 우리는 parent_dir/smpl/models/basicmodel_f_lbs_10_207_0_v1.1.0.pkl 형식
                # 따라서 models 디렉토리를 직접 사용
                try:
                    self.smpl_model = smplx.create(
                        str(model_path),  # models 디렉토리
                        model_type='smpl',
                        gender=gender,
                        ext='pkl',
                        num_betas=16,
                        use_face_contour=False
                    )
                    logger.info(f"SMPL model loaded successfully: {model_file}")
                    return
                except Exception as e1:
                    logger.debug(f"Method 1 failed: {e1}")
            
            # 방법 2: 파일을 직접 전달 (파일 경로 전체)
            try:
                # smplx는 때때로 직접 파일 경로를 받을 수 있음
                # 하지만 파일명이 표준 형식이어야 함
                self.smpl_model = smplx.create(
                    str(model_file),  # 전체 파일 경로
                    model_type='smpl',
                    gender=gender,
                    num_betas=16,
                    use_face_contour=False
                )
                logger.info(f"SMPL model loaded successfully (direct file): {model_file}")
                return
            except Exception as e2:
                logger.debug(f"Method 2 failed: {e2}")
            
            # 방법 3: smplx가 기대하는 구조로 심볼릭 링크 생성
            smplx_format_dir = parent_dir.parent / "smplx_format"
            smplx_format_dir.mkdir(exist_ok=True)
            smplx_smpl_dir = smplx_format_dir / "smpl"
            smplx_smpl_dir.mkdir(exist_ok=True)
            
            # 파일명 매핑
            gender_map = {
                "male": "SMPL_MALE.pkl",
                "female": "SMPL_FEMALE.pkl",
                "neutral": "SMPL_NEUTRAL.pkl"
            }
            target_file = smplx_smpl_dir / gender_map.get(gender, "SMPL_NEUTRAL.pkl")
            
            if not target_file.exists():
                try:
                    import os
                    os.symlink(str(model_file), str(target_file))
                    logger.info(f"Created symlink: {target_file} -> {model_file}")
                except Exception:
                    import shutil
                    shutil.copy2(model_file, target_file)
                    logger.info(f"Copied file: {target_file}")
            
            try:
                self.smpl_model = smplx.create(
                    str(smplx_format_dir),
                    model_type='smpl',
                    gender=gender,
                    num_betas=16,
                    use_face_contour=False
                )
                logger.info(f"SMPL model loaded successfully (symlink method): {model_file}")
                return
            except Exception as e3:
                logger.debug(f"Method 3 failed: {e3}")
            
            # 모든 방법 실패 - 원본 SMPL 코드 시도
            logger.info("Trying original SMPL Python code...")
            try:
                # 원본 SMPL 코드 경로 추가
                smpl_code_path = Path(__file__).parent.parent.parent.parent / "data" / "mapping" / "SMPL_python_v.1.1.0"
                if str(smpl_code_path) not in sys.path:
                    sys.path.insert(0, str(smpl_code_path))
                
                from smpl.smpl_webuser.serialization import load_model
                import pickle
                
                # Python 3 호환성을 위해 pickle 모듈 사용
                # 원본 코드는 cPickle을 사용하지만 Python 3에서는 pickle이 동일
                self.smpl_model_original = load_model(str(model_file))
                logger.info(f"SMPL model loaded successfully using original code: {model_file}")
                self.use_original_smpl = True
                return
            except Exception as e_original:
                logger.debug(f"Original SMPL code failed: {e_original}")
                raise Exception("All SMPL loading methods failed")
            
        except Exception as e:
            logger.warning(f"Failed to load SMPL model: {e}")
            logger.warning("Will use fallback method for joint position calculation.")
            logger.info("Note: Joint positions will be approximate.")
            self.smpl_model = None
            self.smpl_model_original = None
            self.use_original_smpl = False
    
    def load_motion(self, file_path: str) -> Dict[str, np.ndarray]:
        """SMPL 모션 파일을 로드합니다.
        
        Args:
            file_path: .npz 파일 경로
            
        Returns:
            Dictionary containing:
                - 'root_translation': (T, 3) 루트 위치
                - 'root_orientation': (T, 3) 루트 오일러 각도 (또는 axis-angle)
                - 'pose': (T, 72) SMPL 관절 회전 (axis-angle, 루트 제외)
                - 'betas': (16,) SMPL shape parameters
                - 'gender': 성별 문자열
                - 'fps': 프레임레이트
                - 'num_frames': 프레임 수
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Motion file not found: {file_path}")
        
        try:
            # .npz 파일 로드 (context manager로 파일 핸들 자동 닫기)
            with np.load(str(file_path), allow_pickle=True) as data:
                # 데이터 추출
                trans = data['trans'].astype(np.float32)  # (T, 3) 루트 위치
                poses = data['poses'].astype(np.float32)  # (T, 156) 전체 pose 파라미터
                
                # 성별 추출
                gender = str(data['gender'].item()) if 'gender' in data else self.gender
                
                # 프레임레이트 추출
                fps = float(data['mocap_framerate'].item()) if 'mocap_framerate' in data else self.fps
                
                # Shape parameters 추출
                betas = data['betas'].astype(np.float32) if 'betas' in data else np.zeros(16, dtype=np.float32)
            
            # Pose 파라미터 분리
            # poses: (T, 156) = root (3) + body joints (51 * 3 = 153)
            root_pose = poses[:, :3]  # (T, 3) 루트 회전 (axis-angle)
            body_pose = poses[:, 3:72]  # (T, 69) 몸통 관절 (23개 * 3)
            # 나머지 72:156은 손 관절 (사용하지 않음)
            
            # 데이터 형식 검증
            num_frames = trans.shape[0]
            if poses.shape[0] != num_frames:
                raise ValueError(f"Frame count mismatch: trans={num_frames}, poses={poses.shape[0]}")
            
            if trans.shape[1] != 3:
                raise ValueError(f"Invalid root translation shape: {trans.shape}, expected (T, 3)")
            
            if poses.shape[1] != 156:
                raise ValueError(f"Invalid pose shape: {poses.shape}, expected (T, 156)")
            
            logger.info(f"Loaded motion: {file_path.name}, frames={num_frames}, fps={fps}, gender={gender}")
            
            return {
                'root_translation': trans,  # (T, 3)
                'root_orientation': root_pose,  # (T, 3) axis-angle
                'pose': body_pose,  # (T, 69) body joints axis-angle
                'betas': betas,  # (16,)
                'gender': gender,
                'fps': fps,
                'num_frames': num_frames,
                'file_path': str(file_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to load motion file {file_path}: {e}")
            raise
    
    def get_joint_positions(
        self,
        motion_data: Dict[str, np.ndarray],
        use_smplx: bool = True
    ) -> np.ndarray:
        """SMPL 모션 데이터에서 관절 위치를 계산합니다.
        
        Args:
            motion_data: load_motion()의 출력
            use_smplx: smplx 라이브러리 사용 여부 (True 권장)
            
        Returns:
            Joint positions: (T, 24, 3) SMPL 24개 관절의 3D 위치
        """
        if use_smplx and self.smpl_model is None and self.smpl_model_original is None:
            # 모델이 로드되지 않았으면 로드 시도
            gender = motion_data.get('gender', self.gender)
            self._load_smpl_model(gender)
        
        if self.use_original_smpl and self.smpl_model_original is not None:
            return self._get_joint_positions_original(motion_data)
        elif use_smplx and self.smpl_model is not None:
            return self._get_joint_positions_smplx(motion_data)
        else:
            logger.warning("SMPL model not available. Using fallback method.")
            return self._get_joint_positions_fallback(motion_data)
    
    def _get_joint_positions_smplx(self, motion_data: Dict[str, np.ndarray]) -> np.ndarray:
        """smplx 라이브러리를 사용하여 관절 위치 계산.
        
        Args:
            motion_data: load_motion()의 출력
            
        Returns:
            Joint positions: (T, 24, 3)
        """
        num_frames = motion_data['num_frames']
        root_trans = torch.from_numpy(motion_data['root_translation']).float()
        root_pose = torch.from_numpy(motion_data['root_orientation']).float()
        body_pose = torch.from_numpy(motion_data['pose']).float()
        betas = torch.from_numpy(motion_data['betas']).unsqueeze(0).float()  # (1, 16)
        
        # 전체 pose 구성 (루트 + 몸통)
        # body_pose는 (T, 69)이지만 smplx는 (T, 63) 또는 (T, 69)를 받을 수 있음
        # SMPL은 23개 body joints (69차원) + root (3차원) = 72차원
        
        # SMPL pose 형식: (T, 72) = root (3) + body (69)
        full_pose = torch.cat([root_pose, body_pose], dim=1)  # (T, 72)
        
        # 각 프레임에 대해 관절 위치 계산
        joint_positions = []
        
        for t in range(num_frames):
            # SMPL forward pass
            output = self.smpl_model(
                betas=betas,
                body_pose=full_pose[t:t+1, 3:],  # (1, 69)
                global_orient=full_pose[t:t+1, :3],  # (1, 3)
                transl=root_trans[t:t+1]  # (1, 3)
            )
            
            # 관절 위치 추출 (SMPL은 24개 관절)
            joints = output.joints.squeeze(0).detach().numpy()  # (24, 3)
            joint_positions.append(joints)
        
        joint_positions = np.array(joint_positions)  # (T, 24, 3)
        
        return joint_positions
    
    def _get_joint_positions_original(self, motion_data: Dict[str, np.ndarray]) -> np.ndarray:
        """원본 SMPL 코드를 사용하여 관절 위치 계산.
        
        Args:
            motion_data: load_motion()의 출력
            
        Returns:
            Joint positions: (T, 24, 3)
        """
        import chumpy as ch
        import numpy as np
        
        num_frames = motion_data['num_frames']
        root_trans = motion_data['root_translation']  # (T, 3)
        root_pose = motion_data['root_orientation']  # (T, 3) axis-angle
        body_pose = motion_data['pose']  # (T, 69)
        betas = motion_data['betas']  # (16,)
        
        # 전체 pose 구성 (루트 + 몸통)
        full_pose = np.concatenate([root_pose, body_pose], axis=1)  # (T, 72)
        
        joint_positions = []
        
        for t in range(num_frames):
            # 원본 SMPL 모델에 pose와 betas 설정
            self.smpl_model_original.pose[:] = full_pose[t]
            self.smpl_model_original.betas[:] = betas
            self.smpl_model_original.trans[:] = root_trans[t]
            
            # 관절 위치 계산 (J_transformed는 변환된 관절 위치)
            if hasattr(self.smpl_model_original, 'J_transformed'):
                joints = np.array(self.smpl_model_original.J_transformed.r)  # (24, 3)
            elif hasattr(self.smpl_model_original, 'J'):
                # J는 변환 전 관절 위치이므로 trans를 더해야 함
                joints = np.array(self.smpl_model_original.J.r) + root_trans[t]  # (24, 3)
            else:
                # fallback: 루트 위치만 사용
                joints = np.zeros((24, 3))
                joints[0] = root_trans[t]
            
            joint_positions.append(joints)
        
        joint_positions = np.array(joint_positions)  # (T, 24, 3)
        
        return joint_positions.astype(np.float32)
    
    def _get_joint_positions_fallback(self, motion_data: Dict[str, np.ndarray]) -> np.ndarray:
        """Fallback 방법: 루트 위치만 반환 (관절 위치는 근사치).
        
        Args:
            motion_data: load_motion()의 출력
            
        Returns:
            Joint positions: (T, 24, 3) - 루트 위치를 모든 관절에 복사
        """
        num_frames = motion_data['num_frames']
        root_trans = motion_data['root_translation']  # (T, 3)
        
        # 루트 위치를 모든 관절에 복사 (임시 방법)
        # 실제로는 SMPL forward kinematics가 필요함
        joint_positions = np.zeros((num_frames, 24, 3), dtype=np.float32)
        joint_positions[:, 0, :] = root_trans  # 루트 관절
        
        logger.warning("Using fallback method: joint positions are approximate.")
        
        return joint_positions
    
    def get_joint_velocities(
        self,
        joint_positions: np.ndarray,
        fps: Optional[float] = None
    ) -> np.ndarray:
        """관절 위치에서 관절 속도를 계산합니다.
        
        Args:
            joint_positions: (T, 24, 3) 관절 위치
            fps: 프레임레이트 (None이면 self.fps 사용)
            
        Returns:
            Joint velocities: (T, 24, 3) 관절 속도
        """
        if fps is None:
            fps = self.fps
        
        dt = 1.0 / fps
        
        # 차분으로 속도 계산
        velocities = np.zeros_like(joint_positions)
        velocities[1:] = (joint_positions[1:] - joint_positions[:-1]) / dt
        
        # 첫 프레임은 두 번째 프레임과 동일하게 설정
        velocities[0] = velocities[1]
        
        return velocities


def test_smpl_loader():
    """SMPL 로더 테스트 함수."""
    import sys
    
    # 테스트 데이터 경로
    test_file = Path(__file__).parent.parent.parent.parent / "data" / "amass" / "ACCAD" / "Female1Walking_c3d" / "B3 - walk1_poses.npz"
    
    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        return
    
    # 로더 생성
    loader = SMPLLoader(
        data_path=str(test_file.parent.parent.parent),
        fps=30,
        gender="female"
    )
    
    # 모션 로드
    print(f"[TEST] Loading motion: {test_file}")
    motion_data = loader.load_motion(str(test_file))
    
    print(f"[TEST] Motion data keys: {list(motion_data.keys())}")
    print(f"[TEST] Root translation shape: {motion_data['root_translation'].shape}")
    print(f"[TEST] Root orientation shape: {motion_data['root_orientation'].shape}")
    print(f"[TEST] Pose shape: {motion_data['pose'].shape}")
    print(f"[TEST] Betas shape: {motion_data['betas'].shape}")
    print(f"[TEST] FPS: {motion_data['fps']}")
    print(f"[TEST] Gender: {motion_data['gender']}")
    print(f"[TEST] Num frames: {motion_data['num_frames']}")
    
    # 관절 위치 계산
    print(f"[TEST] Computing joint positions...")
    joint_positions = loader.get_joint_positions(motion_data)
    print(f"[TEST] Joint positions shape: {joint_positions.shape}")
    print(f"[TEST] Joint positions range: [{joint_positions.min():.3f}, {joint_positions.max():.3f}]")
    
    # 관절 속도 계산
    print(f"[TEST] Computing joint velocities...")
    joint_velocities = loader.get_joint_velocities(joint_positions, fps=motion_data['fps'])
    print(f"[TEST] Joint velocities shape: {joint_velocities.shape}")
    print(f"[TEST] Joint velocities range: [{joint_velocities.min():.3f}, {joint_velocities.max():.3f}]")
    
    print("[TEST] All tests passed!")


if __name__ == "__main__":
    test_smpl_loader()

