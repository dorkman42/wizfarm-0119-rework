"""
소 얼굴 정렬 모듈
키포인트 기반 얼굴 정렬 및 자세 분류
"""
import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from enum import Enum


class PoseType(Enum):
    """얼굴 자세 타입"""
    FRONTAL = "frontal"    # 정면
    LEFT = "left"          # 왼쪽 측면
    RIGHT = "right"        # 오른쪽 측면
    UNKNOWN = "unknown"    # 알 수 없음


@dataclass
class AlignedFace:
    """정렬된 얼굴 데이터"""
    image: np.ndarray              # 정렬된 얼굴 이미지
    keypoints: np.ndarray          # 정규화된 키포인트
    pose: PoseType                 # 자세 타입
    pose_angle: float              # 자세 각도 (정면=0, 측면=90)
    quality_score: float           # 품질 점수
    original_bbox: np.ndarray      # 원본 바운딩 박스
    transform_matrix: np.ndarray   # 변환 행렬


class FaceAligner:
    """
    소 얼굴 정렬기

    키포인트를 사용하여 얼굴을 정규화된 위치로 정렬하고
    정면/측면 자세를 분류합니다.

    사용법:
        aligner = FaceAligner(output_size=(112, 112))
        aligned = aligner.align(image, keypoints, bbox)
    """

    # 소 얼굴 기준 키포인트 위치 (112x112 기준)
    # [왼쪽 눈, 오른쪽 눈, 코, 왼쪽 귀, 오른쪽 귀]
    REFERENCE_POINTS_112 = np.array([
        [38.2946, 51.6963],   # 왼쪽 눈
        [73.5318, 51.5014],   # 오른쪽 눈
        [56.0252, 71.7366],   # 코
        [26.0, 30.0],         # 왼쪽 귀
        [86.0, 30.0],         # 오른쪽 귀
    ], dtype=np.float32)

    def __init__(
        self,
        output_size: Tuple[int, int] = (112, 112),
        num_keypoints: int = 5,
        pose_threshold: float = 0.3,
    ):
        """
        Args:
            output_size: 출력 얼굴 이미지 크기 (width, height)
            num_keypoints: 키포인트 수
            pose_threshold: 정면/측면 분류 임계값
        """
        self.output_size = output_size
        self.num_keypoints = num_keypoints
        self.pose_threshold = pose_threshold

        # 출력 크기에 맞게 참조 포인트 스케일링
        scale_x = output_size[0] / 112.0
        scale_y = output_size[1] / 112.0
        self.reference_points = self.REFERENCE_POINTS_112.copy()
        self.reference_points[:, 0] *= scale_x
        self.reference_points[:, 1] *= scale_y

    def align(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        bbox: Optional[np.ndarray] = None,
    ) -> AlignedFace:
        """
        단일 얼굴 정렬

        Args:
            image: 입력 이미지
            keypoints: 키포인트 좌표 [N, 2] 또는 [N, 3] (x, y, conf)
            bbox: 바운딩 박스 [x1, y1, x2, y2]

        Returns:
            AlignedFace 객체
        """
        # 키포인트 형식 확인
        kpts = keypoints[:, :2] if keypoints.shape[1] >= 2 else keypoints
        kpts = kpts.astype(np.float32)

        # 자세 분류
        pose, pose_angle = self._classify_pose(kpts)

        # 유효한 키포인트만 사용하여 변환 행렬 계산
        valid_mask = self._get_valid_keypoints(keypoints)

        if np.sum(valid_mask) >= 3:
            # 충분한 키포인트가 있으면 어파인 변환
            src_pts = kpts[valid_mask]
            dst_pts = self.reference_points[valid_mask]
            M = cv2.estimateAffinePartial2D(src_pts, dst_pts)[0]
        elif bbox is not None:
            # 키포인트가 부족하면 바운딩 박스 기반 변환
            M = self._bbox_to_transform(bbox)
        else:
            # 둘 다 없으면 항등 변환
            M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

        # 얼굴 정렬
        aligned_image = cv2.warpAffine(
            image, M, self.output_size,
            borderMode=cv2.BORDER_REPLICATE
        )

        # 키포인트 변환
        aligned_kpts = self._transform_keypoints(kpts, M)

        # 품질 점수 계산
        quality = self._compute_quality(keypoints, pose, aligned_image)

        return AlignedFace(
            image=aligned_image,
            keypoints=aligned_kpts,
            pose=pose,
            pose_angle=pose_angle,
            quality_score=quality,
            original_bbox=bbox if bbox is not None else np.zeros(4),
            transform_matrix=M
        )

    def align_batch(
        self,
        image: np.ndarray,
        keypoints_list: List[np.ndarray],
        bboxes: Optional[List[np.ndarray]] = None,
    ) -> List[AlignedFace]:
        """
        여러 얼굴 일괄 정렬

        Args:
            image: 입력 이미지
            keypoints_list: 키포인트 리스트
            bboxes: 바운딩 박스 리스트

        Returns:
            AlignedFace 객체 리스트
        """
        results = []
        for i, kpts in enumerate(keypoints_list):
            bbox = bboxes[i] if bboxes else None
            aligned = self.align(image, kpts, bbox)
            results.append(aligned)
        return results

    def crop_face(
        self,
        image: np.ndarray,
        bbox: np.ndarray,
        margin: float = 0.2,
    ) -> np.ndarray:
        """
        바운딩 박스 기반 얼굴 크롭

        Args:
            image: 입력 이미지
            bbox: 바운딩 박스 [x1, y1, x2, y2]
            margin: 여백 비율

        Returns:
            크롭된 얼굴 이미지
        """
        x1, y1, x2, y2 = bbox.astype(int)
        w, h = x2 - x1, y2 - y1

        # 여백 추가
        margin_w = int(w * margin)
        margin_h = int(h * margin)

        x1 = max(0, x1 - margin_w)
        y1 = max(0, y1 - margin_h)
        x2 = min(image.shape[1], x2 + margin_w)
        y2 = min(image.shape[0], y2 + margin_h)

        # 크롭
        crop = image[y1:y2, x1:x2]

        # 리사이즈
        resized = cv2.resize(crop, self.output_size)

        return resized

    def _classify_pose(
        self,
        keypoints: np.ndarray
    ) -> Tuple[PoseType, float]:
        """
        자세 분류 (정면/측면)

        키포인트 위치를 분석하여 얼굴의 자세를 분류합니다.
        양쪽 눈 사이의 거리와 눈-코 거리 비율을 사용합니다.
        """
        if len(keypoints) < 3:
            return PoseType.UNKNOWN, 0.0

        left_eye = keypoints[0]
        right_eye = keypoints[1]
        nose = keypoints[2]

        # 눈 사이 거리
        eye_dist = np.linalg.norm(right_eye - left_eye)
        if eye_dist < 1e-6:
            return PoseType.UNKNOWN, 0.0

        # 눈 중심
        eye_center = (left_eye + right_eye) / 2

        # 코가 눈 중심에서 얼마나 벗어났는지
        nose_offset = nose[0] - eye_center[0]
        offset_ratio = nose_offset / eye_dist

        # 자세 각도 계산 (대략적)
        pose_angle = np.arctan(offset_ratio) * 180 / np.pi

        # 분류
        if abs(offset_ratio) < self.pose_threshold:
            return PoseType.FRONTAL, pose_angle
        elif offset_ratio > 0:
            return PoseType.RIGHT, pose_angle
        else:
            return PoseType.LEFT, pose_angle

    def _get_valid_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """유효한 키포인트 마스크 반환"""
        if keypoints.shape[1] >= 3:
            # confidence가 있는 경우
            return keypoints[:, 2] > 0.5
        else:
            # confidence가 없으면 모두 유효
            return np.ones(len(keypoints), dtype=bool)

    def _bbox_to_transform(self, bbox: np.ndarray) -> np.ndarray:
        """바운딩 박스를 변환 행렬로 변환"""
        x1, y1, x2, y2 = bbox
        src_w, src_h = x2 - x1, y2 - y1
        dst_w, dst_h = self.output_size

        scale_x = dst_w / src_w
        scale_y = dst_h / src_h
        scale = min(scale_x, scale_y)

        # 중앙 정렬
        tx = (dst_w - src_w * scale) / 2 - x1 * scale
        ty = (dst_h - src_h * scale) / 2 - y1 * scale

        M = np.array([
            [scale, 0, tx],
            [0, scale, ty]
        ], dtype=np.float32)

        return M

    def _transform_keypoints(
        self,
        keypoints: np.ndarray,
        M: np.ndarray
    ) -> np.ndarray:
        """키포인트 좌표 변환"""
        ones = np.ones((len(keypoints), 1))
        kpts_homo = np.hstack([keypoints[:, :2], ones])
        transformed = (M @ kpts_homo.T).T
        return transformed

    def _compute_quality(
        self,
        keypoints: np.ndarray,
        pose: PoseType,
        aligned_image: np.ndarray
    ) -> float:
        """
        얼굴 품질 점수 계산

        고려 요소:
        - 키포인트 신뢰도
        - 자세 (정면이 높은 점수)
        - 이미지 선명도 (Laplacian variance)
        """
        score = 0.0

        # 키포인트 신뢰도 (있는 경우)
        if keypoints.shape[1] >= 3:
            conf_score = np.mean(keypoints[:, 2])
            score += conf_score * 0.4

        # 자세 점수
        pose_score = {
            PoseType.FRONTAL: 1.0,
            PoseType.LEFT: 0.6,
            PoseType.RIGHT: 0.6,
            PoseType.UNKNOWN: 0.3
        }.get(pose, 0.3)
        score += pose_score * 0.3

        # 이미지 선명도 (Laplacian variance)
        gray = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(laplacian_var / 500, 1.0)  # 정규화
        score += sharpness_score * 0.3

        return score

    def select_best_face(
        self,
        aligned_faces: List[AlignedFace],
        prefer_frontal: bool = True
    ) -> Optional[AlignedFace]:
        """
        가장 좋은 품질의 얼굴 선택

        Args:
            aligned_faces: 정렬된 얼굴 리스트
            prefer_frontal: True면 정면 얼굴 우선

        Returns:
            최고 품질의 얼굴 또는 None
        """
        if not aligned_faces:
            return None

        if prefer_frontal:
            # 정면 얼굴 필터링
            frontal = [f for f in aligned_faces if f.pose == PoseType.FRONTAL]
            if frontal:
                return max(frontal, key=lambda x: x.quality_score)

        # 전체에서 최고 품질 선택
        return max(aligned_faces, key=lambda x: x.quality_score)
