"""
시각화 유틸리티
검출 결과, 키포인트, 인식 결과 시각화
"""
import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict
from pathlib import Path

from ..config import COLORS, SKELETON_CONNECTIONS


def draw_detections(
    image: np.ndarray,
    detections: List,
    show_keypoints: bool = True,
    show_skeleton: bool = True,
    show_labels: bool = True,
    recognition_results: Optional[List] = None,
) -> np.ndarray:
    """
    검출 결과 시각화

    Args:
        image: 입력 이미지
        detections: Detection 객체 리스트
        show_keypoints: 키포인트 표시 여부
        show_skeleton: 스켈레톤 표시 여부
        show_labels: 레이블 표시 여부
        recognition_results: 인식 결과 리스트 (있으면 ID 표시)

    Returns:
        시각화된 이미지
    """
    vis = image.copy()
    num_colors = len(COLORS['id_colors'])

    for i, det in enumerate(detections):
        # 색상 선택
        if det.track_id is not None:
            color = COLORS['id_colors'][det.track_id % num_colors]
        else:
            color = COLORS['bbox']

        # 바운딩 박스
        x1, y1, x2, y2 = det.bbox.astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # 레이블
        if show_labels:
            label_parts = []
            if det.track_id is not None:
                label_parts.append(f"#{det.track_id}")

            # 인식 결과가 있으면 이름 표시
            if recognition_results and i < len(recognition_results):
                rec = recognition_results[i]
                if rec.name:
                    label_parts.append(rec.name)
                label_parts.append(f"{rec.confidence:.2f}")
            else:
                label_parts.append(f"{det.confidence:.2f}")

            label = " ".join(label_parts)
            _draw_label(vis, label, (x1, y1 - 5), color)

        # 키포인트
        if show_keypoints and det.keypoints is not None:
            draw_keypoints(vis, det.keypoints, color=COLORS['keypoint'])

            # 스켈레톤
            if show_skeleton:
                draw_skeleton(vis, det.keypoints, color=COLORS['skeleton'])

    return vis


def draw_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),
    radius: int = 4,
    thickness: int = -1,
) -> np.ndarray:
    """
    키포인트 그리기

    Args:
        image: 입력 이미지 (in-place 수정)
        keypoints: 키포인트 [N, 2] 또는 [N, 3]
        color: 색상 (BGR)
        radius: 원 반지름
        thickness: 선 두께 (-1이면 채움)

    Returns:
        수정된 이미지
    """
    for kpt in keypoints:
        x, y = int(kpt[0]), int(kpt[1])
        conf = kpt[2] if len(kpt) > 2 else 1.0

        if conf > 0.5:
            cv2.circle(image, (x, y), radius, color, thickness)

    return image


def draw_skeleton(
    image: np.ndarray,
    keypoints: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
    connections: Optional[List[Tuple[int, int]]] = None,
) -> np.ndarray:
    """
    스켈레톤 그리기

    Args:
        image: 입력 이미지 (in-place 수정)
        keypoints: 키포인트 [N, 2] 또는 [N, 3]
        color: 색상 (BGR)
        thickness: 선 두께
        connections: 연결 정보 (None이면 기본값)

    Returns:
        수정된 이미지
    """
    if connections is None:
        connections = SKELETON_CONNECTIONS

    for start_idx, end_idx in connections:
        if start_idx >= len(keypoints) or end_idx >= len(keypoints):
            continue

        start_kpt = keypoints[start_idx]
        end_kpt = keypoints[end_idx]

        # 신뢰도 확인
        start_conf = start_kpt[2] if len(start_kpt) > 2 else 1.0
        end_conf = end_kpt[2] if len(end_kpt) > 2 else 1.0

        if start_conf > 0.5 and end_conf > 0.5:
            start_point = (int(start_kpt[0]), int(start_kpt[1]))
            end_point = (int(end_kpt[0]), int(end_kpt[1]))
            cv2.line(image, start_point, end_point, color, thickness)

    return image


def draw_recognition_result(
    image: np.ndarray,
    bbox: np.ndarray,
    cattle_id: Optional[str],
    name: Optional[str],
    confidence: float,
    color: Optional[Tuple[int, int, int]] = None,
) -> np.ndarray:
    """
    인식 결과 그리기

    Args:
        image: 입력 이미지
        bbox: 바운딩 박스
        cattle_id: 개체 ID
        name: 이름
        confidence: 신뢰도
        color: 색상 (None이면 자동)

    Returns:
        시각화된 이미지
    """
    vis = image.copy()
    x1, y1, x2, y2 = bbox.astype(int)

    # 색상 결정
    if color is None:
        if cattle_id:
            # ID가 있으면 해시 기반 색상
            color_idx = hash(cattle_id) % len(COLORS['id_colors'])
            color = COLORS['id_colors'][color_idx]
        else:
            color = (128, 128, 128)  # 미인식은 회색

    # 바운딩 박스
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

    # 레이블
    if name:
        label = f"{name} ({confidence:.2f})"
    elif cattle_id:
        label = f"{cattle_id} ({confidence:.2f})"
    else:
        label = f"Unknown ({confidence:.2f})"

    _draw_label(vis, label, (x1, y1 - 5), color)

    return vis


def _draw_label(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    color: Tuple[int, int, int],
    font_scale: float = 0.6,
    thickness: int = 2,
):
    """레이블 그리기 (배경 포함)"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    x, y = position
    y = max(y, text_h + 5)

    # 배경
    cv2.rectangle(
        image,
        (x, y - text_h - 5),
        (x + text_w + 5, y + baseline),
        color,
        -1
    )

    # 텍스트
    cv2.putText(
        image, text, (x + 2, y - 2),
        font, font_scale, (255, 255, 255), thickness
    )


def save_visualization(
    image: np.ndarray,
    output_path: str,
    create_dir: bool = True,
) -> str:
    """
    시각화 이미지 저장

    Args:
        image: 저장할 이미지
        output_path: 출력 경로
        create_dir: 디렉토리 생성 여부

    Returns:
        저장된 파일 경로
    """
    path = Path(output_path)
    if create_dir:
        path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(path), image)
    return str(path)


def create_gallery_grid(
    images: List[np.ndarray],
    names: Optional[List[str]] = None,
    grid_size: Optional[Tuple[int, int]] = None,
    cell_size: Tuple[int, int] = (112, 112),
    padding: int = 5,
) -> np.ndarray:
    """
    갤러리 그리드 이미지 생성

    Args:
        images: 이미지 리스트
        names: 이름 리스트
        grid_size: (rows, cols), None이면 자동
        cell_size: 각 셀 크기
        padding: 셀 간격

    Returns:
        그리드 이미지
    """
    n = len(images)
    if n == 0:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    if grid_size is None:
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
    else:
        rows, cols = grid_size

    cell_w, cell_h = cell_size
    total_w = cols * cell_w + (cols + 1) * padding
    total_h = rows * cell_h + (rows + 1) * padding

    grid = np.ones((total_h, total_w, 3), dtype=np.uint8) * 255

    for idx, img in enumerate(images):
        if idx >= rows * cols:
            break

        row = idx // cols
        col = idx % cols

        x = padding + col * (cell_w + padding)
        y = padding + row * (cell_h + padding)

        # 리사이즈
        resized = cv2.resize(img, cell_size)
        grid[y:y+cell_h, x:x+cell_w] = resized

        # 이름 표시
        if names and idx < len(names):
            cv2.putText(
                grid, names[idx],
                (x + 5, y + cell_h - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1
            )

    return grid


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """16진수 색상을 RGB로 변환"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_bgr(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """RGB를 BGR로 변환"""
    return (rgb[2], rgb[1], rgb[0])
