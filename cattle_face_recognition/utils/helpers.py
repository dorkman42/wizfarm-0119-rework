"""
헬퍼 유틸리티
IOU 계산, 얼굴 크롭, 키포인트 정규화 등
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Union
from pathlib import Path


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    두 바운딩 박스의 IOU (Intersection over Union) 계산

    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]

    Returns:
        IOU 값 (0~1)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / (union + 1e-6)


def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    두 바운딩 박스 집합 간의 IOU 행렬 계산

    Args:
        boxes1: [N, 4]
        boxes2: [M, 4]

    Returns:
        IOU 행렬 [N, M]
    """
    n = len(boxes1)
    m = len(boxes2)
    iou_matrix = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            iou_matrix[i, j] = compute_iou(boxes1[i], boxes2[j])

    return iou_matrix


def crop_face(
    image: np.ndarray,
    bbox: np.ndarray,
    margin: float = 0.2,
    output_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    바운딩 박스 기반 얼굴 크롭

    Args:
        image: 입력 이미지
        bbox: [x1, y1, x2, y2]
        margin: 여백 비율
        output_size: 출력 크기 (None이면 원본 크기)

    Returns:
        크롭된 이미지
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox.astype(int)

    # 박스 크기
    bw, bh = x2 - x1, y2 - y1

    # 여백 추가
    margin_w = int(bw * margin)
    margin_h = int(bh * margin)

    x1 = max(0, x1 - margin_w)
    y1 = max(0, y1 - margin_h)
    x2 = min(w, x2 + margin_w)
    y2 = min(h, y2 + margin_h)

    # 크롭
    crop = image[y1:y2, x1:x2]

    # 리사이즈
    if output_size:
        crop = cv2.resize(crop, output_size)

    return crop


def normalize_keypoints(
    keypoints: np.ndarray,
    bbox: np.ndarray,
) -> np.ndarray:
    """
    키포인트를 바운딩 박스 기준으로 정규화 (0~1)

    Args:
        keypoints: [N, 2] 또는 [N, 3]
        bbox: [x1, y1, x2, y2]

    Returns:
        정규화된 키포인트
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1

    normalized = keypoints.copy().astype(float)
    normalized[:, 0] = (keypoints[:, 0] - x1) / (w + 1e-6)
    normalized[:, 1] = (keypoints[:, 1] - y1) / (h + 1e-6)

    return normalized


def denormalize_keypoints(
    normalized_keypoints: np.ndarray,
    bbox: np.ndarray,
) -> np.ndarray:
    """
    정규화된 키포인트를 원래 좌표로 복원

    Args:
        normalized_keypoints: 정규화된 키포인트 [N, 2] 또는 [N, 3]
        bbox: [x1, y1, x2, y2]

    Returns:
        원래 좌표의 키포인트
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1

    denormalized = normalized_keypoints.copy()
    denormalized[:, 0] = normalized_keypoints[:, 0] * w + x1
    denormalized[:, 1] = normalized_keypoints[:, 1] * h + y1

    return denormalized


def bbox_to_square(bbox: np.ndarray, padding: float = 0.0) -> np.ndarray:
    """
    바운딩 박스를 정사각형으로 변환

    Args:
        bbox: [x1, y1, x2, y2]
        padding: 패딩 비율

    Returns:
        정사각형 바운딩 박스
    """
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = x2 - x1, y2 - y1

    size = max(w, h) * (1 + padding)
    half_size = size / 2

    return np.array([
        cx - half_size,
        cy - half_size,
        cx + half_size,
        cy + half_size
    ])


def clip_bbox(bbox: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    """
    바운딩 박스를 이미지 크기 내로 클리핑

    Args:
        bbox: [x1, y1, x2, y2]
        image_size: (width, height)

    Returns:
        클리핑된 바운딩 박스
    """
    w, h = image_size
    clipped = bbox.copy()
    clipped[0] = max(0, bbox[0])
    clipped[1] = max(0, bbox[1])
    clipped[2] = min(w, bbox[2])
    clipped[3] = min(h, bbox[3])
    return clipped


def load_image(path: Union[str, Path]) -> np.ndarray:
    """이미지 로드"""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"이미지를 로드할 수 없습니다: {path}")
    return img


def save_image(image: np.ndarray, path: Union[str, Path], create_dir: bool = True):
    """이미지 저장"""
    path = Path(path)
    if create_dir:
        path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def resize_keeping_aspect(
    image: np.ndarray,
    target_size: Tuple[int, int],
    pad_color: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    종횡비를 유지하며 리사이즈 (패딩 추가)

    Args:
        image: 입력 이미지
        target_size: 목표 크기 (width, height)
        pad_color: 패딩 색상

    Returns:
        (리사이즈된 이미지, 스케일, (pad_x, pad_y))
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h))

    # 패딩
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2

    padded = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)
    padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized

    return padded, scale, (pad_x, pad_y)


def match_detections_by_iou(
    prev_detections: List,
    curr_detections: List,
    iou_threshold: float = 0.3,
) -> List[Tuple[int, int]]:
    """
    IOU 기반 검출 매칭 (프레임 간 추적용)

    Args:
        prev_detections: 이전 프레임 검출
        curr_detections: 현재 프레임 검출
        iou_threshold: IOU 임계값

    Returns:
        매칭된 (prev_idx, curr_idx) 튜플 리스트
    """
    if not prev_detections or not curr_detections:
        return []

    prev_boxes = np.array([d.bbox for d in prev_detections])
    curr_boxes = np.array([d.bbox for d in curr_detections])

    iou_matrix = compute_iou_matrix(prev_boxes, curr_boxes)

    matches = []
    used_prev = set()
    used_curr = set()

    # 탐욕적 매칭 (IOU가 높은 순)
    while True:
        # 최대 IOU 찾기
        max_iou = iou_threshold
        max_prev, max_curr = -1, -1

        for i in range(len(prev_detections)):
            if i in used_prev:
                continue
            for j in range(len(curr_detections)):
                if j in used_curr:
                    continue
                if iou_matrix[i, j] > max_iou:
                    max_iou = iou_matrix[i, j]
                    max_prev, max_curr = i, j

        if max_prev < 0:
            break

        matches.append((max_prev, max_curr))
        used_prev.add(max_prev)
        used_curr.add(max_curr)

    return matches
