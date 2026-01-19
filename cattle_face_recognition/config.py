"""
소 얼굴 인식 시스템 설정
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pathlib import Path


def get_device() -> str:
    """사용 가능한 디바이스 자동 감지"""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"  # Apple Silicon
    except ImportError:
        pass
    return "cpu"


# 기본 디바이스 (자동 감지)
DEFAULT_DEVICE = get_device()


@dataclass
class DetectionConfig:
    """얼굴 검출 설정"""
    model_path: str = "yolov8n.pt"  # 기본 YOLOv8 모델 (fine-tuning 필요)
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    image_size: int = 640
    device: str = field(default_factory=lambda: DEFAULT_DEVICE)


@dataclass
class AlignmentConfig:
    """얼굴 정렬 설정"""
    # 키포인트 인덱스 (소 얼굴 기준)
    # 0: 왼쪽 눈, 1: 오른쪽 눈, 2: 코, 3: 왼쪽 귀, 4: 오른쪽 귀
    keypoint_names: List[str] = field(default_factory=lambda: [
        "left_eye", "right_eye", "nose", "left_ear", "right_ear"
    ])
    num_keypoints: int = 5
    output_size: Tuple[int, int] = (112, 112)  # 정렬된 얼굴 크기
    pose_threshold: float = 0.3  # 정면/측면 분류 임계값


@dataclass
class RecognitionConfig:
    """얼굴 인식 설정"""
    model_name: str = "buffalo_l"  # InsightFace 모델
    embedding_size: int = 512
    similarity_threshold: float = 0.4  # 동일 개체 판단 임계값
    device: str = field(default_factory=lambda: DEFAULT_DEVICE)
    custom_model_path: Optional[str] = None  # 자체 학습 모델 경로 (상업용)


@dataclass
class PipelineConfig:
    """전체 파이프라인 설정"""
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)

    # 출력 설정
    output_dir: str = "./output"
    save_visualizations: bool = True
    save_crops: bool = True

    # 추적 설정 (비디오용)
    iou_tracking_threshold: float = 0.3
    max_frames_to_skip: int = 5

    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


# 색상 팔레트 (시각화용)
COLORS = {
    'bbox': (0, 255, 0),        # 초록
    'keypoint': (255, 0, 0),    # 파랑
    'skeleton': (0, 255, 255),  # 노랑
    'text': (255, 255, 255),    # 흰색
    'id_colors': [              # 개체별 색상
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128),
        (128, 128, 0), (128, 0, 128), (0, 128, 128),
    ]
}

# 소 얼굴 키포인트 연결 (스켈레톤)
SKELETON_CONNECTIONS = [
    (0, 1),  # 왼쪽 눈 - 오른쪽 눈
    (0, 2),  # 왼쪽 눈 - 코
    (1, 2),  # 오른쪽 눈 - 코
    (0, 3),  # 왼쪽 눈 - 왼쪽 귀
    (1, 4),  # 오른쪽 눈 - 오른쪽 귀
]
