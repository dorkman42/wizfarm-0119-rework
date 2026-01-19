"""
소 얼굴 인식 시스템 (Cattle Face Recognition System)

YOLOv8 기반 검출 + InsightFace 기반 인식 파이프라인
참조: https://github.com/shujiejulie/An-end-to-end-cattle-face-recognition-system
"""
from .config import PipelineConfig, DetectionConfig, AlignmentConfig, RecognitionConfig
from .detection import CattleFaceDetector
from .alignment import FaceAligner
from .recognition import FaceRecognizer
from .pipeline import CattleFaceRecognitionPipeline

__version__ = "1.0.0"
__all__ = [
    "PipelineConfig",
    "DetectionConfig",
    "AlignmentConfig",
    "RecognitionConfig",
    "CattleFaceDetector",
    "FaceAligner",
    "FaceRecognizer",
    "CattleFaceRecognitionPipeline",
]
