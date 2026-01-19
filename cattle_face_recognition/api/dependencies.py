"""
의존성 주입
"""
from typing import Optional
from pathlib import Path

from .config import settings

# 전역 파이프라인 인스턴스
_pipeline: Optional["CattleFaceRecognitionPipeline"] = None


def init_pipeline():
    """파이프라인 초기화"""
    global _pipeline

    from cattle_face_recognition.pipeline import CattleFaceRecognitionPipeline
    from cattle_face_recognition.config import PipelineConfig, DetectionConfig, RecognitionConfig

    config = PipelineConfig(
        detection=DetectionConfig(
            confidence_threshold=settings.CONFIDENCE_THRESHOLD,
        ),
        recognition=RecognitionConfig(
            embedding_size=256,  # 자체 학습 모델 임베딩 크기
            similarity_threshold=settings.SIMILARITY_THRESHOLD,
            custom_model_path=settings.RECOGNITION_MODEL,
        ),
        output_dir=settings.UPLOAD_DIR,
    )

    _pipeline = CattleFaceRecognitionPipeline(
        config=config,
        detection_model=settings.DETECTION_MODEL,
        gallery_path=settings.GALLERY_FILE,
    )

    print(f"파이프라인 초기화 완료")
    print(f"갤러리 경로: {settings.GALLERY_FILE}")


def get_pipeline() -> "CattleFaceRecognitionPipeline":
    """파이프라인 인스턴스 반환"""
    global _pipeline
    if _pipeline is None:
        init_pipeline()
    return _pipeline
