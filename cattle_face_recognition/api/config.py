"""
API 설정
"""
from pydantic_settings import BaseSettings
from typing import List
from pathlib import Path


class Settings(BaseSettings):
    """API 설정"""
    # 서버 설정
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True

    # CORS 설정
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
    ]

    # 파일 저장 경로
    BASE_DIR: str = str(Path(__file__).parent.parent)
    UPLOAD_DIR: str = str(Path(__file__).parent.parent / "uploads")
    GALLERY_DIR: str = str(Path(__file__).parent.parent / "gallery")
    GALLERY_FILE: str = str(Path(__file__).parent.parent / "gallery" / "cattle_gallery.pkl")

    # 모델 설정
    DETECTION_MODEL: str = str(Path(__file__).parent.parent / "models" / "cattle_face_detector.pt")
    RECOGNITION_MODEL: str = str(Path(__file__).parent.parent / "models" / "cattle_face_recognizer.pt")
    CONFIDENCE_THRESHOLD: float = 0.1  # 소 얼굴 검출용 (학습 모델 특성상 낮게 설정)
    SIMILARITY_THRESHOLD: float = 0.4

    # 파일 업로드 설정
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".webp"]

    class Config:
        env_file = ".env"


settings = Settings()
