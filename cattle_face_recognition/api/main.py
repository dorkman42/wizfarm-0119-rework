"""
소 얼굴 인식 시스템 API 서버
FastAPI 기반 REST API
"""
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# 패키지 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .routes import cattle, recognition, health
from .config import settings
from .dependencies import get_pipeline, init_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 실행"""
    # 시작 시
    print("소 얼굴 인식 API 서버 시작...")
    init_pipeline()

    # 업로드 디렉토리 생성
    Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.GALLERY_DIR).mkdir(parents=True, exist_ok=True)

    yield

    # 종료 시
    print("서버 종료...")


app = FastAPI(
    title="소 얼굴 인식 API",
    description="소 얼굴 검출, 등록, 인식을 위한 REST API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 서빙 (업로드된 이미지)
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")

# 라우터 등록
app.include_router(health.router, tags=["Health"])
app.include_router(cattle.router, prefix="/api/cattle", tags=["Cattle"])
app.include_router(recognition.router, prefix="/api/recognition", tags=["Recognition"])


@app.get("/")
async def root():
    """API 정보"""
    return {
        "name": "소 얼굴 인식 API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
