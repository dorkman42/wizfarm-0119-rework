"""
소 등록/관리 API 라우터
"""
import os
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from pydantic import BaseModel

from ..config import settings
from ..dependencies import get_pipeline

router = APIRouter()


# Pydantic 모델
class CattleCreate(BaseModel):
    """소 등록 요청"""
    name: str
    breed: Optional[str] = None
    age: Optional[int] = None
    notes: Optional[str] = None


class CattleUpdate(BaseModel):
    """소 정보 수정"""
    name: Optional[str] = None
    breed: Optional[str] = None
    age: Optional[int] = None
    notes: Optional[str] = None


class CattleResponse(BaseModel):
    """소 정보 응답"""
    cattle_id: str
    name: str
    num_images: int
    breed: Optional[str] = None
    age: Optional[int] = None
    notes: Optional[str] = None
    registered_at: str
    images: List[str] = []


class CattleListResponse(BaseModel):
    """소 목록 응답"""
    total: int
    cattle: List[CattleResponse]


def save_upload_file(upload_file: UploadFile, cattle_id: str) -> str:
    """업로드 파일 저장"""
    # 파일 확장자 확인
    ext = Path(upload_file.filename).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"허용되지 않는 파일 형식입니다: {ext}")

    # 저장 경로
    filename = f"{uuid.uuid4().hex}{ext}"
    cattle_dir = Path(settings.UPLOAD_DIR) / cattle_id
    cattle_dir.mkdir(parents=True, exist_ok=True)
    file_path = cattle_dir / filename

    # 파일 저장
    with open(file_path, "wb") as f:
        shutil.copyfileobj(upload_file.file, f)

    return str(file_path)


@router.get("", response_model=CattleListResponse)
async def list_cattle(
    pipeline=Depends(get_pipeline)
):
    """등록된 모든 소 목록 조회"""
    identities = pipeline.get_registered_cattle()

    cattle_list = []
    for identity in identities:
        # 이미지 URL 생성 (face crop만, _vis.jpg 제외)
        cattle_id = identity['cattle_id']
        images_dir = Path(settings.UPLOAD_DIR) / cattle_id
        image_urls = []
        if images_dir.exists():
            for img_file in images_dir.glob("*"):
                if img_file.suffix.lower() in settings.ALLOWED_EXTENSIONS:
                    # _vis.jpg 시각화 이미지는 제외
                    if not img_file.stem.endswith('_vis'):
                        image_urls.append(f"/uploads/{cattle_id}/{img_file.name}")

        cattle_list.append(CattleResponse(
            cattle_id=cattle_id,
            name=identity['name'],
            num_images=identity['num_images'],
            breed=identity.get('metadata', {}).get('breed'),
            age=identity.get('metadata', {}).get('age'),
            notes=identity.get('metadata', {}).get('notes'),
            registered_at=identity.get('registered_at', ''),
            images=image_urls,
        ))

    return CattleListResponse(total=len(cattle_list), cattle=cattle_list)


@router.get("/{cattle_id}", response_model=CattleResponse)
async def get_cattle(
    cattle_id: str,
    pipeline=Depends(get_pipeline)
):
    """특정 소 정보 조회"""
    if cattle_id not in pipeline.recognizer.gallery:
        raise HTTPException(404, "소를 찾을 수 없습니다")

    identity = pipeline.recognizer.gallery[cattle_id]

    # 이미지 URL (face crop만, _vis.jpg 제외)
    images_dir = Path(settings.UPLOAD_DIR) / cattle_id
    image_urls = []
    if images_dir.exists():
        for img_file in images_dir.glob("*"):
            if img_file.suffix.lower() in settings.ALLOWED_EXTENSIONS:
                # _vis.jpg 시각화 이미지는 제외
                if not img_file.stem.endswith('_vis'):
                    image_urls.append(f"/uploads/{cattle_id}/{img_file.name}")

    return CattleResponse(
        cattle_id=identity.cattle_id,
        name=identity.name,
        num_images=len(identity.embeddings),
        breed=identity.metadata.get('breed'),
        age=identity.metadata.get('age'),
        notes=identity.metadata.get('notes'),
        registered_at=identity.registered_at,
        images=image_urls,
    )


@router.post("", response_model=CattleResponse)
async def create_cattle(
    name: str = Form(...),
    breed: Optional[str] = Form(None),
    age: Optional[int] = Form(None),
    notes: Optional[str] = Form(None),
    images: List[UploadFile] = File(...),
    pipeline=Depends(get_pipeline)
):
    """새로운 소 등록 (얼굴 검출 후 crop된 이미지 + 바운딩박스 시각화 저장)"""
    if not images:
        raise HTTPException(400, "최소 1개의 이미지가 필요합니다")

    # ID 생성
    cattle_id = pipeline.recognizer.generate_id()
    cattle_dir = Path(settings.UPLOAD_DIR) / cattle_id
    cattle_dir.mkdir(parents=True, exist_ok=True)

    # 얼굴 crop 이미지 및 시각화 저장
    saved_paths = []
    for upload_file in images:
        # 이미지 읽기
        contents = await upload_file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            continue

        # 얼굴 검출
        detections = pipeline.detector.detect(img)
        file_id = uuid.uuid4().hex

        if detections:
            # 가장 큰 얼굴 1개만 선택 (여러 소가 있는 이미지 대응)
            det = max(detections, key=lambda d: (d.bbox[2]-d.bbox[0]) * (d.bbox[3]-d.bbox[1]))
            bx1, by1, bx2, by2 = det.bbox.astype(int)

            # 1. 원본 + 바운딩박스 시각화 이미지 저장
            vis_img = img.copy()
            cv2.rectangle(vis_img, (bx1, by1), (bx2, by2), (0, 255, 0), 3)
            cv2.putText(vis_img, name, (bx1, by1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            vis_path = cattle_dir / f"{file_id}_vis.jpg"
            cv2.imwrite(str(vis_path), vis_img)

            # 2. 얼굴 crop 이미지 저장 (임베딩용)
            h, w = img.shape[:2]
            margin = 0.1
            mw, mh = int((bx2-bx1)*margin), int((by2-by1)*margin)
            x1, y1 = max(0, bx1-mw), max(0, by1-mh)
            x2, y2 = min(w, bx2+mw), min(h, by2+mh)

            face_crop = img[y1:y2, x1:x2]
            if face_crop.size > 0:
                face_path = cattle_dir / f"{file_id}.jpg"
                cv2.imwrite(str(face_path), face_crop)
                saved_paths.append(str(face_path))
        else:
            # 검출 실패 시 원본 저장
            face_path = cattle_dir / f"{file_id}.jpg"
            cv2.imwrite(str(face_path), img)
            saved_paths.append(str(face_path))

    if not saved_paths:
        shutil.rmtree(cattle_dir, ignore_errors=True)
        raise HTTPException(400, "유효한 이미지가 없습니다")

    # 메타데이터
    metadata = {}
    if breed:
        metadata['breed'] = breed
    if age is not None:
        metadata['age'] = age
    if notes:
        metadata['notes'] = notes

    # 소 등록 (이미 crop된 이미지이므로 auto_detect=False)
    try:
        identity = pipeline.register_cattle(
            cattle_id=cattle_id,
            name=name,
            image_source=saved_paths,
            metadata=metadata,
            auto_detect=False,  # 이미 crop된 얼굴
        )
    except Exception as e:
        shutil.rmtree(cattle_dir, ignore_errors=True)
        raise HTTPException(400, f"등록 실패: {str(e)}")

    # 이미지 URL
    image_urls = [f"/uploads/{cattle_id}/{Path(p).name}" for p in saved_paths]

    return CattleResponse(
        cattle_id=identity.cattle_id,
        name=identity.name,
        num_images=len(identity.embeddings),
        breed=metadata.get('breed'),
        age=metadata.get('age'),
        notes=metadata.get('notes'),
        registered_at=identity.registered_at,
        images=image_urls,
    )


@router.put("/{cattle_id}", response_model=CattleResponse)
async def update_cattle(
    cattle_id: str,
    update: CattleUpdate,
    pipeline=Depends(get_pipeline)
):
    """소 정보 수정"""
    if cattle_id not in pipeline.recognizer.gallery:
        raise HTTPException(404, "소를 찾을 수 없습니다")

    identity = pipeline.recognizer.gallery[cattle_id]

    # 정보 업데이트
    if update.name:
        identity.name = update.name
    if update.breed is not None:
        identity.metadata['breed'] = update.breed
    if update.age is not None:
        identity.metadata['age'] = update.age
    if update.notes is not None:
        identity.metadata['notes'] = update.notes

    identity.updated_at = datetime.now().isoformat()

    # 갤러리 저장
    pipeline.save_gallery()

    return await get_cattle(cattle_id, pipeline)


@router.post("/{cattle_id}/images", response_model=CattleResponse)
async def add_cattle_images(
    cattle_id: str,
    images: List[UploadFile] = File(...),
    pipeline=Depends(get_pipeline)
):
    """소에 이미지 추가"""
    if cattle_id not in pipeline.recognizer.gallery:
        raise HTTPException(404, "소를 찾을 수 없습니다")

    # 이미지 저장 및 등록
    for img in images:
        path = save_upload_file(img, cattle_id)
        try:
            pipeline.register_cattle(
                cattle_id=cattle_id,
                name=pipeline.recognizer.gallery[cattle_id].name,
                image_source=path,
                auto_detect=True,
            )
        except Exception as e:
            print(f"이미지 추가 실패: {e}")

    return await get_cattle(cattle_id, pipeline)


@router.delete("/{cattle_id}/images/{filename}")
async def delete_cattle_image(
    cattle_id: str,
    filename: str,
    pipeline=Depends(get_pipeline)
):
    """소의 특정 이미지 삭제"""
    if cattle_id not in pipeline.recognizer.gallery:
        raise HTTPException(404, "소를 찾을 수 없습니다")

    # 이미지 파일 경로
    image_path = Path(settings.UPLOAD_DIR) / cattle_id / filename

    if not image_path.exists():
        raise HTTPException(404, "이미지를 찾을 수 없습니다")

    # 남은 이미지 수 확인
    cattle_dir = Path(settings.UPLOAD_DIR) / cattle_id
    remaining_images = [f for f in cattle_dir.glob("*") if f.suffix.lower() in settings.ALLOWED_EXTENSIONS and f.name != filename]

    if len(remaining_images) == 0:
        raise HTTPException(400, "최소 1개의 이미지는 유지해야 합니다")

    # 이미지 파일 삭제
    image_path.unlink()

    # 임베딩 재계산 (남은 이미지로)
    identity = pipeline.recognizer.gallery[cattle_id]
    identity.embeddings = []  # 기존 임베딩 초기화

    for img_file in remaining_images:
        try:
            pipeline.register_cattle(
                cattle_id=cattle_id,
                name=identity.name,
                image_source=str(img_file),
                auto_detect=True,
            )
        except Exception as e:
            print(f"임베딩 재계산 실패: {e}")

    return {"message": "이미지가 삭제되었습니다", "filename": filename}


@router.delete("/{cattle_id}")
async def delete_cattle(
    cattle_id: str,
    pipeline=Depends(get_pipeline)
):
    """소 삭제"""
    if cattle_id not in pipeline.recognizer.gallery:
        raise HTTPException(404, "소를 찾을 수 없습니다")

    # 갤러리에서 삭제
    pipeline.recognizer.remove_identity(cattle_id)

    # 이미지 파일 삭제
    cattle_dir = Path(settings.UPLOAD_DIR) / cattle_id
    if cattle_dir.exists():
        shutil.rmtree(cattle_dir)

    return {"message": "삭제되었습니다", "cattle_id": cattle_id}
