"""
얼굴 인식 API 라우터
"""
import uuid
import base64
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from pydantic import BaseModel
import cv2
import numpy as np

from ..config import settings
from ..dependencies import get_pipeline

router = APIRouter()


class DetectionResult(BaseModel):
    """검출 결과"""
    bbox: List[float]
    confidence: float
    keypoints: Optional[List[List[float]]] = None


class RecognitionResult(BaseModel):
    """인식 결과"""
    cattle_id: Optional[str]
    name: Optional[str]
    confidence: float
    is_new: bool
    bbox: List[float]
    face_image: Optional[str] = None  # base64 인코딩된 얼굴 이미지 (새로운 소일 때)


class RecognizeResponse(BaseModel):
    """인식 응답"""
    detections: List[DetectionResult]
    recognitions: List[RecognitionResult]
    image_url: Optional[str] = None
    visualization: Optional[str] = None  # base64 인코딩된 시각화 이미지


class SearchResult(BaseModel):
    """검색 결과"""
    cattle_id: str
    name: str
    similarity: float


@router.post("/detect")
async def detect_faces(
    image: UploadFile = File(...),
    pipeline=Depends(get_pipeline)
):
    """이미지에서 소 얼굴 검출"""
    # 이미지 읽기
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(400, "이미지를 읽을 수 없습니다")

    # 검출
    detections = pipeline.detector.detect(img)

    results = []
    for det in detections:
        results.append(DetectionResult(
            bbox=det.bbox.tolist(),
            confidence=det.confidence,
            keypoints=det.keypoints.tolist() if det.keypoints is not None else None,
        ))

    return {"detections": results, "count": len(results)}


@router.post("/recognize", response_model=RecognizeResponse)
async def recognize_faces(
    image: UploadFile = File(...),
    save_image: bool = False,
    auto_register: bool = False,
    pipeline=Depends(get_pipeline)
):
    """이미지에서 소 인식

    auto_register=True: 새로운 소 자동 등록
    """
    import datetime

    # 이미지 읽기
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(400, "이미지를 읽을 수 없습니다")

    # 인식 (항상 시각화 생성)
    result = pipeline.process_image(img, visualize=True)

    # 결과 변환
    detections = []
    recognitions = []

    for i, det in enumerate(result.detections):
        detections.append(DetectionResult(
            bbox=det.bbox.tolist(),
            confidence=det.confidence,
            keypoints=det.keypoints.tolist() if det.keypoints is not None else None,
        ))

        if i < len(result.recognition_results):
            rec = result.recognition_results[i]
            face_image_base64 = None

            # 새로운 소일 때 얼굴 이미지 추출
            if rec.is_new and i < len(result.aligned_faces):
                aligned = result.aligned_faces[i]
                _, buffer = cv2.imencode('.jpg', aligned.image)
                face_image_base64 = base64.b64encode(buffer).decode('utf-8')

                # 자동 등록
                if auto_register:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                    new_id = f"auto_{timestamp}_{uuid.uuid4().hex[:6]}"
                    new_name = f"소_{len(pipeline.recognizer.gallery) + 1}"

                    # 얼굴 이미지 저장
                    cattle_dir = Path(settings.UPLOAD_DIR) / new_id
                    cattle_dir.mkdir(parents=True, exist_ok=True)
                    face_path = cattle_dir / f"{uuid.uuid4().hex}.jpg"
                    cv2.imwrite(str(face_path), aligned.image)

                    # 갤러리에 등록
                    pipeline.register_cattle(
                        cattle_id=new_id,
                        name=new_name,
                        image_source=str(face_path),
                        auto_detect=False,  # 이미 정렬된 얼굴
                    )

                    # 결과 업데이트
                    rec.cattle_id = new_id
                    rec.name = new_name
                    rec.is_new = False
                    rec.confidence = 1.0

            recognitions.append(RecognitionResult(
                cattle_id=rec.cattle_id,
                name=rec.name,
                confidence=rec.confidence,
                is_new=rec.is_new,
                bbox=det.bbox.tolist(),
                face_image=face_image_base64,
            ))

    # 시각화 이미지 base64 인코딩
    visualization_base64 = None
    image_url = None
    if result.visualization is not None:
        # base64 인코딩
        _, buffer = cv2.imencode('.jpg', result.visualization)
        visualization_base64 = base64.b64encode(buffer).decode('utf-8')

        # 파일로도 저장 (선택적)
        if save_image:
            filename = f"result_{uuid.uuid4().hex}.jpg"
            save_path = Path(settings.UPLOAD_DIR) / "results" / filename
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), result.visualization)
            image_url = f"/uploads/results/{filename}"

    return RecognizeResponse(
        detections=detections,
        recognitions=recognitions,
        image_url=image_url,
        visualization=visualization_base64,
    )


@router.post("/search", response_model=List[SearchResult])
async def search_similar(
    image: UploadFile = File(...),
    threshold: float = 0.3,
    pipeline=Depends(get_pipeline)
):
    """유사한 소 검색"""
    # 이미지 읽기
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(400, "이미지를 읽을 수 없습니다")

    # 검색
    matches = pipeline.search_cattle(img, threshold=threshold)

    return [
        SearchResult(cattle_id=cid, name=name, similarity=sim)
        for cid, name, sim in matches
    ]


class RegisterNewRequest(BaseModel):
    """새로운 소 등록 요청"""
    face_image: str  # base64 인코딩된 얼굴 이미지
    name: Optional[str] = None
    breed: Optional[str] = None
    age: Optional[int] = None
    notes: Optional[str] = None


@router.post("/register-new")
async def register_new_cattle(
    request: RegisterNewRequest,
    pipeline=Depends(get_pipeline)
):
    """새로운 소 등록 (얼굴 이미지로부터)"""
    import datetime

    # base64 디코딩
    try:
        img_data = base64.b64decode(request.face_image)
        nparr = np.frombuffer(img_data, np.uint8)
        face_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        raise HTTPException(400, f"이미지 디코딩 실패: {e}")

    if face_img is None:
        raise HTTPException(400, "이미지를 읽을 수 없습니다")

    # ID 생성
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    cattle_id = f"cow_{timestamp}_{uuid.uuid4().hex[:6]}"
    name = request.name or f"소_{len(pipeline.recognizer.gallery) + 1}"

    # 얼굴 이미지 저장
    cattle_dir = Path(settings.UPLOAD_DIR) / cattle_id
    cattle_dir.mkdir(parents=True, exist_ok=True)
    face_path = cattle_dir / f"{uuid.uuid4().hex}.jpg"
    cv2.imwrite(str(face_path), face_img)

    # 갤러리에 등록
    pipeline.register_cattle(
        cattle_id=cattle_id,
        name=name,
        image_source=str(face_path),
        auto_detect=False,  # 이미 정렬된 얼굴
    )

    # 메타데이터 저장 (갤러리에 추가 정보)
    if cattle_id in pipeline.recognizer.gallery:
        identity = pipeline.recognizer.gallery[cattle_id]
        identity.metadata = {
            "breed": request.breed,
            "age": request.age,
            "notes": request.notes,
        }
        pipeline.recognizer.save_gallery(settings.GALLERY_FILE)

    return {
        "cattle_id": cattle_id,
        "name": name,
        "image_url": f"/uploads/{cattle_id}/{face_path.name}",
    }


@router.get("/stats")
async def get_stats(
    pipeline=Depends(get_pipeline)
):
    """통계 정보"""
    identities = pipeline.get_registered_cattle()

    total_images = sum(i['num_images'] for i in identities)

    return {
        "total_cattle": len(identities),
        "total_images": total_images,
        "gallery_path": settings.GALLERY_FILE,
    }
