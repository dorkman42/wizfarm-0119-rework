"""
소 얼굴 인식 모듈
특징 추출, 갤러리 관리, 개체 식별

상업적 사용을 위해 자체 학습 모델 (ResNet + ArcFace) 지원
"""
import cv2
import numpy as np
import pickle
import json
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models


def get_device() -> str:
    """사용 가능한 디바이스 자동 감지"""
    if torch.cuda.is_available():
        return "cuda:0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class CattleFaceEmbedder(nn.Module):
    """
    소 얼굴 특징 추출 모델 (추론용)
    ResNet 백본 + Embedding Layer
    """
    def __init__(
        self,
        backbone: str = "resnet50",
        embedding_size: int = 512,
        pretrained: bool = False,
    ):
        super().__init__()

        # 백본 선택
        if backbone == "resnet18":
            base = models.resnet18(weights=None)
            in_features = 512
        elif backbone == "resnet34":
            base = models.resnet34(weights=None)
            in_features = 512
        elif backbone == "resnet50":
            base = models.resnet50(weights=None)
            in_features = 2048
        elif backbone == "efficientnet_b0":
            base = models.efficientnet_b0(weights=None)
            in_features = 1280
        else:
            raise ValueError(f"지원하지 않는 백본: {backbone}")

        # 마지막 FC 레이어 제거
        self.backbone = nn.Sequential(*list(base.children())[:-1])

        # Embedding layer
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, embedding_size),
            nn.BatchNorm1d(embedding_size),
        )

        self.embedding_size = embedding_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        embeddings = self.embedding(features)
        return embeddings

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        """추론용: 정규화된 임베딩 반환"""
        embeddings = self.forward(x)
        return F.normalize(embeddings, p=2, dim=1)


@dataclass
class CattleIdentity:
    """소 개체 정보"""
    cattle_id: str                          # 고유 ID
    name: str                               # 이름/별명
    embeddings: List[np.ndarray] = field(default_factory=list)  # 특징 벡터들
    images: List[str] = field(default_factory=list)             # 등록된 이미지 경로들
    metadata: Dict = field(default_factory=dict)                # 추가 정보
    registered_at: str = ""                 # 등록 시간
    updated_at: str = ""                    # 마지막 업데이트 시간

    def __post_init__(self):
        now = datetime.now().isoformat()
        if not self.registered_at:
            self.registered_at = now
        self.updated_at = now

    def get_mean_embedding(self) -> Optional[np.ndarray]:
        """평균 특징 벡터 반환"""
        if not self.embeddings:
            return None
        return np.mean(self.embeddings, axis=0)


@dataclass
class RecognitionResult:
    """인식 결과"""
    cattle_id: Optional[str]     # 인식된 개체 ID (새 개체면 None)
    name: Optional[str]          # 이름
    confidence: float            # 인식 신뢰도
    embedding: np.ndarray        # 특징 벡터
    is_new: bool = False         # 새 개체 여부


class FaceRecognizer:
    """
    소 얼굴 인식기

    특징 추출, 갤러리 관리, 개체 식별을 담당합니다.
    새로운 소를 등록하고 기존 소를 인식할 수 있습니다.

    사용법:
        recognizer = FaceRecognizer()
        # 새 소 등록
        recognizer.register("cow_001", "Daisy", face_image)
        # 인식
        result = recognizer.recognize(face_image)
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        embedding_size: int = 512,
        similarity_threshold: float = 0.4,
        device: str = "auto",
        gallery_path: Optional[str] = None,
        custom_model_path: Optional[str] = None,
    ):
        """
        Args:
            model_name: InsightFace 모델 이름 (buffalo_l, buffalo_s 등) - 비상업용
            embedding_size: 특징 벡터 차원
            similarity_threshold: 동일 개체 판단 임계값
            device: 실행 디바이스 (auto, cuda, mps, cpu)
            gallery_path: 갤러리 저장 경로
            custom_model_path: 자체 학습 모델 경로 (.pt 파일) - 상업용
        """
        self.model_name = model_name
        self.embedding_size = embedding_size
        self.similarity_threshold = similarity_threshold
        self.custom_model_path = custom_model_path
        self.gallery_path = gallery_path

        # 디바이스 설정
        if device == "auto":
            self.device = get_device()
        else:
            self.device = device
        self.torch_device = torch.device(self.device)

        # 이미지 전처리 (자체 모델용)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 갤러리 (등록된 소들)
        self.gallery: Dict[str, CattleIdentity] = {}

        # 특징 추출 모델 로드
        self.model = self._load_model()
        self.use_custom_model = custom_model_path is not None and self.model is not None

        # 갤러리 로드
        if gallery_path and Path(gallery_path).exists():
            self.load_gallery(gallery_path)

    def _load_model(self):
        """모델 로드 (자체 학습 모델 우선, 없으면 InsightFace)"""
        # 자체 학습 모델이 지정된 경우
        if self.custom_model_path:
            return self._load_custom_model(self.custom_model_path)

        # InsightFace 모델 시도 (비상업용)
        try:
            from insightface.app import FaceAnalysis

            app = FaceAnalysis(
                name=self.model_name,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            app.prepare(ctx_id=0 if 'cuda' in self.device else -1)
            print(f"InsightFace 모델 로드됨: {self.model_name}")
            return app
        except ImportError:
            print("Warning: InsightFace를 사용할 수 없습니다. 간단한 특징 추출기를 사용합니다.")
            return None

    def _load_custom_model(self, model_path: str):
        """자체 학습 모델 로드 (상업용)"""
        model_path = Path(model_path)
        if not model_path.exists():
            print(f"Warning: 모델 파일이 없습니다: {model_path}")
            return None

        try:
            checkpoint = torch.load(model_path, map_location=self.torch_device, weights_only=False)

            # 모델 설정 추출
            backbone = checkpoint.get('backbone', 'resnet50')
            embedding_size = checkpoint.get('embedding_size', 512)
            self.embedding_size = embedding_size

            # 모델 생성 및 가중치 로드
            model = CattleFaceEmbedder(
                backbone=backbone,
                embedding_size=embedding_size,
                pretrained=False,
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.torch_device)
            model.eval()

            print(f"자체 학습 모델 로드됨: {model_path}")
            print(f"  - 백본: {backbone}")
            print(f"  - 임베딩 크기: {embedding_size}")
            print(f"  - 디바이스: {self.torch_device}")

            return model
        except Exception as e:
            print(f"Warning: 모델 로드 실패: {e}")
            return None

    def extract_embedding(
        self,
        face_image: np.ndarray,
    ) -> np.ndarray:
        """
        얼굴 이미지에서 특징 벡터 추출

        Args:
            face_image: 정렬된 얼굴 이미지 (BGR)

        Returns:
            특징 벡터 (512차원)
        """
        if self.model is not None:
            # 자체 학습 모델 사용 (상업용)
            if self.use_custom_model:
                return self._extract_with_custom_model(face_image)

            # InsightFace 사용 (비상업용)
            faces = self.model.get(face_image)
            if faces:
                return faces[0].embedding
            # 얼굴을 찾지 못하면 직접 특징 추출
            return self._simple_embedding(face_image)
        else:
            return self._simple_embedding(face_image)

    def _extract_with_custom_model(self, face_image: np.ndarray) -> np.ndarray:
        """자체 학습 모델로 특징 추출"""
        # BGR -> RGB 변환
        rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        # 전처리
        input_tensor = self.transform(rgb_image).unsqueeze(0).to(self.torch_device)

        # 추론
        with torch.no_grad():
            embedding = self.model.extract(input_tensor)

        return embedding.cpu().numpy().flatten()

    def _simple_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        간단한 특징 추출 (InsightFace 없을 때 대체용)
        실제 프로덕션에서는 적합한 모델로 교체 필요
        """
        # 이미지를 고정 크기로 리사이즈
        resized = cv2.resize(face_image, (112, 112))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # 히스토그램 특징
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / (hist.sum() + 1e-6)

        # HOG 특징 (간단 버전)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy)
        hog_hist = np.histogram(ang.flatten(), bins=256, range=(0, 2*np.pi), weights=mag.flatten())[0]
        hog_hist = hog_hist / (hog_hist.sum() + 1e-6)

        # 결합하여 512차원으로 맞춤
        embedding = np.concatenate([hist, hog_hist])
        embedding = np.resize(embedding, self.embedding_size)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-6)

        return embedding

    def register(
        self,
        cattle_id: str,
        name: str,
        face_images: Union[np.ndarray, List[np.ndarray]],
        metadata: Optional[Dict] = None,
        image_paths: Optional[List[str]] = None,
    ) -> CattleIdentity:
        """
        새로운 소 등록

        Args:
            cattle_id: 고유 ID
            name: 이름/별명
            face_images: 얼굴 이미지 또는 이미지 리스트
            metadata: 추가 정보 (나이, 품종 등)
            image_paths: 이미지 파일 경로들

        Returns:
            등록된 CattleIdentity 객체
        """
        if isinstance(face_images, np.ndarray):
            face_images = [face_images]

        # 특징 벡터 추출
        embeddings = [self.extract_embedding(img) for img in face_images]

        # 기존 개체가 있으면 업데이트
        if cattle_id in self.gallery:
            identity = self.gallery[cattle_id]
            identity.embeddings.extend(embeddings)
            identity.name = name
            if metadata:
                identity.metadata.update(metadata)
            if image_paths:
                identity.images.extend(image_paths)
            identity.updated_at = datetime.now().isoformat()
        else:
            # 새 개체 생성
            identity = CattleIdentity(
                cattle_id=cattle_id,
                name=name,
                embeddings=embeddings,
                images=image_paths or [],
                metadata=metadata or {},
            )
            self.gallery[cattle_id] = identity

        # 자동 저장
        if self.gallery_path:
            self.save_gallery(self.gallery_path)

        return identity

    def register_from_image(
        self,
        cattle_id: str,
        name: str,
        image_path: Union[str, Path],
        metadata: Optional[Dict] = None,
    ) -> CattleIdentity:
        """
        이미지 파일에서 소 등록

        Args:
            cattle_id: 고유 ID
            name: 이름
            image_path: 이미지 파일 경로
            metadata: 추가 정보

        Returns:
            등록된 CattleIdentity 객체
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")

        return self.register(
            cattle_id=cattle_id,
            name=name,
            face_images=image,
            metadata=metadata,
            image_paths=[str(image_path)]
        )

    def recognize(
        self,
        face_image: np.ndarray,
        top_k: int = 1,
    ) -> List[RecognitionResult]:
        """
        얼굴 인식

        Args:
            face_image: 얼굴 이미지
            top_k: 상위 k개 결과 반환

        Returns:
            인식 결과 리스트
        """
        # 특징 추출
        embedding = self.extract_embedding(face_image)

        if not self.gallery:
            return [RecognitionResult(
                cattle_id=None,
                name=None,
                confidence=0.0,
                embedding=embedding,
                is_new=True
            )]

        # 갤러리와 비교
        similarities = []
        for cattle_id, identity in self.gallery.items():
            mean_emb = identity.get_mean_embedding()
            if mean_emb is not None:
                sim = self._cosine_similarity(embedding, mean_emb)
                similarities.append((cattle_id, identity.name, sim))

        # 정렬
        similarities.sort(key=lambda x: x[2], reverse=True)

        results = []
        for cattle_id, name, sim in similarities[:top_k]:
            is_new = sim < self.similarity_threshold
            results.append(RecognitionResult(
                cattle_id=cattle_id if not is_new else None,
                name=name if not is_new else None,
                confidence=sim,
                embedding=embedding,
                is_new=is_new
            ))

        return results

    def recognize_batch(
        self,
        face_images: List[np.ndarray],
    ) -> List[RecognitionResult]:
        """여러 얼굴 일괄 인식"""
        return [self.recognize(img)[0] for img in face_images]

    def update_identity(
        self,
        cattle_id: str,
        face_image: np.ndarray,
        image_path: Optional[str] = None,
    ) -> bool:
        """
        기존 개체에 새 이미지 추가

        Args:
            cattle_id: 개체 ID
            face_image: 추가할 얼굴 이미지
            image_path: 이미지 파일 경로

        Returns:
            성공 여부
        """
        if cattle_id not in self.gallery:
            return False

        embedding = self.extract_embedding(face_image)
        self.gallery[cattle_id].embeddings.append(embedding)
        if image_path:
            self.gallery[cattle_id].images.append(image_path)
        self.gallery[cattle_id].updated_at = datetime.now().isoformat()

        if self.gallery_path:
            self.save_gallery(self.gallery_path)

        return True

    def remove_identity(self, cattle_id: str) -> bool:
        """개체 삭제"""
        if cattle_id in self.gallery:
            del self.gallery[cattle_id]
            if self.gallery_path:
                self.save_gallery(self.gallery_path)
            return True
        return False

    def search_similar(
        self,
        face_image: np.ndarray,
        threshold: Optional[float] = None,
    ) -> List[Tuple[str, str, float]]:
        """
        유사한 개체 검색

        Args:
            face_image: 검색할 얼굴 이미지
            threshold: 유사도 임계값 (None이면 기본값)

        Returns:
            (cattle_id, name, similarity) 튜플 리스트
        """
        threshold = threshold or self.similarity_threshold
        embedding = self.extract_embedding(face_image)

        results = []
        for cattle_id, identity in self.gallery.items():
            mean_emb = identity.get_mean_embedding()
            if mean_emb is not None:
                sim = self._cosine_similarity(embedding, mean_emb)
                if sim >= threshold:
                    results.append((cattle_id, identity.name, sim))

        results.sort(key=lambda x: x[2], reverse=True)
        return results

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """코사인 유사도 계산"""
        a_norm = a / (np.linalg.norm(a) + 1e-6)
        b_norm = b / (np.linalg.norm(b) + 1e-6)
        return float(np.dot(a_norm, b_norm))

    def save_gallery(self, path: Union[str, Path]):
        """
        갤러리 저장

        Args:
            path: 저장 경로 (.pkl 또는 .json)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # 직렬화 가능한 형태로 변환
        data = {}
        for cattle_id, identity in self.gallery.items():
            data[cattle_id] = {
                'cattle_id': identity.cattle_id,
                'name': identity.name,
                'embeddings': [emb.tolist() for emb in identity.embeddings],
                'images': identity.images,
                'metadata': identity.metadata,
                'registered_at': identity.registered_at,
                'updated_at': identity.updated_at,
            }

        if path.suffix == '.json':
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            with open(path, 'wb') as f:
                pickle.dump(data, f)

        print(f"갤러리 저장됨: {path} ({len(self.gallery)}개 개체)")

    def load_gallery(self, path: Union[str, Path]):
        """
        갤러리 로드

        Args:
            path: 로드 경로
        """
        path = Path(path)
        if not path.exists():
            print(f"갤러리 파일이 없습니다: {path}")
            return

        if path.suffix == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            with open(path, 'rb') as f:
                data = pickle.load(f)

        self.gallery = {}
        for cattle_id, info in data.items():
            self.gallery[cattle_id] = CattleIdentity(
                cattle_id=info['cattle_id'],
                name=info['name'],
                embeddings=[np.array(emb) for emb in info['embeddings']],
                images=info.get('images', []),
                metadata=info.get('metadata', {}),
                registered_at=info.get('registered_at', ''),
                updated_at=info.get('updated_at', ''),
            )

        print(f"갤러리 로드됨: {path} ({len(self.gallery)}개 개체)")

    def get_all_identities(self) -> List[Dict]:
        """모든 등록된 개체 정보 반환"""
        return [
            {
                'cattle_id': identity.cattle_id,
                'name': identity.name,
                'num_images': len(identity.embeddings),
                'metadata': identity.metadata,
                'registered_at': identity.registered_at,
            }
            for identity in self.gallery.values()
        ]

    def generate_id(self) -> str:
        """새 개체 ID 생성"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_part = hashlib.md5(str(np.random.random()).encode()).hexdigest()[:6]
        return f"cow_{timestamp}_{random_part}"
