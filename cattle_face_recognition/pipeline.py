"""
소 얼굴 인식 전체 파이프라인
검출 → 정렬 → 인식 통합
"""
import cv2
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

from .config import PipelineConfig
from .detection import CattleFaceDetector
from .detection.detector import Detection
from .alignment import FaceAligner
from .alignment.aligner import AlignedFace, PoseType
from .recognition import FaceRecognizer
from .recognition.recognizer import RecognitionResult, CattleIdentity
from .utils.visualization import draw_detections, save_visualization, create_gallery_grid
from .utils.helpers import crop_face, match_detections_by_iou


@dataclass
class PipelineResult:
    """파이프라인 처리 결과"""
    detections: List[Detection]
    aligned_faces: List[AlignedFace]
    recognition_results: List[RecognitionResult]
    visualization: Optional[np.ndarray] = None
    frame_idx: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class CattleFaceRecognitionPipeline:
    """
    소 얼굴 인식 전체 파이프라인

    사용법:
        # 초기화
        pipeline = CattleFaceRecognitionPipeline()

        # 새 소 등록
        pipeline.register_cattle("cow_001", "Daisy", "path/to/image.jpg")

        # 이미지에서 인식
        results = pipeline.process_image("path/to/test.jpg")

        # 비디오 처리
        pipeline.process_video("path/to/video.mp4", output_path="output.mp4")
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        detection_model: str = "yolov8n-pose.pt",
        gallery_path: Optional[str] = None,
    ):
        """
        Args:
            config: 파이프라인 설정
            detection_model: 검출 모델 경로
            gallery_path: 갤러리 저장 경로
        """
        self.config = config or PipelineConfig()

        # 출력 디렉토리 생성
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        # 모듈 초기화
        self.detector = CattleFaceDetector(
            model_path=detection_model,
            confidence_threshold=self.config.detection.confidence_threshold,
            iou_threshold=self.config.detection.iou_threshold,
            device=self.config.detection.device,
            image_size=self.config.detection.image_size,
        )

        self.aligner = FaceAligner(
            output_size=self.config.alignment.output_size,
            num_keypoints=self.config.alignment.num_keypoints,
            pose_threshold=self.config.alignment.pose_threshold,
        )

        self.recognizer = FaceRecognizer(
            model_name=self.config.recognition.model_name,
            embedding_size=self.config.recognition.embedding_size,
            similarity_threshold=self.config.recognition.similarity_threshold,
            device=self.config.recognition.device,
            gallery_path=gallery_path,
            custom_model_path=self.config.recognition.custom_model_path,
        )

        # 추적 상태 (비디오용)
        self._prev_detections: List[Detection] = []
        self._track_id_counter = 0

    def process_image(
        self,
        image: Union[np.ndarray, str, Path],
        visualize: bool = True,
        save_crops: bool = False,
    ) -> PipelineResult:
        """
        단일 이미지 처리

        Args:
            image: 입력 이미지 또는 경로
            visualize: 시각화 이미지 생성 여부
            save_crops: 크롭된 얼굴 저장 여부

        Returns:
            PipelineResult 객체
        """
        # 이미지 로드
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"이미지를 로드할 수 없습니다: {image}")
        else:
            img = image.copy()

        # 1. 검출
        detections = self.detector.detect(img)

        # 2. 정렬 및 인식
        aligned_faces = []
        recognition_results = []

        # 검출된 얼굴이 없으면 전체 이미지 사용 (fallback)
        if len(detections) == 0:
            # 전체 이미지를 얼굴로 간주
            h, w = img.shape[:2]
            resized = cv2.resize(img, self.config.alignment.output_size)

            # 가상의 detection 생성
            from .detection.detector import Detection
            fake_det = Detection(
                bbox=np.array([0, 0, w, h], dtype=np.float32),
                confidence=0.0,
                keypoints=None,
                class_id=0,
                track_id=None
            )
            detections = [fake_det]

            aligned = AlignedFace(
                image=resized,
                keypoints=np.zeros((5, 2)),
                pose=PoseType.UNKNOWN,
                pose_angle=0.0,
                quality_score=0.5,
                original_bbox=fake_det.bbox,
                transform_matrix=np.eye(2, 3, dtype=np.float32)
            )
            aligned_faces.append(aligned)

            # 인식
            rec_result = self.recognizer.recognize(aligned.image)[0]
            recognition_results.append(rec_result)

        for det in detections[len(aligned_faces):]:
            # 키포인트가 있으면 정렬, 없으면 크롭
            if det.keypoints is not None and len(det.keypoints) > 0:
                aligned = self.aligner.align(img, det.keypoints, det.bbox)
            else:
                # 크롭만 수행
                face_crop = crop_face(img, det.bbox, margin=0.2, output_size=self.config.alignment.output_size)
                aligned = AlignedFace(
                    image=face_crop,
                    keypoints=np.zeros((5, 2)),
                    pose=PoseType.UNKNOWN,
                    pose_angle=0.0,
                    quality_score=0.5,
                    original_bbox=det.bbox,
                    transform_matrix=np.eye(2, 3, dtype=np.float32)
                )

            aligned_faces.append(aligned)

            # 인식
            rec_result = self.recognizer.recognize(aligned.image)[0]
            recognition_results.append(rec_result)

            # 크롭 저장
            if save_crops:
                self._save_crop(aligned.image, det, rec_result)

        # 시각화
        vis = None
        if visualize:
            vis = draw_detections(
                img, detections,
                show_keypoints=True,
                show_skeleton=True,
                recognition_results=recognition_results
            )

        return PipelineResult(
            detections=detections,
            aligned_faces=aligned_faces,
            recognition_results=recognition_results,
            visualization=vis,
        )

    def process_video(
        self,
        video_path: Union[str, Path],
        output_path: Optional[str] = None,
        track: bool = True,
        show: bool = False,
        max_frames: Optional[int] = None,
        callback: Optional[callable] = None,
    ) -> List[PipelineResult]:
        """
        비디오 처리

        Args:
            video_path: 입력 비디오 경로
            output_path: 출력 비디오 경로
            track: 추적 수행 여부
            show: 실시간 표시 여부
            max_frames: 최대 프레임 수
            callback: 프레임별 콜백 함수 (result를 인자로 받음)

        Returns:
            프레임별 PipelineResult 리스트
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"비디오를 열 수 없습니다: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        all_results = []
        frame_idx = 0
        self._prev_detections = []
        self._track_id_counter = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if max_frames and frame_idx >= max_frames:
                break

            # 프레임 처리
            result = self.process_image(frame, visualize=True, save_crops=False)
            result.frame_idx = frame_idx

            # 추적 (IOU 기반)
            if track and self._prev_detections:
                self._assign_track_ids(result.detections)
            else:
                for det in result.detections:
                    det.track_id = self._track_id_counter
                    self._track_id_counter += 1

            self._prev_detections = result.detections.copy()

            # 시각화 업데이트 (추적 ID 반영)
            if result.visualization is not None:
                result.visualization = draw_detections(
                    frame, result.detections,
                    recognition_results=result.recognition_results
                )

            all_results.append(result)

            # 콜백
            if callback:
                callback(result)

            # 표시/저장
            if result.visualization is not None:
                if show:
                    cv2.imshow('Cattle Face Recognition', result.visualization)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                if writer:
                    writer.write(result.visualization)

            frame_idx += 1

        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()

        return all_results

    def register_cattle(
        self,
        cattle_id: str,
        name: str,
        image_source: Union[np.ndarray, str, Path, List],
        metadata: Optional[Dict] = None,
        auto_detect: bool = True,
    ) -> CattleIdentity:
        """
        새로운 소 등록

        Args:
            cattle_id: 고유 ID
            name: 이름/별명
            image_source: 이미지, 경로, 또는 리스트
            metadata: 추가 정보 (나이, 품종 등)
            auto_detect: True면 이미지에서 얼굴 자동 검출

        Returns:
            등록된 CattleIdentity 객체
        """
        # 이미지 리스트로 변환
        if isinstance(image_source, (np.ndarray, str, Path)):
            image_source = [image_source]

        face_images = []
        image_paths = []

        for src in image_source:
            # 이미지 로드
            if isinstance(src, (str, Path)):
                img = cv2.imread(str(src))
                image_paths.append(str(src))
            else:
                img = src

            if img is None:
                continue

            if auto_detect:
                # 얼굴 검출 및 정렬
                try:
                    detections = self.detector.detect(img)
                except Exception as e:
                    print(f"검출 오류: {e}")
                    detections = []

                if detections:
                    # 가장 큰 얼굴 선택
                    det = max(detections, key=lambda d: (d.bbox[2]-d.bbox[0]) * (d.bbox[3]-d.bbox[1]))

                    if det.keypoints is not None:
                        aligned = self.aligner.align(img, det.keypoints, det.bbox)
                        face_images.append(aligned.image)
                    else:
                        face_crop = crop_face(img, det.bbox, output_size=self.config.alignment.output_size)
                        face_images.append(face_crop)
                else:
                    # 검출 실패 시 이미지 전체를 얼굴로 사용 (fallback)
                    print(f"검출된 얼굴 없음 - 이미지 전체 사용")
                    resized = cv2.resize(img, self.config.alignment.output_size)
                    face_images.append(resized)
            else:
                # 이미지 전체를 얼굴로 사용
                resized = cv2.resize(img, self.config.alignment.output_size)
                face_images.append(resized)

        if not face_images:
            raise ValueError("등록할 얼굴 이미지가 없습니다.")

        return self.recognizer.register(
            cattle_id=cattle_id,
            name=name,
            face_images=face_images,
            metadata=metadata,
            image_paths=image_paths
        )

    def register_from_folder(
        self,
        folder_path: Union[str, Path],
        metadata: Optional[Dict] = None,
    ) -> List[CattleIdentity]:
        """
        폴더에서 소 등록 (폴더명 = 소 이름)

        Args:
            folder_path: 폴더 경로 (하위 폴더마다 한 개체)
            metadata: 공통 메타데이터

        Returns:
            등록된 CattleIdentity 리스트
        """
        folder = Path(folder_path)
        identities = []

        for sub_folder in folder.iterdir():
            if not sub_folder.is_dir():
                continue

            name = sub_folder.name
            cattle_id = self.recognizer.generate_id()

            # 이미지 파일들
            images = list(sub_folder.glob("*.jpg")) + list(sub_folder.glob("*.png"))

            if images:
                identity = self.register_cattle(
                    cattle_id=cattle_id,
                    name=name,
                    image_source=images,
                    metadata=metadata
                )
                identities.append(identity)
                print(f"등록 완료: {name} ({len(images)}개 이미지)")

        return identities

    def search_cattle(
        self,
        image: Union[np.ndarray, str, Path],
        threshold: Optional[float] = None,
    ) -> List[Tuple[str, str, float]]:
        """
        이미지에서 유사한 소 검색

        Args:
            image: 검색할 이미지
            threshold: 유사도 임계값

        Returns:
            (cattle_id, name, similarity) 리스트
        """
        result = self.process_image(image, visualize=False)

        if not result.aligned_faces:
            return []

        # 가장 큰 얼굴로 검색
        best_face = max(result.aligned_faces, key=lambda f: f.quality_score)
        return self.recognizer.search_similar(best_face.image, threshold)

    def get_registered_cattle(self) -> List[Dict]:
        """등록된 모든 소 정보 반환"""
        return self.recognizer.get_all_identities()

    def save_gallery(self, path: Optional[str] = None):
        """갤러리 저장"""
        save_path = path or f"{self.config.output_dir}/gallery.pkl"
        self.recognizer.save_gallery(save_path)

    def load_gallery(self, path: str):
        """갤러리 로드"""
        self.recognizer.load_gallery(path)

    def create_gallery_visualization(self, output_path: Optional[str] = None) -> np.ndarray:
        """등록된 소들의 갤러리 이미지 생성"""
        images = []
        names = []

        for cattle_id, identity in self.recognizer.gallery.items():
            if identity.images:
                # 첫 번째 등록 이미지 사용
                img = cv2.imread(identity.images[0])
                if img is not None:
                    img = cv2.resize(img, (112, 112))
                    images.append(img)
                    names.append(identity.name)

        if not images:
            return np.zeros((100, 100, 3), dtype=np.uint8)

        grid = create_gallery_grid(images, names)

        if output_path:
            save_visualization(grid, output_path)

        return grid

    def _assign_track_ids(self, detections: List[Detection]):
        """IOU 기반 추적 ID 할당"""
        if not self._prev_detections:
            for det in detections:
                det.track_id = self._track_id_counter
                self._track_id_counter += 1
            return

        matches = match_detections_by_iou(
            self._prev_detections,
            detections,
            self.config.iou_tracking_threshold
        )

        matched_curr = set()
        for prev_idx, curr_idx in matches:
            detections[curr_idx].track_id = self._prev_detections[prev_idx].track_id
            matched_curr.add(curr_idx)

        for i, det in enumerate(detections):
            if i not in matched_curr:
                det.track_id = self._track_id_counter
                self._track_id_counter += 1

    def _save_crop(self, face_image: np.ndarray, detection: Detection, rec_result: RecognitionResult):
        """크롭된 얼굴 저장"""
        name = rec_result.name or f"unknown_{self._track_id_counter}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{name}_{timestamp}.jpg"
        output_path = Path(self.config.output_dir) / "crops" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), face_image)
