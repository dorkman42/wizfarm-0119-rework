"""
소 얼굴 검출 모듈
YOLOv8 기반 객체 검출 + 키포인트 추출
"""
import cv2
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("ultralytics 패키지가 필요합니다: pip install ultralytics")


@dataclass
class Detection:
    """검출 결과 데이터 클래스"""
    bbox: np.ndarray          # [x1, y1, x2, y2]
    confidence: float         # 검출 신뢰도
    keypoints: Optional[np.ndarray] = None  # [N, 3] (x, y, conf)
    class_id: int = 0         # 클래스 ID
    track_id: Optional[int] = None  # 추적 ID (비디오용)


class CattleFaceDetector:
    """
    YOLOv8 기반 소 얼굴 검출기

    사용법:
        detector = CattleFaceDetector(model_path="yolov8n-pose.pt")
        detections = detector.detect(image)
    """

    def __init__(
        self,
        model_path: str = "yolov8n-pose.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cuda:0",
        image_size: int = 640,
    ):
        """
        Args:
            model_path: YOLOv8 모델 경로 (.pt 파일)
            confidence_threshold: 검출 신뢰도 임계값
            iou_threshold: NMS IOU 임계값
            device: 실행 디바이스 ('cpu' 또는 'cuda:0')
            image_size: 입력 이미지 크기
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.image_size = image_size

        # 모델 로드
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str) -> YOLO:
        """YOLOv8 모델 로드"""
        model = YOLO(model_path)
        return model

    def detect(
        self,
        image: Union[np.ndarray, str, Path],
        return_image: bool = False
    ) -> Union[List[Detection], Tuple[List[Detection], np.ndarray]]:
        """
        단일 이미지에서 소 얼굴 검출

        Args:
            image: 입력 이미지 (numpy array, 파일 경로, 또는 Path 객체)
            return_image: True면 원본 이미지도 함께 반환

        Returns:
            검출 결과 리스트 (Detection 객체들)
        """
        # 이미지 로드
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"이미지를 로드할 수 없습니다: {image}")
        else:
            img = image.copy()

        # YOLOv8 추론
        results = self.model(
            img,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            device=self.device,
            verbose=False
        )

        # 결과 파싱
        detections = self._parse_results(results[0])

        if return_image:
            return detections, img
        return detections

    def detect_batch(
        self,
        images: List[Union[np.ndarray, str, Path]]
    ) -> List[List[Detection]]:
        """
        배치 이미지에서 소 얼굴 검출

        Args:
            images: 입력 이미지 리스트

        Returns:
            각 이미지별 검출 결과 리스트
        """
        all_detections = []
        for img in images:
            detections = self.detect(img)
            all_detections.append(detections)
        return all_detections

    def detect_video(
        self,
        video_path: Union[str, Path],
        output_path: Optional[str] = None,
        track: bool = True,
        show: bool = False,
        max_frames: Optional[int] = None
    ) -> List[Dict]:
        """
        비디오에서 소 얼굴 검출 및 추적

        Args:
            video_path: 입력 비디오 경로
            output_path: 출력 비디오 경로 (None이면 저장 안함)
            track: True면 객체 추적 수행
            show: True면 실시간 시각화
            max_frames: 최대 처리 프레임 수 (None이면 전체)

        Returns:
            프레임별 검출 결과 리스트
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"비디오를 열 수 없습니다: {video_path}")

        # 비디오 정보
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 출력 비디오 설정
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        all_results = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if max_frames and frame_idx >= max_frames:
                break

            # 검출 또는 추적
            if track:
                results = self.model.track(
                    frame,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    imgsz=self.image_size,
                    device=self.device,
                    persist=True,
                    verbose=False
                )
            else:
                results = self.model(
                    frame,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    imgsz=self.image_size,
                    device=self.device,
                    verbose=False
                )

            detections = self._parse_results(results[0], include_track_id=track)

            all_results.append({
                'frame_idx': frame_idx,
                'detections': detections
            })

            # 시각화
            if show or writer:
                vis_frame = self._visualize_frame(frame, detections)
                if show:
                    cv2.imshow('Cattle Face Detection', vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                if writer:
                    writer.write(vis_frame)

            frame_idx += 1

        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()

        return all_results

    def _parse_results(
        self,
        result,
        include_track_id: bool = False
    ) -> List[Detection]:
        """YOLOv8 결과를 Detection 객체로 변환"""
        detections = []

        if result.boxes is None:
            return detections

        boxes = result.boxes
        keypoints = result.keypoints if hasattr(result, 'keypoints') and result.keypoints is not None else None

        for i in range(len(boxes)):
            # 바운딩 박스
            bbox = boxes.xyxy[i].cpu().numpy()
            confidence = float(boxes.conf[i].cpu().numpy())
            class_id = int(boxes.cls[i].cpu().numpy())

            # 추적 ID
            track_id = None
            if include_track_id and boxes.id is not None:
                track_id = int(boxes.id[i].cpu().numpy())

            # 키포인트
            kpts = None
            if keypoints is not None:
                kpts = keypoints.data[i].cpu().numpy()  # [num_kpts, 3] (x, y, conf)

            detection = Detection(
                bbox=bbox,
                confidence=confidence,
                keypoints=kpts,
                class_id=class_id,
                track_id=track_id
            )
            detections.append(detection)

        return detections

    def _visualize_frame(
        self,
        frame: np.ndarray,
        detections: List[Detection]
    ) -> np.ndarray:
        """프레임에 검출 결과 시각화"""
        vis = frame.copy()

        for det in detections:
            # 바운딩 박스
            x1, y1, x2, y2 = det.bbox.astype(int)
            color = (0, 255, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # 레이블
            label = f"Cow {det.track_id or ''} ({det.confidence:.2f})"
            cv2.putText(vis, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 키포인트
            if det.keypoints is not None:
                for kpt in det.keypoints:
                    x, y, conf = kpt
                    if conf > 0.5:
                        cv2.circle(vis, (int(x), int(y)), 3, (255, 0, 0), -1)

        return vis

    def export_onnx(self, output_path: str = "cattle_detector.onnx"):
        """모델을 ONNX 형식으로 내보내기"""
        self.model.export(format="onnx", imgsz=self.image_size)

    @staticmethod
    def download_pretrained(model_name: str = "yolov8n-pose.pt") -> str:
        """
        사전학습 모델 다운로드

        Args:
            model_name: 모델 이름
                - yolov8n-pose.pt: 가장 작은 모델 (빠름)
                - yolov8s-pose.pt: 작은 모델
                - yolov8m-pose.pt: 중간 모델
                - yolov8l-pose.pt: 큰 모델
                - yolov8x-pose.pt: 가장 큰 모델 (정확함)

        Returns:
            모델 경로
        """
        model = YOLO(model_name)
        return model_name
