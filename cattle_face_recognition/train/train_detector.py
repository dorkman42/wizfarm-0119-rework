"""
소 얼굴 검출기 학습 스크립트
YOLOv8-pose 모델 fine-tuning
"""
import argparse
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("ultralytics 패키지가 필요합니다: pip install ultralytics")
    exit(1)


def train_detector(
    data_yaml: str,
    model_name: str = "yolov8n-pose.pt",
    epochs: int = 100,
    batch_size: int = 16,
    image_size: int = 640,
    device: str = "0",
    project: str = "runs/train",
    name: str = "cattle_face_detector",
):
    """
    소 얼굴 검출기 학습

    Args:
        data_yaml: 데이터셋 설정 파일 경로
        model_name: 기본 모델
        epochs: 학습 에폭 수
        batch_size: 배치 크기
        image_size: 입력 이미지 크기
        device: GPU 디바이스 번호
        project: 결과 저장 디렉토리
        name: 실험 이름
    """
    print("=" * 50)
    print("소 얼굴 검출기 학습")
    print("=" * 50)
    print(f"데이터셋: {data_yaml}")
    print(f"기본 모델: {model_name}")
    print(f"에폭: {epochs}")
    print(f"배치 크기: {batch_size}")
    print(f"이미지 크기: {image_size}")
    print("=" * 50)

    # 모델 로드
    model = YOLO(model_name)

    # 학습
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=image_size,
        device=device,
        project=project,
        name=name,
        # 추가 설정
        patience=20,           # Early stopping patience
        save=True,             # 체크포인트 저장
        save_period=10,        # 10 에폭마다 저장
        val=True,              # 검증 수행
        plots=True,            # 학습 그래프 생성
        # 데이터 증강
        hsv_h=0.015,           # 색조 변환
        hsv_s=0.7,             # 채도 변환
        hsv_v=0.4,             # 밝기 변환
        degrees=10.0,          # 회전
        translate=0.1,         # 이동
        scale=0.5,             # 스케일
        shear=2.0,             # 전단
        flipud=0.0,            # 상하 반전 (소 얼굴에는 비추천)
        fliplr=0.5,            # 좌우 반전
        mosaic=0.8,            # 모자이크 증강
        mixup=0.1,             # MixUp 증강
    )

    print("=" * 50)
    print("학습 완료!")
    print(f"최고 모델: {project}/{name}/weights/best.pt")
    print(f"마지막 모델: {project}/{name}/weights/last.pt")
    print("=" * 50)

    return results


def validate_detector(
    model_path: str,
    data_yaml: str,
    batch_size: int = 16,
    image_size: int = 640,
    device: str = "0",
):
    """
    검출기 검증

    Args:
        model_path: 모델 경로
        data_yaml: 데이터셋 설정 파일
        batch_size: 배치 크기
        image_size: 이미지 크기
        device: GPU 디바이스
    """
    print("=" * 50)
    print("소 얼굴 검출기 검증")
    print("=" * 50)

    model = YOLO(model_path)

    results = model.val(
        data=data_yaml,
        batch=batch_size,
        imgsz=image_size,
        device=device,
    )

    print("검증 완료!")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")

    return results


def export_model(
    model_path: str,
    format: str = "onnx",
    image_size: int = 640,
):
    """
    모델 내보내기

    Args:
        model_path: 모델 경로
        format: 출력 형식 (onnx, torchscript, coreml 등)
        image_size: 이미지 크기
    """
    print(f"모델 내보내기: {format}")

    model = YOLO(model_path)
    model.export(format=format, imgsz=image_size)

    print("내보내기 완료!")


def main():
    parser = argparse.ArgumentParser(description='소 얼굴 검출기 학습')
    subparsers = parser.add_subparsers(dest='command', help='명령어')

    # 학습 명령
    train_parser = subparsers.add_parser('train', help='모델 학습')
    train_parser.add_argument('--data', type=str, required=True, help='데이터셋 YAML 파일')
    train_parser.add_argument('--model', type=str, default='yolov8n-pose.pt', help='기본 모델')
    train_parser.add_argument('--epochs', type=int, default=100, help='에폭 수')
    train_parser.add_argument('--batch', type=int, default=16, help='배치 크기')
    train_parser.add_argument('--imgsz', type=int, default=640, help='이미지 크기')
    train_parser.add_argument('--device', type=str, default='0', help='GPU 디바이스')
    train_parser.add_argument('--name', type=str, default='cattle_face', help='실험 이름')

    # 검증 명령
    val_parser = subparsers.add_parser('val', help='모델 검증')
    val_parser.add_argument('--model', type=str, required=True, help='모델 경로')
    val_parser.add_argument('--data', type=str, required=True, help='데이터셋 YAML 파일')
    val_parser.add_argument('--batch', type=int, default=16, help='배치 크기')
    val_parser.add_argument('--device', type=str, default='0', help='GPU 디바이스')

    # 내보내기 명령
    export_parser = subparsers.add_parser('export', help='모델 내보내기')
    export_parser.add_argument('--model', type=str, required=True, help='모델 경로')
    export_parser.add_argument('--format', type=str, default='onnx', help='출력 형식')

    args = parser.parse_args()

    if args.command == 'train':
        train_detector(
            data_yaml=args.data,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
            image_size=args.imgsz,
            device=args.device,
            name=args.name,
        )
    elif args.command == 'val':
        validate_detector(
            model_path=args.model,
            data_yaml=args.data,
            batch_size=args.batch,
            device=args.device,
        )
    elif args.command == 'export':
        export_model(
            model_path=args.model,
            format=args.format,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
