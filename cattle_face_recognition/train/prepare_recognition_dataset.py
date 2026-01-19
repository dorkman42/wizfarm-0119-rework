"""
소 얼굴 인식 모델용 데이터셋 준비 스크립트

검출된 소 얼굴 이미지를 개체별(ID별) 폴더로 구성하여
ResNet + ArcFace 학습용 데이터셋을 생성합니다.

사용법:
    # 파이프라인으로 검출된 얼굴 이미지 정리
    python prepare_recognition_dataset.py --input crops/ --output dataset/recognition

    # train/val 분할
    python prepare_recognition_dataset.py --input crops/ --output dataset/recognition --split 0.8
"""
import os
import shutil
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import json

import cv2
from tqdm import tqdm


def organize_by_cattle_id(
    input_dir: str,
    output_dir: str,
    id_mapping: Optional[Dict[str, str]] = None,
) -> Dict[str, List[str]]:
    """
    얼굴 이미지를 개체 ID별로 정리

    Args:
        input_dir: 입력 이미지 디렉토리
        output_dir: 출력 디렉토리
        id_mapping: 파일명 -> 개체ID 매핑 (없으면 폴더명 또는 파일명에서 추출)

    Returns:
        개체별 이미지 경로 딕셔너리
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cattle_images: Dict[str, List[str]] = defaultdict(list)

    # 이미지 파일 탐색
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(input_path.rglob(ext))

    print(f"총 {len(image_files)}개 이미지 발견")

    for img_path in tqdm(image_files, desc="이미지 정리 중"):
        # 개체 ID 추출
        if id_mapping and img_path.name in id_mapping:
            cattle_id = id_mapping[img_path.name]
        elif img_path.parent.name != input_path.name:
            # 폴더명이 개체 ID인 경우 (예: cow_001/image.jpg)
            cattle_id = img_path.parent.name
        else:
            # 파일명에서 개체 ID 추출 (예: cow_001_image_001.jpg)
            filename = img_path.stem
            parts = filename.split('_')
            if len(parts) >= 2:
                cattle_id = '_'.join(parts[:2])  # cow_001
            else:
                cattle_id = parts[0]

        # 출력 폴더 생성
        cattle_dir = output_path / cattle_id
        cattle_dir.mkdir(parents=True, exist_ok=True)

        # 파일 복사
        new_filename = f"{cattle_id}_{len(cattle_images[cattle_id]):04d}{img_path.suffix}"
        new_path = cattle_dir / new_filename
        shutil.copy2(img_path, new_path)

        cattle_images[cattle_id].append(str(new_path))

    print(f"\n{len(cattle_images)}개 개체로 정리됨")
    for cattle_id, images in sorted(cattle_images.items())[:10]:
        print(f"  {cattle_id}: {len(images)}장")
    if len(cattle_images) > 10:
        print(f"  ... 외 {len(cattle_images) - 10}개")

    return dict(cattle_images)


def split_dataset(
    organized_dir: str,
    train_dir: str,
    val_dir: str,
    train_ratio: float = 0.8,
    min_images_per_cattle: int = 2,
    seed: int = 42,
) -> Tuple[int, int, int]:
    """
    데이터셋을 train/val로 분할

    Args:
        organized_dir: 정리된 이미지 디렉토리
        train_dir: 학습용 출력 디렉토리
        val_dir: 검증용 출력 디렉토리
        train_ratio: 학습 데이터 비율
        min_images_per_cattle: 최소 이미지 수 (미만이면 제외)
        seed: 랜덤 시드

    Returns:
        (총 개체 수, 학습 이미지 수, 검증 이미지 수)
    """
    random.seed(seed)

    organized_path = Path(organized_dir)
    train_path = Path(train_dir)
    val_path = Path(val_dir)

    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)

    total_cattle = 0
    total_train = 0
    total_val = 0

    # 개체별 폴더 순회
    cattle_dirs = [d for d in organized_path.iterdir() if d.is_dir()]

    for cattle_dir in tqdm(cattle_dirs, desc="데이터 분할 중"):
        images = list(cattle_dir.glob("*"))
        images = [f for f in images if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]

        # 최소 이미지 수 확인
        if len(images) < min_images_per_cattle:
            print(f"건너뜀: {cattle_dir.name} ({len(images)}장 < {min_images_per_cattle})")
            continue

        total_cattle += 1

        # 셔플 후 분할
        random.shuffle(images)
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        # 최소 1장씩 보장
        if len(val_images) == 0 and len(train_images) > 1:
            val_images = [train_images.pop()]

        # 학습 폴더에 복사
        train_cattle_dir = train_path / cattle_dir.name
        train_cattle_dir.mkdir(parents=True, exist_ok=True)
        for img in train_images:
            shutil.copy2(img, train_cattle_dir / img.name)
            total_train += 1

        # 검증 폴더에 복사
        val_cattle_dir = val_path / cattle_dir.name
        val_cattle_dir.mkdir(parents=True, exist_ok=True)
        for img in val_images:
            shutil.copy2(img, val_cattle_dir / img.name)
            total_val += 1

    print(f"\n분할 완료:")
    print(f"  총 개체: {total_cattle}")
    print(f"  학습 이미지: {total_train}")
    print(f"  검증 이미지: {total_val}")

    return total_cattle, total_train, total_val


def create_from_gallery(
    gallery_path: str,
    output_dir: str,
    train_ratio: float = 0.8,
):
    """
    기존 갤러리 데이터에서 학습 데이터셋 생성

    Args:
        gallery_path: 갤러리 JSON/PKL 파일 경로
        output_dir: 출력 디렉토리
        train_ratio: 학습 데이터 비율
    """
    gallery_file = Path(gallery_path)
    output_path = Path(output_dir)

    # 갤러리 로드
    if gallery_file.suffix == '.json':
        with open(gallery_file, 'r', encoding='utf-8') as f:
            gallery = json.load(f)
    else:
        import pickle
        with open(gallery_file, 'rb') as f:
            gallery = pickle.load(f)

    print(f"갤러리에서 {len(gallery)}개 개체 로드")

    train_dir = output_path / 'train'
    val_dir = output_path / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    total_train = 0
    total_val = 0

    for cattle_id, info in tqdm(gallery.items(), desc="데이터셋 생성 중"):
        images = info.get('images', [])
        if not images:
            continue

        random.shuffle(images)
        split_idx = max(1, int(len(images) * train_ratio))

        # 학습 데이터
        train_cattle_dir = train_dir / cattle_id
        train_cattle_dir.mkdir(parents=True, exist_ok=True)
        for i, img_path in enumerate(images[:split_idx]):
            src = Path(img_path)
            if src.exists():
                dst = train_cattle_dir / f"{cattle_id}_{i:04d}{src.suffix}"
                shutil.copy2(src, dst)
                total_train += 1

        # 검증 데이터
        if len(images) > split_idx:
            val_cattle_dir = val_dir / cattle_id
            val_cattle_dir.mkdir(parents=True, exist_ok=True)
            for i, img_path in enumerate(images[split_idx:]):
                src = Path(img_path)
                if src.exists():
                    dst = val_cattle_dir / f"{cattle_id}_{i:04d}{src.suffix}"
                    shutil.copy2(src, dst)
                    total_val += 1

    print(f"\n데이터셋 생성 완료:")
    print(f"  학습: {train_dir} ({total_train}장)")
    print(f"  검증: {val_dir} ({total_val}장)")


def validate_dataset(dataset_dir: str) -> Dict:
    """
    데이터셋 유효성 검증

    Args:
        dataset_dir: 데이터셋 디렉토리

    Returns:
        검증 결과 딕셔너리
    """
    dataset_path = Path(dataset_dir)

    stats = {
        'total_cattle': 0,
        'total_images': 0,
        'images_per_cattle': {},
        'min_images': float('inf'),
        'max_images': 0,
        'invalid_images': [],
    }

    if not dataset_path.exists():
        print(f"디렉토리가 존재하지 않음: {dataset_path}")
        return stats

    cattle_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    stats['total_cattle'] = len(cattle_dirs)

    for cattle_dir in tqdm(cattle_dirs, desc="검증 중"):
        images = list(cattle_dir.glob("*"))
        images = [f for f in images if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]

        valid_count = 0
        for img_path in images:
            # 이미지 로드 테스트
            try:
                img = cv2.imread(str(img_path))
                if img is not None and img.size > 0:
                    valid_count += 1
                else:
                    stats['invalid_images'].append(str(img_path))
            except Exception as e:
                stats['invalid_images'].append(str(img_path))

        stats['images_per_cattle'][cattle_dir.name] = valid_count
        stats['total_images'] += valid_count
        stats['min_images'] = min(stats['min_images'], valid_count)
        stats['max_images'] = max(stats['max_images'], valid_count)

    if stats['min_images'] == float('inf'):
        stats['min_images'] = 0

    print(f"\n검증 결과:")
    print(f"  총 개체: {stats['total_cattle']}")
    print(f"  총 이미지: {stats['total_images']}")
    print(f"  개체당 이미지: {stats['min_images']} ~ {stats['max_images']}")
    print(f"  평균 이미지/개체: {stats['total_images'] / max(1, stats['total_cattle']):.1f}")
    if stats['invalid_images']:
        print(f"  유효하지 않은 이미지: {len(stats['invalid_images'])}개")

    return stats


def main():
    parser = argparse.ArgumentParser(description='소 얼굴 인식 학습용 데이터셋 준비')

    subparsers = parser.add_subparsers(dest='command', help='명령')

    # organize 명령
    org_parser = subparsers.add_parser('organize', help='이미지를 개체별로 정리')
    org_parser.add_argument('--input', type=str, required=True, help='입력 이미지 디렉토리')
    org_parser.add_argument('--output', type=str, required=True, help='출력 디렉토리')

    # split 명령
    split_parser = subparsers.add_parser('split', help='train/val 분할')
    split_parser.add_argument('--input', type=str, required=True, help='정리된 이미지 디렉토리')
    split_parser.add_argument('--train-dir', type=str, required=True, help='학습 데이터 출력')
    split_parser.add_argument('--val-dir', type=str, required=True, help='검증 데이터 출력')
    split_parser.add_argument('--ratio', type=float, default=0.8, help='학습 데이터 비율')
    split_parser.add_argument('--min-images', type=int, default=2, help='최소 이미지 수')
    split_parser.add_argument('--seed', type=int, default=42, help='랜덤 시드')

    # from-gallery 명령
    gallery_parser = subparsers.add_parser('from-gallery', help='갤러리에서 데이터셋 생성')
    gallery_parser.add_argument('--gallery', type=str, required=True, help='갤러리 파일 경로')
    gallery_parser.add_argument('--output', type=str, required=True, help='출력 디렉토리')
    gallery_parser.add_argument('--ratio', type=float, default=0.8, help='학습 데이터 비율')

    # validate 명령
    val_parser = subparsers.add_parser('validate', help='데이터셋 검증')
    val_parser.add_argument('--dir', type=str, required=True, help='데이터셋 디렉토리')

    args = parser.parse_args()

    if args.command == 'organize':
        organize_by_cattle_id(args.input, args.output)
    elif args.command == 'split':
        split_dataset(
            args.input,
            args.train_dir,
            args.val_dir,
            train_ratio=args.ratio,
            min_images_per_cattle=args.min_images,
            seed=args.seed,
        )
    elif args.command == 'from-gallery':
        create_from_gallery(args.gallery, args.output, args.ratio)
    elif args.command == 'validate':
        validate_dataset(args.dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
