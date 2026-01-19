"""
소 얼굴 인식 데이터셋 다운로드 및 준비 스크립트
scidb.cn에서 이미지를 다운로드하고 개체별로 정리합니다.
"""
import os
import asyncio
import aiohttp
import aiofiles
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict
import argparse
import random
import shutil
from tqdm import tqdm


async def download_file(
    session: aiohttp.ClientSession,
    url: str,
    output_path: Path,
    semaphore: asyncio.Semaphore,
) -> bool:
    """단일 파일 다운로드"""
    async with semaphore:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as response:
                if response.status == 200:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    async with aiofiles.open(output_path, 'wb') as f:
                        await f.write(await response.read())
                    return True
                return False
        except Exception as e:
            return False


async def download_images(
    urls: List[Tuple[str, Path]],
    max_concurrent: int = 20,
) -> int:
    """이미지 다운로드"""
    semaphore = asyncio.Semaphore(max_concurrent)
    connector = aiohttp.TCPConnector(limit=max_concurrent)

    success_count = 0

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for url, path in urls:
            if not path.exists():
                tasks.append(download_file(session, url, path, semaphore))

        if tasks:
            pbar = tqdm(total=len(tasks), desc="다운로드 중")
            for coro in asyncio.as_completed(tasks):
                result = await coro
                if result:
                    success_count += 1
                pbar.update(1)
            pbar.close()

    return success_count


def parse_url_file(url_file: str, images_only: bool = True) -> Dict[str, List[Tuple[str, str]]]:
    """
    URL 파일 파싱 및 개체별 그룹화

    Returns:
        cattle_id -> [(url, filename), ...] 딕셔너리
    """
    cattle_data = defaultdict(list)

    with open(url_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('http'):
                continue

            # 이미지만 필터링
            if images_only and '.jpg' not in line.lower() and '.png' not in line.lower():
                continue

            # 파일명 추출
            if 'fileName=' not in line:
                continue

            filename = line.split('fileName=')[-1]

            # 개체 ID 추출 (처음 6자리)
            if len(filename) >= 6:
                cattle_id = filename[:6]
                cattle_data[cattle_id].append((line, filename))

    return dict(cattle_data)


def prepare_recognition_dataset(
    url_file: str,
    output_dir: str,
    train_ratio: float = 0.8,
    max_images_per_cattle: int = None,
    max_cattle: int = None,
    download: bool = True,
    seed: int = 42,
):
    """
    인식 학습용 데이터셋 준비

    Args:
        url_file: URL 목록 파일
        output_dir: 출력 디렉토리
        train_ratio: 학습 데이터 비율
        max_images_per_cattle: 개체당 최대 이미지 수
        max_cattle: 최대 개체 수
        download: 다운로드 실행 여부
        seed: 랜덤 시드
    """
    random.seed(seed)
    output_path = Path(output_dir)

    # URL 파싱
    print("URL 파일 파싱 중...")
    cattle_data = parse_url_file(url_file, images_only=True)
    print(f"총 {len(cattle_data)}개 개체 발견")

    # 개체 수 제한
    cattle_ids = list(cattle_data.keys())
    if max_cattle and len(cattle_ids) > max_cattle:
        cattle_ids = random.sample(cattle_ids, max_cattle)
        cattle_data = {k: cattle_data[k] for k in cattle_ids}
        print(f"  -> {max_cattle}개 개체로 제한")

    # 통계 출력
    total_images = sum(len(v) for v in cattle_data.values())
    print(f"총 이미지: {total_images}개")

    # 다운로드 목록 생성 및 train/val 분할
    train_dir = output_path / "train"
    val_dir = output_path / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    download_list = []
    train_count = 0
    val_count = 0

    for cattle_id, items in cattle_data.items():
        # 이미지 수 제한
        if max_images_per_cattle and len(items) > max_images_per_cattle:
            items = random.sample(items, max_images_per_cattle)

        # train/val 분할
        random.shuffle(items)
        split_idx = max(1, int(len(items) * train_ratio))
        train_items = items[:split_idx]
        val_items = items[split_idx:] if len(items) > 1 else []

        # train 폴더
        cattle_train_dir = train_dir / cattle_id
        cattle_train_dir.mkdir(parents=True, exist_ok=True)
        for url, filename in train_items:
            local_path = cattle_train_dir / filename
            download_list.append((url, local_path))
            train_count += 1

        # val 폴더
        if val_items:
            cattle_val_dir = val_dir / cattle_id
            cattle_val_dir.mkdir(parents=True, exist_ok=True)
            for url, filename in val_items:
                local_path = cattle_val_dir / filename
                download_list.append((url, local_path))
                val_count += 1

    print(f"\n데이터셋 분할:")
    print(f"  학습: {train_count}개 이미지")
    print(f"  검증: {val_count}개 이미지")

    # 이미 다운로드된 파일 제외
    download_list = [(url, path) for url, path in download_list if not path.exists()]
    print(f"\n다운로드 필요: {len(download_list)}개 파일")

    if download and download_list:
        print("\n다운로드 시작...")
        success = asyncio.run(download_images(download_list))
        print(f"다운로드 완료: {success}/{len(download_list)}")

    print(f"\n데이터셋 준비 완료!")
    print(f"  학습 데이터: {train_dir}")
    print(f"  검증 데이터: {val_dir}")

    return train_dir, val_dir


def main():
    parser = argparse.ArgumentParser(description='소 얼굴 인식 데이터셋 다운로드 및 준비')
    parser.add_argument('--url-file', type=str, required=True, help='URL 목록 파일')
    parser.add_argument('--output', type=str, default='./dataset/recognition', help='출력 디렉토리')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='학습 데이터 비율')
    parser.add_argument('--max-images', type=int, default=None, help='개체당 최대 이미지 수')
    parser.add_argument('--max-cattle', type=int, default=None, help='최대 개체 수')
    parser.add_argument('--no-download', action='store_true', help='다운로드 건너뛰기')
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드')

    args = parser.parse_args()

    prepare_recognition_dataset(
        url_file=args.url_file,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        max_images_per_cattle=args.max_images,
        max_cattle=args.max_cattle,
        download=not args.no_download,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
