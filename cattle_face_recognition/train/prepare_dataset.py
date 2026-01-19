"""
데이터셋 준비 스크립트
scidb.cn 데이터셋 URL 파일에서 데이터 다운로드 및 YOLO 형식 변환
"""
import os
import sys
import asyncio
import aiohttp
import aiofiles
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
import argparse


async def download_file(session: aiohttp.ClientSession, url: str, output_path: Path) -> bool:
    """단일 파일 다운로드"""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as response:
            if response.status == 200:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                async with aiofiles.open(output_path, 'wb') as f:
                    await f.write(await response.read())
                return True
            else:
                return False
    except Exception as e:
        print(f"다운로드 실패: {url} - {e}")
        return False


async def download_batch(urls: List[Tuple[str, Path]], max_concurrent: int = 10):
    """배치 다운로드"""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def download_with_semaphore(session, url, path):
        async with semaphore:
            return await download_file(session, url, path)

    connector = aiohttp.TCPConnector(limit=max_concurrent)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [download_with_semaphore(session, url, path) for url, path in urls]

        results = []
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="다운로드 중"):
            result = await task
            results.append(result)

        return results


def parse_url_file(url_file: str) -> List[Tuple[str, str, str]]:
    """
    URL 파일 파싱

    Returns:
        (url, split, filename) 튜플 리스트
    """
    entries = []

    with open(url_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith('http'):
                continue

            # URL에서 정보 추출
            # 예: ...path=/V2/奶牛脸部及关键点检测数据集/images/train/000051000067.jpg...
            if 'images/train/' in line:
                split = 'train'
            elif 'images/val/' in line:
                split = 'val'
            elif 'labels/train/' in line:
                split = 'train'
            elif 'labels/val/' in line:
                split = 'val'
            else:
                continue

            # 파일명 추출
            if 'fileName=' in line:
                filename = line.split('fileName=')[-1]
            else:
                continue

            # 이미지/라벨 구분
            if '.jpg' in line or '.png' in line:
                file_type = 'images'
            elif '.txt' in line:
                file_type = 'labels'
            else:
                continue

            entries.append((line, split, file_type, filename))

    return entries


def prepare_yolo_dataset(
    url_file: str,
    output_dir: str,
    max_files: int = None,
    download: bool = True,
):
    """
    YOLO 형식 데이터셋 준비

    Args:
        url_file: URL 목록 파일 경로
        output_dir: 출력 디렉토리
        max_files: 최대 파일 수 (테스트용)
        download: 실제 다운로드 수행 여부
    """
    output_path = Path(output_dir)

    # URL 파싱
    print("URL 파일 파싱 중...")
    entries = parse_url_file(url_file)
    print(f"총 {len(entries)}개 파일 발견")

    if max_files:
        entries = entries[:max_files]

    # 다운로드 목록 생성
    download_list = []
    for url, split, file_type, filename in entries:
        local_path = output_path / file_type / split / filename
        if not local_path.exists():
            download_list.append((url, local_path))

    print(f"다운로드 필요: {len(download_list)}개 파일")

    if download and download_list:
        # 비동기 다운로드
        asyncio.run(download_batch(download_list))

    # YOLO 데이터셋 설정 파일 생성
    create_dataset_yaml(output_path)

    print(f"데이터셋 준비 완료: {output_path}")


def create_dataset_yaml(dataset_path: Path):
    """YOLO 데이터셋 설정 파일 생성"""
    yaml_content = f"""# 소 얼굴 검출 데이터셋
path: {dataset_path.absolute()}
train: images/train
val: images/val

# 클래스
nc: 1
names:
  0: cattle_face

# 키포인트 (5개)
kpt_shape: [5, 3]  # [num_keypoints, (x, y, visibility)]
"""

    yaml_path = dataset_path / "cattle_face.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print(f"데이터셋 설정 파일 생성: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description='소 얼굴 데이터셋 준비')
    parser.add_argument('--url-file', type=str, required=True, help='URL 목록 파일')
    parser.add_argument('--output', type=str, default='./dataset', help='출력 디렉토리')
    parser.add_argument('--max-files', type=int, default=None, help='최대 파일 수')
    parser.add_argument('--no-download', action='store_true', help='다운로드 건너뛰기')

    args = parser.parse_args()

    prepare_yolo_dataset(
        url_file=args.url_file,
        output_dir=args.output,
        max_files=args.max_files,
        download=not args.no_download,
    )


if __name__ == "__main__":
    main()
