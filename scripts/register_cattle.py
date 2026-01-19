"""
소 등록 스크립트 - 데이터셋에서 소를 API를 통해 등록합니다 (시각화 이미지 포함)
"""
import os
import sys
import random
import requests
from pathlib import Path


API_BASE = "http://localhost:8000/api"


def register_cattle(name: str, images: list, breed: str = None):
    """API를 통해 소 등록"""
    url = f"{API_BASE}/cattle"

    files = [("images", (img.name, open(img, "rb"), "image/jpeg")) for img in images]
    data = {"name": name}
    if breed:
        data["breed"] = breed

    response = requests.post(url, files=files, data=data)

    # 파일 핸들 닫기
    for _, (_, f, _) in files:
        f.close()

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"등록 실패: {response.status_code} - {response.text}")


def main():
    # 데이터셋 경로
    dataset_dir = Path(__file__).parent.parent / "cattle_face_recognition/dataset/recognition/train"

    # 사용 가능한 소 ID 목록
    cattle_ids = sorted([d.name for d in dataset_dir.iterdir() if d.is_dir()])
    print(f"총 {len(cattle_ids)}마리의 소 데이터 발견")

    # 20마리만 선택 (랜덤)
    random.seed(42)  # 재현 가능
    selected_ids = random.sample(cattle_ids, min(20, len(cattle_ids)))
    print(f"등록할 소: {selected_ids}")

    registered = 0
    failed = 0

    # 각 소 등록
    for cattle_id in selected_ids:
        cattle_dir = dataset_dir / cattle_id
        images = list(cattle_dir.glob("*.jpg"))

        if not images:
            print(f"[SKIP] {cattle_id}: 이미지 없음")
            continue

        # 최대 3개 이미지 선택
        selected_images = random.sample(images, min(3, len(images)))

        print(f"\n[등록 중] {cattle_id}")
        print(f"  이미지: {len(selected_images)}개")

        try:
            result = register_cattle(
                name=f"소{cattle_id}",
                images=selected_images,
                breed="한우",
            )
            print(f"  [완료] ID: {result['cattle_id']}, 임베딩: {result['num_images']}개")
            registered += 1
        except Exception as e:
            print(f"  [실패] {str(e)}")
            failed += 1

    print(f"\n=== 등록 완료 ===")
    print(f"성공: {registered}마리")
    print(f"실패: {failed}마리")


if __name__ == "__main__":
    main()
