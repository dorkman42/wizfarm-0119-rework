#!/usr/bin/env python3
"""
소 일괄 등록 스크립트
각 소마다 1장은 등록용, 나머지는 테스트용으로 분리
"""
import os
import sys
import requests
from pathlib import Path
import json
import random

DATASET_DIR = Path(__file__).parent.parent / "dataset" / "recognition" / "train"
API_BASE = "http://localhost:8000/api"

def get_cattle_with_images(min_images=2):
    """이미지가 min_images개 이상인 소 목록 반환"""
    cattle = []
    for cow_dir in sorted(DATASET_DIR.iterdir()):
        if not cow_dir.is_dir():
            continue
        images = list(cow_dir.glob("*.jpg"))
        if len(images) >= min_images:
            cattle.append({
                "cow_id": cow_dir.name,
                "images": sorted([str(img) for img in images])
            })
    return cattle

def clear_gallery():
    """기존 갤러리 초기화"""
    # 등록된 소 목록 조회
    res = requests.get(f"{API_BASE}/cattle")
    if res.ok:
        data = res.json()
        for c in data.get("cattle", []):
            cid = c["cattle_id"]
            print(f"  삭제: {cid}")
            requests.delete(f"{API_BASE}/cattle/{cid}")
    print("갤러리 초기화 완료")

def register_cattle(cow_id: str, name: str, image_paths: list):
    """소 등록"""
    files = [("images", (Path(p).name, open(p, "rb"), "image/jpeg")) for p in image_paths]
    data = {
        "name": name,
        "notes": f"Dataset ID: {cow_id}"
    }
    try:
        res = requests.post(f"{API_BASE}/cattle", data=data, files=files)
        return res.json() if res.ok else None
    finally:
        for _, (_, f, _) in files:
            f.close()

def test_recognition(image_path: str, expected_id: str):
    """인식 테스트"""
    with open(image_path, "rb") as f:
        res = requests.post(
            f"{API_BASE}/recognition/recognize",
            files={"image": f}
        )
    if not res.ok:
        return {"success": False, "error": "API error"}

    data = res.json()
    recognitions = data.get("recognitions", [])

    for rec in recognitions:
        if rec.get("cattle_id") == expected_id:
            return {
                "success": True,
                "confidence": rec["confidence"],
                "is_new": rec["is_new"]
            }

    # 해당 소를 찾지 못함
    return {
        "success": False,
        "found": [r.get("cattle_id") for r in recognitions],
        "expected": expected_id
    }

def main():
    print("=" * 60)
    print("소 일괄 등록 및 인식 테스트")
    print("=" * 60)

    # 데이터셋 분석
    cattle_list = get_cattle_with_images(min_images=3)
    print(f"\n이미지 3개 이상인 소: {len(cattle_list)}마리")

    if not cattle_list:
        print("테스트할 소가 없습니다.")
        return

    # 기존 갤러리 초기화 (선택적)
    if len(sys.argv) > 1 and sys.argv[1] == "--clear":
        print("\n기존 갤러리 초기화 중...")
        clear_gallery()

    # 등록 및 테스트
    results = {
        "registered": 0,
        "tested": 0,
        "correct": 0,
        "failed": 0
    }

    registered_cattle = {}  # dataset_id -> registered_id 매핑

    print("\n[1/2] 소 등록 중...")
    for cow in cattle_list:
        cow_id = cow["cow_id"]
        images = cow["images"]

        # 첫 번째 이미지 2개로 등록 (나머지는 테스트용)
        register_images = images[:2]
        test_images = images[2:]

        name = f"소_{cow_id}"
        result = register_cattle(cow_id, name, register_images)

        if result and "cattle_id" in result:
            registered_cattle[cow_id] = {
                "cattle_id": result["cattle_id"],
                "name": name,
                "test_images": test_images
            }
            results["registered"] += 1
            print(f"  ✓ {name} 등록 완료 (ID: {result['cattle_id']})")
        else:
            print(f"  ✗ {name} 등록 실패: {result}")

    print(f"\n등록 완료: {results['registered']}마리")

    # 인식 테스트
    print("\n[2/2] 인식 테스트 중...")
    for cow_id, info in registered_cattle.items():
        cattle_id = info["cattle_id"]
        test_images = info["test_images"]

        for img_path in test_images[:2]:  # 각 소당 최대 2개 이미지 테스트
            results["tested"] += 1
            test_result = test_recognition(img_path, cattle_id)

            if test_result["success"]:
                results["correct"] += 1
                conf = test_result["confidence"]
                print(f"  ✓ {info['name']}: {conf*100:.1f}% ({Path(img_path).name})")
            else:
                results["failed"] += 1
                print(f"  ✗ {info['name']}: 인식 실패 ({Path(img_path).name})")
                if "found" in test_result:
                    print(f"    발견된 소: {test_result['found']}")

    # 결과 요약
    print("\n" + "=" * 60)
    print("결과 요약")
    print("=" * 60)
    print(f"등록된 소: {results['registered']}마리")
    print(f"테스트 수: {results['tested']}개")
    print(f"정확 인식: {results['correct']}개")
    print(f"인식 실패: {results['failed']}개")
    if results["tested"] > 0:
        accuracy = results["correct"] / results["tested"] * 100
        print(f"인식 정확도: {accuracy:.1f}%")

if __name__ == "__main__":
    main()
