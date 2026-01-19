"""
소 얼굴 인식 시스템 기본 사용 예제
"""
import sys
from pathlib import Path

# 패키지 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cattle_face_recognition import (
    CattleFaceRecognitionPipeline,
    PipelineConfig,
    DetectionConfig,
)


def example_basic_usage():
    """기본 사용법"""
    print("=" * 50)
    print("1. 기본 사용법")
    print("=" * 50)

    # 파이프라인 초기화
    pipeline = CattleFaceRecognitionPipeline(
        detection_model="yolov8n-pose.pt",  # 또는 fine-tuned 모델
        gallery_path="./gallery.pkl"
    )

    # 이미지 처리
    # result = pipeline.process_image("path/to/image.jpg")
    # print(f"검출된 소: {len(result.detections)}마리")

    print("파이프라인 초기화 완료")


def example_register_cattle():
    """새로운 소 등록"""
    print("\n" + "=" * 50)
    print("2. 새로운 소 등록")
    print("=" * 50)

    pipeline = CattleFaceRecognitionPipeline()

    # 단일 이미지로 등록
    # identity = pipeline.register_cattle(
    #     cattle_id="cow_001",
    #     name="Daisy",
    #     image_source="path/to/daisy.jpg",
    #     metadata={"breed": "Holstein", "age": 3}
    # )
    # print(f"등록 완료: {identity.name} (ID: {identity.cattle_id})")

    # 여러 이미지로 등록 (더 정확한 인식)
    # identity = pipeline.register_cattle(
    #     cattle_id="cow_002",
    #     name="Bella",
    #     image_source=[
    #         "path/to/bella_front.jpg",
    #         "path/to/bella_side.jpg",
    #         "path/to/bella_another.jpg",
    #     ],
    #     metadata={"breed": "Jersey", "age": 2}
    # )

    # 폴더에서 일괄 등록
    # identities = pipeline.register_from_folder("path/to/cattle_images/")
    # print(f"총 {len(identities)}마리 등록 완료")

    print("등록 예제 준비 완료 (이미지 경로 수정 필요)")


def example_recognize():
    """소 인식"""
    print("\n" + "=" * 50)
    print("3. 소 인식")
    print("=" * 50)

    pipeline = CattleFaceRecognitionPipeline(gallery_path="./gallery.pkl")

    # 이미지에서 소 인식
    # result = pipeline.process_image("path/to/test.jpg", visualize=True)
    #
    # for i, rec in enumerate(result.recognition_results):
    #     if rec.is_new:
    #         print(f"소 {i+1}: 새로운 개체 (신뢰도: {rec.confidence:.2f})")
    #     else:
    #         print(f"소 {i+1}: {rec.name} (ID: {rec.cattle_id}, 신뢰도: {rec.confidence:.2f})")
    #
    # # 시각화 저장
    # if result.visualization is not None:
    #     import cv2
    #     cv2.imwrite("result.jpg", result.visualization)

    print("인식 예제 준비 완료 (이미지 경로 수정 필요)")


def example_video_processing():
    """비디오 처리"""
    print("\n" + "=" * 50)
    print("4. 비디오 처리")
    print("=" * 50)

    pipeline = CattleFaceRecognitionPipeline(gallery_path="./gallery.pkl")

    # 비디오 처리 (추적 포함)
    # results = pipeline.process_video(
    #     video_path="path/to/video.mp4",
    #     output_path="output.mp4",
    #     track=True,
    #     show=True,  # 실시간 표시
    #     max_frames=1000,  # 최대 1000프레임
    # )
    # print(f"총 {len(results)} 프레임 처리 완료")

    print("비디오 처리 예제 준비 완료 (비디오 경로 수정 필요)")


def example_search_similar():
    """유사한 소 검색"""
    print("\n" + "=" * 50)
    print("5. 유사한 소 검색")
    print("=" * 50)

    pipeline = CattleFaceRecognitionPipeline(gallery_path="./gallery.pkl")

    # 유사한 소 검색
    # matches = pipeline.search_cattle(
    #     image="path/to/query.jpg",
    #     threshold=0.3
    # )
    #
    # for cattle_id, name, similarity in matches:
    #     print(f"- {name} (ID: {cattle_id}): 유사도 {similarity:.2f}")

    print("검색 예제 준비 완료 (이미지 경로 수정 필요)")


def example_gallery_management():
    """갤러리 관리"""
    print("\n" + "=" * 50)
    print("6. 갤러리 관리")
    print("=" * 50)

    pipeline = CattleFaceRecognitionPipeline()

    # 등록된 소 목록 확인
    cattle_list = pipeline.get_registered_cattle()
    print(f"등록된 소: {len(cattle_list)}마리")
    for cattle in cattle_list:
        print(f"  - {cattle['name']} (ID: {cattle['cattle_id']}, 이미지: {cattle['num_images']}장)")

    # 갤러리 저장
    # pipeline.save_gallery("./my_gallery.pkl")

    # 갤러리 로드
    # pipeline.load_gallery("./my_gallery.pkl")

    # 갤러리 시각화
    # grid = pipeline.create_gallery_visualization("gallery_preview.jpg")


def main():
    """모든 예제 실행"""
    print("소 얼굴 인식 시스템 예제")
    print("=" * 50)

    example_basic_usage()
    example_register_cattle()
    example_recognize()
    example_video_processing()
    example_search_similar()
    example_gallery_management()

    print("\n" + "=" * 50)
    print("모든 예제 완료!")
    print("=" * 50)


if __name__ == "__main__":
    main()
