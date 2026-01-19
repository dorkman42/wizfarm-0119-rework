# 소 얼굴 인식 시스템 (Cattle Face Recognition)

YOLOv8 기반 소 얼굴 검출 및 인식 시스템

## 주요 기능

- 소 얼굴 검출 (YOLOv8 기반)
- 소 얼굴 인식 (ResNet + ArcFace)
- 소 등록/관리 API
- 웹 기반 관리 인터페이스

## 설치 방법

### 1. Python 환경 설정

```bash
# Python 3.10+ 필요
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 모델 파일 준비

`cattle_face_recognition/models/` 디렉토리에 다음 파일 배치:
- `cattle_face_detector.pt` - 소 얼굴 검출 모델
- `cattle_face_recognizer.pt` - 소 얼굴 인식 모델

### 3. 웹 프론트엔드 설정

```bash
cd cattle_face_recognition/web
npm install
npm run build
```

## 실행 방법

### API 서버 실행

```bash
cd cattle_face_recognition
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 웹 프론트엔드 실행

```bash
cd cattle_face_recognition/web
npm run dev
```

## API 문서

서버 실행 후 접속:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 주요 API 엔드포인트

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/api/cattle` | GET | 등록된 소 목록 조회 |
| `/api/cattle` | POST | 새 소 등록 |
| `/api/cattle/{id}` | GET | 소 상세 정보 조회 |
| `/api/cattle/{id}` | DELETE | 소 삭제 |
| `/api/recognition/detect` | POST | 얼굴 검출 |
| `/api/recognition/recognize` | POST | 얼굴 인식 |

## 프로젝트 구조

```
cattle_face_recognition/
├── api/                 # FastAPI 백엔드
│   ├── routes/          # API 라우터
│   ├── config.py        # 설정
│   └── main.py          # 앱 진입점
├── detection/           # 얼굴 검출 모듈
├── recognition/         # 얼굴 인식 모듈
├── alignment/           # 얼굴 정렬 모듈
├── models/              # 모델 파일
├── web/                 # Next.js 프론트엔드
└── pipeline.py          # 통합 파이프라인
```

## 라이센스

이 프로젝트는 Gendive 라이센스를 따릅니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

**상업적 사용 불가** - 비상업적 용도로만 사용할 수 있습니다.
