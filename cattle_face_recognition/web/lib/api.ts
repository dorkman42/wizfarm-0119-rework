/**
 * API 유틸리티
 */
const API_BASE = '/api';

export interface Cattle {
  cattle_id: string;
  name: string;
  num_images: number;
  breed?: string;
  age?: number;
  notes?: string;
  registered_at: string;
  images: string[];
}

export interface CattleListResponse {
  total: number;
  cattle: Cattle[];
}

export interface RecognitionResult {
  cattle_id: string | null;
  name: string | null;
  confidence: number;
  is_new: boolean;
  bbox: number[];
  face_image: string | null;  // base64 인코딩된 얼굴 이미지 (새로운 소일 때)
}

export interface RecognizeResponse {
  detections: any[];
  recognitions: RecognitionResult[];
  image_url: string | null;
  visualization: string | null;  // base64 인코딩된 시각화 이미지
}

export interface DetectResponse {
  detections: any[];
  count: number;
}

export const api = {
  // 얼굴 검출 (미리보기용)
  async detect(formData: FormData): Promise<DetectResponse> {
    const res = await fetch(`${API_BASE}/recognition/detect`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) throw new Error('검출 실패');
    return res.json();
  },

  // 소 목록 조회
  async getCattleList(): Promise<CattleListResponse> {
    const res = await fetch(`${API_BASE}/cattle`);
    if (!res.ok) throw new Error('목록 조회 실패');
    return res.json();
  },

  // 소 상세 조회
  async getCattle(cattleId: string): Promise<Cattle> {
    const res = await fetch(`${API_BASE}/cattle/${cattleId}`);
    if (!res.ok) throw new Error('조회 실패');
    return res.json();
  },

  // 소 등록
  async createCattle(formData: FormData): Promise<Cattle> {
    const res = await fetch(`${API_BASE}/cattle`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const error = await res.json();
      throw new Error(error.detail || '등록 실패');
    }
    return res.json();
  },

  // 소 정보 수정
  async updateCattle(
    cattleId: string,
    data: Partial<Cattle>
  ): Promise<Cattle> {
    const res = await fetch(`${API_BASE}/cattle/${cattleId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error('수정 실패');
    return res.json();
  },

  // 소 이미지 추가
  async addCattleImages(cattleId: string, formData: FormData): Promise<Cattle> {
    const res = await fetch(`${API_BASE}/cattle/${cattleId}/images`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) throw new Error('이미지 추가 실패');
    return res.json();
  },

  // 소 삭제
  async deleteCattle(cattleId: string): Promise<void> {
    const res = await fetch(`${API_BASE}/cattle/${cattleId}`, {
      method: 'DELETE',
    });
    if (!res.ok) throw new Error('삭제 실패');
  },

  // 소 이미지 삭제
  async deleteCattleImage(cattleId: string, filename: string): Promise<void> {
    const res = await fetch(`${API_BASE}/cattle/${cattleId}/images/${filename}`, {
      method: 'DELETE',
    });
    if (!res.ok) {
      const error = await res.json();
      throw new Error(error.detail || '이미지 삭제 실패');
    }
  },

  // 얼굴 인식
  async recognize(formData: FormData, autoRegister: boolean = false): Promise<RecognizeResponse> {
    const res = await fetch(`${API_BASE}/recognition/recognize?save_image=true&auto_register=${autoRegister}`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) throw new Error('인식 실패');
    return res.json();
  },

  // 새로운 소 등록 (얼굴 이미지로부터)
  async registerNewCattle(data: {
    face_image: string;
    name?: string;
    breed?: string;
    age?: number;
    notes?: string;
  }): Promise<{ cattle_id: string; name: string; image_url: string }> {
    const res = await fetch(`${API_BASE}/recognition/register-new`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    if (!res.ok) {
      const error = await res.json();
      throw new Error(error.detail || '등록 실패');
    }
    return res.json();
  },

  // 통계
  async getStats(): Promise<{ total_cattle: number; total_images: number }> {
    const res = await fetch(`${API_BASE}/recognition/stats`);
    if (!res.ok) throw new Error('통계 조회 실패');
    return res.json();
  },
};
