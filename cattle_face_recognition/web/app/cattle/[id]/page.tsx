'use client';

import { useState, useEffect } from 'react';
import { useParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import {
  ArrowLeft,
  Edit,
  Trash2,
  Plus,
  Save,
  X,
  Loader2,
  ZoomIn,
} from 'lucide-react';
import { api, Cattle } from '@/lib/api';
import { ImageDropzone } from '@/components/ImageDropzone';

export default function CattleDetailPage() {
  const params = useParams();
  const router = useRouter();
  const cattleId = params.id as string;

  const [cattle, setCattle] = useState<Cattle | null>(null);
  const [loading, setLoading] = useState(true);
  const [editing, setEditing] = useState(false);
  const [addingImages, setAddingImages] = useState(false);
  const [saving, setSaving] = useState(false);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);

  // 편집 상태
  const [editName, setEditName] = useState('');
  const [editBreed, setEditBreed] = useState('');
  const [editAge, setEditAge] = useState('');
  const [editNotes, setEditNotes] = useState('');
  const [newFiles, setNewFiles] = useState<File[]>([]);

  useEffect(() => {
    loadCattle();
  }, [cattleId]);

  const loadCattle = async () => {
    try {
      const data = await api.getCattle(cattleId);
      setCattle(data);
      setEditName(data.name);
      setEditBreed(data.breed || '');
      setEditAge(data.age?.toString() || '');
      setEditNotes(data.notes || '');
    } catch (error) {
      console.error('조회 실패:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    if (!editName.trim()) {
      alert('이름을 입력해주세요');
      return;
    }

    setSaving(true);
    try {
      const updated = await api.updateCattle(cattleId, {
        name: editName,
        breed: editBreed || undefined,
        age: editAge ? parseInt(editAge) : undefined,
        notes: editNotes || undefined,
      });
      setCattle(updated);
      setEditing(false);
    } catch (error) {
      console.error('저장 실패:', error);
      alert('저장에 실패했습니다');
    } finally {
      setSaving(false);
    }
  };

  const handleAddImages = async () => {
    if (newFiles.length === 0) return;

    setSaving(true);
    try {
      const formData = new FormData();
      newFiles.forEach((file) => formData.append('images', file));

      const updated = await api.addCattleImages(cattleId, formData);
      setCattle(updated);
      setAddingImages(false);
      setNewFiles([]);
    } catch (error) {
      console.error('이미지 추가 실패:', error);
      alert('이미지 추가에 실패했습니다');
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async () => {
    if (!confirm('정말 삭제하시겠습니까? 이 작업은 되돌릴 수 없습니다.')) return;

    try {
      await api.deleteCattle(cattleId);
      router.push('/cattle');
    } catch (error) {
      console.error('삭제 실패:', error);
      alert('삭제에 실패했습니다');
    }
  };

  // 얼굴 crop 이미지 URL -> 시각화 이미지 URL 변환
  const getVisImageUrl = (cropUrl: string): string => {
    // /uploads/cattle_id/abc123.jpg -> /uploads/cattle_id/abc123_vis.jpg
    const parts = cropUrl.split('.');
    if (parts.length >= 2) {
      const ext = parts.pop();
      return `${parts.join('.')}_vis.${ext}`;
    }
    return cropUrl;
  };

  const handleDeleteImage = async (imageUrl: string) => {
    if (!confirm('이 이미지를 삭제하시겠습니까?')) return;

    // URL에서 파일명 추출
    const filename = imageUrl.split('/').pop();
    if (!filename) return;

    setSaving(true);
    try {
      await api.deleteCattleImage(cattleId, filename);
      await loadCattle(); // 데이터 새로고침
    } catch (error: any) {
      console.error('이미지 삭제 실패:', error);
      alert(error.message || '이미지 삭제에 실패했습니다');
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className='text-center py-12'>
        <Loader2 className='w-8 h-8 animate-spin mx-auto text-gray-400' />
        <p className='text-gray-500 mt-2'>로딩 중...</p>
      </div>
    );
  }

  if (!cattle) {
    return (
      <div className='text-center py-12'>
        <p className='text-gray-500'>소를 찾을 수 없습니다</p>
        <Link href='/cattle' className='btn btn-primary mt-4'>
          목록으로 돌아가기
        </Link>
      </div>
    );
  }

  return (
    <div>
      {/* 헤더 */}
      <div className='flex items-center justify-between mb-8'>
        <div className='flex items-center space-x-4'>
          <Link href='/cattle' className='p-2 hover:bg-gray-100 rounded-lg'>
            <ArrowLeft className='w-5 h-5' />
          </Link>
          <h1 className='text-2xl font-bold text-gray-900'>{cattle.name}</h1>
        </div>
        <div className='flex space-x-2'>
          {!editing && (
            <button
              onClick={() => setEditing(true)}
              className='btn btn-secondary'>
              <Edit className='w-4 h-4 mr-2 inline' />
              수정
            </button>
          )}
          <button onClick={handleDelete} className='btn btn-danger'>
            <Trash2 className='w-4 h-4 mr-2 inline' />
            삭제
          </button>
        </div>
      </div>

      <div className='grid grid-cols-1 lg:grid-cols-3 gap-8'>
        {/* 기본 정보 */}
        <div className='card'>
          <div className='card-header flex justify-between items-center'>
            <h2 className='text-lg font-semibold'>기본 정보</h2>
            {editing && (
              <div className='flex space-x-2'>
                <button
                  onClick={() => setEditing(false)}
                  className='p-1 hover:bg-gray-200 rounded'>
                  <X className='w-4 h-4' />
                </button>
              </div>
            )}
          </div>
          <div className='card-body'>
            {editing ? (
              <div className='space-y-4'>
                <div>
                  <label className='block text-sm font-medium text-gray-700 mb-1'>
                    이름
                  </label>
                  <input
                    type='text'
                    value={editName}
                    onChange={(e) => setEditName(e.target.value)}
                    className='input'
                  />
                </div>
                <div>
                  <label className='block text-sm font-medium text-gray-700 mb-1'>
                    품종
                  </label>
                  <input
                    type='text'
                    value={editBreed}
                    onChange={(e) => setEditBreed(e.target.value)}
                    className='input'
                  />
                </div>
                <div>
                  <label className='block text-sm font-medium text-gray-700 mb-1'>
                    나이 (개월)
                  </label>
                  <input
                    type='number'
                    value={editAge}
                    onChange={(e) => setEditAge(e.target.value)}
                    className='input'
                  />
                </div>
                <div>
                  <label className='block text-sm font-medium text-gray-700 mb-1'>
                    메모
                  </label>
                  <textarea
                    value={editNotes}
                    onChange={(e) => setEditNotes(e.target.value)}
                    className='input'
                    rows={3}
                  />
                </div>
                <button
                  onClick={handleSave}
                  disabled={saving}
                  className='btn btn-primary w-full'>
                  {saving ? (
                    <Loader2 className='w-4 h-4 animate-spin inline mr-2' />
                  ) : (
                    <Save className='w-4 h-4 inline mr-2' />
                  )}
                  저장
                </button>
              </div>
            ) : (
              <dl className='space-y-3'>
                <div>
                  <dt className='text-sm text-gray-500'>ID</dt>
                  <dd className='font-mono text-sm'>{cattle.cattle_id}</dd>
                </div>
                <div>
                  <dt className='text-sm text-gray-500'>이름</dt>
                  <dd className='font-medium'>{cattle.name}</dd>
                </div>
                {cattle.breed && (
                  <div>
                    <dt className='text-sm text-gray-500'>품종</dt>
                    <dd>{cattle.breed}</dd>
                  </div>
                )}
                {cattle.age && (
                  <div>
                    <dt className='text-sm text-gray-500'>나이</dt>
                    <dd>{cattle.age}개월</dd>
                  </div>
                )}
                {cattle.notes && (
                  <div>
                    <dt className='text-sm text-gray-500'>메모</dt>
                    <dd className='text-gray-700'>{cattle.notes}</dd>
                  </div>
                )}
                <div>
                  <dt className='text-sm text-gray-500'>등록일</dt>
                  <dd>
                    {new Date(cattle.registered_at).toLocaleDateString('ko-KR')}
                  </dd>
                </div>
              </dl>
            )}
          </div>
        </div>

        {/* 이미지 갤러리 */}
        <div className='lg:col-span-2 card'>
          <div className='card-header flex justify-between items-center'>
            <h2 className='text-lg font-semibold'>
              등록된 이미지 ({cattle.images.length})
            </h2>
            {!addingImages && (
              <button
                onClick={() => setAddingImages(true)}
                className='btn btn-secondary text-sm'>
                <Plus className='w-4 h-4 mr-1 inline' />
                이미지 추가
              </button>
            )}
          </div>
          <div className='card-body'>
            {addingImages && (
              <div className='mb-6 p-4 bg-gray-50 rounded-lg'>
                <ImageDropzone
                  files={newFiles}
                  onFilesChange={setNewFiles}
                  multiple={true}
                  maxFiles={5}
                />
                <div className='flex justify-end space-x-2 mt-4'>
                  <button
                    onClick={() => {
                      setAddingImages(false);
                      setNewFiles([]);
                    }}
                    className='btn btn-secondary'>
                    취소
                  </button>
                  <button
                    onClick={handleAddImages}
                    disabled={saving || newFiles.length === 0}
                    className='btn btn-primary'>
                    {saving ? (
                      <Loader2 className='w-4 h-4 animate-spin inline mr-2' />
                    ) : null}
                    추가하기
                  </button>
                </div>
              </div>
            )}

            {cattle.images.length === 0 ? (
              <p className='text-gray-500 text-center py-8'>
                등록된 이미지가 없습니다
              </p>
            ) : (
              <div className='grid grid-cols-2 md:grid-cols-3 gap-4'>
                {cattle.images.map((url, index) => (
                  <div
                    key={index}
                    className='relative group aspect-square rounded-lg overflow-hidden bg-gray-100'>
                    <img
                      src={url}
                      alt={`${cattle.name} ${index + 1}`}
                      className='w-full h-full object-cover cursor-pointer'
                      onClick={() => setSelectedImage(url)}
                    />
                    {/* 확대 버튼 (hover 시 표시) */}
                    <button
                      onClick={() => setSelectedImage(url)}
                      className='absolute bottom-2 left-2 p-1.5 bg-black/50 text-white rounded-full
                                 opacity-0 group-hover:opacity-100 transition-opacity hover:bg-black/70'
                      title='원본 이미지 보기'>
                      <ZoomIn className='w-4 h-4' />
                    </button>
                    {/* 삭제 버튼 (hover 시 표시) */}
                    <button
                      onClick={() => handleDeleteImage(url)}
                      disabled={saving || cattle.images.length <= 1}
                      className='absolute top-2 right-2 p-1.5 bg-red-500 text-white rounded-full
                                 opacity-0 group-hover:opacity-100 transition-opacity
                                 hover:bg-red-600 disabled:bg-gray-400 disabled:cursor-not-allowed'
                      title={cattle.images.length <= 1 ? '최소 1개의 이미지는 유지해야 합니다' : '이미지 삭제'}>
                      <X className='w-4 h-4' />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* 이미지 모달 (원본 + 바운딩박스 시각화) */}
      {selectedImage && (
        <div
          className='fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4'
          onClick={() => setSelectedImage(null)}>
          <div className='relative max-w-5xl w-full max-h-[90vh]'>
            {/* 닫기 버튼 */}
            <button
              onClick={() => setSelectedImage(null)}
              className='absolute -top-10 right-0 p-2 text-white hover:text-gray-300'>
              <X className='w-6 h-6' />
            </button>

            {/* 이미지 컨테이너 */}
            <div className='grid grid-cols-1 md:grid-cols-2 gap-4'>
              {/* 얼굴 crop 이미지 */}
              <div className='bg-gray-900 rounded-lg p-4'>
                <p className='text-white text-sm mb-2 text-center'>얼굴 이미지 (임베딩용)</p>
                <img
                  src={selectedImage}
                  alt='Face crop'
                  className='max-h-[70vh] w-full object-contain rounded'
                  onClick={(e) => e.stopPropagation()}
                />
              </div>

              {/* 원본 + 바운딩박스 시각화 */}
              <div className='bg-gray-900 rounded-lg p-4'>
                <p className='text-white text-sm mb-2 text-center'>원본 + 얼굴 검출 영역</p>
                <img
                  src={getVisImageUrl(selectedImage)}
                  alt='Original with bounding box'
                  className='max-h-[70vh] w-full object-contain rounded'
                  onClick={(e) => e.stopPropagation()}
                  onError={(e) => {
                    // vis 이미지가 없으면 원본 이미지로 대체
                    (e.target as HTMLImageElement).src = selectedImage;
                  }}
                />
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
