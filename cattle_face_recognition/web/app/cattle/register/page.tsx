'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { ArrowLeft, Save, Loader2, Eye } from 'lucide-react';
import Link from 'next/link';
import { ImageDropzone } from '@/components/ImageDropzone';
import { api } from '@/lib/api';

export default function RegisterCattlePage() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [name, setName] = useState('');
  const [breed, setBreed] = useState('');
  const [age, setAge] = useState('');
  const [notes, setNotes] = useState('');
  const [files, setFiles] = useState<File[]>([]);
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const [detectionCount, setDetectionCount] = useState<number | null>(null);

  const handlePreview = async () => {
    if (files.length === 0) {
      setError('이미지를 먼저 업로드해주세요');
      return;
    }

    setPreviewLoading(true);
    setError(null);
    setPreviewImage(null);
    setDetectionCount(null);

    try {
      const formData = new FormData();
      formData.append('image', files[0]);

      const response = await api.recognize(formData);
      if (response.visualization) {
        setPreviewImage(`data:image/jpeg;base64,${response.visualization}`);
      }
      setDetectionCount(response.detections.length);
    } catch (err: any) {
      setError(err.message || '미리보기에 실패했습니다');
    } finally {
      setPreviewLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!name.trim()) {
      setError('이름을 입력해주세요');
      return;
    }
    if (files.length === 0) {
      setError('최소 1개의 이미지를 업로드해주세요');
      return;
    }

    setLoading(true);

    try {
      const formData = new FormData();
      formData.append('name', name.trim());
      if (breed) formData.append('breed', breed);
      if (age) formData.append('age', age);
      if (notes) formData.append('notes', notes);
      files.forEach((file) => formData.append('images', file));

      const result = await api.createCattle(formData);
      router.push(`/cattle/${result.cattle_id}`);
    } catch (err: any) {
      setError(err.message || '등록에 실패했습니다');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      {/* 헤더 */}
      <div className='flex items-center space-x-4 mb-8'>
        <Link href='/cattle' className='p-2 hover:bg-gray-100 rounded-lg'>
          <ArrowLeft className='w-5 h-5' />
        </Link>
        <h1 className='text-2xl font-bold text-gray-900'>새 소 등록</h1>
      </div>

      <form onSubmit={handleSubmit}>
        <div className='grid grid-cols-1 lg:grid-cols-2 gap-8'>
          {/* 기본 정보 */}
          <div className='card'>
            <div className='card-header'>
              <h2 className='text-lg font-semibold'>기본 정보</h2>
            </div>
            <div className='card-body space-y-4'>
              <div>
                <label className='block text-sm font-medium text-gray-700 mb-1'>
                  이름 <span className='text-red-500'>*</span>
                </label>
                <input
                  type='text'
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className='input'
                  placeholder='소 이름을 입력하세요'
                />
              </div>

              <div>
                <label className='block text-sm font-medium text-gray-700 mb-1'>
                  품종
                </label>
                <input
                  type='text'
                  value={breed}
                  onChange={(e) => setBreed(e.target.value)}
                  className='input'
                  placeholder='예: 한우, 홀스타인'
                />
              </div>

              <div>
                <label className='block text-sm font-medium text-gray-700 mb-1'>
                  나이
                </label>
                <input
                  type='number'
                  value={age}
                  onChange={(e) => setAge(e.target.value)}
                  className='input'
                  placeholder='나이 (개월)'
                  min='0'
                />
              </div>

              <div>
                <label className='block text-sm font-medium text-gray-700 mb-1'>
                  메모
                </label>
                <textarea
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  className='input'
                  rows={3}
                  placeholder='추가 정보를 입력하세요'
                />
              </div>
            </div>
          </div>

          {/* 이미지 업로드 */}
          <div className='card'>
            <div className='card-header'>
              <h2 className='text-lg font-semibold'>
                얼굴 이미지 <span className='text-red-500'>*</span>
              </h2>
              <p className='text-sm text-gray-500'>
                소의 얼굴이 잘 보이는 이미지를 업로드하세요
              </p>
            </div>
            <div className='card-body'>
              <ImageDropzone
                files={files}
                onFilesChange={(newFiles) => {
                  setFiles(newFiles);
                  setPreviewImage(null);
                  setDetectionCount(null);
                }}
                multiple={true}
                maxFiles={10}
              />

              {files.length > 0 && (
                <button
                  type='button'
                  onClick={handlePreview}
                  disabled={previewLoading}
                  className='btn btn-secondary mt-4 w-full'>
                  {previewLoading ? (
                    <>
                      <Loader2 className='w-4 h-4 mr-2 animate-spin inline' />
                      검출 중...
                    </>
                  ) : (
                    <>
                      <Eye className='w-4 h-4 mr-2 inline' />
                      얼굴 검출 미리보기
                    </>
                  )}
                </button>
              )}

              {/* 미리보기 결과 */}
              {previewImage && (
                <div className='mt-4'>
                  <div className='rounded-lg overflow-hidden bg-gray-100 mb-2'>
                    <img
                      src={previewImage}
                      alt='Detection preview'
                      className='w-full'
                    />
                  </div>
                  <p className={`text-sm ${detectionCount && detectionCount > 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {detectionCount && detectionCount > 0
                      ? `✓ ${detectionCount}개의 소 얼굴이 검출되었습니다`
                      : '소 얼굴이 검출되지 않았습니다. 다른 이미지를 사용해주세요.'}
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* 에러 메시지 */}
        {error && (
          <div className='mt-6 p-4 bg-red-50 border border-red-200 rounded-lg text-red-600'>
            {error}
          </div>
        )}

        {/* 제출 버튼 */}
        <div className='mt-8 flex justify-end space-x-4'>
          <Link href='/cattle' className='btn btn-secondary'>
            취소
          </Link>
          <button type='submit' className='btn btn-primary' disabled={loading}>
            {loading ? (
              <>
                <Loader2 className='w-4 h-4 mr-2 animate-spin inline' />
                등록 중...
              </>
            ) : (
              <>
                <Save className='w-4 h-4 mr-2 inline' />
                등록하기
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
}
