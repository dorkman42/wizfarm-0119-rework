'use client';

import { useState } from 'react';
import Link from 'next/link';
import { Camera, Loader2, CheckCircle, AlertCircle, Plus, Zap } from 'lucide-react';
import { ImageDropzone } from '@/components/ImageDropzone';
import { api, RecognitionResult } from '@/lib/api';

export default function RecognizePage() {
  const [files, setFiles] = useState<File[]>([]);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<RecognitionResult[]>([]);
  const [resultImageUrl, setResultImageUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [autoRegister, setAutoRegister] = useState(false);
  const [registeringIndex, setRegisteringIndex] = useState<number | null>(null);

  const handleRecognize = async () => {
    if (files.length === 0) {
      setError('이미지를 선택해주세요');
      return;
    }

    setLoading(true);
    setError(null);
    setResults([]);
    setResultImageUrl(null);

    try {
      const formData = new FormData();
      formData.append('image', files[0]);

      const response = await api.recognize(formData, autoRegister);
      setResults(response.recognitions);
      // base64 시각화 이미지 또는 image_url 사용
      if (response.visualization) {
        setResultImageUrl(`data:image/jpeg;base64,${response.visualization}`);
      } else {
        setResultImageUrl(response.image_url);
      }
    } catch (err: any) {
      setError(err.message || '인식에 실패했습니다');
    } finally {
      setLoading(false);
    }
  };

  const handleRegisterNew = async (index: number, result: RecognitionResult) => {
    if (!result.face_image) {
      alert('얼굴 이미지가 없습니다');
      return;
    }

    setRegisteringIndex(index);
    try {
      const registered = await api.registerNewCattle({
        face_image: result.face_image,
      });

      // 결과 업데이트
      setResults(prev =>
        prev.map((r, i) =>
          i === index
            ? { ...r, cattle_id: registered.cattle_id, name: registered.name, is_new: false, confidence: 1.0 }
            : r
        )
      );
    } catch (err: any) {
      alert(err.message || '등록에 실패했습니다');
    } finally {
      setRegisteringIndex(null);
    }
  };

  const reset = () => {
    setFiles([]);
    setResults([]);
    setResultImageUrl(null);
    setError(null);
  };

  const newCount = results.filter(r => r.is_new).length;

  return (
    <div>
      <h1 className='text-2xl font-bold text-gray-900 mb-8'>소 인식</h1>

      <div className='grid grid-cols-1 lg:grid-cols-2 gap-8'>
        {/* 이미지 업로드 */}
        <div className='card'>
          <div className='card-header'>
            <h2 className='text-lg font-semibold'>이미지 업로드</h2>
            <p className='text-sm text-gray-500'>
              소가 포함된 이미지를 업로드하세요
            </p>
          </div>
          <div className='card-body'>
            <ImageDropzone
              files={files}
              onFilesChange={setFiles}
              multiple={false}
              maxFiles={1}
            />

            {/* 자동 등록 옵션 */}
            <label className='flex items-center mt-4 cursor-pointer'>
              <input
                type='checkbox'
                checked={autoRegister}
                onChange={(e) => setAutoRegister(e.target.checked)}
                className='w-4 h-4 text-blue-600 rounded focus:ring-blue-500'
              />
              <span className='ml-2 text-sm text-gray-700 flex items-center'>
                <Zap className='w-4 h-4 mr-1 text-yellow-500' />
                새로운 소 자동 등록
              </span>
            </label>

            {error && (
              <div className='mt-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-600 text-sm'>
                {error}
              </div>
            )}

            <div className='mt-6 flex space-x-2'>
              <button
                onClick={handleRecognize}
                disabled={loading || files.length === 0}
                className='btn btn-primary flex-1'>
                {loading ? (
                  <>
                    <Loader2 className='w-4 h-4 mr-2 animate-spin inline' />
                    분석 중...
                  </>
                ) : (
                  <>
                    <Camera className='w-4 h-4 mr-2 inline' />
                    인식하기
                  </>
                )}
              </button>
              {(results.length > 0 || resultImageUrl || error) && (
                <button onClick={reset} className='btn btn-secondary'>
                  초기화
                </button>
              )}
            </div>
          </div>
        </div>

        {/* 인식 결과 */}
        <div className='card'>
          <div className='card-header'>
            <h2 className='text-lg font-semibold'>인식 결과</h2>
            {results.length > 0 && (
              <p className='text-sm text-gray-500'>
                {results.length}개 검출 {newCount > 0 && `(새로운 소 ${newCount}개)`}
              </p>
            )}
          </div>
          <div className='card-body'>
            {loading ? (
              <div className='text-center py-12'>
                <Loader2 className='w-8 h-8 animate-spin mx-auto text-gray-400' />
                <p className='text-gray-500 mt-2'>이미지 분석 중...</p>
              </div>
            ) : results.length === 0 && !resultImageUrl ? (
              <div className='text-center py-12 text-gray-500'>
                이미지를 업로드하고 인식 버튼을 눌러주세요
              </div>
            ) : (
              <div className='space-y-4'>
                {/* 결과 이미지 */}
                {resultImageUrl && (
                  <div className='rounded-lg overflow-hidden bg-gray-100 mb-4'>
                    <img
                      src={resultImageUrl}
                      alt='Recognition result'
                      className='w-full'
                    />
                  </div>
                )}

                {/* 인식된 소들 */}
                {results.map((result, index) => (
                  <div
                    key={index}
                    className={`p-4 rounded-lg border ${
                      result.is_new
                        ? 'bg-yellow-50 border-yellow-200'
                        : 'bg-green-50 border-green-200'
                    }`}>
                    <div className='flex items-start'>
                      {result.is_new ? (
                        <AlertCircle className='w-5 h-5 text-yellow-500 mr-3 mt-0.5 flex-shrink-0' />
                      ) : (
                        <CheckCircle className='w-5 h-5 text-green-500 mr-3 mt-0.5 flex-shrink-0' />
                      )}
                      <div className='flex-1 min-w-0'>
                        <div className='flex justify-between items-start'>
                          <div>
                            <h3 className='font-semibold text-gray-900'>
                              {result.is_new
                                ? '새로운 개체'
                                : result.name || 'Unknown'}
                            </h3>
                            {!result.is_new && result.cattle_id && (
                              <p className='text-sm text-gray-500 font-mono truncate'>
                                ID: {result.cattle_id}
                              </p>
                            )}
                          </div>
                          <span
                            className={`text-sm font-medium ml-2 ${
                              result.confidence > 0.7
                                ? 'text-green-600'
                                : result.confidence > 0.4
                                ? 'text-yellow-600'
                                : 'text-red-600'
                            }`}>
                            {(result.confidence * 100).toFixed(1)}%
                          </span>
                        </div>

                        {/* 새로운 소: 즉시 등록 버튼 */}
                        {result.is_new && result.face_image && (
                          <div className='mt-3 flex items-center space-x-2'>
                            <button
                              onClick={() => handleRegisterNew(index, result)}
                              disabled={registeringIndex === index}
                              className='btn btn-sm bg-yellow-500 hover:bg-yellow-600 text-white'>
                              {registeringIndex === index ? (
                                <>
                                  <Loader2 className='w-3 h-3 mr-1 animate-spin inline' />
                                  등록 중...
                                </>
                              ) : (
                                <>
                                  <Plus className='w-3 h-3 mr-1 inline' />
                                  바로 등록
                                </>
                              )}
                            </button>
                            {/* 얼굴 미리보기 */}
                            <img
                              src={`data:image/jpeg;base64,${result.face_image}`}
                              alt='Face'
                              className='w-10 h-10 rounded object-cover'
                            />
                          </div>
                        )}

                        {result.is_new && !result.face_image && (
                          <p className='text-sm text-yellow-700 mt-2'>
                            등록되지 않은 소입니다.
                            <Link
                              href='/cattle/register'
                              className='underline ml-1'>
                              등록하기
                            </Link>
                          </p>
                        )}

                        {!result.is_new && result.cattle_id && (
                          <Link
                            href={`/cattle/${result.cattle_id}`}
                            className='text-sm text-green-700 underline mt-2 inline-block'>
                            상세 보기
                          </Link>
                        )}
                      </div>
                    </div>
                  </div>
                ))}

                {results.length === 0 && resultImageUrl && (
                  <div className='text-center py-4 text-gray-500'>
                    등록된 소와 일치하는 개체가 없습니다
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
