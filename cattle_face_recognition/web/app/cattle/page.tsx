'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { Plus, Search, Trash2, Edit, Eye, Loader2 } from 'lucide-react';
import { api, Cattle } from '@/lib/api';

export default function CattleListPage() {
  const [cattle, setCattle] = useState<Cattle[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [deleteId, setDeleteId] = useState<string | null>(null);

  useEffect(() => {
    loadCattle();
  }, []);

  const loadCattle = async () => {
    try {
      const data = await api.getCattleList();
      setCattle(data.cattle);
    } catch (error) {
      console.error('목록 로드 실패:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (cattleId: string) => {
    if (!confirm('정말 삭제하시겠습니까?')) return;

    setDeleteId(cattleId);
    try {
      await api.deleteCattle(cattleId);
      setCattle(cattle.filter((c) => c.cattle_id !== cattleId));
    } catch (error) {
      console.error('삭제 실패:', error);
      alert('삭제에 실패했습니다');
    } finally {
      setDeleteId(null);
    }
  };

  const filteredCattle = cattle.filter(
    (c) =>
      c.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      c.breed?.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div>
      {/* 헤더 */}
      <div className='flex justify-between items-center mb-8'>
        <h1 className='text-2xl font-bold text-gray-900'>소 관리</h1>
        <Link href='/cattle/register' className='btn btn-primary'>
          <Plus className='w-4 h-4 mr-2 inline' />
          새 소 등록
        </Link>
      </div>

      {/* 검색 */}
      <div className='card mb-6'>
        <div className='card-body'>
          <div className='relative'>
            <Search className='absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5' />
            <input
              type='text'
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className='input pl-10'
              placeholder='이름 또는 품종으로 검색...'
            />
          </div>
        </div>
      </div>

      {/* 목록 */}
      {loading ? (
        <div className='text-center py-12'>
          <Loader2 className='w-8 h-8 animate-spin mx-auto text-gray-400' />
          <p className='text-gray-500 mt-2'>로딩 중...</p>
        </div>
      ) : filteredCattle.length === 0 ? (
        <div className='text-center py-12 card'>
          <p className='text-gray-500'>
            {searchTerm ? '검색 결과가 없습니다' : '등록된 소가 없습니다'}
          </p>
          <Link href='/cattle/register' className='btn btn-primary mt-4'>
            <Plus className='w-4 h-4 mr-2 inline' />
            첫 소 등록하기
          </Link>
        </div>
      ) : (
        <div className='grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6'>
          {filteredCattle.map((item) => (
            <div key={item.cattle_id} className='card hover:shadow-lg transition-shadow'>
              {/* 이미지 */}
              <div className='aspect-video bg-gray-100 relative'>
                {item.images.length > 0 ? (
                  <img
                    src={item.images[0]}
                    alt={item.name}
                    className='w-full h-full object-cover'
                  />
                ) : (
                  <div className='flex items-center justify-center h-full text-gray-400'>
                    이미지 없음
                  </div>
                )}
                <div className='absolute top-2 right-2 bg-black/50 text-white text-xs px-2 py-1 rounded'>
                  {item.num_images}장
                </div>
              </div>

              {/* 정보 */}
              <div className='p-4'>
                <h3 className='text-lg font-semibold text-gray-900'>
                  {item.name}
                </h3>
                <div className='text-sm text-gray-500 mt-1 space-y-1'>
                  {item.breed && <p>품종: {item.breed}</p>}
                  {item.age && <p>나이: {item.age}개월</p>}
                </div>

                {/* 액션 버튼 */}
                <div className='flex space-x-2 mt-4'>
                  <Link
                    href={`/cattle/${item.cattle_id}`}
                    className='btn btn-secondary flex-1 text-center text-sm'>
                    <Eye className='w-4 h-4 mr-1 inline' />
                    상세
                  </Link>
                  <button
                    onClick={() => handleDelete(item.cattle_id)}
                    disabled={deleteId === item.cattle_id}
                    className='btn btn-danger text-sm'>
                    {deleteId === item.cattle_id ? (
                      <Loader2 className='w-4 h-4 animate-spin' />
                    ) : (
                      <Trash2 className='w-4 h-4' />
                    )}
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
