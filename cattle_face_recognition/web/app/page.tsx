'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { Beef, Camera, Users, TrendingUp } from 'lucide-react';
import { api } from '@/lib/api';

interface Stats {
  total_cattle: number;
  total_images: number;
}

export default function HomePage() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadStats();
  }, []);

  const loadStats = async () => {
    try {
      const data = await api.getStats();
      setStats(data);
    } catch (error) {
      console.error('통계 로드 실패:', error);
    } finally {
      setLoading(false);
    }
  };

  const statCards = [
    {
      title: '등록된 소',
      value: stats?.total_cattle ?? 0,
      icon: Beef,
      color: 'bg-green-500',
      href: '/cattle',
    },
    {
      title: '총 이미지',
      value: stats?.total_images ?? 0,
      icon: Camera,
      color: 'bg-blue-500',
      href: '/cattle',
    },
  ];

  return (
    <div>
      <h1 className='text-3xl font-bold text-gray-900 mb-8'>
        소 얼굴 인식 시스템
      </h1>

      {/* 통계 카드 */}
      <div className='grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8'>
        {statCards.map((card) => (
          <Link key={card.title} href={card.href}>
            <div className='card hover:shadow-lg transition-shadow cursor-pointer'>
              <div className='card-body flex items-center'>
                <div className={`${card.color} p-3 rounded-lg`}>
                  <card.icon className='w-6 h-6 text-white' />
                </div>
                <div className='ml-4'>
                  <p className='text-sm text-gray-500'>{card.title}</p>
                  <p className='text-2xl font-bold text-gray-900'>
                    {loading ? '-' : card.value}
                  </p>
                </div>
              </div>
            </div>
          </Link>
        ))}
      </div>

      {/* 빠른 작업 */}
      <div className='card'>
        <div className='card-header'>
          <h2 className='text-lg font-semibold'>빠른 작업</h2>
        </div>
        <div className='card-body'>
          <div className='grid grid-cols-1 md:grid-cols-3 gap-4'>
            <Link href='/cattle/register'>
              <div className='p-6 border-2 border-dashed border-gray-300 rounded-lg hover:border-blue-500 hover:bg-blue-50 transition-colors cursor-pointer text-center'>
                <Users className='w-12 h-12 mx-auto text-gray-400 mb-2' />
                <h3 className='font-medium text-gray-900'>새 소 등록</h3>
                <p className='text-sm text-gray-500'>
                  새로운 소를 시스템에 등록합니다
                </p>
              </div>
            </Link>

            <Link href='/recognize'>
              <div className='p-6 border-2 border-dashed border-gray-300 rounded-lg hover:border-green-500 hover:bg-green-50 transition-colors cursor-pointer text-center'>
                <Camera className='w-12 h-12 mx-auto text-gray-400 mb-2' />
                <h3 className='font-medium text-gray-900'>소 인식</h3>
                <p className='text-sm text-gray-500'>
                  이미지에서 소를 인식합니다
                </p>
              </div>
            </Link>

            <Link href='/cattle'>
              <div className='p-6 border-2 border-dashed border-gray-300 rounded-lg hover:border-purple-500 hover:bg-purple-50 transition-colors cursor-pointer text-center'>
                <TrendingUp className='w-12 h-12 mx-auto text-gray-400 mb-2' />
                <h3 className='font-medium text-gray-900'>소 관리</h3>
                <p className='text-sm text-gray-500'>
                  등록된 소를 조회하고 관리합니다
                </p>
              </div>
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
