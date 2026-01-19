'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Home, Beef, UserPlus, Camera, Settings } from 'lucide-react';

const menuItems = [
  { href: '/', label: '홈', icon: Home },
  { href: '/cattle', label: '소 관리', icon: Beef },
  { href: '/cattle/register', label: '소 등록', icon: UserPlus },
  { href: '/recognize', label: '소 인식', icon: Camera },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className='fixed left-0 top-0 h-full w-64 bg-white border-r border-gray-200 shadow-sm'>
      {/* 로고 */}
      <div className='p-6 border-b border-gray-200'>
        <Link href='/' className='flex items-center space-x-2'>
          <Beef className='w-8 h-8 text-green-600' />
          <span className='text-xl font-bold text-gray-900'>CattleFace</span>
        </Link>
      </div>

      {/* 메뉴 */}
      <nav className='p-4'>
        <ul className='space-y-2'>
          {menuItems.map((item) => {
            const isActive = pathname === item.href;
            return (
              <li key={item.href}>
                <Link
                  href={item.href}
                  className={`flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors ${
                    isActive
                      ? 'bg-green-100 text-green-700'
                      : 'text-gray-600 hover:bg-gray-100'
                  }`}>
                  <item.icon className='w-5 h-5' />
                  <span className='font-medium'>{item.label}</span>
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>

      {/* 하단 정보 */}
      <div className='absolute bottom-0 left-0 right-0 p-4 border-t border-gray-200'>
        <p className='text-xs text-gray-500 text-center'>
          소 얼굴 인식 시스템 v1.0
        </p>
      </div>
    </aside>
  );
}
