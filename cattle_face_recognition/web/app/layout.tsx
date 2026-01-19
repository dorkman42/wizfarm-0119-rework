import type { Metadata } from 'next';
import './globals.css';
import { Sidebar } from '@/components/Sidebar';

export const metadata: Metadata = {
  title: '소 얼굴 인식 시스템',
  description: '소 개체 등록 및 인식 관리 시스템',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang='ko'>
      <body className='min-h-screen bg-gray-50'>
        <div className='flex'>
          <Sidebar />
          <main className='flex-1 ml-64 p-8'>{children}</main>
        </div>
      </body>
    </html>
  );
}
