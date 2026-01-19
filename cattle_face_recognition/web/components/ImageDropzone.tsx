'use client';

import { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, X, Image as ImageIcon } from 'lucide-react';

interface ImageDropzoneProps {
  files: File[];
  onFilesChange: (files: File[]) => void;
  multiple?: boolean;
  maxFiles?: number;
}

export function ImageDropzone({
  files,
  onFilesChange,
  multiple = true,
  maxFiles = 10,
}: ImageDropzoneProps) {
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (multiple) {
        const newFiles = [...files, ...acceptedFiles].slice(0, maxFiles);
        onFilesChange(newFiles);
      } else {
        onFilesChange(acceptedFiles.slice(0, 1));
      }
    },
    [files, onFilesChange, multiple, maxFiles]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpg', '.jpeg', '.png', '.webp'],
    },
    multiple,
    maxFiles,
  });

  const removeFile = (index: number) => {
    const newFiles = files.filter((_, i) => i !== index);
    onFilesChange(newFiles);
  };

  return (
    <div>
      {/* 드롭존 */}
      <div
        {...getRootProps()}
        className={`dropzone ${isDragActive ? 'active' : ''}`}>
        <input {...getInputProps()} />
        <Upload className='w-12 h-12 mx-auto text-gray-400 mb-4' />
        {isDragActive ? (
          <p className='text-blue-600'>파일을 여기에 놓으세요</p>
        ) : (
          <div>
            <p className='text-gray-600'>
              클릭하거나 파일을 드래그하여 업로드
            </p>
            <p className='text-sm text-gray-400 mt-1'>
              JPG, PNG, WEBP (최대 {maxFiles}개)
            </p>
          </div>
        )}
      </div>

      {/* 선택된 파일 목록 */}
      {files.length > 0 && (
        <div className='mt-4 grid grid-cols-2 md:grid-cols-4 gap-4'>
          {files.map((file, index) => (
            <div key={index} className='relative group'>
              <div className='aspect-square rounded-lg overflow-hidden bg-gray-100'>
                <img
                  src={URL.createObjectURL(file)}
                  alt={file.name}
                  className='w-full h-full object-cover'
                />
              </div>
              <button
                type='button'
                onClick={() => removeFile(index)}
                className='absolute top-2 right-2 p-1 bg-red-500 text-white rounded-full opacity-0 group-hover:opacity-100 transition-opacity'>
                <X className='w-4 h-4' />
              </button>
              <p className='text-xs text-gray-500 truncate mt-1'>{file.name}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
