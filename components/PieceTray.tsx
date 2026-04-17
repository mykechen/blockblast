'use client';

import { useRef, useCallback } from 'react';
import { Piece } from '@/lib/engine/types';

type PieceTrayProps = {
  pieces: (Piece | null)[];
  onDragStart: (pieceIndex: number, offsetX: number, offsetY: number) => void;
  onDragMove: (clientX: number, clientY: number) => void;
  onDragEnd: (clientX: number, clientY: number) => void;
  draggingIndex: number | null;
  onPointerType?: (isTouch: boolean) => void;
  suggestedIndex?: number | null;
};

const TRAY_CELL_SIZE = 28;

function PiecePreview({
  piece,
  index,
  onDragStart,
  isDragging,
}: {
  piece: Piece;
  index: number;
  onDragStart: (pieceIndex: number, offsetX: number, offsetY: number, isTouch: boolean) => void;
  isDragging: boolean;
}) {
  const ref = useRef<HTMLDivElement>(null);

  const handlePointerDown = useCallback(
    (e: React.PointerEvent) => {
      e.preventDefault();
      const el = ref.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      const pieceW = piece.shape[0].length * TRAY_CELL_SIZE;
      const pieceH = piece.shape.length * TRAY_CELL_SIZE;
      const offsetX = e.clientX - rect.left - pieceW / 2;
      const offsetY = e.clientY - rect.top - pieceH / 2;
      const isTouch = e.pointerType === 'touch';
      onDragStart(index, offsetX, offsetY, isTouch);
    },
    [piece, index, onDragStart]
  );

  const rows = piece.shape.length;
  const cols = piece.shape[0].length;

  return (
    <div
      ref={ref}
      className="flex items-center justify-center p-2 cursor-grab active:cursor-grabbing transition-all duration-150"
      style={{
        opacity: isDragging ? 0.3 : 1,
        transform: isDragging ? 'scale(0.9)' : 'scale(1)',
        touchAction: 'none',
      }}
      onPointerDown={handlePointerDown}
    >
      <div
        style={{
          display: 'grid',
          gridTemplateRows: `repeat(${rows}, ${TRAY_CELL_SIZE}px)`,
          gridTemplateColumns: `repeat(${cols}, ${TRAY_CELL_SIZE}px)`,
          gap: '1px',
        }}
      >
        {piece.shape.flatMap((row, r) =>
          row.map((filled, c) => (
            <div
              key={`${r}-${c}`}
              style={{
                width: TRAY_CELL_SIZE,
                height: TRAY_CELL_SIZE,
                borderRadius: 3,
                backgroundColor: filled ? piece.color : 'transparent',
              }}
            />
          ))
        )}
      </div>
    </div>
  );
}

export default function PieceTray({
  pieces,
  onDragStart,
  onDragMove,
  onDragEnd,
  draggingIndex,
  onPointerType,
  suggestedIndex,
}: PieceTrayProps) {
  const handlePointerMove = useCallback(
    (e: PointerEvent) => {
      e.preventDefault();
      onDragMove(e.clientX, e.clientY);
    },
    [onDragMove]
  );

  const handlePointerUp = useCallback(
    (e: PointerEvent) => {
      e.preventDefault();
      onDragEnd(e.clientX, e.clientY);
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', handlePointerUp);
    },
    [onDragEnd, handlePointerMove]
  );

  const wrappedDragStart = useCallback(
    (pieceIndex: number, offsetX: number, offsetY: number, isTouch: boolean) => {
      onPointerType?.(isTouch);
      onDragStart(pieceIndex, offsetX, offsetY);
      window.addEventListener('pointermove', handlePointerMove);
      window.addEventListener('pointerup', handlePointerUp);
    },
    [onDragStart, handlePointerMove, handlePointerUp, onPointerType]
  );

  return (
    <div className="flex items-center justify-center gap-4 w-full max-w-[480px] py-4">
      {pieces.map((piece, i) => (
        <div
          key={i}
          className="flex-1 flex items-center justify-center min-h-[100px] rounded-xl transition-all duration-300"
          style={{
            boxShadow: suggestedIndex === i
              ? '0 0 16px rgba(68,255,136,0.4), inset 0 0 8px rgba(68,255,136,0.1)'
              : 'none',
            border: suggestedIndex === i
              ? '1px solid rgba(68,255,136,0.35)'
              : '1px solid transparent',
          }}
        >
          {piece ? (
            <PiecePreview
              piece={piece}
              index={i}
              onDragStart={wrappedDragStart}
              isDragging={draggingIndex === i}
            />
          ) : (
            <div className="w-8 h-8 rounded opacity-10" style={{ background: 'var(--grid-line)' }} />
          )}
        </div>
      ))}
    </div>
  );
}
