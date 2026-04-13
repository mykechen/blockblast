'use client';

import { useRef, useEffect, useCallback } from 'react';
import { BoardState, Piece, BOARD_SIZE } from '@/lib/engine/types';
import { canPlacePiece } from '@/lib/engine/board';

type ClearAnimation = {
  rows: number[];
  cols: number[];
  startTime: number;
};

type ComboPopup = {
  text: string;
  x: number;
  y: number;
  startTime: number;
  color: string;
};

type GameBoardProps = {
  board: BoardState;
  boardColors: (string | null)[][];
  ghostPiece: Piece | null;
  ghostRow: number;
  ghostCol: number;
  clearAnimation: ClearAnimation | null;
  comboPopups: ComboPopup[];
  onBoardLayout: (rect: DOMRect, cellSize: number) => void;
};

const BG_BOARD = '#1a1a2e';
const GRID_LINE = '#2a2a3e';
const GHOST_VALID = 'rgba(68, 221, 136, 0.35)';
const GHOST_INVALID = 'rgba(255, 68, 85, 0.35)';
const CLEAR_FLASH = '#ffffff';
const CELL_RADIUS = 3;
const CLEAR_DURATION = 500;
const POPUP_DURATION = 1000;

export default function GameBoard({
  board,
  boardColors,
  ghostPiece,
  ghostRow,
  ghostCol,
  clearAnimation,
  comboPopups,
  onBoardLayout,
}: GameBoardProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const rafRef = useRef<number>(0);
  const dprRef = useRef(1);

  const getCellSize = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return 0;
    return canvas.width / dprRef.current / BOARD_SIZE;
  }, []);

  // Report layout to parent for drag-and-drop coordinate mapping
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const observer = new ResizeObserver(() => {
      const rect = container.getBoundingClientRect();
      const cellSize = rect.width / BOARD_SIZE;
      onBoardLayout(rect, cellSize);
    });

    observer.observe(container);
    return () => observer.disconnect();
  }, [onBoardLayout]);

  // Resize canvas to match container
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const observer = new ResizeObserver(() => {
      const rect = container.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      dprRef.current = dpr;
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;
    });

    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  // Render loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animating = true;

    function render() {
      if (!animating || !ctx || !canvas) return;

      const dpr = dprRef.current;
      const w = canvas.width / dpr;
      const h = canvas.height / dpr;
      const cellSize = w / BOARD_SIZE;
      const padding = 1;

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);

      // Draw board background
      ctx.fillStyle = BG_BOARD;
      roundRect(ctx, 0, 0, w, h, 8);
      ctx.fill();

      // Draw grid lines
      ctx.strokeStyle = GRID_LINE;
      ctx.lineWidth = 0.5;
      for (let i = 1; i < BOARD_SIZE; i++) {
        ctx.beginPath();
        ctx.moveTo(i * cellSize, 0);
        ctx.lineTo(i * cellSize, h);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(0, i * cellSize);
        ctx.lineTo(w, i * cellSize);
        ctx.stroke();
      }

      // Determine which cells are being cleared (for animation)
      const clearingCells = new Set<string>();
      let clearProgress = 0;
      if (clearAnimation) {
        const elapsed = performance.now() - clearAnimation.startTime;
        clearProgress = Math.min(elapsed / CLEAR_DURATION, 1);
        for (const r of clearAnimation.rows) {
          for (let c = 0; c < BOARD_SIZE; c++) {
            clearingCells.add(`${r},${c}`);
          }
        }
        for (const c of clearAnimation.cols) {
          for (let r = 0; r < BOARD_SIZE; r++) {
            clearingCells.add(`${r},${c}`);
          }
        }
      }

      // Draw placed cells
      for (let r = 0; r < BOARD_SIZE; r++) {
        for (let c = 0; c < BOARD_SIZE; c++) {
          if (!board[r][c]) continue;
          const key = `${r},${c}`;
          const isClearing = clearingCells.has(key);

          if (isClearing) {
            if (clearProgress < 0.3) {
              const flashAlpha = 1 - (clearProgress / 0.3) * 0.5;
              ctx.fillStyle = CLEAR_FLASH;
              ctx.globalAlpha = flashAlpha;
            } else {
              const fadeProgress = (clearProgress - 0.3) / 0.7;
              const color = boardColors[r][c] || '#4466ff';
              ctx.fillStyle = color;
              ctx.globalAlpha = 1 - fadeProgress;
            }
          } else {
            ctx.fillStyle = boardColors[r][c] || '#4466ff';
            ctx.globalAlpha = 1;
          }

          roundRect(
            ctx,
            c * cellSize + padding,
            r * cellSize + padding,
            cellSize - padding * 2,
            cellSize - padding * 2,
            CELL_RADIUS
          );
          ctx.fill();
          ctx.globalAlpha = 1;
        }
      }

      // Draw ghost preview
      if (ghostPiece && ghostRow >= -ghostPiece.shape.length && ghostCol >= -ghostPiece.shape[0].length) {
        const isValid = canPlacePiece(board, ghostPiece, ghostRow, ghostCol);
        ctx.fillStyle = isValid ? GHOST_VALID : GHOST_INVALID;

        for (let r = 0; r < ghostPiece.shape.length; r++) {
          for (let c = 0; c < ghostPiece.shape[r].length; c++) {
            if (!ghostPiece.shape[r][c]) continue;
            const br = ghostRow + r;
            const bc = ghostCol + c;
            if (br < 0 || br >= BOARD_SIZE || bc < 0 || bc >= BOARD_SIZE) continue;

            roundRect(
              ctx,
              bc * cellSize + padding,
              br * cellSize + padding,
              cellSize - padding * 2,
              cellSize - padding * 2,
              CELL_RADIUS
            );
            ctx.fill();
          }
        }
      }

      // Draw combo popups
      const now = performance.now();
      for (const popup of comboPopups) {
        const elapsed = now - popup.startTime;
        if (elapsed > POPUP_DURATION) continue;
        const progress = elapsed / POPUP_DURATION;
        const yOffset = progress * 40;
        const alpha = 1 - progress;

        ctx.globalAlpha = alpha;
        ctx.fillStyle = popup.color;
        ctx.font = 'bold 18px var(--font-inter), Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(popup.text, popup.x, popup.y - yOffset);
        ctx.globalAlpha = 1;
      }

      const needsAnimation =
        clearAnimation !== null ||
        comboPopups.some(p => now - p.startTime < POPUP_DURATION);

      if (needsAnimation) {
        rafRef.current = requestAnimationFrame(render);
      }
    }

    render();
    rafRef.current = requestAnimationFrame(render);

    return () => {
      animating = false;
      cancelAnimationFrame(rafRef.current);
    };
  }, [board, boardColors, ghostPiece, ghostRow, ghostCol, clearAnimation, comboPopups]);

  return (
    <div
      ref={containerRef}
      className="w-full aspect-square max-w-[480px] rounded-lg overflow-hidden"
    >
      <canvas ref={canvasRef} className="w-full h-full block" />
    </div>
  );
}

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number
) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}
