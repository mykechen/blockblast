'use client';

import { useRef, useEffect, useCallback, useState } from 'react';
import { BoardState, Piece, BOARD_SIZE } from '@/lib/engine/types';
import { canPlacePiece, placePiece, findCompletedLines } from '@/lib/engine/board';

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
const GRID_LINE = '#2e2e50';
const GHOST_VALID = 'rgba(68, 221, 136, 0.35)';
const GHOST_INVALID = 'rgba(255, 68, 85, 0.35)';
const CLEAR_FLASH = '#ffffff';
const LINE_PREVIEW_COLOR = 'rgba(255, 204, 0, 0.15)';
const LINE_PREVIEW_BORDER = 'rgba(255, 204, 0, 0.5)';
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
  const [canvasReady, setCanvasReady] = useState(false);

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
      setCanvasReady(true);
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
      ctx.lineWidth = 1;
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

      // Draw line-completion preview (highlight rows/cols that will clear)
      if (ghostPiece && ghostRow >= 0 && ghostCol >= 0) {
        const isValid = canPlacePiece(board, ghostPiece, ghostRow, ghostCol);
        if (isValid) {
          // Simulate placement to find what lines would clear
          const simBoard = placePiece(board, ghostPiece, ghostRow, ghostCol);
          const { rows: previewRows, cols: previewCols } = findCompletedLines(simBoard);

          if (previewRows.length > 0 || previewCols.length > 0) {
            // Draw a glowing highlight over the rows/cols that will clear
            const pulse = 0.5 + 0.5 * Math.sin(performance.now() / 200);
            const alpha = 0.1 + pulse * 0.12;

            ctx.fillStyle = LINE_PREVIEW_COLOR;
            ctx.globalAlpha = alpha + 0.1;

            for (const r of previewRows) {
              roundRect(ctx, padding, r * cellSize + padding, w - padding * 2, cellSize - padding * 2, 2);
              ctx.fill();
            }
            for (const c of previewCols) {
              roundRect(ctx, c * cellSize + padding, padding, cellSize - padding * 2, h - padding * 2, 2);
              ctx.fill();
            }

            // Draw border lines on those rows/cols
            ctx.strokeStyle = LINE_PREVIEW_BORDER;
            ctx.lineWidth = 1.5;
            ctx.globalAlpha = 0.4 + pulse * 0.3;

            for (const r of previewRows) {
              ctx.beginPath();
              ctx.moveTo(0, r * cellSize);
              ctx.lineTo(w, r * cellSize);
              ctx.stroke();
              ctx.beginPath();
              ctx.moveTo(0, (r + 1) * cellSize);
              ctx.lineTo(w, (r + 1) * cellSize);
              ctx.stroke();
            }
            for (const c of previewCols) {
              ctx.beginPath();
              ctx.moveTo(c * cellSize, 0);
              ctx.lineTo(c * cellSize, h);
              ctx.stroke();
              ctx.beginPath();
              ctx.moveTo((c + 1) * cellSize, 0);
              ctx.lineTo((c + 1) * cellSize, h);
              ctx.stroke();
            }

            ctx.globalAlpha = 1;

            // Need continuous animation for the pulse
            rafRef.current = requestAnimationFrame(render);
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
  }, [board, boardColors, ghostPiece, ghostRow, ghostCol, clearAnimation, comboPopups, canvasReady]);

  return (
    <div
      ref={containerRef}
      className="w-full aspect-square max-w-[480px] rounded-lg overflow-hidden border border-[#2a2a4e]"
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
