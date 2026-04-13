'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import { GameState, Piece, BOARD_SIZE } from '@/lib/engine/types';
import { initGame, handlePlacement } from '@/lib/engine/game';
import { canPlacePiece } from '@/lib/engine/board';
import { playPlace, playClear, playCombo, playGameOver, playPickup, playInvalid } from '@/lib/sounds';
import GameBoard, { Particle } from './GameBoard';
import PieceTray from './PieceTray';
import ScoreDisplay from './ScoreDisplay';
import GameOverOverlay from './GameOverOverlay';

type DragState = {
  pieceIndex: number;
  piece: Piece;
  clientX: number;
  clientY: number;
};

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

// Vertical offset (px) to lift the dragged piece above the finger on touch
const TOUCH_LIFT_OFFSET = 60;

export default function BlockBlastGame() {
  const [game, setGame] = useState<GameState | null>(null);
  const [boardColors, setBoardColors] = useState<(string | null)[][]>(
    Array.from({ length: BOARD_SIZE }, () => Array(BOARD_SIZE).fill(null))
  );
  const [dragState, setDragState] = useState<DragState | null>(null);
  const [clearAnimation, setClearAnimation] = useState<ClearAnimation | null>(null);
  const [comboPopups, setComboPopups] = useState<ComboPopup[]>([]);
  const [particles, setParticles] = useState<Particle[]>([]);
  const [showConfirm, setShowConfirm] = useState(false);

  const boardRectRef = useRef<DOMRect | null>(null);
  const cellSizeRef = useRef(0);
  const dragOverlayRef = useRef<HTMLDivElement>(null);
  const isTouchRef = useRef(false);

  // Refs to avoid stale closures in window event handlers
  const dragStateRef = useRef<DragState | null>(null);
  const gameRef = useRef<GameState | null>(null);
  const boardColorsRef = useRef(boardColors);

  // Keep refs in sync with state
  dragStateRef.current = dragState;
  gameRef.current = game;
  boardColorsRef.current = boardColors;

  // Initialize game on mount (client-side only)
  useEffect(() => {
    setGame(initGame());
  }, []);

  const handleBoardLayout = useCallback((rect: DOMRect, cellSize: number) => {
    boardRectRef.current = rect;
    cellSizeRef.current = cellSize;
  }, []);

  // Convert client coordinates to board grid position
  const clientToGrid = useCallback(
    (clientX: number, clientY: number, piece: Piece): { row: number; col: number } => {
      const rect = boardRectRef.current;
      const cellSize = cellSizeRef.current;
      if (!rect || cellSize === 0) return { row: -100, col: -100 };

      const pieceW = piece.shape[0].length * cellSize;
      const pieceH = piece.shape.length * cellSize;
      // Apply touch offset — shift the grid target up so the piece
      // lands above the finger, matching the visual overlay position
      const yOffset = isTouchRef.current ? TOUCH_LIFT_OFFSET : 0;
      const x = clientX - rect.left - pieceW / 2;
      const y = (clientY - yOffset) - rect.top - pieceH / 2;

      return {
        row: Math.round(y / cellSize),
        col: Math.round(x / cellSize),
      };
    },
    []
  );

  // Spawn particles from cleared cells
  function spawnClearParticles(clearedRows: number[], clearedCols: number[], colors: (string | null)[][]) {
    const cellSize = cellSizeRef.current;
    const newParticles: Particle[] = [];
    const cells = new Set<string>();

    for (const r of clearedRows) {
      for (let c = 0; c < BOARD_SIZE; c++) cells.add(`${r},${c}`);
    }
    for (const c of clearedCols) {
      for (let r = 0; r < BOARD_SIZE; r++) cells.add(`${r},${c}`);
    }

    for (const key of cells) {
      const [r, c] = key.split(',').map(Number);
      const color = colors[r]?.[c] || '#ffffff';
      const cx = c * cellSize + cellSize / 2;
      const cy = r * cellSize + cellSize / 2;

      // 3-5 particles per cell
      const count = 3 + Math.floor(Math.random() * 3);
      for (let i = 0; i < count; i++) {
        const angle = Math.random() * Math.PI * 2;
        const speed = 40 + Math.random() * 100;
        newParticles.push({
          x: cx + (Math.random() - 0.5) * cellSize * 0.5,
          y: cy + (Math.random() - 0.5) * cellSize * 0.5,
          vx: Math.cos(angle) * speed,
          vy: Math.sin(angle) * speed - 30,
          size: 3 + Math.random() * 4,
          color,
          life: 1,
          startTime: performance.now(),
          duration: 400 + Math.random() * 300,
        });
      }
    }

    setParticles(prev => [...prev, ...newParticles]);
    setTimeout(() => {
      setParticles(prev => prev.filter(p => performance.now() - p.startTime < p.duration));
    }, 800);
  }

  const handleDragStart = useCallback(
    (pieceIndex: number, _offsetX: number, _offsetY: number) => {
      if (!game || game.isGameOver) return;
      const piece = game.currentPieces[pieceIndex];
      if (!piece) return;
      playPickup();
      const newDrag = { pieceIndex, piece, clientX: 0, clientY: 0 };
      dragStateRef.current = newDrag;
      setDragState(newDrag);
    },
    [game]
  );

  const handleDragMove = useCallback(
    (clientX: number, clientY: number) => {
      setDragState(prev => {
        if (!prev) return prev;
        const updated = { ...prev, clientX, clientY };
        dragStateRef.current = updated;
        return updated;
      });

      // Update floating piece overlay position with touch offset
      if (dragOverlayRef.current) {
        const yOffset = isTouchRef.current ? TOUCH_LIFT_OFFSET : 0;
        dragOverlayRef.current.style.left = `${clientX}px`;
        dragOverlayRef.current.style.top = `${clientY - yOffset}px`;
      }
    },
    []
  );

  const handleDragEnd = useCallback(
    (clientX: number, clientY: number) => {
      const currentDrag = dragStateRef.current;
      const currentGame = gameRef.current;
      const currentColors = boardColorsRef.current;

      if (!currentDrag || !currentGame) {
        dragStateRef.current = null;
        setDragState(null);
        return;
      }

      const { row, col } = clientToGrid(clientX, clientY, currentDrag.piece);
      if (canPlacePiece(currentGame.board, currentDrag.piece, row, col)) {
        const newState = handlePlacement(currentGame, currentDrag.pieceIndex, row, col);

        if (newState) {
          // Update board colors
          const newColors = currentColors.map(r => [...r]);
          for (let r = 0; r < currentDrag.piece.shape.length; r++) {
            for (let c = 0; c < currentDrag.piece.shape[r].length; c++) {
              if (currentDrag.piece.shape[r][c]) {
                newColors[row + r][col + c] = currentDrag.piece.color;
              }
            }
          }

          // Check which lines were cleared
          const clearedRows: number[] = [];
          const clearedCols: number[] = [];

          for (let r = 0; r < BOARD_SIZE; r++) {
            if (newColors[r].every(c => c !== null)) clearedRows.push(r);
          }
          for (let c = 0; c < BOARD_SIZE; c++) {
            if (newColors.every(row => row[c] !== null)) clearedCols.push(c);
          }

          const linesCleared = clearedRows.length + clearedCols.length;

          // Spawn particles before clearing colors
          if (linesCleared > 0) {
            spawnClearParticles(clearedRows, clearedCols, newColors);
          }

          // Clear colors for cleared lines
          for (const r of clearedRows) {
            for (let c = 0; c < BOARD_SIZE; c++) newColors[r][c] = null;
          }
          for (const c of clearedCols) {
            for (let r = 0; r < BOARD_SIZE; r++) newColors[r][c] = null;
          }

          // Sound effects
          if (linesCleared > 0) {
            if (newState.combo > 1) {
              playCombo(newState.combo);
            } else {
              playClear();
            }
          } else {
            playPlace();
          }

          // Clear animation + popups
          if (linesCleared > 0) {
            setClearAnimation({
              rows: clearedRows,
              cols: clearedCols,
              startTime: performance.now(),
            });
            setTimeout(() => setClearAnimation(null), 500);

            const cellSize = cellSizeRef.current;
            const centerX = (BOARD_SIZE * cellSize) / 2;
            const centerY = (BOARD_SIZE * cellSize) / 2;

            if (newState.combo > 0) {
              setComboPopups(prev => [
                ...prev,
                {
                  text: `${newState.combo + 1}x COMBO!`,
                  x: centerX,
                  y: centerY - 20,
                  startTime: performance.now(),
                  color: '#ff8800',
                },
              ]);
            } else if (linesCleared >= 2) {
              setComboPopups(prev => [
                ...prev,
                {
                  text: `${linesCleared} LINES!`,
                  x: centerX,
                  y: centerY - 20,
                  startTime: performance.now(),
                  color: '#44dd88',
                },
              ]);
            }

            const pointsEarned = newState.score - (currentGame.score ?? 0);
            if (pointsEarned > 10) {
              setComboPopups(prev => [
                ...prev,
                {
                  text: `+${pointsEarned}`,
                  x: centerX,
                  y: centerY + 15,
                  startTime: performance.now(),
                  color: '#ffffff',
                },
              ]);
            }

            setTimeout(() => {
              setComboPopups(prev => prev.filter(p => performance.now() - p.startTime < 1000));
            }, 1100);
          }

          setBoardColors(newColors);
          setGame(newState);

          // Game over sound
          if (newState.isGameOver) {
            setTimeout(() => playGameOver(), 300);
          }
        }
      } else {
        // Invalid placement — play error sound if over the board
        const rect = boardRectRef.current;
        if (rect && clientX >= rect.left && clientX <= rect.right && clientY >= rect.top && clientY <= rect.bottom) {
          playInvalid();
        }
      }

      dragStateRef.current = null;
      setDragState(null);
    },
    [clientToGrid]
  );

  const handleRestart = useCallback(() => {
    setGame(initGame());
    setBoardColors(
      Array.from({ length: BOARD_SIZE }, () => Array(BOARD_SIZE).fill(null))
    );
    setClearAnimation(null);
    setComboPopups([]);
    setParticles([]);
    setShowConfirm(false);
  }, []);

  const handleNewGameClick = useCallback(() => {
    if (!game) return;
    // If game is over, restart immediately
    if (game.isGameOver) {
      handleRestart();
      return;
    }
    // If game is in progress, show confirmation
    if (game.score > 0) {
      setShowConfirm(true);
    } else {
      handleRestart();
    }
  }, [game, handleRestart]);

  // Detect touch vs mouse for the lift offset
  const handleDragStartWithTouch = useCallback(
    (pieceIndex: number, offsetX: number, offsetY: number) => {
      // isTouchRef is set by PieceTray's pointer event type
      handleDragStart(pieceIndex, offsetX, offsetY);
    },
    [handleDragStart]
  );

  // Expose touch detection to PieceTray via a callback
  const handlePointerType = useCallback((isTouch: boolean) => {
    isTouchRef.current = isTouch;
  }, []);

  // Compute ghost position from drag state
  let ghostPiece: Piece | null = null;
  let ghostRow = -100;
  let ghostCol = -100;
  if (dragState && dragState.clientX !== 0) {
    ghostPiece = dragState.piece;
    const grid = clientToGrid(dragState.clientX, dragState.clientY, dragState.piece);
    ghostRow = grid.row;
    ghostCol = grid.col;
  }

  if (!game) return null;

  return (
    <div className="relative flex flex-col items-center w-full max-w-[520px] px-4 gap-2">
      <div className="flex items-center justify-between w-full max-w-[480px]">
        <ScoreDisplay
          score={game.score}
          highScore={game.highScore}
          combo={game.combo}
        />
      </div>

      <GameBoard
        board={game.board}
        boardColors={boardColors}
        ghostPiece={ghostPiece}
        ghostRow={ghostRow}
        ghostCol={ghostCol}
        clearAnimation={clearAnimation}
        comboPopups={comboPopups}
        particles={particles}
        onBoardLayout={handleBoardLayout}
      />

      <PieceTray
        pieces={game.currentPieces}
        onDragStart={handleDragStartWithTouch}
        onDragMove={handleDragMove}
        onDragEnd={handleDragEnd}
        draggingIndex={dragState?.pieceIndex ?? null}
        onPointerType={handlePointerType}
      />

      {/* New Game button */}
      <button
        onClick={handleNewGameClick}
        className="mt-1 px-4 py-1.5 rounded-lg text-xs uppercase tracking-wider font-semibold transition-all duration-150 hover:brightness-125 active:scale-95 cursor-pointer"
        style={{
          background: 'rgba(255, 255, 255, 0.06)',
          color: 'var(--text-secondary)',
          border: '1px solid rgba(255, 255, 255, 0.08)',
        }}
      >
        New Game
      </button>

      {/* Floating drag overlay */}
      {dragState && dragState.clientX !== 0 && (
        <div
          ref={dragOverlayRef}
          className="fixed pointer-events-none z-40"
          style={{
            left: dragState.clientX,
            top: dragState.clientY - (isTouchRef.current ? TOUCH_LIFT_OFFSET : 0),
            transform: 'translate(-50%, -50%) scale(1.05)',
            filter: 'drop-shadow(0 4px 12px rgba(0,0,0,0.5))',
          }}
        >
          <div
            style={{
              display: 'grid',
              gridTemplateRows: `repeat(${dragState.piece.shape.length}, ${cellSizeRef.current}px)`,
              gridTemplateColumns: `repeat(${dragState.piece.shape[0].length}, ${cellSizeRef.current}px)`,
              gap: '1px',
            }}
          >
            {dragState.piece.shape.flatMap((row, r) =>
              row.map((filled, c) => (
                <div
                  key={`${r}-${c}`}
                  style={{
                    width: cellSizeRef.current,
                    height: cellSizeRef.current,
                    borderRadius: 3,
                    backgroundColor: filled ? dragState.piece.color : 'transparent',
                  }}
                />
              ))
            )}
          </div>
        </div>
      )}

      {/* New game confirmation dialog */}
      {showConfirm && (
        <div
          className="absolute inset-0 flex items-center justify-center z-50"
          style={{ backgroundColor: 'rgba(10, 10, 15, 0.85)' }}
        >
          <div
            className="flex flex-col items-center gap-4 p-6 rounded-2xl max-w-[280px] w-full mx-4"
            style={{ backgroundColor: '#1a1a2e', animation: 'popIn 0.2s ease-out' }}
          >
            <h3 className="text-lg font-bold">New Game?</h3>
            <p className="text-sm text-center" style={{ color: 'var(--text-secondary)' }}>
              Your current progress will be lost.
            </p>
            <div className="flex gap-3 w-full">
              <button
                onClick={() => setShowConfirm(false)}
                className="flex-1 py-2 rounded-lg font-semibold text-sm cursor-pointer transition-all hover:brightness-125"
                style={{ background: 'rgba(255,255,255,0.08)', color: 'var(--text-secondary)' }}
              >
                Cancel
              </button>
              <button
                onClick={handleRestart}
                className="flex-1 py-2 rounded-lg font-semibold text-sm cursor-pointer transition-all hover:brightness-110"
                style={{ background: 'linear-gradient(135deg, #4466ff, #8844ff)', color: '#fff' }}
              >
                Restart
              </button>
            </div>
          </div>
        </div>
      )}

      {game.isGameOver && (
        <GameOverOverlay
          score={game.score}
          highScore={game.highScore}
          totalLinesCleared={game.totalLinesCleared}
          highestCombo={game.highestCombo}
          onRestart={handleRestart}
        />
      )}
    </div>
  );
}
