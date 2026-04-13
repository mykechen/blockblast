'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import { GameState, Piece, BOARD_SIZE } from '@/lib/engine/types';
import { initGame, handlePlacement } from '@/lib/engine/game';
import { canPlacePiece } from '@/lib/engine/board';
import GameBoard from './GameBoard';
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

export default function BlockBlastGame() {
  const [game, setGame] = useState<GameState | null>(null);
  const [boardColors, setBoardColors] = useState<(string | null)[][]>(
    Array.from({ length: BOARD_SIZE }, () => Array(BOARD_SIZE).fill(null))
  );
  const [dragState, setDragState] = useState<DragState | null>(null);
  const [clearAnimation, setClearAnimation] = useState<ClearAnimation | null>(null);
  const [comboPopups, setComboPopups] = useState<ComboPopup[]>([]);

  const boardRectRef = useRef<DOMRect | null>(null);
  const cellSizeRef = useRef(0);
  const dragOverlayRef = useRef<HTMLDivElement>(null);

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

      // Center the piece on the cursor
      const pieceW = piece.shape[0].length * cellSize;
      const pieceH = piece.shape.length * cellSize;
      const x = clientX - rect.left - pieceW / 2;
      const y = clientY - rect.top - pieceH / 2;

      return {
        row: Math.round(y / cellSize),
        col: Math.round(x / cellSize),
      };
    },
    []
  );

  const handleDragStart = useCallback(
    (pieceIndex: number, _offsetX: number, _offsetY: number) => {
      if (!game || game.isGameOver) return;
      const piece = game.currentPieces[pieceIndex];
      if (!piece) return;
      setDragState({ pieceIndex, piece, clientX: 0, clientY: 0 });
    },
    [game]
  );

  const handleDragMove = useCallback(
    (clientX: number, clientY: number) => {
      setDragState(prev => {
        if (!prev) return prev;
        return { ...prev, clientX, clientY };
      });

      // Update floating piece overlay position
      if (dragOverlayRef.current) {
        dragOverlayRef.current.style.left = `${clientX}px`;
        dragOverlayRef.current.style.top = `${clientY}px`;
      }
    },
    []
  );

  const handleDragEnd = useCallback(
    (clientX: number, clientY: number) => {
      if (!dragState || !game) {
        setDragState(null);
        return;
      }

      const { row, col } = clientToGrid(clientX, clientY, dragState.piece);

      if (canPlacePiece(game.board, dragState.piece, row, col)) {
        const newState = handlePlacement(game, dragState.pieceIndex, row, col);

        if (newState) {
          // Update board colors — keep existing colors, add new piece color
          const newColors = boardColors.map(r => [...r]);
          for (let r = 0; r < dragState.piece.shape.length; r++) {
            for (let c = 0; c < dragState.piece.shape[r].length; c++) {
              if (dragState.piece.shape[r][c]) {
                newColors[row + r][col + c] = dragState.piece.color;
              }
            }
          }

          // Check which lines were cleared
          const clearedRows: number[] = [];
          const clearedCols: number[] = [];

          for (let r = 0; r < BOARD_SIZE; r++) {
            const rowFull = newColors[r].every(c => c !== null);
            if (rowFull) clearedRows.push(r);
          }
          for (let c = 0; c < BOARD_SIZE; c++) {
            const colFull = newColors.every(row => row[c] !== null);
            if (colFull) clearedCols.push(c);
          }

          // Clear colors for cleared lines
          for (const r of clearedRows) {
            for (let c = 0; c < BOARD_SIZE; c++) {
              newColors[r][c] = null;
            }
          }
          for (const c of clearedCols) {
            for (let r = 0; r < BOARD_SIZE; r++) {
              newColors[r][c] = null;
            }
          }

          // Trigger clear animation if lines were cleared
          if (clearedRows.length > 0 || clearedCols.length > 0) {
            setClearAnimation({
              rows: clearedRows,
              cols: clearedCols,
              startTime: performance.now(),
            });

            setTimeout(() => setClearAnimation(null), 500);

            // Add combo popup
            if (newState.combo > 0) {
              const cellSize = cellSizeRef.current;
              const centerX = (BOARD_SIZE * cellSize) / 2;
              const centerY = (BOARD_SIZE * cellSize) / 2;
              setComboPopups(prev => [
                ...prev,
                {
                  text: `${newState.combo}x COMBO!`,
                  x: centerX,
                  y: centerY,
                  startTime: performance.now(),
                  color: '#ff8800',
                },
              ]);
              setTimeout(() => {
                setComboPopups(prev => prev.filter(p => performance.now() - p.startTime < 1000));
              }, 1100);
            }
          }

          setBoardColors(newColors);
          setGame(newState);
        }
      }

      setDragState(null);
    },
    [dragState, game, boardColors, clientToGrid]
  );

  const handleRestart = useCallback(() => {
    setGame(initGame());
    setBoardColors(
      Array.from({ length: BOARD_SIZE }, () => Array(BOARD_SIZE).fill(null))
    );
    setClearAnimation(null);
    setComboPopups([]);
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

  // Don't render until client-side init
  if (!game) return null;

  return (
    <div className="relative flex flex-col items-center w-full max-w-[520px] px-4 gap-2">
      <ScoreDisplay
        score={game.score}
        highScore={game.highScore}
        combo={game.combo}
        streak={game.streak}
      />

      <GameBoard
        board={game.board}
        boardColors={boardColors}
        ghostPiece={ghostPiece}
        ghostRow={ghostRow}
        ghostCol={ghostCol}
        clearAnimation={clearAnimation}
        comboPopups={comboPopups}
        onBoardLayout={handleBoardLayout}
      />

      <PieceTray
        pieces={game.currentPieces}
        onDragStart={handleDragStart}
        onDragMove={handleDragMove}
        onDragEnd={handleDragEnd}
        draggingIndex={dragState?.pieceIndex ?? null}
      />

      {/* Floating drag overlay */}
      {dragState && dragState.clientX !== 0 && (
        <div
          ref={dragOverlayRef}
          className="fixed pointer-events-none z-40"
          style={{
            left: dragState.clientX,
            top: dragState.clientY,
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

      {game.isGameOver && (
        <GameOverOverlay
          score={game.score}
          highScore={game.highScore}
          totalLinesCleared={game.totalLinesCleared}
          longestStreak={game.longestStreak}
          highestCombo={game.highestCombo}
          onRestart={handleRestart}
        />
      )}
    </div>
  );
}
