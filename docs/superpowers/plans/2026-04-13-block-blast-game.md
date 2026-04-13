# Block Blast Game — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fully playable, polished Block Blast puzzle game as a responsive web app with smooth drag-and-drop, Canvas rendering, and line-clear animations.

**Architecture:** Pure TypeScript game engine in `/lib/engine/` (no UI deps) with immutable functions. Canvas-based board rendering in React components. DOM-based drag-and-drop for pieces (better touch handling) with Canvas ghost preview on the board. React `useReducer` for game state management.

**Tech Stack:** Next.js 16, TypeScript, HTML5 Canvas, Tailwind CSS, localStorage for high score persistence.

---

## File Structure

```
lib/engine/
  types.ts          — All game types (Cell, Piece, BoardState, GameState, PlacementResult)
  pieces.ts         — Piece definitions catalog and random selection
  board.ts          — Board logic (placement validation, line detection, clearing)
  scoring.ts        — Score calculation with combo/streak
  game.ts           — Game state orchestrator (init, placement, game-over check)

components/
  BlockBlastGame.tsx — Main game container, state management, drag coordination
  GameBoard.tsx      — Canvas board renderer with ghost preview and animations
  PieceTray.tsx      — Draggable piece tray (DOM-based drag)
  ScoreDisplay.tsx   — Score, combo, streak counters with animations
  GameOverOverlay.tsx — Game over screen with stats and restart

app/
  page.tsx           — Main page, renders BlockBlastGame
  layout.tsx         — HTML head, fonts, metadata, global styles
  globals.css        — Tailwind imports + canvas/game-specific styles
```

---

### Task 1: Game Types

**Files:**
- Create: `lib/engine/types.ts`

- [ ] **Step 1: Define all game types**

```typescript
// lib/engine/types.ts

export type Cell = {
  row: number;
  col: number;
};

export type Piece = {
  shape: boolean[][];
  id: string;
  color: string;
};

export type BoardState = boolean[][];

export type GameState = {
  board: BoardState;
  currentPieces: (Piece | null)[];
  score: number;
  combo: number;
  streak: number;
  isGameOver: boolean;
  turnNumber: number;
  totalLinesCleared: number;
  longestStreak: number;
  highestCombo: number;
  highScore: number;
};

export type PlacementResult = {
  newBoard: BoardState;
  clearedRows: number[];
  clearedCols: number[];
  pointsEarned: number;
  newCombo: number;
  newStreak: number;
};

export const BOARD_SIZE = 8;
```

- [ ] **Step 2: Commit**

```bash
git add lib/engine/types.ts
git commit -m "feat: add game engine types"
```

---

### Task 2: Piece Definitions

**Files:**
- Create: `lib/engine/pieces.ts`

- [ ] **Step 1: Define the piece catalog**

```typescript
// lib/engine/pieces.ts
import { Piece } from './types';

// Color palette for pieces — vibrant, saturated, distinct
const COLORS = {
  cyan: '#00d4ff',
  blue: '#4466ff',
  purple: '#8844ff',
  magenta: '#ff44cc',
  pink: '#ff6699',
  orange: '#ff8800',
  yellow: '#ffcc00',
  green: '#44dd88',
  teal: '#00ccaa',
  red: '#ff4455',
};

// Helper to create a piece from a string template for readability
function p(id: string, color: string, ...rows: string[]): Piece {
  return {
    id,
    color,
    shape: rows.map(row => [...row].map(c => c === 'X')),
  };
}

export const PIECE_CATALOG: Piece[] = [
  // 1x1
  p('dot', COLORS.yellow, 'X'),

  // 1x2, 2x1
  p('h2', COLORS.cyan, 'XX'),
  p('v2', COLORS.cyan, 'X', 'X'),

  // 1x3, 3x1
  p('h3', COLORS.blue, 'XXX'),
  p('v3', COLORS.blue, 'X', 'X', 'X'),

  // 1x4, 4x1
  p('h4', COLORS.purple, 'XXXX'),
  p('v4', COLORS.purple, 'X', 'X', 'X', 'X'),

  // 1x5, 5x1
  p('h5', COLORS.magenta, 'XXXXX'),
  p('v5', COLORS.magenta, 'X', 'X', 'X', 'X', 'X'),

  // 2x2 square
  p('sq2', COLORS.orange, 'XX', 'XX'),

  // 3x3 square
  p('sq3', COLORS.red, 'XXX', 'XXX', 'XXX'),

  // 2x3, 3x2 rectangles
  p('r2x3', COLORS.teal, 'XXX', 'XXX'),
  p('r3x2', COLORS.teal, 'XX', 'XX', 'XX'),

  // L-shape (4 rotations)
  p('L0', COLORS.green, 'X.', 'X.', 'XX'),
  p('L1', COLORS.green, 'XXX', 'X..'),
  p('L2', COLORS.green, 'XX', '.X', '.X'),
  p('L3', COLORS.green, '..X', 'XXX'),

  // T-shape (4 rotations)
  p('T0', COLORS.pink, 'XXX', '.X.'),
  p('T1', COLORS.pink, '.X', 'XX', '.X'),
  p('T2', COLORS.pink, '.X.', 'XXX'),
  p('T3', COLORS.pink, 'X.', 'XX', 'X.'),

  // S-shape (2 orientations)
  p('S0', COLORS.orange, '.XX', 'XX.'),
  p('S1', COLORS.orange, 'X.', 'XX', '.X'),

  // Z-shape (2 orientations)
  p('Z0', COLORS.red, 'XX.', '.XX'),
  p('Z1', COLORS.red, '.X', 'XX', 'X.'),
];

export function getRandomPieces(count: number): Piece[] {
  const pieces: Piece[] = [];
  for (let i = 0; i < count; i++) {
    const index = Math.floor(Math.random() * PIECE_CATALOG.length);
    // Deep clone so each instance is independent
    const source = PIECE_CATALOG[index];
    pieces.push({
      id: source.id,
      color: source.color,
      shape: source.shape.map(row => [...row]),
    });
  }
  return pieces;
}
```

- [ ] **Step 2: Commit**

```bash
git add lib/engine/pieces.ts
git commit -m "feat: add piece catalog with all Block Blast shapes"
```

---

### Task 3: Board Logic

**Files:**
- Create: `lib/engine/board.ts`

- [ ] **Step 1: Implement board functions**

```typescript
// lib/engine/board.ts
import { BoardState, Piece, BOARD_SIZE } from './types';

export function createEmptyBoard(): BoardState {
  return Array.from({ length: BOARD_SIZE }, () =>
    Array.from({ length: BOARD_SIZE }, () => false)
  );
}

export function canPlacePiece(
  board: BoardState,
  piece: Piece,
  row: number,
  col: number
): boolean {
  for (let r = 0; r < piece.shape.length; r++) {
    for (let c = 0; c < piece.shape[r].length; c++) {
      if (!piece.shape[r][c]) continue;
      const br = row + r;
      const bc = col + c;
      if (br < 0 || br >= BOARD_SIZE || bc < 0 || bc >= BOARD_SIZE) return false;
      if (board[br][bc]) return false;
    }
  }
  return true;
}

export function placePiece(
  board: BoardState,
  piece: Piece,
  row: number,
  col: number
): BoardState {
  const newBoard = board.map(r => [...r]);
  for (let r = 0; r < piece.shape.length; r++) {
    for (let c = 0; c < piece.shape[r].length; c++) {
      if (piece.shape[r][c]) {
        newBoard[row + r][col + c] = true;
      }
    }
  }
  return newBoard;
}

export function findCompletedLines(
  board: BoardState
): { rows: number[]; cols: number[] } {
  const rows: number[] = [];
  const cols: number[] = [];

  for (let r = 0; r < BOARD_SIZE; r++) {
    if (board[r].every(cell => cell)) rows.push(r);
  }
  for (let c = 0; c < BOARD_SIZE; c++) {
    if (board.every(row => row[c])) cols.push(c);
  }

  return { rows, cols };
}

export function clearLines(
  board: BoardState,
  rows: number[],
  cols: number[]
): BoardState {
  const newBoard = board.map(r => [...r]);
  for (const r of rows) {
    for (let c = 0; c < BOARD_SIZE; c++) {
      newBoard[r][c] = false;
    }
  }
  for (const c of cols) {
    for (let r = 0; r < BOARD_SIZE; r++) {
      newBoard[r][c] = false;
    }
  }
  return newBoard;
}

export function hasValidPlacement(
  board: BoardState,
  pieces: (Piece | null)[]
): boolean {
  for (const piece of pieces) {
    if (!piece) continue;
    for (let r = 0; r < BOARD_SIZE; r++) {
      for (let c = 0; c < BOARD_SIZE; c++) {
        if (canPlacePiece(board, piece, r, c)) return true;
      }
    }
  }
  return false;
}
```

- [ ] **Step 2: Commit**

```bash
git add lib/engine/board.ts
git commit -m "feat: add board logic — placement, line detection, clearing"
```

---

### Task 4: Scoring Logic

**Files:**
- Create: `lib/engine/scoring.ts`

- [ ] **Step 1: Implement scoring functions**

```typescript
// lib/engine/scoring.ts

export function calculateScore(
  cellsCleared: number,
  combo: number,
  streak: number
): { points: number; newCombo: number; newStreak: number } {
  if (cellsCleared === 0) {
    return { points: 0, newCombo: 0, newStreak: streak };
  }

  const comboMultiplier = Math.max(1, combo + 1);
  const streakMultiplier = Math.min(2, 1 + streak * 0.1);
  const points = Math.round(cellsCleared * 10 * comboMultiplier * streakMultiplier);

  return {
    points,
    newCombo: combo + 1,
    newStreak: streak + 1,
  };
}
```

- [ ] **Step 2: Commit**

```bash
git add lib/engine/scoring.ts
git commit -m "feat: add scoring with combo and streak multipliers"
```

---

### Task 5: Game State Orchestrator

**Files:**
- Create: `lib/engine/game.ts`

- [ ] **Step 1: Implement game orchestrator**

```typescript
// lib/engine/game.ts
import { GameState, BOARD_SIZE } from './types';
import { createEmptyBoard, canPlacePiece, placePiece, findCompletedLines, clearLines, hasValidPlacement } from './board';
import { getRandomPieces } from './pieces';
import { calculateScore } from './scoring';

function loadHighScore(): number {
  if (typeof window === 'undefined') return 0;
  try {
    const saved = localStorage.getItem('blockblast-highscore');
    return saved ? parseInt(saved, 10) : 0;
  } catch {
    return 0;
  }
}

function saveHighScore(score: number): void {
  if (typeof window === 'undefined') return;
  try {
    localStorage.setItem('blockblast-highscore', String(score));
  } catch {
    // silently fail if localStorage unavailable
  }
}

export function initGame(): GameState {
  const pieces = getRandomPieces(3);
  return {
    board: createEmptyBoard(),
    currentPieces: pieces,
    score: 0,
    combo: 0,
    streak: 0,
    isGameOver: false,
    turnNumber: 1,
    totalLinesCleared: 0,
    longestStreak: 0,
    highestCombo: 0,
    highScore: loadHighScore(),
  };
}

export function handlePlacement(
  state: GameState,
  pieceIndex: number,
  row: number,
  col: number
): GameState | null {
  const piece = state.currentPieces[pieceIndex];
  if (!piece) return null;
  if (!canPlacePiece(state.board, piece, row, col)) return null;

  // Place the piece
  let newBoard = placePiece(state.board, piece, row, col);

  // Check for completed lines
  const { rows, cols } = findCompletedLines(newBoard);
  const cellsCleared = rows.length * BOARD_SIZE + cols.length * BOARD_SIZE
    - rows.length * cols.length; // subtract intersections counted twice

  // Clear lines
  if (rows.length > 0 || cols.length > 0) {
    newBoard = clearLines(newBoard, rows, cols);
  }

  // Calculate score
  const { points, newCombo, newStreak } = calculateScore(
    cellsCleared,
    state.combo,
    cellsCleared > 0 ? state.streak : 0
  );

  const totalLinesCleared = state.totalLinesCleared + rows.length + cols.length;
  const longestStreak = Math.max(state.longestStreak, newStreak);
  const highestCombo = Math.max(state.highestCombo, newCombo);

  // Remove piece from tray
  const newPieces = [...state.currentPieces];
  newPieces[pieceIndex] = null;

  // Check if all pieces placed — deal new ones
  const allPlaced = newPieces.every(p => p === null);
  let finalPieces = newPieces;
  let newTurnNumber = state.turnNumber;
  let finalCombo = newCombo;
  let finalStreak = newStreak;

  if (allPlaced) {
    finalPieces = getRandomPieces(3);
    newTurnNumber = state.turnNumber + 1;
    finalCombo = 0; // reset combo at start of new turn
    // streak persists across turns if player cleared on last placement
    if (cellsCleared === 0) {
      finalStreak = 0;
    }
  }

  const newScore = state.score + points;
  const highScore = Math.max(state.highScore, newScore);

  // Save high score
  if (highScore > state.highScore) {
    saveHighScore(highScore);
  }

  // Check game over
  const isGameOver = !hasValidPlacement(newBoard, finalPieces);

  return {
    board: newBoard,
    currentPieces: finalPieces,
    score: newScore,
    combo: cellsCleared > 0 ? finalCombo : 0,
    streak: cellsCleared > 0 ? finalStreak : 0,
    isGameOver,
    turnNumber: newTurnNumber,
    totalLinesCleared,
    longestStreak,
    highestCombo,
    highScore,
  };
}
```

- [ ] **Step 2: Commit**

```bash
git add lib/engine/game.ts
git commit -m "feat: add game state orchestrator with placement, scoring, game-over"
```

---

### Task 6: App Layout and Global Styles

**Files:**
- Modify: `app/layout.tsx`
- Modify: `app/globals.css`
- Modify: `app/page.tsx`

- [ ] **Step 1: Update layout.tsx**

Replace the contents of `app/layout.tsx` with:

```typescript
// app/layout.tsx
import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
});

export const metadata: Metadata = {
  title: 'Block Blast',
  description: 'A polished Block Blast puzzle game',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={`${inter.variable} font-sans antialiased`}>
        {children}
      </body>
    </html>
  );
}
```

- [ ] **Step 2: Update globals.css**

Replace the contents of `app/globals.css` with:

```css
@import "tailwindcss";

:root {
  --bg-primary: #0a0a0f;
  --bg-board: #1a1a2e;
  --grid-line: #2a2a3e;
  --text-primary: #ffffff;
  --text-secondary: #a0a0b0;
}

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

html, body {
  background-color: var(--bg-primary);
  color: var(--text-primary);
  overflow: hidden;
  height: 100%;
  width: 100%;
  touch-action: none;
  user-select: none;
  -webkit-user-select: none;
}

/* Prevent pull-to-refresh and overscroll on mobile */
body {
  overscroll-behavior: none;
}

@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes slideUp {
  from {
    opacity: 0;
    transform: translateY(40px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes popIn {
  0% {
    opacity: 0;
    transform: scale(0.8);
  }
  60% {
    transform: scale(1.05);
  }
  100% {
    opacity: 1;
    transform: scale(1);
  }
}

@keyframes floatUp {
  from {
    opacity: 1;
    transform: translateY(0);
  }
  to {
    opacity: 0;
    transform: translateY(-40px);
  }
}
```

- [ ] **Step 3: Update page.tsx**

Replace the contents of `app/page.tsx` with:

```typescript
// app/page.tsx
import BlockBlastGame from '@/components/BlockBlastGame';

export default function Home() {
  return (
    <main className="flex items-center justify-center h-screen w-screen">
      <BlockBlastGame />
    </main>
  );
}
```

- [ ] **Step 4: Commit**

```bash
git add app/layout.tsx app/globals.css app/page.tsx
git commit -m "feat: set up app layout, global styles, and main page"
```

---

### Task 7: ScoreDisplay Component

**Files:**
- Create: `components/ScoreDisplay.tsx`

- [ ] **Step 1: Build the score display**

```typescript
// components/ScoreDisplay.tsx
'use client';

import { useEffect, useRef, useState } from 'react';

type ScoreDisplayProps = {
  score: number;
  highScore: number;
  combo: number;
  streak: number;
};

function AnimatedNumber({ value }: { value: number }) {
  const [display, setDisplay] = useState(value);
  const rafRef = useRef<number>(0);
  const startRef = useRef<number>(0);
  const fromRef = useRef<number>(value);

  useEffect(() => {
    if (value === display) return;
    fromRef.current = display;
    startRef.current = performance.now();
    const duration = 400;

    function animate(now: number) {
      const elapsed = now - startRef.current;
      const progress = Math.min(elapsed / duration, 1);
      // Ease out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = Math.round(fromRef.current + (value - fromRef.current) * eased);
      setDisplay(current);
      if (progress < 1) {
        rafRef.current = requestAnimationFrame(animate);
      }
    }

    rafRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(rafRef.current);
  }, [value]);

  return <span>{display.toLocaleString()}</span>;
}

export default function ScoreDisplay({ score, highScore, combo, streak }: ScoreDisplayProps) {
  return (
    <div className="flex items-center justify-between w-full max-w-[480px] px-4 py-2">
      <div className="flex flex-col items-start">
        <span className="text-xs uppercase tracking-wider" style={{ color: 'var(--text-secondary)' }}>
          Score
        </span>
        <span className="text-2xl font-bold tabular-nums">
          <AnimatedNumber value={score} />
        </span>
      </div>

      <div className="flex gap-3 items-center">
        {combo > 0 && (
          <div
            className="px-2 py-1 rounded text-sm font-bold"
            style={{
              animation: 'popIn 0.3s ease-out',
              background: 'rgba(255, 136, 0, 0.2)',
              color: '#ff8800',
            }}
          >
            {combo}x COMBO
          </div>
        )}
        {streak > 1 && (
          <div
            className="px-2 py-1 rounded text-sm font-bold"
            style={{
              animation: 'popIn 0.3s ease-out',
              background: 'rgba(68, 221, 136, 0.2)',
              color: '#44dd88',
            }}
          >
            {streak} STREAK
          </div>
        )}
      </div>

      <div className="flex flex-col items-end">
        <span className="text-xs uppercase tracking-wider" style={{ color: 'var(--text-secondary)' }}>
          Best
        </span>
        <span className="text-lg font-semibold tabular-nums" style={{ color: 'var(--text-secondary)' }}>
          <AnimatedNumber value={highScore} />
        </span>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add components/ScoreDisplay.tsx
git commit -m "feat: add ScoreDisplay with animated counters and combo/streak badges"
```

---

### Task 8: GameBoard Component (Canvas Renderer)

**Files:**
- Create: `components/GameBoard.tsx`

This is the most complex component. It renders the 8x8 board on a Canvas, draws placed cells with colors, shows the ghost preview during drag, and handles line-clear animations.

- [ ] **Step 1: Build the GameBoard component**

```typescript
// components/GameBoard.tsx
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
            // Flash white then fade out
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

      // Continue animation loop if needed
      const needsAnimation =
        clearAnimation !== null ||
        comboPopups.some(p => now - p.startTime < POPUP_DURATION);

      if (needsAnimation) {
        rafRef.current = requestAnimationFrame(render);
      }
    }

    render();

    // Re-render whenever props change
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
```

- [ ] **Step 2: Commit**

```bash
git add components/GameBoard.tsx
git commit -m "feat: add Canvas-based GameBoard with ghost preview and clear animations"
```

---

### Task 9: PieceTray Component (Drag and Drop)

**Files:**
- Create: `components/PieceTray.tsx`

- [ ] **Step 1: Build the PieceTray component with drag support**

```typescript
// components/PieceTray.tsx
'use client';

import { useRef, useCallback } from 'react';
import { Piece } from '@/lib/engine/types';

type PieceTrayProps = {
  pieces: (Piece | null)[];
  onDragStart: (pieceIndex: number, offsetX: number, offsetY: number) => void;
  onDragMove: (clientX: number, clientY: number) => void;
  onDragEnd: (clientX: number, clientY: number) => void;
  draggingIndex: number | null;
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
  onDragStart: (pieceIndex: number, offsetX: number, offsetY: number) => void;
  isDragging: boolean;
}) {
  const ref = useRef<HTMLDivElement>(null);

  const handlePointerDown = useCallback(
    (e: React.PointerEvent) => {
      e.preventDefault();
      const el = ref.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      // Calculate offset from piece center for centering on cursor
      const pieceW = piece.shape[0].length * TRAY_CELL_SIZE;
      const pieceH = piece.shape.length * TRAY_CELL_SIZE;
      const offsetX = e.clientX - rect.left - pieceW / 2;
      const offsetY = e.clientY - rect.top - pieceH / 2;
      onDragStart(index, offsetX, offsetY);
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
    (pieceIndex: number, offsetX: number, offsetY: number) => {
      onDragStart(pieceIndex, offsetX, offsetY);
      window.addEventListener('pointermove', handlePointerMove);
      window.addEventListener('pointerup', handlePointerUp);
    },
    [onDragStart, handlePointerMove, handlePointerUp]
  );

  return (
    <div className="flex items-center justify-center gap-4 w-full max-w-[480px] py-4">
      {pieces.map((piece, i) => (
        <div key={i} className="flex-1 flex items-center justify-center min-h-[100px]">
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
```

- [ ] **Step 2: Commit**

```bash
git add components/PieceTray.tsx
git commit -m "feat: add PieceTray with pointer-based drag support"
```

---

### Task 10: GameOverOverlay Component

**Files:**
- Create: `components/GameOverOverlay.tsx`

- [ ] **Step 1: Build the game over overlay**

```typescript
// components/GameOverOverlay.tsx
'use client';

type GameOverOverlayProps = {
  score: number;
  highScore: number;
  totalLinesCleared: number;
  longestStreak: number;
  highestCombo: number;
  onRestart: () => void;
};

export default function GameOverOverlay({
  score,
  highScore,
  totalLinesCleared,
  longestStreak,
  highestCombo,
  onRestart,
}: GameOverOverlayProps) {
  const isNewHighScore = score >= highScore && score > 0;

  return (
    <div
      className="absolute inset-0 flex items-center justify-center z-50"
      style={{ backgroundColor: 'rgba(10, 10, 15, 0.85)' }}
    >
      <div
        className="flex flex-col items-center gap-6 p-8 rounded-2xl max-w-[320px] w-full mx-4"
        style={{
          backgroundColor: '#1a1a2e',
          animation: 'slideUp 0.4s ease-out',
        }}
      >
        <h2 className="text-2xl font-bold">Game Over</h2>

        {isNewHighScore && (
          <div
            className="text-sm font-bold px-3 py-1 rounded-full"
            style={{
              background: 'rgba(255, 204, 0, 0.2)',
              color: '#ffcc00',
              animation: 'popIn 0.5s ease-out',
            }}
          >
            NEW HIGH SCORE!
          </div>
        )}

        <div className="text-4xl font-bold tabular-nums">
          {score.toLocaleString()}
        </div>

        <div className="w-full grid grid-cols-3 gap-3 text-center">
          <div className="flex flex-col">
            <span className="text-xs uppercase tracking-wider" style={{ color: 'var(--text-secondary)' }}>
              Lines
            </span>
            <span className="text-lg font-semibold tabular-nums">{totalLinesCleared}</span>
          </div>
          <div className="flex flex-col">
            <span className="text-xs uppercase tracking-wider" style={{ color: 'var(--text-secondary)' }}>
              Best Streak
            </span>
            <span className="text-lg font-semibold tabular-nums">{longestStreak}</span>
          </div>
          <div className="flex flex-col">
            <span className="text-xs uppercase tracking-wider" style={{ color: 'var(--text-secondary)' }}>
              Best Combo
            </span>
            <span className="text-lg font-semibold tabular-nums">{highestCombo}x</span>
          </div>
        </div>

        <button
          onClick={onRestart}
          className="w-full py-3 rounded-xl font-semibold text-lg transition-all duration-150 hover:brightness-110 active:scale-95 cursor-pointer"
          style={{
            background: 'linear-gradient(135deg, #4466ff, #8844ff)',
            color: '#ffffff',
          }}
        >
          Play Again
        </button>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add components/GameOverOverlay.tsx
git commit -m "feat: add GameOverOverlay with stats and restart"
```

---

### Task 11: BlockBlastGame — Main Orchestrator

**Files:**
- Create: `components/BlockBlastGame.tsx`

This is the central component that wires together all sub-components, manages game state, and coordinates drag-and-drop between the PieceTray and GameBoard.

- [ ] **Step 1: Build the main game component**

```typescript
// components/BlockBlastGame.tsx
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
        const prevBoard = game.board;
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

          // Check which lines were cleared by comparing boards
          // Find cleared rows/cols from the placement result
          const clearedRows: number[] = [];
          const clearedCols: number[] = [];

          // A row is cleared if it was full after placement but empty in newState
          const boardAfterPlace = newColors.map(r => [...r]);
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

            // Auto-clear animation after duration
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
              // Clean up old popups
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
```

- [ ] **Step 2: Commit**

```bash
git add components/BlockBlastGame.tsx
git commit -m "feat: add BlockBlastGame orchestrator with drag-and-drop coordination"
```

---

### Task 12: Clean Up Boilerplate and Test

**Files:**
- Modify: `app/page.tsx` (already done in Task 6)
- Remove: any leftover Next.js boilerplate

- [ ] **Step 1: Remove boilerplate favicon and images**

```bash
rm -f public/file.svg public/globe.svg public/next.svg public/vercel.svg public/window.svg
```

- [ ] **Step 2: Run the dev server and verify**

```bash
npm run dev
```

Open http://localhost:3000 in a browser. Verify:
- Board renders as 8x8 grid with dark theme
- 3 pieces appear in tray below board
- Pieces are draggable with ghost preview on board
- Placing pieces works (snaps to grid)
- Line clearing works with animation
- Score updates with animated counter
- Combo/streak badges appear
- Game over overlay appears when no moves remain
- "Play Again" restarts the game
- High score persists across restarts (localStorage)
- Works on both mouse and touch

- [ ] **Step 3: Commit final state**

```bash
git add -A
git commit -m "feat: Block Blast game — fully playable with polish"
```

---

### Task 13: Visual Polish and Animation Tuning

**Files:**
- Modify: `components/GameBoard.tsx`
- Modify: `components/BlockBlastGame.tsx`
- Modify: `components/PieceTray.tsx`

This task is for tuning after initial testing. Address any visual issues found during Task 12 testing:

- [ ] **Step 1: Test and fix drag-and-drop feel**

Verify the floating piece follows the cursor smoothly at 60fps. If janky, ensure pointer events use `requestAnimationFrame` batching and the floating overlay uses `transform` (GPU-accelerated) instead of `left/top` for movement.

- [ ] **Step 2: Test and fix ghost preview alignment**

Verify the ghost preview aligns correctly with the grid cells. The piece should snap to the nearest grid cell and be centered under the cursor. Fix `clientToGrid` math if alignment is off.

- [ ] **Step 3: Test responsive layout**

Resize browser to mobile width (375px). Verify:
- Board fills width with padding
- Pieces in tray don't overflow
- Touch drag works (no scroll interference)
- Score display doesn't wrap awkwardly

- [ ] **Step 4: Fix any issues and commit**

```bash
git add -A
git commit -m "fix: polish visual feel and responsive layout"
```

---

## Summary

| Task | Component | Depends On |
|------|-----------|------------|
| 1 | Types | — |
| 2 | Piece Catalog | 1 |
| 3 | Board Logic | 1 |
| 4 | Scoring | 1 |
| 5 | Game Orchestrator | 2, 3, 4 |
| 6 | Layout & Styles | — |
| 7 | ScoreDisplay | 1 |
| 8 | GameBoard (Canvas) | 1, 3 |
| 9 | PieceTray (Drag) | 1 |
| 10 | GameOverOverlay | — |
| 11 | BlockBlastGame | 5, 7, 8, 9, 10 |
| 12 | Clean Up & Test | 6, 11 |
| 13 | Visual Polish | 12 |

Tasks 1-4 can be parallelized. Tasks 6-10 can be parallelized. Task 11 requires all prior components. Tasks 12-13 are sequential.
