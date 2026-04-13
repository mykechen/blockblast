import { Piece, BoardState, BOARD_SIZE } from './types';
import { canPlacePiece, placePiece } from './board';

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

// --- Piece catalog grouped by difficulty ---

// Easy pieces: small, simple shapes that fill gaps easily
const EASY_PIECES: Piece[] = [
  p('dot', COLORS.yellow, 'X'),
  p('h2', COLORS.cyan, 'XX'),
  p('v2', COLORS.cyan, 'X', 'X'),
  p('h3', COLORS.blue, 'XXX'),
  p('v3', COLORS.blue, 'X', 'X', 'X'),
  p('sq2', COLORS.orange, 'XX', 'XX'),
];

// Medium pieces: larger lines and rectangles
const MEDIUM_PIECES: Piece[] = [
  p('h4', COLORS.purple, 'XXXX'),
  p('v4', COLORS.purple, 'X', 'X', 'X', 'X'),
  p('r2x3', COLORS.teal, 'XXX', 'XXX'),
  p('r3x2', COLORS.teal, 'XX', 'XX', 'XX'),
  p('L0', COLORS.green, 'X.', 'X.', 'XX'),
  p('L1', COLORS.green, 'XXX', 'X..'),
  p('L2', COLORS.green, 'XX', '.X', '.X'),
  p('L3', COLORS.green, '..X', 'XXX'),
];

// Hard pieces: large, awkward shapes
const HARD_PIECES: Piece[] = [
  p('h5', COLORS.magenta, 'XXXXX'),
  p('v5', COLORS.magenta, 'X', 'X', 'X', 'X', 'X'),
  p('sq3', COLORS.red, 'XXX', 'XXX', 'XXX'),
  p('T0', COLORS.pink, 'XXX', '.X.'),
  p('T1', COLORS.pink, '.X', 'XX', '.X'),
  p('T2', COLORS.pink, '.X.', 'XXX'),
  p('T3', COLORS.pink, 'X.', 'XX', 'X.'),
  p('S0', COLORS.orange, '.XX', 'XX.'),
  p('S1', COLORS.orange, 'X.', 'XX', '.X'),
  p('Z0', COLORS.red, 'XX.', '.XX'),
  p('Z1', COLORS.red, '.X', 'XX', 'X.'),
];

export const PIECE_CATALOG: Piece[] = [...EASY_PIECES, ...MEDIUM_PIECES, ...HARD_PIECES];

function clonePiece(piece: Piece): Piece {
  return {
    id: piece.id,
    color: piece.color,
    shape: piece.shape.map(row => [...row]),
  };
}

function pickRandom<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)];
}

// Count how many filled cells a piece has
function pieceCellCount(piece: Piece): number {
  let count = 0;
  for (const row of piece.shape) {
    for (const cell of row) {
      if (cell) count++;
    }
  }
  return count;
}

// Check if a single piece can be placed anywhere on the board
function canFitPiece(board: BoardState, piece: Piece): boolean {
  for (let r = 0; r < BOARD_SIZE; r++) {
    for (let c = 0; c < BOARD_SIZE; c++) {
      if (canPlacePiece(board, piece, r, c)) return true;
    }
  }
  return false;
}

/**
 * Get difficulty-scaled random pieces.
 *
 * - score < 200:  mostly easy pieces with occasional medium
 * - score 200-800: mix of easy and medium, occasional hard
 * - score 800-2000: mix of medium and hard
 * - score > 2000:  mostly hard, occasional medium
 *
 * After selecting pieces, validates that all 3 can fit on the board.
 * If any can't fit, swaps it for the smallest piece that can.
 */
export function getSmartPieces(count: number, score: number, inputBoard: BoardState): Piece[] {
  let board = inputBoard;
  const pieces: Piece[] = [];

  for (let i = 0; i < count; i++) {
    let pool: Piece[];

    if (score < 200) {
      // 80% easy, 20% medium
      pool = Math.random() < 0.8 ? EASY_PIECES : MEDIUM_PIECES;
    } else if (score < 800) {
      // 40% easy, 45% medium, 15% hard
      const r = Math.random();
      pool = r < 0.4 ? EASY_PIECES : r < 0.85 ? MEDIUM_PIECES : HARD_PIECES;
    } else if (score < 2000) {
      // 15% easy, 40% medium, 45% hard
      const r = Math.random();
      pool = r < 0.15 ? EASY_PIECES : r < 0.55 ? MEDIUM_PIECES : HARD_PIECES;
    } else {
      // 5% easy, 30% medium, 65% hard
      const r = Math.random();
      pool = r < 0.05 ? EASY_PIECES : r < 0.35 ? MEDIUM_PIECES : HARD_PIECES;
    }

    pieces.push(clonePiece(pickRandom(pool)));
  }

  // Solvability check: verify each piece can fit, accounting for the
  // space consumed by prior pieces in the set. We simulate placing each
  // piece at its first valid position to check if subsequent pieces still fit.
  const sortedBySize = [...PIECE_CATALOG].sort(
    (a, b) => pieceCellCount(a) - pieceCellCount(b)
  );

  for (let i = 0; i < pieces.length; i++) {
    if (!canFitPiece(board, pieces[i])) {
      // Swap for the smallest piece that fits
      let replaced = false;
      for (const candidate of sortedBySize) {
        if (canFitPiece(board, candidate)) {
          pieces[i] = clonePiece(candidate);
          replaced = true;
          break;
        }
      }
      // If nothing fits at all, game is truly over
      if (!replaced) continue;
    }

    // Simulate placing this piece at its first valid spot to check
    // whether remaining pieces can still fit on the resulting board
    if (i < pieces.length - 1) {
      let simBoard: BoardState | null = null;
      for (let r = 0; r < BOARD_SIZE && !simBoard; r++) {
        for (let c = 0; c < BOARD_SIZE && !simBoard; c++) {
          if (canPlacePiece(board, pieces[i], r, c)) {
            simBoard = placePiece(board, pieces[i], r, c);
          }
        }
      }
      if (simBoard) {
        // Use the simulated board for checking the next piece
        board = simBoard;
      }
    }
  }

  return pieces;
}

// Legacy function for backwards compatibility
export function getRandomPieces(count: number): Piece[] {
  const pieces: Piece[] = [];
  for (let i = 0; i < count; i++) {
    pieces.push(clonePiece(pickRandom(PIECE_CATALOG)));
  }
  return pieces;
}
