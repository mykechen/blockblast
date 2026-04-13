import { GameState, BOARD_SIZE } from './types';
import { createEmptyBoard, canPlacePiece, placePiece, findCompletedLines, clearLines, hasValidPlacement } from './board';
import { getSmartPieces, getRandomPieces } from './pieces';
import { calculatePlacementScore, calculateClearScore } from './scoring';

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
    // silently fail
  }
}

export function initGame(): GameState {
  const board = createEmptyBoard();
  const pieces = getRandomPieces(3);
  return {
    board,
    currentPieces: pieces,
    score: 0,
    combo: 0,
    placementsSinceLastClear: 0,
    isGameOver: false,
    turnNumber: 1,
    totalLinesCleared: 0,
    highestCombo: 0,
    highScore: loadHighScore(),
  };
}

function countPieceCells(shape: boolean[][]): number {
  let count = 0;
  for (const row of shape) {
    for (const cell of row) {
      if (cell) count++;
    }
  }
  return count;
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

  let newBoard = placePiece(state.board, piece, row, col);

  // Base points for placing the piece
  const cellsPlaced = countPieceCells(piece.shape);
  const placementPoints = calculatePlacementScore(cellsPlaced);

  // Check for completed lines
  const { rows, cols } = findCompletedLines(newBoard);
  const linesCleared = rows.length + cols.length;
  const cellsCleared = rows.length * BOARD_SIZE + cols.length * BOARD_SIZE
    - rows.length * cols.length;

  if (linesCleared > 0) {
    newBoard = clearLines(newBoard, rows, cols);
  }

  // Combo logic:
  // - Clear a line → combo increments, placement counter resets
  // - No clear → placement counter increments
  // - 3 placements without a clear → combo resets to 0
  let newCombo = state.combo;
  let newPlacementsSinceLastClear = state.placementsSinceLastClear;

  if (linesCleared > 0) {
    // Line cleared: increment combo, reset placement counter
    newCombo = state.combo + 1;
    newPlacementsSinceLastClear = 0;
  } else {
    // No clear: increment placement counter
    newPlacementsSinceLastClear = state.placementsSinceLastClear + 1;
    if (newPlacementsSinceLastClear >= 3) {
      // 3 placements without a clear — combo expires
      newCombo = 0;
      newPlacementsSinceLastClear = 0;
    }
  }

  // Calculate clear bonus using the combo BEFORE this clear incremented it
  // (so the first clear in a combo uses multiplier 1x, second uses 2x, etc.)
  const clearPoints = calculateClearScore(linesCleared, cellsCleared, state.combo);

  const totalLinesCleared = state.totalLinesCleared + linesCleared;
  const highestCombo = Math.max(state.highestCombo, newCombo);

  // Remove piece from tray
  const newPieces = [...state.currentPieces];
  newPieces[pieceIndex] = null;

  // Deal new pieces if all 3 placed
  const allPlaced = newPieces.every(p => p === null);
  let finalPieces = newPieces;
  let newTurnNumber = state.turnNumber;

  const totalPoints = placementPoints + clearPoints;
  const newScore = state.score + totalPoints;

  if (allPlaced) {
    finalPieces = getSmartPieces(3, newScore, newBoard);
    newTurnNumber = state.turnNumber + 1;
  }

  const highScore = Math.max(state.highScore, newScore);
  if (highScore > state.highScore) {
    saveHighScore(highScore);
  }

  const isGameOver = !hasValidPlacement(newBoard, finalPieces);

  return {
    board: newBoard,
    currentPieces: finalPieces,
    score: newScore,
    combo: newCombo,
    placementsSinceLastClear: newPlacementsSinceLastClear,
    isGameOver,
    turnNumber: newTurnNumber,
    totalLinesCleared,
    highestCombo,
    highScore,
  };
}
