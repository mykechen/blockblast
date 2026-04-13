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
    // silently fail if localStorage unavailable
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
    streak: 0,
    isGameOver: false,
    turnNumber: 1,
    totalLinesCleared: 0,
    longestStreak: 0,
    highestCombo: 0,
    highScore: loadHighScore(),
    clearedThisTurn: false,
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

  // Place the piece
  let newBoard = placePiece(state.board, piece, row, col);

  // Base points for placing the piece
  const cellsPlaced = countPieceCells(piece.shape);
  const placementPoints = calculatePlacementScore(cellsPlaced);

  // Check for completed lines
  const { rows, cols } = findCompletedLines(newBoard);
  const linesCleared = rows.length + cols.length;
  const cellsCleared = rows.length * BOARD_SIZE + cols.length * BOARD_SIZE
    - rows.length * cols.length; // subtract intersections

  // Clear lines
  if (linesCleared > 0) {
    newBoard = clearLines(newBoard, rows, cols);
  }

  // Calculate clear bonus (uses current combo for multiplier)
  const clearPoints = calculateClearScore(linesCleared, cellsCleared, state.combo);

  // Track if we cleared any line this turn
  const clearedThisTurn = state.clearedThisTurn || linesCleared > 0;

  const totalLinesCleared = state.totalLinesCleared + linesCleared;

  // Remove piece from tray
  const newPieces = [...state.currentPieces];
  newPieces[pieceIndex] = null;

  // Check if all 3 pieces have been placed — deal new set
  const allPlaced = newPieces.every(p => p === null);
  let finalPieces = newPieces;
  let newTurnNumber = state.turnNumber;
  let newCombo = state.combo;
  let newClearedThisTurn = clearedThisTurn;

  const totalPoints = placementPoints + clearPoints;
  const newScore = state.score + totalPoints;

  if (allPlaced) {
    // End of turn: update combo based on whether we cleared any line this turn
    if (clearedThisTurn) {
      // Cleared at least one line this turn — combo continues
      newCombo = state.combo + 1;
    } else {
      // No clears this turn — combo resets
      newCombo = 0;
    }

    finalPieces = getSmartPieces(3, newScore, newBoard);
    newTurnNumber = state.turnNumber + 1;
    newClearedThisTurn = false; // reset for new turn
  }

  const highestCombo = Math.max(state.highestCombo, newCombo);
  // Streak = same as combo for display purposes
  const streak = newCombo;
  const longestStreak = Math.max(state.longestStreak, streak);
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
    streak,
    isGameOver,
    turnNumber: newTurnNumber,
    totalLinesCleared,
    longestStreak,
    highestCombo,
    highScore,
    clearedThisTurn: newClearedThisTurn,
  };
}
