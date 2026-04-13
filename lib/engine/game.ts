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
  };
}

// Count filled cells in a piece
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

  // Base points for placing the piece (every placement scores)
  const cellsPlaced = countPieceCells(piece.shape);
  const placementPoints = calculatePlacementScore(cellsPlaced);

  // Check for completed lines
  const { rows, cols } = findCompletedLines(newBoard);
  const cellsCleared = rows.length * BOARD_SIZE + cols.length * BOARD_SIZE
    - rows.length * cols.length; // subtract intersections counted twice

  // Clear lines
  if (rows.length > 0 || cols.length > 0) {
    newBoard = clearLines(newBoard, rows, cols);
  }

  // Calculate clear bonus score
  const { points: clearPoints, newCombo, newStreak } = calculateClearScore(
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

  // Check if all pieces placed — deal new ones (difficulty-scaled)
  const allPlaced = newPieces.every(p => p === null);
  let finalPieces = newPieces;
  let newTurnNumber = state.turnNumber;
  let finalCombo = newCombo;
  let finalStreak = newStreak;

  const totalPoints = placementPoints + clearPoints;
  const newScore = state.score + totalPoints;

  if (allPlaced) {
    finalPieces = getSmartPieces(3, newScore, newBoard);
    newTurnNumber = state.turnNumber + 1;
    finalCombo = 0;
    if (cellsCleared === 0) {
      finalStreak = 0;
    }
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
