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
