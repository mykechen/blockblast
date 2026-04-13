// Points awarded per cell when placing a piece (no line clear needed)
const BASE_POINTS_PER_CELL = 1;

// Points awarded per cell cleared via line completion
const CLEAR_POINTS_PER_CELL = 10;

export function calculatePlacementScore(cellsPlaced: number): number {
  return cellsPlaced * BASE_POINTS_PER_CELL;
}

export function calculateClearScore(
  cellsCleared: number,
  combo: number,
  streak: number
): { points: number; newCombo: number; newStreak: number } {
  if (cellsCleared === 0) {
    return { points: 0, newCombo: 0, newStreak: streak };
  }

  const comboMultiplier = Math.max(1, combo + 1);
  const streakMultiplier = Math.min(2, 1 + streak * 0.1);
  const points = Math.round(cellsCleared * CLEAR_POINTS_PER_CELL * comboMultiplier * streakMultiplier);

  return {
    points,
    newCombo: combo + 1,
    newStreak: streak + 1,
  };
}
