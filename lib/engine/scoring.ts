// Points awarded per cell when placing a piece (every placement scores)
const BASE_POINTS_PER_CELL = 1;

// Points awarded per cell cleared via line completion
const CLEAR_POINTS_PER_CELL = 10;

export function calculatePlacementScore(cellsPlaced: number): number {
  return cellsPlaced * BASE_POINTS_PER_CELL;
}

/**
 * Combo scoring for line clears.
 *
 * Combo rules:
 * - Every line clear increments the combo counter
 * - Combo persists across placements as long as you clear again
 *   within 3 placements (3 blocks placed without a clear = combo resets)
 * - Combo multiplier applied to line clear points: (1 + combo)
 *
 * @param linesCleared - number of lines (rows + cols) cleared in this placement
 * @param cellsCleared - total cells cleared (for point calculation)
 * @param combo - current combo counter before this clear
 */
export function calculateClearScore(
  linesCleared: number,
  cellsCleared: number,
  combo: number
): number {
  if (cellsCleared === 0) return 0;

  const basePoints = cellsCleared * CLEAR_POINTS_PER_CELL;

  // Multi-line bonus: clearing 2+ lines at once
  const lineBonusMultiplier = linesCleared >= 2 ? 1 + (linesCleared - 1) * 0.5 : 1;

  // Combo multiplier
  const comboMultiplier = 1 + combo;

  return Math.round(basePoints * lineBonusMultiplier * comboMultiplier);
}
