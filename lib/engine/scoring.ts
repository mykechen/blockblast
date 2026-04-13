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
 * Combo rules (matching Block Blast):
 * - Combo tracks consecutive turns where you clear at least one line
 * - A "turn" = one set of 3 pieces
 * - If you clear any line during a turn, the combo continues to the next turn
 * - If a full turn passes with no clears, combo resets to 0
 * - Combo multiplier: combo 0 = 1x, combo 1 = 2x, combo 2 = 3x, etc.
 *
 * @param linesCleared - number of lines (rows + cols) cleared in this placement
 * @param combo - current combo counter (number of consecutive turns with clears)
 */
export function calculateClearScore(
  linesCleared: number,
  cellsCleared: number,
  combo: number
): number {
  if (cellsCleared === 0) return 0;

  // Base: 10 points per cell cleared
  const basePoints = cellsCleared * CLEAR_POINTS_PER_CELL;

  // Multi-line bonus: clearing 2+ lines at once gives extra
  const lineBonusMultiplier = linesCleared >= 2 ? 1 + (linesCleared - 1) * 0.5 : 1;

  // Combo multiplier: each consecutive turn with clears adds 1x
  const comboMultiplier = 1 + combo;

  return Math.round(basePoints * lineBonusMultiplier * comboMultiplier);
}
