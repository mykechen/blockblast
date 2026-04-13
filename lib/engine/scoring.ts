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
