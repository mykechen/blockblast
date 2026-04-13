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
  // How many placements since the last line clear (combo expires at 3)
  placementsSinceLastClear: number;
  isGameOver: boolean;
  turnNumber: number;
  totalLinesCleared: number;
  highestCombo: number;
  highScore: number;
};

export type PlacementResult = {
  newBoard: BoardState;
  clearedRows: number[];
  clearedCols: number[];
  pointsEarned: number;
  newCombo: number;
};

export const BOARD_SIZE = 8;
