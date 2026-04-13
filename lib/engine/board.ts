import { BoardState, Piece, BOARD_SIZE } from './types';

// Creates an 8x8 board of all false
export function createEmptyBoard(): BoardState {
  return Array(BOARD_SIZE)
    .fill(null)
    .map(() => Array(BOARD_SIZE).fill(false));
}

// Returns true if piece fits at (row, col) — all piece cells land on empty squares within bounds
export function canPlacePiece(
  board: BoardState,
  piece: Piece,
  row: number,
  col: number
): boolean {
  const { shape } = piece;

  for (let r = 0; r < shape.length; r++) {
    for (let c = 0; c < shape[r].length; c++) {
      // Skip false cells in the piece shape
      if (!shape[r][c]) continue;

      const boardRow = row + r;
      const boardCol = col + c;

      // Check bounds
      if (boardRow < 0 || boardRow >= BOARD_SIZE || boardCol < 0 || boardCol >= BOARD_SIZE) {
        return false;
      }

      // Check board occupancy
      if (board[boardRow][boardCol]) {
        return false;
      }
    }
  }

  return true;
}

// Returns NEW board with piece placed (immutable — no mutation)
export function placePiece(
  board: BoardState,
  piece: Piece,
  row: number,
  col: number
): BoardState {
  // Clone board rows
  const newBoard = board.map((row) => [...row]);
  const { shape } = piece;

  for (let r = 0; r < shape.length; r++) {
    for (let c = 0; c < shape[r].length; c++) {
      // Skip false cells in the piece shape
      if (!shape[r][c]) continue;

      const boardRow = row + r;
      const boardCol = col + c;

      // Set cells to true
      newBoard[boardRow][boardCol] = true;
    }
  }

  return newBoard;
}

// Returns arrays of row indices and column indices that are completely filled
export function findCompletedLines(board: BoardState): { rows: number[]; cols: number[] } {
  const rows: number[] = [];
  const cols: number[] = [];

  // Check each row with .every()
  for (let r = 0; r < BOARD_SIZE; r++) {
    if (board[r].every((cell) => cell === true)) {
      rows.push(r);
    }
  }

  // Check each column by iterating all rows
  for (let c = 0; c < BOARD_SIZE; c++) {
    if (board.every((row) => row[c] === true)) {
      cols.push(c);
    }
  }

  return { rows, cols };
}

// Returns NEW board with specified rows and cols cleared to false
export function clearLines(board: BoardState, rows: number[], cols: number[]): BoardState {
  // Clone board
  const newBoard = board.map((row) => [...row]);

  // Set cleared row cells to false
  for (const r of rows) {
    for (let c = 0; c < BOARD_SIZE; c++) {
      newBoard[r][c] = false;
    }
  }

  // Set cleared column cells to false
  for (const c of cols) {
    for (let r = 0; r < BOARD_SIZE; r++) {
      newBoard[r][c] = false;
    }
  }

  return newBoard;
}

// Returns true if ANY piece in the array can be legally placed anywhere on the board
export function hasValidPlacement(
  board: BoardState,
  pieces: (Piece | null)[]
): boolean {
  for (const piece of pieces) {
    // Skip null pieces
    if (!piece) continue;

    // Try every (r, c) position
    for (let row = 0; row < BOARD_SIZE; row++) {
      for (let col = 0; col < BOARD_SIZE; col++) {
        if (canPlacePiece(board, piece, row, col)) {
          return true;
        }
      }
    }
  }

  return false;
}
