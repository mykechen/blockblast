import { Piece } from './types';

// Color palette
const COLORS = {
  cyan: '#00d4ff',
  blue: '#4466ff',
  purple: '#8844ff',
  magenta: '#ff44cc',
  pink: '#ff6699',
  orange: '#ff8800',
  yellow: '#ffcc00',
  green: '#44dd88',
  teal: '#00ccaa',
  red: '#ff4455',
};

/**
 * Helper function to create a piece from string templates
 * 'X' = filled cell, '.' = empty cell
 */
function p(id: string, color: string, ...rows: string[]): Piece {
  const shape = rows.map((row) => row.split('').map((char) => char === 'X'));
  return { id, color, shape };
}

// Piece catalog - contains all Block Blast pieces
export const PIECE_CATALOG: Piece[] = [
  // Dot (1x1)
  p('dot', COLORS.yellow, 'X'),

  // Horizontal 2 (1x2)
  p('h2', COLORS.cyan, 'XX'),
  // Vertical 2 (2x1)
  p('v2', COLORS.cyan, 'X', 'X'),

  // Horizontal 3 (1x3)
  p('h3', COLORS.blue, 'XXX'),
  // Vertical 3 (3x1)
  p('v3', COLORS.blue, 'X', 'X', 'X'),

  // Horizontal 4 (1x4)
  p('h4', COLORS.purple, 'XXXX'),
  // Vertical 4 (4x1)
  p('v4', COLORS.purple, 'X', 'X', 'X', 'X'),

  // Horizontal 5 (1x5)
  p('h5', COLORS.magenta, 'XXXXX'),
  // Vertical 5 (5x1)
  p('v5', COLORS.magenta, 'X', 'X', 'X', 'X', 'X'),

  // Square 2x2
  p('sq2', COLORS.orange, 'XX', 'XX'),

  // Square 3x3
  p('sq3', COLORS.red, 'XXX', 'XXX', 'XXX'),

  // Rectangle 2x3
  p('r2x3', COLORS.teal, 'XXX', 'XXX'),
  // Rectangle 3x2
  p('r3x2', COLORS.teal, 'XX', 'XX', 'XX'),

  // L-pieces (4 rotations)
  p('L0', COLORS.green, 'X.', 'X.', 'XX'),
  p('L1', COLORS.green, 'XXX', 'X..'),
  p('L2', COLORS.green, 'XX', '.X', '.X'),
  p('L3', COLORS.green, '..X', 'XXX'),

  // T-pieces (4 rotations)
  p('T0', COLORS.pink, 'XXX', '.X.'),
  p('T1', COLORS.pink, '.X', 'XX', '.X'),
  p('T2', COLORS.pink, '.X.', 'XXX'),
  p('T3', COLORS.pink, 'X.', 'XX', 'X.'),

  // S-pieces (2 rotations)
  p('S0', COLORS.orange, '.XX', 'XX.'),
  p('S1', COLORS.orange, 'X.', 'XX', '.X'),

  // Z-pieces (2 rotations)
  p('Z0', COLORS.red, 'XX.', '.XX'),
  p('Z1', COLORS.red, '.X', 'XX', 'X.'),
];

/**
 * Get random pieces from the catalog
 * Returns deep clones to prevent mutations
 */
export function getRandomPieces(count: number): Piece[] {
  const pieces: Piece[] = [];
  for (let i = 0; i < count; i++) {
    const randomIndex = Math.floor(Math.random() * PIECE_CATALOG.length);
    const piece = PIECE_CATALOG[randomIndex];
    // Deep clone the piece
    pieces.push({
      id: piece.id,
      color: piece.color,
      shape: piece.shape.map((row) => [...row]),
    });
  }
  return pieces;
}
