'use client';

type GameOverOverlayProps = {
  score: number;
  highScore: number;
  totalLinesCleared: number;
  longestStreak: number;
  highestCombo: number;
  onRestart: () => void;
};

export default function GameOverOverlay({
  score,
  highScore,
  totalLinesCleared,
  longestStreak,
  highestCombo,
  onRestart,
}: GameOverOverlayProps) {
  const isNewHighScore = score >= highScore && score > 0;

  return (
    <div
      className="absolute inset-0 flex items-center justify-center z-50"
      style={{ backgroundColor: 'rgba(10, 10, 15, 0.85)' }}
    >
      <div
        className="flex flex-col items-center gap-6 p-8 rounded-2xl max-w-[320px] w-full mx-4"
        style={{
          backgroundColor: '#1a1a2e',
          animation: 'slideUp 0.4s ease-out',
        }}
      >
        <h2 className="text-2xl font-bold">Game Over</h2>

        {isNewHighScore && (
          <div
            className="text-sm font-bold px-3 py-1 rounded-full"
            style={{
              background: 'rgba(255, 204, 0, 0.2)',
              color: '#ffcc00',
              animation: 'popIn 0.5s ease-out',
            }}
          >
            NEW HIGH SCORE!
          </div>
        )}

        <div className="text-4xl font-bold tabular-nums">
          {score.toLocaleString()}
        </div>

        <div className="w-full grid grid-cols-3 gap-3 text-center">
          <div className="flex flex-col">
            <span className="text-xs uppercase tracking-wider" style={{ color: 'var(--text-secondary)' }}>
              Lines
            </span>
            <span className="text-lg font-semibold tabular-nums">{totalLinesCleared}</span>
          </div>
          <div className="flex flex-col">
            <span className="text-xs uppercase tracking-wider" style={{ color: 'var(--text-secondary)' }}>
              Best Streak
            </span>
            <span className="text-lg font-semibold tabular-nums">{longestStreak}</span>
          </div>
          <div className="flex flex-col">
            <span className="text-xs uppercase tracking-wider" style={{ color: 'var(--text-secondary)' }}>
              Best Combo
            </span>
            <span className="text-lg font-semibold tabular-nums">{highestCombo}x</span>
          </div>
        </div>

        <button
          onClick={onRestart}
          className="w-full py-3 rounded-xl font-semibold text-lg transition-all duration-150 hover:brightness-110 active:scale-95 cursor-pointer"
          style={{
            background: 'linear-gradient(135deg, #4466ff, #8844ff)',
            color: '#ffffff',
          }}
        >
          Play Again
        </button>
      </div>
    </div>
  );
}
