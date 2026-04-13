'use client';

import { useEffect, useRef, useState } from 'react';

type ScoreDisplayProps = {
  score: number;
  highScore: number;
  combo: number;
  streak: number;
};

function AnimatedNumber({ value }: { value: number }) {
  const [display, setDisplay] = useState(value);
  const rafRef = useRef<number>(0);
  const startRef = useRef<number>(0);
  const fromRef = useRef<number>(value);

  useEffect(() => {
    if (value === display) return;
    fromRef.current = display;
    startRef.current = performance.now();
    const duration = 400;

    function animate(now: number) {
      const elapsed = now - startRef.current;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = Math.round(fromRef.current + (value - fromRef.current) * eased);
      setDisplay(current);
      if (progress < 1) {
        rafRef.current = requestAnimationFrame(animate);
      }
    }

    rafRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(rafRef.current);
  }, [value]);

  return <span>{display.toLocaleString()}</span>;
}

export default function ScoreDisplay({ score, highScore, combo, streak }: ScoreDisplayProps) {
  return (
    <div className="flex items-center justify-between w-full max-w-[480px] px-4 py-2">
      <div className="flex flex-col items-start">
        <span className="text-xs uppercase tracking-wider" style={{ color: 'var(--text-secondary)' }}>
          Score
        </span>
        <span className="text-2xl font-bold tabular-nums">
          <AnimatedNumber value={score} />
        </span>
      </div>

      <div className="flex gap-3 items-center">
        {combo > 0 && (
          <div
            className="px-2 py-1 rounded text-sm font-bold"
            style={{
              animation: 'popIn 0.3s ease-out',
              background: 'rgba(255, 136, 0, 0.2)',
              color: '#ff8800',
            }}
          >
            {combo}x COMBO
          </div>
        )}
        {streak > 1 && (
          <div
            className="px-2 py-1 rounded text-sm font-bold"
            style={{
              animation: 'popIn 0.3s ease-out',
              background: 'rgba(68, 221, 136, 0.2)',
              color: '#44dd88',
            }}
          >
            {streak} STREAK
          </div>
        )}
      </div>

      <div className="flex flex-col items-end">
        <span className="text-xs uppercase tracking-wider" style={{ color: 'var(--text-secondary)' }}>
          Best
        </span>
        <span className="text-lg font-semibold tabular-nums" style={{ color: 'var(--text-secondary)' }}>
          <AnimatedNumber value={highScore} />
        </span>
      </div>
    </div>
  );
}
