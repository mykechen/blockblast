// Web Audio API synthesized sound effects — no audio files needed

let ctx: AudioContext | null = null;

function getCtx(): AudioContext {
  if (!ctx) {
    ctx = new AudioContext();
  }
  return ctx;
}

function playTone(
  freq: number,
  duration: number,
  type: OscillatorType = 'sine',
  volume = 0.15,
  ramp?: { to: number; duration: number }
) {
  try {
    const c = getCtx();
    const osc = c.createOscillator();
    const gain = c.createGain();
    osc.type = type;
    osc.frequency.value = freq;
    gain.gain.value = volume;
    gain.gain.exponentialRampToValueAtTime(0.001, c.currentTime + duration);
    if (ramp) {
      osc.frequency.exponentialRampToValueAtTime(ramp.to, c.currentTime + ramp.duration);
    }
    osc.connect(gain);
    gain.connect(c.destination);
    osc.start();
    osc.stop(c.currentTime + duration);
  } catch {
    // Audio not available
  }
}

export function playPlace() {
  playTone(400, 0.08, 'sine', 0.1);
  setTimeout(() => playTone(500, 0.06, 'sine', 0.08), 30);
}

export function playClear() {
  playTone(523, 0.15, 'sine', 0.12);
  setTimeout(() => playTone(659, 0.15, 'sine', 0.12), 80);
  setTimeout(() => playTone(784, 0.2, 'sine', 0.1), 160);
}

export function playCombo(comboLevel: number) {
  const baseFreq = 523 + comboLevel * 80;
  playTone(baseFreq, 0.12, 'sine', 0.12);
  setTimeout(() => playTone(baseFreq * 1.25, 0.12, 'sine', 0.12), 60);
  setTimeout(() => playTone(baseFreq * 1.5, 0.15, 'sine', 0.1), 120);
  setTimeout(() => playTone(baseFreq * 2, 0.2, 'triangle', 0.08), 180);
}

export function playGameOver() {
  playTone(400, 0.3, 'sine', 0.12, { to: 200, duration: 0.3 });
  setTimeout(() => playTone(300, 0.4, 'sine', 0.1, { to: 150, duration: 0.4 }), 200);
}

export function playPickup() {
  playTone(600, 0.05, 'sine', 0.06);
}

export function playInvalid() {
  playTone(200, 0.1, 'square', 0.05);
}
