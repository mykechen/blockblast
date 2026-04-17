# Block Blast DQN — Training an Agent to Play a Puzzle Game From Scratch

A portfolio writeup of how I built and trained a Dueling Double DQN agent with prioritized experience replay to play Block Blast, a tetris-adjacent puzzle game I'd already implemented as a Next.js web app.

**Headline result:** After 2M training steps (~97k games), the learned policy scores **7.3x** a random-play baseline — mean score of **1,434** on 100 held-out seeds vs. **197** for a uniform-random-valid-action policy. Best single game: **10,002**. This was trained on an M-series MacBook using the MPS backend, no cloud, no GPU cluster.

The interesting part isn't the number — it's what shaped it. This doc is about the decisions and the iteration.

---

## The problem

Block Blast is deceptively hard for an RL agent:

- **The board is 8×8** (64 cells, 2^64 possible configurations — much too big to tabulate).
- **Every turn, the agent is handed 3 random pieces** from a catalog of tetris-like shapes, which must be placed before 3 new pieces arrive. Action space per turn: up to 64 positions × 3 pieces = **192 discrete actions**, most of them illegal on any given board.
- **Reward is delayed.** Clearing lines only happens when a row or column is completely filled; survival depends on planning multiple pieces ahead.
- **Games are short and bursty.** A random player averages ~12 pieces before losing. A trained agent doubles that. One good decision early can mean 10x the final score.

I already had a working JavaScript implementation of the game. The challenge was turning it into something an RL algorithm could learn from.

---

## System design

```
app/           — Next.js React game (original, human-playable)
lib/           — TypeScript game logic (board, pieces, scoring)
training/
  env/         — Python port of the game engine, wrapped as a Gymnasium env
  agent/       — Dueling DQN model, prioritized replay buffer, training loop
  scripts/     — train.py, eval.py
  tests/       — 65 unit + parity tests
  configs/     — hyperparameters as YAML
```

The layering matters:

1. **TS engine** stays the source of truth for the human-playable game.
2. **Python port of the engine** is a line-for-line rewrite of the rules (`board.py`, `pieces.py`, `game.py`, `scoring.py`). I wrote **38 parity tests** that seed both engines identically and assert byte-equal outputs — this caught two off-by-one bugs in line-clearing before they poisoned training.
3. **Gymnasium env** (`BlockBlastEnv`) wraps the Python engine with a standard `reset`/`step` interface, an observation encoder, an action mask, and a configurable reward function.
4. **Agent + trainer** consumes the env. Algorithm-agnostic enough to swap in PPO later without touching the env.

Total: ~2,000 lines of Python, 65 tests, split across 17 files.

---

## Observation encoding

A key decision: what does the agent see each turn?

The observation is a **(7, 8, 8) float32 tensor**:

- **Channel 0:** the 8×8 board, 1 where occupied.
- **Channels 1–3:** the shape masks of the 3 current pieces (padded to 8×8; pieces max out at 5×5).
- **Channels 4–6:** presence indicators — all ones if piece *i* is still available, all zeros if already placed.

Why channels for piece presence? Because once a piece is placed, that slot is empty until all three are used and a fresh trio is drawn. The model needs to distinguish "piece 2 is the shape X" from "piece 2 has been used." The binary presence channels make this trivial for a conv net to learn.

A CNN was the right fit: shapes are translation-invariant. A piece placed at (0,0) looks the same as at (5,5) locally.

---

## Action space and masking

Naïve action space: 192 discrete actions = 3 pieces × 8 rows × 8 cols. Tons of these are illegal on any given turn (piece out of bounds, overlap with existing blocks, piece already used).

I tried two approaches in my head before building:

1. **Output valid actions only** — model produces Q-values over a variable-length list.
2. **Fixed 192-dim output + action mask** — mask invalid actions to `-inf` before argmax.

Went with #2. It matches the DQN math (fixed action space, simple argmax), the mask is cheap to compute (`get_action_mask`, ~37 lines), and I could use the same mask in the replay buffer when computing Double DQN targets for next-state actions.

**Lesson learned:** the mask needs to be applied in *two* places — during action selection AND when computing the target network's best-next-action in the Bellman update. I forgot the second one initially; the test suite caught it (the target net was proposing invalid actions, which destabilized training targets). Fixed in `dqn.py:74-76`.

---

## Algorithm: Dueling Double DQN + prioritized replay

Three well-known DQN enhancements, stacked:

### Dueling architecture (Wang et al. 2016)

The network splits into value and advantage streams after a shared CNN trunk:

```
Q(s, a) = V(s) + A(s, a) − mean_a A(s, a)
```

Intuition: in Block Blast, most states aren't sensitive to *which* action you pick — they're just good or bad positions. The dueling head lets the value stream learn "this board is healthy" without entangling that signal with advantage-per-action.

### Double DQN (van Hasselt et al. 2016)

Standard DQN maxes Q from the same network that's being trained, which overestimates action values. Double DQN decouples: policy network picks the action, target network evaluates it. Meaningful in a 192-dim action space where even small overestimation accumulates.

### Prioritized experience replay (Schaul et al. 2016)

Sample transitions proportional to `(|TD_error| + ε)^α` instead of uniformly. Implemented with a **sum-tree** for O(log n) sampling over a 500k-transition buffer (`replay_buffer.py`). Importance-sampling weights anneal β from 0.4 → 1.0 over training to correct for the non-uniform sampling.

The buffer stores `(state, action, reward, next_state, done, next_action_mask)` — the mask has to travel with the transition because it's needed in the Double DQN update.

### Architecture

```
Input: (batch, 7, 8, 8)
Conv 7→32, 32→64, 64→128 (3×3, BatchNorm, ReLU)
Flatten → Linear(8192, 512) → ReLU → Dropout(0.1)
Value head: Linear(512, 256, 1)
Advantage head: Linear(512, 256, 192)
```

~4.4M parameters. Small by today's standards, but the game is small too.

---

## Reward shaping

Shaping is the single most underrated lever in RL on games with sparse scores. My current reward per step:

```
cells_cleared × 10.0              # primary: clearing 8 cells = +80
multi_line_bonus × 15.0           # combo bonus for 2+ lines in one move
board_cleanliness × 0.1           # mild pull toward emptiness
holes × −2.0                      # discourage unreachable cavities
survival × +0.5                   # small per-step bonus
game_over × −50.0                 # terminal penalty
```

A hole is defined as an empty cell with 3+ filled-or-out-of-bounds orthogonal neighbors. Simple, fast, captures the "trapped pocket" intuition.

**Things I considered and rejected:**

- **Raw game score as reward.** Too sparse — a bad game yields almost nothing, and the variance across games would make TD learning noisy.
- **Potential-based shaping F(s,s') = γφ(s') − φ(s).** Theoretically the right thing (preserves optimal policy), but tuning φ turned out harder than just adding direct shaping terms with small weights.
- **Curriculum learning.** Start with easier piece distributions. Overkill for a game this size — the agent learns fine from the full distribution.

---

## Training run — what the numbers show

**Hyperparameters:**

- 2M environment steps
- Epsilon greedy 1.0 → 0.05 linearly over the first 500k steps (so 75% of training is at ε=0.05)
- γ = 0.99, batch 128, lr 1e-4
- Target net sync every 2000 steps, train every 4 steps
- Replay buffer 500k, min fill 10k before learning starts

Trained for ~3.5 hours on MPS.

**Progression (TensorBoard logs):**

| Milestone       | Mean score (last 100 ep) | Mean pieces placed | Loss |
| --------------- | ------------------------ | ------------------ | ---- |
| ~step 10k       | ~220                     | ~12                | 31.2 |
| step 250k       | ~700                     | ~17                | ~12  |
| step 1M         | ~1,180                   | ~20                | ~7   |
| step 2M (final) | ~1,155                   | ~21                | 4.6  |

(Numbers include ε=0.05 exploration noise. Greedy eval below.)

**Deterministic evaluation** (100 games at ε=0, same seeds across checkpoints):

| Checkpoint            | Mean | Median | p75  | Max    |
| --------------------- | ---- | ------ | ---- | ------ |
| Random baseline       | 197  | 122    | 240  | 1,205  |
| checkpoint_50k        | 246  | 132    | 295  | 1,132  |
| checkpoint_100k       | 318  | 131    | 320  | 2,990  |
| checkpoint_1M         | 1,180| 700    | 1,668| 10,001 |
| checkpoint_1.45M      | 1,289| 1,092  | 1,723| 4,698  |
| checkpoint_1.75M      | 1,430| 1,058  | 1,793| 9,468  |
| checkpoint_1.9M       | 1,274| 884    | 1,692| 5,608  |
| checkpoint_1.95M      | 1,327| 806    | 1,468| 10,002 |
| **final_model.pt**    | **1,434** | **1,044** | **1,887** | **5,634** |

Two readable takeaways:

1. **Performance is noisy across late checkpoints.** `1.75M`, `1.9M`, `1.95M`, and `final` all land within one std dev of each other. The mean moved from 1,180 at step 1M to 1,434 at step 2M — roughly +250 over a full additional million steps. That's diminishing returns.
2. **The right tail is fat.** Median is ~1,040 but max is ~5,600 and occasional runs hit 10k+. The agent clearly *knows* how to play well; it just doesn't do so consistently. That's a variance problem, not a capability problem.

---

## What worked

- **Action masking in both the selection path AND the target computation.** The training got dramatically more stable after I fixed the target-side mask — target Q-values stopped being poisoned by "imagine placing an invalid piece" estimates.
- **Parity tests between the TS and Python engines.** I spent a day on 38 tests that all ran seeded games in both engines and asserted matching board states. This sounds paranoid; it saved me later when I discovered a line-clearing edge case only in the TS version. Without parity tests, any training divergence could have been "my algorithm" OR "my env is subtly wrong" — impossible to separate.
- **Prioritized replay over uniform.** Even in a 2M-step budget, priority sampling noticeably accelerated early-phase learning. The buffer is full of boring "survived one more step" transitions; priority focuses learning on the moments where the model was actually wrong.
- **Small-but-deep CNN.** 3 conv layers + 1 shared linear + 2 heads. On an 8×8 input, going deeper doesn't help — you run out of spatial resolution. 4.4M params trains fast on MPS.
- **YAML config files.** Not glamorous, but every hyperparameter in `configs/default.yaml` means experiments reduce to `--config configs/exp_X.yaml` without touching code. Makes ablations trivial.

## What didn't work / what I'm still fighting

- **`hole_penalty=2.0` is probably too weak.** Relative to `cells_cleared=10.0`, a hole costs less than a quarter of a single cell clear. The agent will happily create pockets if it means placing a piece cleanly. I'm currently running an experiment with `hole_penalty=5.0` resumed from `final_model.pt` for +500k steps to test this.
- **Epsilon decay finishes too early.** Linear decay to 0.05 by step 500k means the remaining 1.5M steps are mostly exploitation. For a game where exploration matters (the agent doesn't know how to set up multi-line combos until it's stumbled into them), this is probably suboptimal. Next experiment: extend `epsilon_decay_steps` to 1M.
- **No `mean_q` logging.** I log loss, epsilon, buffer size — but I'd have learned more from Q-value statistics over time. A rising mean Q is a healthy sign; a flat one suggests the value function is collapsed. Fixing this is a one-line change.
- **Observation has no lookahead.** The web game shows only the 3 current pieces, not a preview of what's coming next. That's a hard information ceiling for the policy — even a perfect Q function can't plan 4 pieces ahead if it can't see piece 4.
- **No self-play, no population-based training, no MCTS.** This is vanilla DQN. A stronger variant would be AlphaZero-style MCTS using the learned network as a policy/value prior — probably worth 2-3x more score.

---

## What I'd do next

In order of expected bang-for-buck:

1. **Reward-shape sweep.** 3-4 configs with different `hole_penalty` × `cells_cleared_weight` ratios, 500k steps each resumed from `final_model.pt`. Cheap. Running experiment #1 now.
2. **Longer epsilon decay + longer total training.** 1M-step decay, 4M total. ~7 hours.
3. **Log Q-value stats and per-action reward attribution** so I can actually see *what* the agent values, not just whether it's winning.
4. **PPO baseline.** DQN on a discrete action space with masking is a decent fit, but PPO with action masking and GAE would be a fair comparison — and in stochastic-reward games PPO tends to have lower variance.
5. **MCTS + value network.** The real endgame. Use the trained DQN as an AlphaZero-style prior and do 50-100 rollouts per move. Expensive at inference but would push past the plateau.
6. **Wire it into the web UI.** Export the model to ONNX, serve via onnxruntime-web, and let it play the actual Next.js game in the browser as a demo.

---

## What I took away

Two things I'll carry to the next RL project:

1. **Build the tooling first.** Parity tests between your "truth" environment and your training environment. A config system that makes experiments free. Eval scripts that give a single comparable number across checkpoints and baselines. I wrote all three before running the big training run and they paid for themselves within a day.
2. **Evaluate deterministically, report distributions.** Training curves include exploration noise and are a terrible basis for comparing checkpoints. Fix seeds, set ε=0, run 100 episodes, report mean/median/p25/p75/max. That's the number to trust — and the right shape of the answer, since in a game like this the mean hides a fat right tail.

The plateau at ~1,400 mean score isn't a failure — it's the signal that tells me which lever to pull next. Diminishing returns on step count means the bottleneck is upstream: reward shaping, observation richness, or algorithm. That's a much more interesting problem than "run it longer."

---

## Follow-up: porting to CUDA and pushing the ceiling

After the MacBook run, I ported the training stack to a desktop with an RTX 4070 and ran a series of experiments to see how far the DQN architecture could go.

### Porting to CUDA

The `uv.lock` was generated on macOS with CPU/MPS PyTorch wheels. Replacing them with CUDA 12.4 wheels meant pinning a custom index in `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu124"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu124" }
```

A one-off `uv pip install --force-reinstall` doesn't stick — the next `uv run` re-syncs the lockfile. The only persistent fix is lockfile-level. Also: `get_device()` was MPS-first, swapped to CUDA → MPS → CPU. One-line change, all 65 tests still passed.

Throughput on the 4070: ~220 SPS sustained, 35-40% GPU utilization, ~1.7GB VRAM. The env is CPU-bound (step is pure NumPy), not GPU-bound — meaning the 4070 was idle most of the time, and background apps eating CPU on the host mattered more than I expected.

### Tier 2: a stronger base model

Same approach as the MacBook run but with a residual model (96 channels × 4 blocks, 768 FC) and 2M steps. Config at `configs/tier2.yaml`. 28 minutes wall time.

| Checkpoint | Mean (100 held-out eps) | vs Random (222) |
|---|---|---|
| `checkpoint_1850000.pt` | **2,495** | 11.2× |
| `final_model.pt` | 2,190 | 9.9× |

**74% improvement over the writeup's 1,434** with the same step budget, same algorithm, just the bigger model. The best checkpoint wasn't the final one — late-training instability is a known DQN quirk and worth sweeping for.

### What inference-time search *didn't* do

Implemented 1-step lookahead (simulate each valid action, pick `argmax(r + γ · max Q(s'))`) expecting +15-25% as the literature suggests. Result on tier2:

```
Greedy:           mean 2495.4
1-step lookahead: mean 2495.4  (identical to the decimal)
2-step pruned (k=8): mean 2495.4  (also identical)
```

**Zero gain.** Not a bug. This is what happens when the Q-net is tightly converged: `argmax_a Q(s,a)` already equals `argmax_a [r(s,a) + γ·max Q(s',a')]` because the Bellman optimality condition is nearly satisfied. Inference-time search only helps when the value function is noisy or miscalibrated — it's a correction for imperfect training, not an amplifier for good training.

Gains from search would require either:
1. A value head that provides a *different* signal than the policy argmax (AlphaZero-style separation), or
2. Deep MCTS with many simulations to discover trajectories the greedy rollout wouldn't,
3. Distributional RL where the full return distribution informs search.

Lesson for the doc I wish I'd found earlier: *"If your Q-net is already strong, don't expect shallow lookahead to stack on top. It's a fix for a different failure mode."*

### What n-step returns *didn't* do

Added 3-step return aggregation in the replay buffer — stock Rainbow ingredient, well-documented gain on Atari. Expected +10-25%.

Combined with a fresh 5M-step run (`configs/tier3.yaml`) that kept the residual model and added n-step. At 90% through training, the training-mean score was **~900 vs tier2's 1,586 at 2M**. The gamble didn't pay off.

Likely causes, in order of plausibility:
1. **3-step averaging blurs credit on bursty reward signals** — line clears are spike-like; averaging 3 steps of reward before bootstrapping diffuses the gradient signal for the move that actually caused the clear.
2. **Longer ε decay (1.5M vs 750k)** — twice the exploration budget, half the effective exploitation time per step.
3. **Replay buffer staleness** — 500k buffer against 5M training steps means the oldest experience is 10× off-policy by the end, which can destabilize convergence.

I ran it anyway on the "you don't learn which levers fail without pulling them" principle. Also a useful negative result: **the recipes that compose well on Atari don't all compose on every MDP**, especially ones with sparse bursty rewards and short episodes.

The first instinct — hole_penalty shaping at -5/hole — was killed even earlier. The agent settled at ~200 training score with ε near minimum because the -5 penalty dominated cell-clear reward, so the policy that minimized hole creation was "place pieces conservatively and lose quickly." Reward shaping in RL is alignment-adjacent; a stronger penalty can create a new optimum that wasn't what you wanted to optimize.

### Demo search: where the fat tail lives

What actually produced demo-grade games was a **seed sweep harness** (`scripts/demo_search.py`): eval the greedy policy on N seeds, sort by score, save the top-K with their action sequences to JSON. Replay any game deterministically with `scripts/replay_demo.py`.

On tier2's `checkpoint_1850000.pt` over 10,000 seeds:
- Mean: 2,075 (close to the 100-ep eval, confirms the sample is representative)
- Median: 1,541
- p95: 5,700
- p99: 9,076
- **Max: 32,147** (seed 5,008,089, 62 moves)

The max-over-10k is the important number for a demo. The tail is much heavier than exponential — one-in-10k already clears 32k, extrapolating suggests one-in-1M clears ~100k. That's cheap search (~25 hours of greedy rollouts on the 4070) compared to the alternative of training a much stronger model.

This framing changed how I think about the agent. **Mean score is what you train for; tail score is what you demo.** They're related but not the same metric, and the right compute budget split between "train longer" vs "sweep seeds" depends on which one you care about.

### Updated results

| Setup | Training steps | Mean (100-ep eval) | Demo-search max (N seeds) |
|---|---|---|---|
| Random policy | — | 222 | 1,815 (n=100) |
| Original writeup (MacBook, MPS) | 2M | 1,434 | 10,002 (n=100) |
| Tier 1 (RTX 4070, small model) | 500k | 1,133 | 6,046 (n=100) |
| Tier 2 (RTX 4070, residual) | 2M | **2,495** | **32,147** (n=10,000) |
| Tier 3 (residual + n-step=3) | 5M | underperformed tier 2 | — |

### What the chain said about next steps

After all the extensions, the honest lever-ranking has shifted:

1. **Seed search scales linearly with compute and produces demo-grade artifacts reliably.** 100k scores are probably 1-in-1M seeds on tier2 — feasible with an overnight run.
2. **Algorithmic extensions** (C51 distributional, MCTS with value head, PPO) remain the ceiling-raisers but each costs a day+ of engineering and doesn't obviously compound with the next one.
3. **Longer training of the same recipe** is the lowest-yield lever at this point. Diminishing returns past 2M steps on the same hyperparameters.

The thing I learned most on the extension: *a result that disproves an intuition is worth more than one that confirms it.* n-step failing here, 2-step lookahead failing here, hole_penalty failing here — those each took an evening and each told me something concrete about where the remaining gains aren't.
