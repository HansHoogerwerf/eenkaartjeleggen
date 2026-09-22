# Roem-aware neural retrain (September 2026)

This runbook covers the second-generation neural card player: why it exists,
what changed in the engine, and the exact commands to regenerate it.

## 1. What was wrong

Play-testing against the public `neural` opponent showed it handing over roem
(honours) for no reason: completing an opponent's sequence with a free discard,
or putting *stuk* on the table under an outstanding trump Jack, up to 70 roem
at a time (10-J-Q-K of trump = sequence 50 + stuk 20).

The root cause was not the net itself but everything it learned from:

| Layer | Roem handling before | Now |
|---|---|---|
| Heuristic card play (`AIPlayer._lead/_try_win/_discard_for_partner/_play_safe_discard`) | Only card points; the roem a card *completes on the table* was never looked at | `_roem_added_by()` penalises completing the opponents' roem, rewards completing the partner's; roem already on the table raises the trick's value |
| Exact endgame solver (`_endgame_minimax`, last 3 cards, also used by the neural hybrid) | Points + last-trick bonus only | Adds `trick_roem_points()` per completed trick |
| 3-trick lookahead (`_lookahead_minimax`, `advanced` profile) | Points only | Same |
| CUDA training engine (`tools/gpu_engine.py`) | Roem computed **from the initial hands at deal time** (an old rule set); nat decided on trick points alone (`<= 81`) | Roem per completed trick to the trick winner; nat = declarer's trick points + roem must strictly exceed the opponents' |
| Neural features (`neural/features.py`) | Nothing about roem on the table; the MLP had to rediscover `find_roem` from one-hot cards | v2 layout adds 33 features: roem each legal card would add, and roem already on the table |
| Imitation teacher | `expert` heuristics (roem-blind) | Mythos (`model_players/mythos_player.py`), whose alpha-beta scores trick roem, nat and pit |

In the four scripted situations covered by `tests/test_ai_roem.py`, the old
engine picked the roem-gifting card about 55% of the time (a coin flip on the
tie-break); the patched engine never does.

Measured on 512 identical rounds of four-seat heuristic self-play (`beginner`
profile, replays analysed for decisions where a roem-free legal card existed):

| | Roem handed to opponents (avoidable) | Events | Avoidable roem completed for own team |
|---|---|---|---|
| Before | 4380 pts (855 per 100 rounds) | 186 | 2210 pts |
| After | 60 pts (12 per 100 rounds) | 3 | 4030 pts |

## 2. Feature layouts

`neural/features.py` now has two layouts. Offsets of the original blocks are
unchanged, so the legal-move mask stays at offset 190 in both.

| Version | Width | Extra blocks |
|---|---|---|
| v1 | 267 | – (all pre-existing checkpoints, incl. `models/neural_best.pt`) |
| v2 | 300 | 34. per-card roem delta if played now, legal cards only, /100 (32) · 35. roem already on the table, /100 (1) |

`KlaverjasNet.from_state_dict()` reads the width from a checkpoint, and
`neural/player.py` picks the matching encoder, so v1 and v2 checkpoints can be
served by the same code. `KlaverjasGPUEngine(feature_version=...)` and
`tools/train_gpu_selfplay.py` follow the checkpoint automatically.

## 3. Pipeline

All steps are wrapped by `tools/run_full_pipeline.py`; the individual commands:

```bash
# 1. Record Mythos self-play (CPU only, ~4.5 s/round/worker; 8000 rounds ≈ 1 h on 14 workers).
#    Skips forced (single-legal-card) decisions; ~21 samples per round.
python tools/generate_training_data.py --teacher mythos --rounds 8000 --workers 14 \
    --output models/training_data_mythos_v2.npz

# 2. Imitation training (needs torch + CUDA). Width is read from the data.
python tools/train_neural.py --data models/training_data_mythos_v2.npz \
    --output models/neural_mythos_v2.pt --epochs 80 --batch-size 512 --patience 15

# 3. Benchmark card play only: same Mythos bidder on both sides.
python tools/ai_benchmark.py --candidate-strength neural_mythosbid --baseline-strength mythos \
    --rounds 512 --workers 15 --model models/neural_mythos_v2.pt

# 4. PPO self-play on the CUDA engine, benchmarking vs Mythos every 25 epochs.
python tools/train_gpu_selfplay.py --model models/neural_mythos_v2.pt \
    --output models/neural_mythos_v2_rl.pt --epochs 300 \
    --benchmark-baseline mythos --benchmark-candidate neural_mythosbid --benchmark-rounds 256

# 5. Try a checkpoint in the web app / show-off without replacing neural_best.pt.
NEURAL_MODEL_PATH=models/neural_mythos_v2_rl.pt python model_players/showoff.py 4 8
```

Promote by copying the winner over `models/neural_best.pt` (or
`run_full_pipeline.py --promote`).

Several datasets can be concatenated: `--data a.npz b.npz` (same width).
Generation writes a checkpoint of the `.npz` every 25 games, so a killed run
still leaves a usable file.

## 4. Training log (21 Sept 2026) — what worked and what did not

Benchmarks: neural card play + Mythos bidder vs Mythos, 512 rounds, seed 7,
`avg_point_diff` per 16-round game (negative = Mythos ahead). Roem audit: 96
rounds of four-seat hybrid self-play, avoidable roem handed to opponents.

| Checkpoint | How it was made | Val. agreement | vs Mythos (win rate / pts per game) | Roem gifted / 96 rounds |
|---|---|---|---|---|
| `neural_best.pt` (old, 267 inputs) | imitation of `expert_v2` + PPO self-play (previous generation) | – | 0.50 / −64 | 430 (18 events) |
| `neural_mythos_v2.pt` | from scratch, 6000 Mythos rounds | 57.3% | 0.03 / −448 | – |
| `neural_mythos_v2_small.pt` | as above, 512-256-128 net | 58.3% | 0.16 / −322 | – |
| `neural_mythos_v2_ft.pt` | old net warm-started (zero-padded) + Mythos fine-tune | 57.3% | 0.31 / −273 | – |
| `neural_mythos_v2_mix.pt` | warm start + 3M roem-upgraded expert_v2 samples + Mythos ×8 | 73.7% | 0.22 / −370 | 450 (18 events) |
| `neural_best.pt` **+ roem guard** | no retraining | – | **0.47 / +17** | **60 (3 events, 2 of them the endgame solver)** |
| *Mythos itself (reference)* | – | – | – | 160 (5 events) |
| `neural_roem_rl.pt` (PPO best, epoch 125) | 300 epochs CUDA self-play from `neural_best_v2pad.pt`, roem rewards | – | 0.50 / −56 (guard on) | 440 guard off, 40 guard on |
| `neural_roem_rl_final.pt` (epoch 300) | as above | – | 0.38 / −98 (guard on) | 470 guard off, 20 guard on |
| `neural_distill_v2.pt` | **self-distillation**: 100k rounds of guarded-hybrid self-play (`--teacher neural_mythosbid`, 2.0M decisions, 7.7 min to record) imitated into a fresh 300-input net | 74.2% | 0.53 / −44 | **0 guard off, 0 guard on** |
| `neural_distill_v2_rl.pt` (PPO from the distilled net, best = epoch 150) | 512-round Mythos benchmarks every 50 epochs on seed 42: −156 (start) → −148 → **+28 → +30** → −155; run killed by the host at epoch 250 (low system memory) | – | **0.53 / +19** | **20 guard off (1 event), 20 guard on** |

Mythos's own figure shows the audit counts some justified cases (a 20-roem
card can be the cheapest loss); the guarded net now sits below that rate.

Lessons:

1. **Imitation alone cannot reach the old net.** Every imitation variant loses
   heavily to Mythos; the old net's strength comes from its self-play stage,
   and fine-tuning on imitation data erases it (catastrophic forgetting).
   The roem-aware retrain must therefore start from the old weights
   (`models/neural_best_v2pad.pt`, the old net zero-padded to 300 inputs, which
   plays identically) and go straight to PPO self-play on the roem-aware CUDA
   engine, with the v2 features available to learn from.
2. **Six thousand Mythos rounds are too few to imitate a search player**
   (57% agreement, strong overfitting). Mythos data is expensive (~4.5 s per
   round per core); if more imitation is wanted, generate overnight.
3. **The gifting pattern is systematic and specific:** the net "saves" a high
   card by throwing the low card that completes the opponents' sequence or
   stuk (Q of trump next to their K to keep the 10, an 8 that makes
   8-9-10-J instead of a bare Ace). `tools/upgrade_dataset_v2.py` and the
   `@k` oversampling in `train_neural.py` did not remove it. The roem guard
   (`AIPlayer._roem_guard`) does, deterministically, and is on by default for
   every neural player.

4. **300 PPO epochs (~20M transitions) from the padded old net neither beat
   the guarded old net nor learned to avoid gifts on their own.** The periodic
   24-game benchmarks swung between −116 and +136 per game; the "+136" best was
   noise (−56 on the clean 512-round seed). Round-end rewards diluted over 32
   plies give too little credit to a single 20-point gift.

6. **Distil, then self-play, is the recipe.** PPO from the distilled net
   reached +19 per game vs Mythos on the common seed (old net −64, old net +
   guard +17) while keeping the roem discipline in its weights (one avoidable
   gift in 96 rounds, guard off). Its 512-round benchmarks still swing by
   ±150 between epochs on the run's own seed, so always re-check a saved best
   on a second seed before promoting it. `models/neural_distill_v2_rl.pt` is
   that checkpoint (epoch 150 of an interrupted 300-epoch run).

5. **Self-distillation works.** Recording the guarded hybrid's own play (fast:
   ~0.05 s per round per core) and imitating it into a fresh 300-input net
   gives a net that never gifts avoidable roem *without* the guard, at the old
   net's strength. The roem features make the guard's rule directly learnable.

Recommended next steps for a net that carries roem discipline in its weights:

* **Self-distillation (done, see table):**
  `python tools/generate_training_data.py --teacher neural_mythosbid --rounds 100000 --workers 14 --output models/training_data_guarded_v2.npz`
  then `python tools/train_neural.py --data models/training_data_guarded_v2.npz --output models/neural_distill_v2.pt --epochs 40 --batch-size 1024 --patience 6`.
  Iterate: distil again from the improved net whenever the guard or solver is
  changed, and follow with PPO self-play.
* **Denser rewards:** emit per-trick point + roem deltas from
  `KlaverjasGPUEngine.step()` instead of only the round total, so PPO can
  attribute a gifted sequence to the card that completed it.
* **More Mythos data** (overnight) if Mythos-style play is the goal; 6000
  rounds gave 57% agreement.

Self-play from the padded old net (the run described above):

```bash
python tools/train_gpu_selfplay.py --model models/neural_best_v2pad.pt \
    --output models/neural_roem_rl.pt --epochs 300 --benchmark-interval 25 \
    --benchmark-rounds 384 --benchmark-baseline mythos --benchmark-candidate neural_mythosbid
```

## 4b. Neural v3 (22 Sept 2026): exact, nat-aware endgame for the hybrid

Full log with every benchmark: [neural-v3-campaign.md](neural-v3-campaign.md).

The biggest lever turned out not to be the net but what finishes the round
for it. The shipped hybrid solved the last 3 cards with a points-only
minimax over a single guessed layout of the hidden cards. Replacing that:

| Endgame of the hybrid (net + Mythos bidder, `neural_best.pt` unchanged) | vs Mythos, 512 rounds, pts/game, seeds 7 / 11 / 13 / 17 | mean |
|---|---|---|
| old solver (3 cards, points only, first consistent deal) | +19 / −103 / · / · | −42 |
| solver v2, 3 cards (sampled deals, nat/pit terminal, alpha-beta) | +62 / +143 / −14 / +131 | +81 |
| solver v2, 4 cards | +204 / +31 / +81 / +99 | +104 |
| **Mythos search from 5 cards** (`NEURAL_ENDGAME_ENGINE=mythos`, now the default) | **+220 / +129 / +146 / +61** | **+139** |

Mythos's int-encoded alpha-beta is exact to the end of the round at ≤ 5
cards, averages up to 10 sampled consistent deals and applies nat and pit
at the terminal; the hybrid now hands it the round from 5 cards down
(~0.3 s per decision). `AIPlayer._endgame_minimax` (the Python solver) got
the same ideas — sampled deals, nat/pit-aware terminal, alpha-beta — and
stays the engine for the internal profiles and as the fallback.

What did **not** help, each checked on two seeds at 512 rounds: PPO
self-play from the shipped net with dense per-trick rewards (`--dense-rewards`,
best snapshot +25 vs +102 for the unchanged net in the same setup), a ladder
run from that snapshot at a lower learning rate, and a sparse-reward control
at the lower rate. The net itself is therefore unchanged in v3; the tooling
(dense rewards, `--save-every` snapshots) stays for future runs.

Knobs on the hybrid (environment): `NEURAL_ENDGAME_ENGINE` (mythos | solver),
`NEURAL_ENDGAME_CARDS`, `NEURAL_ENDGAME_SAMPLES`, and the experimental
`NEURAL_MIDGAME=search` (net ranks, Mythos search picks among the top
`NEURAL_MIDGAME_TOPK` within `NEURAL_MIDGAME_BUDGET` seconds).

## 5. Verifying the CUDA engine

`tests/test_gpu_engine_roem.py` pins the tensor implementation to the Python
rules (random tricks vs `find_roem`, v2 feature block vs `encode_state`, nat
rule). It is skipped automatically when torch cannot be imported; run it once
torch works:

```bash
python -m unittest tests.test_gpu_engine_roem -v
```

## 6. Troubleshooting: `WinError 4551` importing torch

```
OSError: [WinError 4551] An Application Control policy has blocked this file.
Error loading "...\site-packages\torch\lib\torch.dll"
```

Windows **Smart App Control** (Windows Security → App & browser control) is
refusing to load the unsigned `torch.dll` from the pip wheel. It is a
machine-wide policy, not a Python problem; the Code Integrity event log
(`Microsoft-Windows-CodeIntegrity/Operational`, event 3077) names the blocked
file. Until it is allowed, only the CPU-only stages work (generation,
`ai_benchmark.py` with the heuristic fallback, the unit tests); the neural
opponents silently fall back to heuristic play.
