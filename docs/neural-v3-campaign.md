# Neural v3 campaign log (22 Sept 2026)

Branch `feature/neural-v3`, started 00:57 local. Goal: a stronger neural
opponent than the one shipped in PR #57 (`neural_best.pt` = distilled +
150 PPO epochs, +19 pts/game vs Mythos on seed 7, 1 avoidable roem gift per
96 rounds).

Every change is kept only if it holds up on **two** 512-round benchmarks
(seeds 7 and 11) against Mythos with the same bidder on both sides, plus the
roem audit (`neural card play + Mythos bidder`, guard off).

Baseline (shipped `neural_best.pt`, unmodified develop code in a separate worktree):
seed 7 → 0.53 / +19, seed 11 → 0.375 / −103 (mean −42). Seed spread is ~120
points per game, so a change needs to move both seeds to count.

## Plan

1. Endgame solver v2 (hybrid's last cards): average over sampled consistent
   deals, nat/pit-aware terminal, try 4 cards instead of 3.
2. Dense per-trick rewards in the CUDA engine; longer self-play from the
   shipped net.
3. Ladder self-play (opponent = previous best); re-distil into a wider net.
4. Two-seed benchmarks, audits, docs, PR.

## Log

| Time | Step | Result |
|---|---|---|
| 00:57 | branch created from develop | – |
| 01:05 | endgame solver v2 (sampled deals, nat/pit terminal, `endgame_cards` knob) + tests | tests OK; A/B benchmarks running |
| 01:20 | dense per-trick rewards in the CUDA engine (`--dense-rewards`), snapshot option (`--save-every`) + parity test | tests OK |
| 01:25 | dense-reward self-play from `neural_best.pt`, 300 epochs, snapshots every 25 | running on GPU |
| 01:30 | baseline seed 11 | 0.375 / −103 |
| 01:50 | solver v2, 3 cards, 8 deals | seed 7: 0.53 / **+62** (nats 51 vs 52); seed 11: 0.625 / **+143** (nats 38 vs 52) → mean +102 vs baseline −42 |
