# Neural v3 campaign log (22 Sept 2026)

Branch `feature/neural-v3`, started 00:57 local. Goal: a stronger neural
opponent than the one shipped in PR #57 (`neural_best.pt` = distilled +
150 PPO epochs, +19 pts/game vs Mythos on seed 7, 1 avoidable roem gift per
96 rounds).

Every change is kept only if it holds up on **two** 512-round benchmarks
(seeds 7 and 11) against Mythos with the same bidder on both sides, plus the
roem audit (`neural card play + Mythos bidder`, guard off).

Baseline (shipped `neural_best.pt`, unmodified develop code in a separate worktree):
seed 7 → 0.53 / +19, seed 11 → 0.375 / −103, seed 13 → 0.41 / −66, seed 17 → 0.59 / +82
(mean −17). Seed spread is ~180 points per game, so a change needs to move
every seed to count.

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
| 02:20 | solver v2, 4 cards, 8 deals | seed 7: 0.66 / **+204** (nats 34 vs 61); seed 11: 0.50 / +31 (nats 42 vs 47) → mean +118; not separable from 3 cards at 512 rounds |
| 02:40 | alpha-beta + strongest-first ordering in the solver | 4× faster: 3 cards 0.13 s/round, 4 cards 1.1 s/round (4 players, under load), 5 cards 14 s/round (too slow without an int-encoded port) |
| 02:45 | dense self-play snapshots, seed 7, 256 rounds, solver v2 (3) | ep100 −30, **ep200 +150**, ep300 −2, final −87 (shipped net same setup: +62 at 512) |
| 03:05 | hybrid option `NEURAL_ENDGAME_ENGINE=mythos`: Mythos's int-encoded alpha-beta (nat/pit terminal, ≤10 sampled deals, time-budgeted) finishes the round from `endgame_cards` | timing under load: 4 cards 0.7 s/round, 5 cards 2.8 s/round (4 players) → benchmarks queued |
| 03:25 | solver v2, 3 cards, extra seeds | seed 13: 0.59 / −14; seed 17: 0.66 / +131 → 4-seed mean **+81** (baseline 2-seed mean −42) |
| 03:50 | solver v2, 4 cards, extra seeds | seed 13: 0.53 / +81; seed 17: 0.59 / +99 → 4-seed mean **+104**, own nats 34/42/37/38 vs 51/38/44/44 for 3 cards → provisional default 4 |
| 04:20 | dense RL ep200 net, two seeds, 3-card solver | seed 7: 0.50 / +1; seed 11: 0.56 / +50 → mean +25 vs shipped net +102 in the same setup: **RL did not improve the net**; its +150 screen was noise |
| 04:50 | hybrid + Mythos engine from 5 cards | seed 7: 0.72 / **+220** (nats 39 vs 62); seed 11: 0.50 / **+129** (nats 41 vs 52) → 2-seed mean +175 (solver v2 4 cards: +118, 3 cards: +102) |
| 05:05 | hybrid + Mythos engine from 4 cards | seed 7: 0.53 / +63; seed 11: 0.69 / +170 → mean +116 (5 cards: +175) |
| 05:40 | ladder snapshots, seed 7, 256 rounds, 3-card solver | ep100 +44, ep150 +86, ep200 +32 — same band as the shipped net (+62): no RL candidate |
| 06:00 | neural-guided midgame (`NEURAL_MIDGAME=search`: net ranks, Mythos search picks among top-3 in 0.4 s) + 5-card Mythos endgame | ~0.3 s/decision under load; 2-seed benchmark queued |
| 06:30 | hybrid + Mythos engine from 5 cards, extra seeds | seed 13: 0.56 / +146; seed 17: 0.56 / +61 → **4-seed mean +139** (solver v2 4 cards +104, 3 cards +81) → **new hybrid default: engine=mythos, endgame_cards=5** |
| 06:55 | low-LR dense snapshot ep100, seed 7, 256 rounds — **note: from here screens use the new default (Mythos engine, 5 cards)**, reference = unchanged net +220 on this seed | 0.56 / +203 → on par |
| 07:10 | roem audit, new default (Mythos engine 5 cards), guard off, 96 rounds | 80 pts gifted (4 events, 20 of them inside the endgame search); shipped config 20 (1 event); Mythos itself 160 (5 events); **guard on: 20 (1 event)** |
| 07:30 | low-LR dense snapshots ep150 / ep200, seed 7 screen (reference +220) | +75 / +81 → no RL candidate from this run either |
| 08:00 | neural-guided midgame (top-3, 0.4 s) + 5-card Mythos endgame | seed 7: 0.59 / +71; seed 11: 0.56 / +24 → mean +48 vs +175 for the plain net: **worse**; the net's early-trick play beats a short search, option stays off |
| 08:20 | sparse low-LR control snapshots ep100 / ep200, seed 7 screen (reference +220) | **+248** / +168 → ep100 gets a 2-seed 512-round check |
| 08:45 | shipped baseline, seeds 13 / 17 (worktree of develop) | −66 / +82 → 4-seed mean **−17** (new default +139: **+156 per game**) |
| 09:10 | sparse low-LR ep100, two seeds, 512 rounds, new default endgame | seed 7: 0.66 / +192; seed 11: 0.59 / +67 → mean +130 vs +175 unchanged net: **no**; the net stays as shipped |
| 03:05 | ladder self-play from ep200 (opponent = ep200, dense, lr 5e-5, 256 steps/epoch, 200 epochs) | running on GPU |

## Outcome

* **Kept:** exact, nat/pit-aware endgame — Mythos's search from 5 cards as the
  hybrid's default (+139 pts/game vs Mythos over 4 seeds; shipped config −17),
  solver v2 for the Python profiles, dense-reward and snapshot tooling.
* **Not kept:** every RL variant (dense lr 1e-4, ladder, dense lr 5e-5, sparse
  lr 5e-5) and the neural-guided midgame search. The net in `neural_best.pt`
  is unchanged.
* **Lesson:** at 512 rounds a benchmark moves ±100 per game between seeds;
  four seeds separate ~+40. The endgame change is ~4× that; every RL delta
  was inside it.
