# Neural v4 campaign log (22 Sept 2026): imitate Mythos, then self-improve

Branch `feature/neural-v4-imitation` (from `feature/neural-v3`). Goal: a
stronger **net** — measured as the deployed hybrid (net + Mythos endgame
from 5 cards + Mythos bidder) vs Mythos, 512 rounds per seed, at least two
seeds, compared with the unchanged `neural_best.pt` in the same setup
(seed 7 → +220, seed 11 → +129, seed 13 → +146, seed 17 → +61, seed 29 → +119
(1024 r), seed 31 → +49 (1024 r)).

Why the earlier imitation attempts failed (see neural-roem-retrain.md):
6000 Mythos rounds gave 57% agreement and weak nets; fine-tuning the RL net on
that data erased its self-play skills (−273 vs +19 at the time). This time:

1. **Data at scale:** 50 000 Mythos rounds (~1M decisions), recorded with the
   v2 features while the GPU runs a long self-play in parallel.
2. **Gentle imitation:** warm-start from the current net, low learning rate,
   snapshots every epoch, and the checkpoint chosen by *benchmark* (hybrid vs
   Mythos), not by validation accuracy.
3. **Self-improvement:** PPO self-play from the best imitation net (dense
   rewards, low LR, snapshots), screened the same way.
4. Only a net that beats the current one on two seeds is promoted.

## Log

| Time | Step | Result |
|---|---|---|
| 08:08 | branch from v3; recording 50 000 Mythos rounds (14 workers, budget 1.0 s) | running, ETA ~5 h |
| 08:08 | long PPO from `neural_best.pt`: 600 epochs, lr 3e-5, dense rewards, 256 steps/epoch, snapshots every 50 | running on GPU |
| 08:15 | `train_neural.py --save-every-epoch` for benchmark-based checkpoint selection | – |
| 08:40 | recipe trial on the first data checkpoint (3200 Mythos rounds, 66k decisions): warm start from `neural_best.pt`, lr 3e-5, 6 epochs, per-epoch snapshots; screen epochs 2/4/6 at 256 rounds (seed 7, 6 workers so recording keeps most of the CPU) | running |
| 08:45 | long PPO at epoch 540/600 (~4 s/epoch); snapshots 200/400/600 queued for the same screen | running |
| 09:05 | trial fine-tune (3200 Mythos rounds, lr 3e-5, warm start) — val agreement 52% → 55%; seed-7 screens, 256 rounds: ep2 0.75 / +274, **ep4 1.00 / +429** (nats 12 vs 39, declare ok 0.89), ep6 0.56 / −38 | screens too noisy to rank; 512-round confirmations on seeds 11/13 queued |
| 09:05 | long PPO (600 epochs, lr 3e-5) finished; snapshots 200/400/600 being screened | – |
| 09:20 | fine-tune `v4_ft_c` on the 6400-round checkpoint (132k decisions), same recipe | val agreement 53.6% → 55.5%; epochs 2/4/6 queued for 512-round confirmation on seeds 11/13 |
| 09:20 | PPO long snapshot ep200, seed-7 screen | 0.375 / −119 (reference +220) |
| 10:05 | PPO long snapshots ep400 / ep600, seed-7 screens | +213 / +108 (reference +220): no gain from 600 epochs at lr 3e-5 |
| 10:45 | trial fine-tune ep4, 512 rounds | seed 11: 0.53 / +33 (ref +129); seed 13: 0.59 / +101 (ref +146) → ~70/game **below** the current net; the +429 screen was a 16-game fluke |
| 10:55 | anchored fine-tune `v4_ft_mix`: guarded self-play data (2.0M) + Mythos 6400 rounds ×4, warm start, lr 3e-5, 3 epochs | val agreement 76.9% (mixed val set); epochs 1/3 queued for 512-round confirmation on seeds 11/13 |
| 11:30 | trial fine-tune ep2, 512 rounds | seed 11: +39 (ref +129); seed 13: +44 (ref +146) → ~90/game below the current net. Consistent with v3's guided-midgame result: **the net's early-trick play is already better than Mythos's, so imitating Mythos there hurts** |
| 11:50 | trial fine-tune ep6, seed 11 | +144 (ref +129) — epochs differ wildly on the same data; no consistent gain |
| 12:10 | **net-rollout PIMC** (`neural/pimc.py`, `PIMCNetPlayer`): for early tricks, 32 sampled deals × up to 8 candidates played to the end of the round by the net on the GPU engine (CUDA-graph captured, ~120 ms/decision); search overrides the net in ~50% of early decisions | screening vs the plain hybrid on seed 7 |
