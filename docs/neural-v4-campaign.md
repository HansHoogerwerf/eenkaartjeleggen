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
| 12:45 | stopped the Mythos recording at 655 games (9600-round checkpoint kept as `training_data_mythos_v2_50k.npz`): two fine-tunes had already shown imitation lowers early-trick strength, and its 14 workers were slowing the decisive benchmarks 3× | – |
| 12:55 | trial fine-tune, all epochs, 512 rounds, seeds 11 / 13 (ref +129 / +146) | ep2 +39 / +44, ep4 +33 / +101, ep6 +144 / +36 → every epoch below the current net (means +42 / +67 / +90 vs +138) |
| 13:10 | PIMC (32 deals, greedy rollouts) vs plain hybrid, seed 7, 256 rounds, same load | PIMC 0.56 / +72 vs hybrid 0.56 / +125: **no gain**; trying 64 deals + an 8-point override margin at 512 rounds |
| 13:50 | PIMC v2 (64 deals, margin 8) vs plain hybrid, seed 7, 512 rounds, back-to-back under the same (heavy) load | **PIMC 0.63 / +165** vs plain 0.53 / −14 (the plain hybrid scored +220 here on a quiet machine: benchmarks are load-sensitive, only paired runs count) → repeating the pair on seeds 11 / 13 |
| 14:20 | 6400-round fine-tune `v4_ft_c`, 512 rounds, seeds 11 / 13, under heavy load | ep2 +69 / +0, ep4 +65 / +9, ep6 +78 / +82. **Caveat:** the plain hybrid measured −14 on seed 7 under this load vs +220 quiet, so comparisons against quiet-machine references are invalid; a quiet, sequential tournament (plain, ft_c ep6, ft_mix, PIMC v2; seeds 7 / 11) is queued after the running batches |
| 14:45 | PIMC v2 vs plain, seed 11 pair (512 rounds, same load) | **PIMC 0.66 / +199** (nats 29 vs 53, declare ok 0.88) vs plain 0.50 / +72 (44 vs 43, 0.83) → second pair won, by +127 |
| 14:50 | started recording PIMC self-play (`--teacher pimc`, 64 deals, margin 8, 2 GPU workers, 4000 rounds) for expert-iteration distillation | running |
| 15:10 | anchored fine-tune `v4_ft_mix`, 512 rounds, seeds 11 / 13, under load | ep1 +24 / +142; **ep3 +220 / +94** (plain hybrid under the same load: +72 on seed 11) → first fine-tune that may beat the net; quiet tournament decides |
| 15:10 | PIMC v2, seed 13 (paired control pending) | 0.56 / +58 |
| 15:35 | PIMC v2 vs plain, seed 13 pair | PIMC +58 vs plain +47 → three pairs: +180 / +127 / +11 in PIMC's favour (mean +106) |
| 16:05 | quiet tournament controls (plain hybrid, 12 workers + 2 recording workers) | seed 7: 0.47 / −7; seed 11: 0.59 / +38 — the reference for the rows that follow |
| 16:10 | first expert-iteration trial `v4_ft_pimc_a`: guarded 2M + PIMC self-play 640 rounds ×16, warm start, lr 3e-5, 3 epochs | val 79.5%; paired checks vs plain queued after the tournament |
| 16:45 | tournament rows so far (control −7 / +38 on seeds 7 / 11) | ft_c ep6: +114 / +39; ft_mix ep3: +60 / (pending) |
| 17:05 | tournament rows (control −7 / +38) | ft_c ep6 +114 / +39; ft_mix ep3 +60 / +21; ft_mix ep1 +114 / (pending); PIMC pending |
| 17:30 | tournament (control −7 / +38 on seeds 7 / 11) | ft_c ep6 +114 / +39; ft_mix ep3 +60 / +21; ft_mix ep1 +114 / +13 → every fine-tune beats the control on seed 7 by ~120 and ties on seed 11 (mean gain ~+50, still inside noise) |
| 17:30 | **host stopped three jobs for low memory:** the tournament before its PIMC rows, the paired checks of both PIMC-distilled trials (`v4_ft_pimc_a_epoch003.pt`, `v4_ft_pimc_b_epoch003.pt`, both trained, val 79.5% / 79.1%). The PIMC recording survived (100/250 games). Not restarted automatically | pending |

## Where this stands

* **Imitating Mythos directly** (3200 / 6400 rounds, gentle warm-start fine-tunes)
  did not beat the current net in paired or quiet comparisons; on the same
  seeds a short Mythos search choosing among the net's cards also lost. The
  net's early-trick play is already stronger than Mythos's.
* **Anchored fine-tunes** (own guarded self-play + Mythos) and the 6400-round
  fine-tune are the first candidates that came out *ahead* of the control in
  a tournament (by ~120 on seed 7, level on seed 11) — promising, not proven.
* **Self-improvement:** six PPO runs in v3/v4 never beat the net. What did:
  **net-rollout PIMC** (`neural/pimc.py`, `PIMCNetPlayer`) — the net used as
  the rollout policy of a determinized search on the GPU — beat the plain
  hybrid in three paired 512-round runs (+180 / +127 / +11). It needs a GPU,
  so the deployable form is a net distilled from PIMC self-play (recording in
  `models/training_data_pimc_v2.npz`, `--teacher pimc`); the first two
  distilled trials are trained and await their paired checks.
* **Protocol:** benchmarks vs Mythos move ±100/game with machine load; only
  paired runs under equal load or a single sequential tournament are
  comparable (`tools/screen_checkpoints.py`).

To resume (each ~25 min at 12 workers, run one at a time):

```bash
python tools/screen_checkpoints.py models/v4_ft_pimc_b_epoch003.pt models/neural_best.pt --rounds 512 --seeds 7 11 --workers 12
NEURAL_PIMC_DEALS=64 NEURAL_PIMC_MARGIN=8 python tools/screen_checkpoints.py models/neural_best.pt --candidate pimc --rounds 512 --seeds 7 11 --workers 12
```
