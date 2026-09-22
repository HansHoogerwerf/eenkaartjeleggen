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

Steps are in order. The clock column of earlier versions of this log was
an estimate and ran ahead of the wall clock: by commit and log-file times the
campaign up to the reopen ran 08:08–13:30 and the reopened attempt from 13:30.

| # | Step | Result |
|---|---|---|
| 1 | branch from v3; recording 50 000 Mythos rounds (14 workers, budget 1.0 s) | running, ETA ~5 h |
| 2 | long PPO from `neural_best.pt`: 600 epochs, lr 3e-5, dense rewards, 256 steps/epoch, snapshots every 50 | running on GPU |
| 3 | `train_neural.py --save-every-epoch` for benchmark-based checkpoint selection | – |
| 4 | recipe trial on the first data checkpoint (3200 Mythos rounds, 66k decisions): warm start from `neural_best.pt`, lr 3e-5, 6 epochs, per-epoch snapshots; screen epochs 2/4/6 at 256 rounds (seed 7, 6 workers so recording keeps most of the CPU) | running |
| 5 | long PPO at epoch 540/600 (~4 s/epoch); snapshots 200/400/600 queued for the same screen | running |
| 6 | trial fine-tune (3200 Mythos rounds, lr 3e-5, warm start) — val agreement 52% → 55%; seed-7 screens, 256 rounds: ep2 0.75 / +274, **ep4 1.00 / +429** (nats 12 vs 39, declare ok 0.89), ep6 0.56 / −38 | screens too noisy to rank; 512-round confirmations on seeds 11/13 queued |
| 7 | long PPO (600 epochs, lr 3e-5) finished; snapshots 200/400/600 being screened | – |
| 8 | fine-tune `v4_ft_c` on the 6400-round checkpoint (132k decisions), same recipe | val agreement 53.6% → 55.5%; epochs 2/4/6 queued for 512-round confirmation on seeds 11/13 |
| 9 | PPO long snapshot ep200, seed-7 screen | 0.375 / −119 (reference +220) |
| 10 | PPO long snapshots ep400 / ep600, seed-7 screens | +213 / +108 (reference +220): no gain from 600 epochs at lr 3e-5 |
| 11 | trial fine-tune ep4, 512 rounds | seed 11: 0.53 / +33 (ref +129); seed 13: 0.59 / +101 (ref +146) → ~70/game **below** the current net; the +429 screen was a 16-game fluke |
| 12 | anchored fine-tune `v4_ft_mix`: guarded self-play data (2.0M) + Mythos 6400 rounds ×4, warm start, lr 3e-5, 3 epochs | val agreement 76.9% (mixed val set); epochs 1/3 queued for 512-round confirmation on seeds 11/13 |
| 13 | trial fine-tune ep2, 512 rounds | seed 11: +39 (ref +129); seed 13: +44 (ref +146) → ~90/game below the current net. Consistent with v3's guided-midgame result: **the net's early-trick play is already better than Mythos's, so imitating Mythos there hurts** |
| 14 | trial fine-tune ep6, seed 11 | +144 (ref +129) — epochs differ wildly on the same data; no consistent gain |
| 15 | **net-rollout PIMC** (`neural/pimc.py`, `PIMCNetPlayer`): for early tricks, 32 sampled deals × up to 8 candidates played to the end of the round by the net on the GPU engine (CUDA-graph captured, ~120 ms/decision); search overrides the net in ~50% of early decisions | screening vs the plain hybrid on seed 7 |
| 16 | stopped the Mythos recording at 655 games (9600-round checkpoint kept as `training_data_mythos_v2_50k.npz`): two fine-tunes had already shown imitation lowers early-trick strength, and its 14 workers were slowing the decisive benchmarks 3× | – |
| 17 | trial fine-tune, all epochs, 512 rounds, seeds 11 / 13 (ref +129 / +146) | ep2 +39 / +44, ep4 +33 / +101, ep6 +144 / +36 → every epoch below the current net (means +42 / +67 / +90 vs +138) |
| 18 | PIMC (32 deals, greedy rollouts) vs plain hybrid, seed 7, 256 rounds, same load | PIMC 0.56 / +72 vs hybrid 0.56 / +125: **no gain**; trying 64 deals + an 8-point override margin at 512 rounds |
| 19 | PIMC v2 (64 deals, margin 8) vs plain hybrid, seed 7, 512 rounds, back-to-back under the same (heavy) load | **PIMC 0.63 / +165** vs plain 0.53 / −14 (the plain hybrid scored +220 here on a quiet machine: benchmarks are load-sensitive, only paired runs count) → repeating the pair on seeds 11 / 13 |
| 20 | 6400-round fine-tune `v4_ft_c`, 512 rounds, seeds 11 / 13, under heavy load | ep2 +69 / +0, ep4 +65 / +9, ep6 +78 / +82. **Caveat:** the plain hybrid measured −14 on seed 7 under this load vs +220 quiet, so comparisons against quiet-machine references are invalid; a quiet, sequential tournament (plain, ft_c ep6, ft_mix, PIMC v2; seeds 7 / 11) is queued after the running batches |
| 21 | PIMC v2 vs plain, seed 11 pair (512 rounds, same load) | **PIMC 0.66 / +199** (nats 29 vs 53, declare ok 0.88) vs plain 0.50 / +72 (44 vs 43, 0.83) → second pair won, by +127 |
| 22 | started recording PIMC self-play (`--teacher pimc`, 64 deals, margin 8, 2 GPU workers, 4000 rounds) for expert-iteration distillation | running |
| 23 | anchored fine-tune `v4_ft_mix`, 512 rounds, seeds 11 / 13, under load | ep1 +24 / +142; **ep3 +220 / +94** (plain hybrid under the same load: +72 on seed 11) → first fine-tune that may beat the net; quiet tournament decides |
| 24 | PIMC v2, seed 13 (paired control pending) | 0.56 / +58 |
| 25 | PIMC v2 vs plain, seed 13 pair | PIMC +58 vs plain +47 → three pairs: +180 / +127 / +11 in PIMC's favour (mean +106) |
| 26 | quiet tournament controls (plain hybrid, 12 workers + 2 recording workers) | seed 7: 0.47 / −7; seed 11: 0.59 / +38 — the reference for the rows that follow |
| 27 | first expert-iteration trial `v4_ft_pimc_a`: guarded 2M + PIMC self-play 640 rounds ×16, warm start, lr 3e-5, 3 epochs | val 79.5%; paired checks vs plain queued after the tournament |
| 28 | tournament rows so far (control −7 / +38 on seeds 7 / 11) | ft_c ep6: +114 / +39; ft_mix ep3: +60 / (pending) |
| 29 | tournament rows (control −7 / +38) | ft_c ep6 +114 / +39; ft_mix ep3 +60 / +21; ft_mix ep1 +114 / (pending); PIMC pending |
| 30 | tournament (control −7 / +38 on seeds 7 / 11) | ft_c ep6 +114 / +39; ft_mix ep3 +60 / +21; ft_mix ep1 +114 / +13 → every fine-tune beats the control on seed 7 by ~120 and ties on seed 11 (mean gain ~+50, still inside noise) |
| 31 | **host stopped three jobs for low memory:** the tournament before its PIMC rows, the paired checks of both PIMC-distilled trials (`v4_ft_pimc_a_epoch003.pt`, `v4_ft_pimc_b_epoch003.pt`, both trained, val 79.5% / 79.1%). The PIMC recording survived (100/250 games). Not restarted automatically | pending |
| 32 | PIMC recording done: 4000 rounds, 81 842 decisions (`training_data_pimc_v2.npz`, 126 min on 2 GPU workers) | – |
| 33 | final distillation `v4_ft_pimc_full` (guarded 2M + PIMC ×8, warm start, lr 3e-5, 3 epochs) | val 77.8%; paired checks vs plain on seeds 7 / 11 running, one benchmark at a time |
| 34 | `v4_ft_pimc_full` ep3 vs plain, paired, quiet machine | seed 7: +42 vs +143; seed 11: +108 vs +27 → mean −10: **distillation did not transfer the search's edge** |
| 35 | final quiet paired test of the best remaining candidate, `v4_ft_c_epoch006.pt`, vs plain on seeds 7 / 11 / 13 | running |
| 36 | `v4_ft_c_epoch006.pt` vs plain, paired, quiet | seed 7: −84 vs +58; seed 11: −36 vs +64; seed 13: +99 vs +167 → **not an improvement** (−142 / −100 / −69) |
| 37 | **reopened** (goal check): the failed distillation mixed search labels with the net's own contradicting labels on the same states. New attempt: fine-tune on search data *only* (net-consistent where the search agrees) and record much more of it | 16k-round PIMC recording started (3 GPU workers); search-only fine-tunes lr 3e-5 (val 60.7%, barely moved) and lr 1e-4 × 8 epochs → epochs 4 / 8 paired vs plain on seeds 7 / 11 |
| 38 | search-only fine-tune lr 1e-4, paired vs plain (seeds 7 / 11) | ep4 +2 / −18, ep8 +100 / +49 vs plain +163 / +27 → ep8 mean −20: **no** |
| 39 | 15:05 (wall clock) early-trick-only fine-tune (`--min-hand 6`, warm start, lr 1e-4, 8 epochs) on the PIMC hard labels of 8000 rounds (63k decisions with ≥ 6 cards in hand) | agreement with the search stays at 65.0 % before and after training (val loss falls, accuracy does not): **the hard labels are not learnable by this net**, so more of the same data will not help → the 16k hard-label recording stopped at 4000 rounds (`training_data_pimc_v2_b.npz`) |
| 40 | 15:10 the search's per-card values are now recorded (`values` array, NaN for unscored cards); new recording of 12 000 PIMC rounds with values (4 GPU workers, 64 deals, margin 8); `train_neural.py --soft-temp T` distils from softmax(values/T) | running |
| 41 | 15:20 net-rollout search timed on the CPU (4 threads, under load): 8 / 16 / 32 deals → 0.42 / 0.52 / 0.65 s per decision — the fixed engine-step cost dominates, so it is deployable without a GPU | paired CPU benchmark (64 deals, 1 thread per worker) vs the plain hybrid on seed 7 running |
| 42 | 15:55 CPU search player (64 deals, margin 8, 1 thread/worker) vs plain hybrid, seed 7, paired under the recording load | **+35 vs +162** → −127; four pairs now read +180 / +127 / +11 / −127 (mean +48): the search's edge over the net is unproven |
| 43 | 16:00 parity check: the CPU (eager) and CUDA (graph) evaluators return identical values for identical deals, except the first CUDA call, which was the graph-capture call and returned garbage for one decision (fixed: capture, then reload the state). So the seed-7 CPU result is benchmark noise, not a CPU/GPU difference | head-to-head `pimc` vs `neural_mythosbid` (same bidder and endgame; only the early tricks differ) on seeds 7 / 11 running to measure the search's edge without Mythos's load sensitivity |

## Outcome (interim, before the reopened attempt)

**The net in `neural_best.pt` stays.** No candidate from this campaign beat it
in paired, equal-load benchmarks against Mythos:

| Route | What was tried | Paired result vs the current net |
|---|---|---|
| Imitate Mythos | gentle warm-start fine-tunes on 3200 / 6400 Mythos rounds (lr 3e-5, per-epoch snapshots chosen by benchmark) | every epoch below or level; final quiet pairs of the best one: −142 / −100 / −69 |
| Imitate Mythos, anchored | own guarded self-play (2M) + Mythos ×4 | ahead by ~100 on seed 7 in a loaded tournament, level on seed 11 → inside noise |
| Improve by itself (PPO) | 600 epochs, lr 3e-5, dense rewards, snapshots | snapshots −119 / +213 / +108 vs +220 reference: noise around the start |
| Improve by itself (search) | **net-rollout PIMC**: the net as rollout policy of a 64-deal determinized search on the GPU (`neural/pimc.py`) | **+180 / +127 / +11 as a player** (three pairs) — but needs a GPU |
| Distil the search back into the net | guarded 2M + 4000 rounds of PIMC self-play ×8, warm start | −101 / +82 → no transfer |

Why: the current net's early-trick play is already better than Mythos's
(imitating Mythos pulls it down), self-play has been at a plateau for seven
runs, and the search's edge lives in per-position lookahead that a
300-input MLP does not absorb from 80k labelled decisions.

What the branch leaves behind for the next attempt: `PIMCNetPlayer` (a
stronger opponent wherever a GPU is available), the `pimc` recording teacher,
`tools/screen_checkpoints.py` and the paired-benchmark protocol, per-epoch
snapshots in `train_neural.py`, and the datasets
(`training_data_mythos_v2_50k.npz` 9600 rounds, `training_data_pimc_v2.npz`
4000 rounds). The most promising untried step is a larger PIMC recording
(50k+ rounds, cheap on GPU) distilled into a *wider* net, then PPO from it.
