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
