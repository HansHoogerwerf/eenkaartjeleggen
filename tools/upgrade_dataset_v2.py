"""Upgrade a v1 (267-feature) imitation dataset to the roem-aware v2 layout.

The v1 vector already contains the current trick cards (3 one-hot slots at
offset 64), the trump suit (one-hot at 160) and the legal-move mask (190), so
the 33 trick-roem features can be recomputed exactly; the labels are kept.

Because the old teachers (`expert`, `expert_v2`) were roem-blind, samples in
which a roem choice existed (some legal card would add roem, or roem was
already on the table) carry unreliable labels for the new features.  With
``--drop-roem-decisions`` (recommended) those samples are removed, leaving the
roem-relevant decisions to a roem-aware teacher such as Mythos.

The roem block is computed with the GPU engine's tensor implementation in
chunks, so 3M samples take seconds on CUDA (or a minute or two on CPU).

Example:
  python tools/upgrade_dataset_v2.py models/training_data_v2_100k.npz \\
      models/training_data_expert_v2_100k_roem.npz --drop-roem-decisions
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural.features import LEGAL_MASK_OFFSET, NUM_FEATURES, NUM_FEATURES_V2  # noqa: E402
from tools.gpu_engine import KlaverjasGPUEngine  # noqa: E402

TRICK_OFFSET = 64      # 3 x 32 one-hot slots
TRUMP_OFFSET = 160     # 4 one-hot


def roem_block(engine: KlaverjasGPUEngine, X: torch.Tensor) -> torch.Tensor:
    """[N,267+] v1 features -> [N,33] roem block (already /100)."""
    N = X.shape[0]
    presence = (X[:, TRICK_OFFSET:TRICK_OFFSET + 32]
                + X[:, TRICK_OFFSET + 32:TRICK_OFFSET + 64]
                + X[:, TRICK_OFFSET + 64:TRICK_OFFSET + 96]).clamp(max=1.0)   # [N,32]
    trump = X[:, TRUMP_OFFSET:TRUMP_OFFSET + 4].argmax(dim=1)                  # [N]
    legal = X[:, LEGAL_MASK_OFFSET:LEGAL_MASK_OFFSET + 32]                     # [N,32]
    has_trick = presence.sum(dim=1) > 0
    base = engine._roem_from_presence(presence, trump)                          # [N]
    cand = (presence.unsqueeze(1) + engine._EYE32.unsqueeze(0)).clamp(max=1.0)  # [N,32,32]
    cand_roem = engine._roem_from_presence(cand, trump.unsqueeze(1).expand(N, 32))
    delta = (cand_roem - base.unsqueeze(1)).float() * legal * has_trick.unsqueeze(1).float()
    out = torch.empty(N, 33, dtype=torch.float32, device=X.device)
    out[:, :32] = delta / 100.0
    out[:, 32] = base.float() / 100.0
    return out


def main_cli() -> None:
    ap = argparse.ArgumentParser(description="Upgrade a 267-feature dataset to the 300-feature v2 layout.")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--drop-roem-decisions", action="store_true",
                    help="Drop samples where any legal card adds roem or roem is on the table.")
    ap.add_argument("--chunk", type=int, default=65536)
    ap.add_argument("--teacher", default="expert_v2", help="Teacher label stored in the output.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(args.input)
    X1, y = data["features"], data["labels"]
    if X1.shape[1] != NUM_FEATURES:
        raise SystemExit(f"{args.input} has {X1.shape[1]} features; expected v1 width {NUM_FEATURES}")
    N = X1.shape[0]
    print(f"Loaded {N} v1 samples from {args.input}; device {device}")

    engine = KlaverjasGPUEngine(batch_size=1, device=device, feature_version=2)
    X2 = np.empty((N, NUM_FEATURES_V2), dtype=np.float32)
    keep = np.ones(N, dtype=bool)
    for start in range(0, N, args.chunk):
        xb = torch.from_numpy(X1[start:start + args.chunk]).to(device)
        rb = roem_block(engine, xb)
        X2[start:start + args.chunk, :NUM_FEATURES] = X1[start:start + args.chunk]
        X2[start:start + args.chunk, NUM_FEATURES:] = rb.cpu().numpy()
        if args.drop_roem_decisions:
            involved = (rb[:, :32].sum(dim=1) > 0) | (rb[:, 32] > 0)
            keep[start:start + args.chunk] = ~involved.cpu().numpy()
        if (start // args.chunk) % 10 == 0:
            print(f"  {min(start + args.chunk, N)}/{N}")
    del X1

    if args.drop_roem_decisions:
        print(f"Dropping {int((~keep).sum())} roem-relevant samples ({(~keep).mean() * 100:.1f}%)")
        X2, y = X2[keep], y[keep]

    np.savez_compressed(args.output, features=X2, labels=y,
                        feature_version=np.int64(2), teacher=np.array(args.teacher),
                        rounds=np.int64(int(data["rounds"]) if "rounds" in data.files else 0))
    print(f"Saved {X2.shape[0]} samples x {X2.shape[1]} features -> {args.output}")


if __name__ == "__main__":
    main_cli()
