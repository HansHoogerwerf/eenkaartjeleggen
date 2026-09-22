"""Train the Klaverjassen neural network from imitation learning data.

Data files are .npz with ``features`` / ``labels`` (see
tools/generate_training_data.py).  The feature width of the data decides the
network's input layer (267 = v1 layout, 300 = roem-aware v2 layout).

Examples
--------
  # From scratch on one dataset
  python tools/train_neural.py --data models/training_data_mythos_v2.npz \\
      --output models/neural_mythos_v2.pt

  # Warm-start the old 267-input net (zero-padded to 300 inputs) and fine-tune
  # on a big heuristic set plus an 8x oversampled Mythos set
  python tools/train_neural.py --init models/neural_best.pt \\
      --data models/training_data_expert_v2_100k_roem.npz models/training_data_mythos_v2.npz@8 \\
      --output models/neural_mythos_v2_mix.pt --lr 3e-4
"""

import argparse
import sys
from functools import partial
from pathlib import Path

print = partial(print, flush=True)

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural.features import LEGAL_MASK_OFFSET, feature_version_for_size
from neural.model import KlaverjasNet, infer_net_shape


def load_datasets(
    specs: list[str], val_split: float = 0.1, seed: int | None = None, min_hand: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load one or more datasets and return (X_train, y_train, X_val, y_val).

    Each spec is ``path`` or ``path@k``: the dataset's *training* part is
    repeated k times (oversampling a small, high-value set such as Mythos
    roem decisions next to a large heuristic set).  The train/val split is
    made per dataset before repetition, so no sample leaks into validation.

    ``min_hand`` keeps only decisions made with at least that many cards in
    hand (the own-hand one-hot is the first 32 features).  The deployed hybrid
    hands the last tricks to the Mythos endgame search, so for it only the
    early-trick decisions of a dataset matter.
    """
    rng = np.random.default_rng(seed)
    tr_x, tr_y, va_x, va_y, tr_v, va_v = [], [], [], [], [], []
    for spec in specs:
        path, _, rep = spec.partition("@")
        repeat = int(rep) if rep else 1
        data = np.load(path)
        X, y = data["features"], data["labels"]
        # Per-decision search values (32 floats, NaN where the teacher scored
        # no card), the soft targets of --soft-temp; all-NaN without them.
        V = data["values"] if "values" in data.files else np.full((X.shape[0], 32), np.nan, np.float32)
        version = int(data["feature_version"]) if "feature_version" in data.files else 1
        teacher = str(data["teacher"]) if "teacher" in data.files else "?"
        if min_hand > 1:
            keep = X[:, :32].sum(axis=1) >= min_hand
            print(f"  {path}: keeping {int(keep.sum())} of {X.shape[0]} decisions with >= {min_hand} cards in hand")
            X, y, V = X[keep], y[keep], V[keep]
        n = X.shape[0]
        perm = rng.permutation(n)
        val_n = int(n * val_split)
        va_idx, tr_idx = perm[:val_n], perm[val_n:]
        print(f"  {path}: {n} samples x {X.shape[1]} features (feature_version={version}, "
              f"teacher={teacher}) -> train {len(tr_idx)} x{repeat}, val {len(va_idx)}")
        for _ in range(repeat):
            tr_x.append(X[tr_idx])
            tr_y.append(y[tr_idx])
            tr_v.append(V[tr_idx])
        va_x.append(X[va_idx])
        va_y.append(y[va_idx])
        va_v.append(V[va_idx])
    widths = {X.shape[1] for X in tr_x}
    if len(widths) != 1:
        raise ValueError(f"Datasets have different feature widths: {sorted(widths)}")
    return (np.concatenate(tr_x), np.concatenate(tr_y),
            np.concatenate(va_x), np.concatenate(va_y),
            np.concatenate(tr_v), np.concatenate(va_v))


def warm_start(state_dict: dict, num_features: int) -> KlaverjasNet:
    """Build a net for *num_features* inputs initialised from *state_dict*.

    When the checkpoint has fewer inputs (a 267-wide v1 net) the first Linear
    layer's weight is zero-padded for the extra (roem) columns, so the net
    starts out playing exactly like the old one and learns the new features.
    """
    hidden_sizes, in_features = infer_net_shape(state_dict)
    if in_features > num_features:
        raise ValueError(f"init checkpoint has {in_features} inputs > data width {num_features}")
    net = KlaverjasNet(hidden_sizes=hidden_sizes, in_features=num_features)
    sd = dict(state_dict)
    if in_features < num_features:
        w = sd["net.0.weight"]
        pad = torch.zeros(w.shape[0], num_features - in_features, dtype=w.dtype, device=w.device)
        sd["net.0.weight"] = torch.cat([w, pad], dim=1)
    net.load_state_dict(sd)
    return net


def train(
    data_path: str | list[str],
    output_path: str,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 5,
    val_split: float = 0.1,
    hidden_sizes: tuple[int, ...] = (512, 256, 128),
    init_path: str | None = None,
    weight_decay: float = 0.0,
    save_every_epoch: bool = False,
    min_hand: int = 0,
    soft_temp: float = 0.0,
    split_seed: int | None = None,
) -> None:
    # Load data (one or more files; width decides the feature layout).
    # The train/val split happens per dataset inside load_datasets.
    paths = [data_path] if isinstance(data_path, str) else list(data_path)
    X_tr, y_tr, X_va, y_va, V_tr, V_va = load_datasets(paths, val_split=val_split, seed=split_seed,
                                                       min_hand=min_hand)
    num_features = X_tr.shape[1]
    feature_version = feature_version_for_size(num_features)
    print(f"Loaded {X_tr.shape[0]} train + {X_va.shape[0]} val samples, "
          f"{num_features} features (layout v{feature_version})")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Move full dataset to GPU once — RTX 5080 has 16 GB VRAM, dataset fits easily.
    # This eliminates per-batch CPU→GPU transfers so the GPU is never starved.
    legal_mask_offset = LEGAL_MASK_OFFSET
    X_train = torch.from_numpy(X_tr).to(device)
    y_train = torch.from_numpy(y_tr).to(device)
    X_val   = torch.from_numpy(X_va).to(device)
    y_val   = torch.from_numpy(y_va).to(device)
    # Soft targets: softmax(values / T) over the cards the search scored; rows
    # without at least two scored cards fall back to the one-hot label.
    T_train = _soft_targets(V_tr, y_tr, soft_temp, device)
    # Search argmax per validation row (-1 where the teacher scored < 2 cards):
    # with a PIMC override margin the played card is often the net's own
    # choice, so agreement with the search's best card is the metric that
    # tells whether distillation moves the net toward the search.
    y_val_search = _search_argmax(V_va, device)
    # Regret: search points lost by the card the net picks vs the search's best
    # card (the top cards are often near-ties, so regret, not argmax agreement,
    # measures what a policy loses).  Cards the search did not score count as
    # the worst scored card.
    V_val = torch.from_numpy(np.ascontiguousarray(V_va)).to(device)
    del X_tr, y_tr, X_va, y_va, V_tr, V_va
    y_train_search = None
    if T_train is not None:
        print(f"Soft targets (T={soft_temp}) on {int(T_train[1].sum())} of {T_train[0].size(0)} train samples")
        T_train = T_train[0]
        y_train_search = T_train.argmax(dim=-1)   # the search's best card (label on hard rows)
    has_search = bool((y_val_search >= 0).any())

    def validate() -> tuple[float, float, float, float]:
        model.eval()
        v_loss = torch.tensor(0.0, device=device)
        v_correct = torch.tensor(0, device=device)
        s_correct = torch.tensor(0, device=device)
        s_n = torch.tensor(0, device=device)
        regret_sum = torch.tensor(0.0, device=device)
        with torch.no_grad():
            for start in range(0, n_val, batch_size):
                xb = X_val[start:start + batch_size]
                yb = y_val[start:start + batch_size]
                sb = y_val_search[start:start + batch_size]
                vb = V_val[start:start + batch_size]
                mask_b = legal_mask_val[start:start + batch_size]
                logits = model(xb)
                v_loss += criterion(logits, yb) * xb.size(0)
                pred = logits.masked_fill(mask_b == 0, float("-inf")).argmax(dim=-1)
                v_correct += (pred == yb).sum()
                scored = sb >= 0
                s_correct += ((pred == sb) & scored).sum()
                s_n += scored.sum()
                if bool(scored.any()):
                    vv = vb[scored]
                    best = vv.nan_to_num(nan=float("-inf")).max(dim=1).values
                    worst = vv.nan_to_num(nan=float("inf")).min(dim=1).values
                    chosen = vv.gather(1, pred[scored][:, None]).squeeze(1)
                    chosen = torch.where(torch.isnan(chosen), worst, chosen)
                    regret_sum += (best - chosen).sum()
        n_s = max(1, int(s_n))
        return ((v_loss / n_val).item(), (v_correct / n_val).item(),
                (s_correct / n_s).item(), (regret_sum / n_s).item())

    legal_mask_train = X_train[:, legal_mask_offset:legal_mask_offset + 32]
    legal_mask_val   = X_val[:,   legal_mask_offset:legal_mask_offset + 32]

    n_train = X_train.size(0)
    n_val = X_val.size(0)
    print(f"Train: {n_train}, Val: {n_val}")

    # Model (optionally warm-started from an existing checkpoint)
    if init_path:
        init_sd = torch.load(init_path, map_location=device, weights_only=True)
        model = warm_start(init_sd, num_features).to(device)
        print(f"Warm-started from {init_path} ({model.in_features} inputs)")
    else:
        model = KlaverjasNet(hidden_sizes=hidden_sizes, in_features=num_features).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    if has_search:
        _, a0, s0, r0 = validate()
        print(f"Epoch   0/{epochs}  val_acc={a0:.4f}  val_search_acc={s0:.4f}  val_regret={r0:.2f}")

    best_val_acc = float("-inf")
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # Train — pure GPU batching, no DataLoader overhead
        model.train()
        train_loss = torch.tensor(0.0, device=device)
        train_correct = torch.tensor(0, device=device)

        perm = torch.randperm(n_train, device=device)
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb, mask_b = X_train[idx], y_train[idx], legal_mask_train[idx]

            logits = model(xb)
            if T_train is not None:
                loss = -(T_train[idx] * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            else:
                loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.detach() * idx.size(0)
            masked_logits = logits.detach().masked_fill(mask_b == 0, float("-inf"))
            pred = masked_logits.argmax(dim=-1)
            train_correct += (pred == (y_train_search[idx] if y_train_search is not None else yb)).sum()

        scheduler.step()
        train_loss = (train_loss / n_train).item()
        train_acc = (train_correct / n_train).item()

        # Validate
        val_loss, hard_acc, search_acc, regret = validate()
        # Model selection: lowest regret vs the search when values exist,
        # otherwise agreement with the label.
        val_acc = (-regret) if has_search else hard_acc

        lr_now = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={hard_acc:.4f}  "
            + (f"val_search_acc={search_acc:.4f}  val_regret={regret:.2f}  " if has_search else "")
            + f"lr={lr_now:.6f}"
        )

        if save_every_epoch:
            snap = str(Path(output_path).with_suffix("")) + f"_epoch{epoch:03d}.pt"
            torch.save(model.state_dict(), snap)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(), output_path)
            print(f"  -> Saved best model (val_acc={val_acc:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {output_path}")


def _search_argmax(V: np.ndarray, device) -> torch.Tensor:
    """Index of the teacher's best-valued card per row, -1 where < 2 cards were scored."""
    V = torch.from_numpy(np.ascontiguousarray(V)).to(device)
    scored = ~torch.isnan(V)
    best = V.nan_to_num(nan=float("-inf")).argmax(dim=-1)
    return torch.where(scored.sum(dim=1) >= 2, best, torch.full_like(best, -1))


def _soft_targets(V: np.ndarray, y: np.ndarray, temp: float, device) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return (targets [N,32], is_soft [N]) or None when soft targets are off/absent."""
    if temp <= 0:
        return None
    V = torch.from_numpy(np.ascontiguousarray(V)).to(device)
    scored = ~torch.isnan(V)
    is_soft = scored.sum(dim=1) >= 2
    if not bool(is_soft.any()):
        return None
    logits = torch.where(scored, V / temp, torch.full_like(V, float("-inf")))
    soft = torch.softmax(logits.nan_to_num(nan=float("-inf")), dim=-1)
    hard = torch.nn.functional.one_hot(torch.from_numpy(y).to(device), 32).to(soft.dtype)
    targets = torch.where(is_soft[:, None], soft, hard)
    return targets, is_soft


def main_cli() -> None:
    parser = argparse.ArgumentParser(description="Train neural Klaverjassen AI.")
    parser.add_argument("--data", nargs="+", default=[str(ROOT / "models" / "training_data.npz")],
                        help="One or more training data .npz files (same feature width); "
                             "append @k to repeat a file's training part k times.")
    parser.add_argument("--output", default=str(ROOT / "models" / "neural_best.pt"), help="Output model path.")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[1024, 512, 256],
                        help="Hidden layer sizes (default: 1024 512 256; ignored with --init)")
    parser.add_argument("--init", default=None,
                        help="Warm-start from this checkpoint (a 267-input net is zero-padded to the data width).")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay.")
    parser.add_argument("--save-every-epoch", action="store_true",
                        help="Also write <output>_epochNNN.pt after every epoch, so checkpoints can be "
                             "picked by benchmark instead of validation accuracy.")
    parser.add_argument("--min-hand", type=int, default=0,
                        help="Train only on decisions with at least this many cards in hand "
                             "(e.g. 6 for the tricks the hybrid plays with the net).")
    parser.add_argument("--split-seed", type=int, default=None,
                        help="Seed of the train/val split (fixed seed = comparable runs).")
    parser.add_argument("--soft-temp", type=float, default=0.0,
                        help="Distil from the teacher's per-card search values (datasets with a "
                             "'values' array, e.g. --teacher pimc): the target is "
                             "softmax(values / T) in round points instead of the one-hot label.")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    train(
        data_path=args.data,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        val_split=args.val_split,
        hidden_sizes=tuple(args.hidden_sizes),
        init_path=args.init,
        weight_decay=args.weight_decay,
        save_every_epoch=args.save_every_epoch,
        min_hand=args.min_hand,
        soft_temp=args.soft_temp,
        split_seed=args.split_seed,
    )


if __name__ == "__main__":
    main_cli()
