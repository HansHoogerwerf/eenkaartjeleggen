"""Train the Klaverjassen bidding neural network from imitation learning data."""

import argparse
import sys
from functools import partial
from pathlib import Path

print = partial(print, flush=True)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural.bid_features import NUM_BID_FEATURES
from neural.model import KlaverjassBidNet


def train(
    data_path: str,
    output_path: str,
    epochs: int = 50,
    batch_size: int = 512,
    lr: float = 1e-3,
    patience: int = 10,
    val_split: float = 0.1,
) -> None:
    data = np.load(data_path)
    X = data["features"]
    y = data["labels"]

    n_declare = int(y.sum())
    n_pass = len(y) - n_declare
    print(f"Loaded {X.shape[0]} samples, {NUM_BID_FEATURES} features")
    print(f"  Declares: {n_declare} ({100*n_declare/len(y):.1f}%)")
    print(f"  Passes:   {n_pass} ({100*n_pass/len(y):.1f}%)")

    # Split train/val
    n = X.shape[0]
    indices = np.random.permutation(n)
    val_n = int(n * val_split)
    val_idx, train_idx = indices[:val_n], indices[val_n:]

    X_train = torch.from_numpy(X[train_idx])
    y_train = torch.from_numpy(y[train_idx]).float()
    X_val = torch.from_numpy(X[val_idx])
    y_val = torch.from_numpy(y[val_idx]).float()

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = KlaverjassBidNet().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Class-weighted loss to handle imbalance (more passes than declares)
    pos_weight = torch.tensor([n_pass / max(n_declare, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"pos_weight: {pos_weight.item():.2f}")

    best_val_loss = float("inf")
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_tp = train_fp = train_fn = train_tn = 0
        train_total = 0

        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb).squeeze(-1)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_correct += (preds == yb).sum().item()
            train_tp += ((preds == 1) & (yb == 1)).sum().item()
            train_fp += ((preds == 1) & (yb == 0)).sum().item()
            train_fn += ((preds == 0) & (yb == 1)).sum().item()
            train_tn += ((preds == 0) & (yb == 0)).sum().item()
            train_total += xb.size(0)

        scheduler.step()
        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_tp = val_fp = val_fn = val_tn = 0
        val_total = 0

        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb).squeeze(-1)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += (preds == yb).sum().item()
                val_tp += ((preds == 1) & (yb == 1)).sum().item()
                val_fp += ((preds == 1) & (yb == 0)).sum().item()
                val_fn += ((preds == 0) & (yb == 1)).sum().item()
                val_tn += ((preds == 0) & (yb == 0)).sum().item()
                val_total += xb.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total
        val_prec = val_tp / max(val_tp + val_fp, 1)
        val_rec = val_tp / max(val_tp + val_fn, 1)
        val_f1 = 2 * val_prec * val_rec / max(val_prec + val_rec, 1e-8)

        lr_now = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"t_loss={train_loss:.4f}  t_acc={train_acc:.4f}  "
            f"v_loss={val_loss:.4f}  v_acc={val_acc:.4f}  "
            f"v_prec={val_prec:.3f}  v_rec={val_rec:.3f}  v_f1={val_f1:.3f}  "
            f"lr={lr_now:.6f}"
        )

        # Early stopping on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), output_path)
            print(f"  -> Saved best model (val_loss={val_loss:.4f}, val_f1={val_f1:.3f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    # Final confusion matrix on validation set
    model.load_state_dict(torch.load(output_path, map_location=device, weights_only=True))
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in val_dl:
            xb = xb.to(device)
            logits = model(xb).squeeze(-1)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            all_preds.append(preds.cpu())
            all_labels.append(yb)

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    print(f"\nConfusion matrix (validation set):")
    print(f"                Predicted Pass  Predicted Declare")
    print(f"  Actual Pass   {tn:>10}      {fp:>10}")
    print(f"  Actual Decl   {fn:>10}      {tp:>10}")
    print(f"\nBest val_loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_path}")


def main_cli() -> None:
    parser = argparse.ArgumentParser(description="Train neural Klaverjassen bidding AI.")
    parser.add_argument("--data", default=str(ROOT / "models" / "bid_training_data.npz"), help="Training data path.")
    parser.add_argument("--output", default=str(ROOT / "models" / "bid_neural_v1.pt"), help="Output model path.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--val-split", type=float, default=0.1)
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
    )


if __name__ == "__main__":
    main_cli()
