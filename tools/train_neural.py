"""Train the Klaverjassen neural network from imitation learning data."""

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

from neural.features import NUM_FEATURES
from neural.model import KlaverjasNet


def train(
    data_path: str,
    output_path: str,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 5,
    val_split: float = 0.1,
    hidden_sizes: tuple[int, ...] = (512, 256, 128),
) -> None:
    # Load data
    data = np.load(data_path)
    X = data["features"]
    y = data["labels"]
    print(f"Loaded {X.shape[0]} samples, {NUM_FEATURES} features")

    # Split train/val
    n = X.shape[0]
    indices = np.random.permutation(n)
    val_n = int(n * val_split)
    val_idx, train_idx = indices[:val_n], indices[val_n:]

    X_train = torch.from_numpy(X[train_idx])
    y_train = torch.from_numpy(y[train_idx])
    X_val = torch.from_numpy(X[val_idx])
    y_val = torch.from_numpy(y[val_idx])

    # Extract legal masks from features (offset 190, length 32)
    legal_mask_offset = 190
    legal_mask_train = X_train[:, legal_mask_offset:legal_mask_offset + 32]
    legal_mask_val = X_val[:, legal_mask_offset:legal_mask_offset + 32]

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # DataLoader
    train_ds = TensorDataset(X_train, y_train, legal_mask_train)
    val_ds = TensorDataset(X_val, y_val, legal_mask_val)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model
    model = KlaverjasNet(hidden_sizes=hidden_sizes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for xb, yb, mask_b in train_dl:
            xb, yb, mask_b = xb.to(device), yb.to(device), mask_b.to(device)
            logits = model(xb)
            # Mask illegal moves for accuracy calculation, but train on all logits
            # so gradients flow through the full network
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)
            # Accuracy with legal masking
            masked_logits = logits.masked_fill(mask_b == 0, float("-inf"))
            preds = masked_logits.argmax(dim=-1)
            train_correct += (preds == yb).sum().item()
            train_total += xb.size(0)

        scheduler.step()
        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for xb, yb, mask_b in val_dl:
                xb, yb, mask_b = xb.to(device), yb.to(device), mask_b.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                masked_logits = logits.masked_fill(mask_b == 0, float("-inf"))
                preds = masked_logits.argmax(dim=-1)
                val_correct += (preds == yb).sum().item()
                val_total += xb.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        lr_now = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"lr={lr_now:.6f}"
        )

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


def main_cli() -> None:
    parser = argparse.ArgumentParser(description="Train neural Klaverjassen AI.")
    parser.add_argument("--data", default=str(ROOT / "models" / "training_data.npz"), help="Training data path.")
    parser.add_argument("--output", default=str(ROOT / "models" / "neural_v1.pt"), help="Output model path.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
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
