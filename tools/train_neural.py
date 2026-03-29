"""Train the Klaverjassen neural network from imitation learning data."""

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

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Move full dataset to GPU once — RTX 5080 has 16 GB VRAM, dataset fits easily.
    # This eliminates per-batch CPU→GPU transfers so the GPU is never starved.
    legal_mask_offset = 190
    X_train = torch.from_numpy(X[train_idx]).to(device)
    y_train = torch.from_numpy(y[train_idx]).to(device)
    X_val   = torch.from_numpy(X[val_idx]).to(device)
    y_val   = torch.from_numpy(y[val_idx]).to(device)
    legal_mask_train = X_train[:, legal_mask_offset:legal_mask_offset + 32]
    legal_mask_val   = X_val[:,   legal_mask_offset:legal_mask_offset + 32]

    n_train = X_train.size(0)
    n_val = X_val.size(0)
    print(f"Train: {n_train}, Val: {n_val}")

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
        # Train — pure GPU batching, no DataLoader overhead
        model.train()
        train_loss = torch.tensor(0.0, device=device)
        train_correct = torch.tensor(0, device=device)

        perm = torch.randperm(n_train, device=device)
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb, mask_b = X_train[idx], y_train[idx], legal_mask_train[idx]

            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.detach() * idx.size(0)
            masked_logits = logits.detach().masked_fill(mask_b == 0, float("-inf"))
            train_correct += (masked_logits.argmax(dim=-1) == yb).sum()

        scheduler.step()
        train_loss = (train_loss / n_train).item()
        train_acc = (train_correct / n_train).item()

        # Validate
        model.eval()
        val_loss = torch.tensor(0.0, device=device)
        val_correct = torch.tensor(0, device=device)

        with torch.no_grad():
            for start in range(0, n_val, batch_size):
                xb = X_val[start:start + batch_size]
                yb = y_val[start:start + batch_size]
                mask_b = legal_mask_val[start:start + batch_size]
                logits = model(xb)
                val_loss += criterion(logits, yb) * xb.size(0)
                masked_logits = logits.masked_fill(mask_b == 0, float("-inf"))
                val_correct += (masked_logits.argmax(dim=-1) == yb).sum()

        val_loss = (val_loss / n_val).item()
        val_acc = (val_correct / n_val).item()

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
    parser.add_argument("--output", default=str(ROOT / "models" / "neural_best.pt"), help="Output model path.")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[1024, 512, 256],
                        help="Hidden layer sizes (default: 1024 512 256)")
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
    )


if __name__ == "__main__":
    main_cli()
