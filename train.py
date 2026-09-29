import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import (
    load_cached,
    group_split,
    XODataset
)

from models import build


def evaluate(model, loader, device):
    model.eval()

    correct = 0
    total = 0
    total_loss = 0.0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    avg_loss = total_loss / total

    return avg_loss, accuracy


def train(model_name, epochs=25, batch_size=32, learning_rate=0.001):
    print(f"\nTraining model: {model_name}")

    # -------------------------
    # Load preprocessed dataset
    # -------------------------

    X, y, paths = load_cached()

    print(f"Total usable images: {len(y)}")

    # -------------------------
    # Split into train / val
    # -------------------------

    train_idx, val_idx = group_split(
        y,
        paths,
        val_frac=0.2,
        seed=42
    )

    X_train = X[train_idx]
    y_train = y[train_idx]

    X_val = X[val_idx]
    y_val = y[val_idx]

    print(f"Training images: {len(y_train)}")
    print(f"Validation images: {len(y_val)}")

    # -------------------------
    # Create datasets
    # -------------------------

    train_dataset = XODataset(
        X_train,
        y_train,
        augment=True,
        copies=3
    )

    val_dataset = XODataset(
        X_val,
        y_val,
        augment=False,
        copies=1
    )

    # -------------------------
    # Data loaders
    # -------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # -------------------------
    # Device
    # -------------------------

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    # -------------------------
    # Build model
    # -------------------------

    model = build(model_name).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # -------------------------
    # Training loop
    # -------------------------

    best_val_accuracy = 0.0

    os.makedirs("models", exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_accuracy = correct / total

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            device
        )

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss {train_loss:.4f} | "
            f"train acc {train_accuracy:.3f} | "
            f"val loss {val_loss:.4f} | "
            f"val acc {val_accuracy:.3f}"
        )

        # -------------------------
        # Save best model
        # -------------------------

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            model_path = f"models/{model_name}.pt"

            torch.save(
                model.state_dict(),
                model_path
            )

            print(
                f"  saved new best model "
                f"({best_val_accuracy:.3f})"
            )

    print(
        f"\nBest validation accuracy for "
        f"{model_name}: {best_val_accuracy:.3f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=["mlp", "cnn"],
        required=True
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=25
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001
    )

    args = parser.parse_args()

    train(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )