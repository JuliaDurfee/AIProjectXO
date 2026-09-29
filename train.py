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

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def train(
    model_name,
    epochs=100,
    batch_size=32,
    learning_rate=0.001,
    patience=5,
    min_delta=0.001
):
    print(f"\nTraining model: {model_name}")

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------

    X, y, paths = load_cached()

    print(f"Total usable images: {len(y)}")

    # --------------------------------------------------
    # Split by source image
    #
    # This prevents crops from the SAME source photo
    # appearing in both training and validation.
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Datasets
    #
    # Training data gets augmentation.
    # Validation data NEVER gets augmentation.
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Data loaders
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = build(model_name).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # --------------------------------------------------
    # Early stopping variables
    # --------------------------------------------------

    best_val_loss = float("inf")
    best_val_accuracy = 0.0

    epochs_without_improvement = 0
    best_epoch = 0

    os.makedirs("models", exist_ok=True)

    model_path = f"models/{model_name}.pt"

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    for epoch in range(1, epochs + 1):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            # Remove gradients from previous batch
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

            running_loss += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        train_loss = running_loss / total
        train_accuracy = correct / total

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            device
        )

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train loss {train_loss:.4f} | "
            f"train acc {train_accuracy:.3f} | "
            f"val loss {val_loss:.4f} | "
            f"val acc {val_accuracy:.3f}"
        )

        # --------------------------------------------------
        # Early stopping
        #
        # We use VALIDATION LOSS to decide whether
        # the model is still improving.
        # --------------------------------------------------

        if val_loss < best_val_loss - min_delta:

            best_val_loss = val_loss
            best_val_accuracy = val_accuracy

            best_epoch = epoch

            epochs_without_improvement = 0

            # Save BEST model, not last model
            torch.save(
                model.state_dict(),
                model_path
            )

            print(
                f"  ✓ validation improved "
                f"(loss {best_val_loss:.4f}, "
                f"acc {best_val_accuracy:.3f})"
            )

            print(
                f"  ✓ saved {model_path}"
            )

        else:

            epochs_without_improvement += 1

            print(
                f"  no validation improvement "
                f"({epochs_without_improvement}/{patience})"
            )

            if epochs_without_improvement >= patience:

                print("\nEARLY STOPPING")

                print(
                    f"Best epoch: {best_epoch}"
                )

                print(
                    f"Best validation loss: "
                    f"{best_val_loss:.4f}"
                )

                print(
                    f"Validation accuracy at best epoch: "
                    f"{best_val_accuracy:.3f}"
                )

                break

    print("\nTraining finished.")

    print(
        f"Best model saved at: {model_path}"
    )

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best validation loss: "
        f"{best_val_loss:.4f}"
    )

    print(
        f"Validation accuracy at best epoch: "
        f"{best_val_accuracy:.3f}"
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
        default=100
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

    parser.add_argument(
        "--patience",
        type=int,
        default=5
    )

    args = parser.parse_args()

    train(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience
    )