"""""
Train the MLP or the CNN.
 
    python train.py --model mlp
    python train.py --model cnn --epochs 60
 
Saves the best model (by validation accuracy) to models/<model>.pt
"""
import argparse
import os
 
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
 
from dataset import XODataset, load_cached, stratified_split
from models import build
 
 
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu()
            correct += (pred == y).sum().item(); total += len(y)
    return correct / total
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mlp", "cnn"], required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--copies", type=int, default=10, help="augmented views of each image per epoch")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
 
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
 
    X, y, _ = load_cached()
    tr, va = stratified_split(y, seed=args.seed)
    train_dl = DataLoader(XODataset(X[tr], y[tr], augment=True, copies=args.copies), batch_size=64, shuffle=True)
    val_dl = DataLoader(XODataset(X[va], y[va]), batch_size=256)
    print(f"train {len(tr)} images (x{args.copies} augmented), val {len(va)} images")
 
    model = build(args.model).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    loss_fn = nn.CrossEntropyLoss()
 
    os.makedirs("models", exist_ok=True)
    best = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, t in train_dl:
            x, t = x.to(device), t.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), t)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(t)
        sched.step()
        acc = evaluate(model, val_dl, device)
        flag = ""
        if acc >= best:
            best = acc
            torch.save(model.state_dict(), f"models/{args.model}.pt")
            flag = "  <- saved"
        print(f"epoch {ep:3d}  loss {total_loss / len(train_dl.dataset):.4f}  val acc {acc:.3f}{flag}")
    print(f"best val accuracy: {best:.3f}")
 
 
if __name__ == "__main__":
    main()