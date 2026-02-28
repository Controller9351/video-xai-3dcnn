import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from src.dataset import UCF101BinaryDataset
from src.model import build_r3d18, freeze_all, unfreeze_fc, unfreeze_layers, count_trainable_params
from src.utils import set_seed, LABEL_MAP


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--videos_root", type=str, required=True,
                   help="Path to UCF-101 root folder containing class subfolders (e.g., .../UCF-101)")
    p.add_argument("--splits_root", type=str, required=True,
                   help="Path to ucfTrainTestlist folder containing trainlist01.txt and testlist01.txt")
    p.add_argument("--epochs_head", type=int, default=1)
    p.add_argument("--epochs_ft", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--lr_ft", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--out", type=str, default="best_r3d18_ucf_binary.pth")
    return p.parse_args()


def load_lines(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def build_dfs(videos_root, splits_root):
    """
    Returns train_df, val_df, test_df for Basketball vs BasketballDunk using official split list 01.
    """
    train_file = os.path.join(splits_root, "trainlist01.txt")
    test_file  = os.path.join(splits_root, "testlist01.txt")

    train_lines = load_lines(train_file)  # "Class/file.avi classIndex"
    test_lines  = load_lines(test_file)   # "Class/file.avi"

    target_classes = set(LABEL_MAP.keys())

    train_keep = [l for l in train_lines if l.split()[0].split("/")[0] in target_classes]
    test_keep  = [l for l in test_lines  if l.split("/")[0] in target_classes]

    def parse_train(line):
        rel = line.split()[0]
        cls = rel.split("/")[0]
        abspath = os.path.join(videos_root, rel)
        return abspath, LABEL_MAP[cls], cls, rel

    def parse_test(line):
        rel = line.strip()
        cls = rel.split("/")[0]
        abspath = os.path.join(videos_root, rel)
        return abspath, LABEL_MAP[cls], cls, rel

    train_df = pd.DataFrame([parse_train(l) for l in train_keep], columns=["path","label","class","rel"])
    test_df  = pd.DataFrame([parse_test(l)  for l in test_keep],  columns=["path","label","class","rel"])

    # sanity check existence
    if (~train_df["path"].apply(os.path.exists)).any():
        missing = train_df.loc[~train_df["path"].apply(os.path.exists), "path"].tolist()[:5]
        raise FileNotFoundError(f"Missing train files, e.g.: {missing}")
    if (~test_df["path"].apply(os.path.exists)).any():
        missing = test_df.loc[~test_df["path"].apply(os.path.exists), "path"].tolist()[:5]
        raise FileNotFoundError(f"Missing test files, e.g.: {missing}")

    # stratified val split from train
    train_df2, val_df = train_test_split(
        train_df,
        test_size=0.2,
        random_state=42,
        stratify=train_df["label"]
    )

    return train_df2.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * y.size(0)

        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


def train_one_epoch(model, loader, device, optimizer, scaler, criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in tqdm(loader, leave=False):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * y.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_df, val_df, test_df = build_dfs(args.videos_root, args.splits_root)
    print("Train:", len(train_df), "Val:", len(val_df), "Test:", len(test_df))
    print("Train label counts:\n", train_df["label"].value_counts().to_dict())
    print("Val label counts:\n", val_df["label"].value_counts().to_dict())
    print("Test label counts:\n", test_df["label"].value_counts().to_dict())

    train_ds = UCF101BinaryDataset(train_df)
    val_ds   = UCF101BinaryDataset(val_df)
    test_ds  = UCF101BinaryDataset(test_df)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    model = build_r3d18(num_classes=2, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    # ---- Stage 1: Head-only ----
    freeze_all(model)
    unfreeze_fc(model)
    print("Stage1 trainable params:", count_trainable_params(model))

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=args.lr_head, weight_decay=args.weight_decay)

    best_val = 0.0
    best_path = args.out

    for epoch in range(1, args.epochs_head + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, optimizer, scaler, criterion)
        va_loss, va_acc = evaluate(model, val_loader, device, criterion)
        print(f"[Head] Epoch {epoch}/{args.epochs_head} | train_acc={tr_acc:.4f} val_acc={va_acc:.4f}")

        if va_acc > best_val:
            best_val = va_acc
            torch.save({"model": model.state_dict(), "val_acc": va_acc, "stage": "head", "epoch": epoch}, best_path)

    # ---- Stage 2: Fine-tune layer3+layer4+fc ----
    unfreeze_layers(model, ("layer3", "layer4", "fc"))
    print("Stage2 trainable params:", count_trainable_params(model))

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=args.lr_ft, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs_ft + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, optimizer, scaler, criterion)
        va_loss, va_acc = evaluate(model, val_loader, device, criterion)
        print(f"[FT] Epoch {epoch}/{args.epochs_ft} | train_acc={tr_acc:.4f} val_acc={va_acc:.4f}")

        if va_acc > best_val:
            best_val = va_acc
            torch.save({"model": model.state_dict(), "val_acc": va_acc, "stage": "ft", "epoch": epoch}, best_path)

    # ---- Final Test ----
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    te_loss, te_acc = evaluate(model, test_loader, device, criterion)

    print("\nBest checkpoint:", best_path, "| val_acc:", ckpt["val_acc"], "| stage:", ckpt["stage"], "| epoch:", ckpt["epoch"])
    print(f"TEST acc={te_acc:.4f} loss={te_loss:.4f}")


if __name__ == "__main__":
    main()