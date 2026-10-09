"""Train the LSTM + attention sarcasm detector.

Examples:
    python train.py                                # downloads Kaggle data, 200k sample, 6 epochs
    python train.py --max-samples 50000 --epochs 4 # quicker
    python train.py --no-context                   # ablation: ignore the parent/previous message
    python train.py --csv data/train-balanced-sarcasm.csv
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import prepare_data
from src.model import ContextAttentionLSTM
from src.utils import (best_threshold, compute_metrics, get_device, predict_loader,
                       save_checkpoint, set_seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=None, help="path to train-balanced-sarcasm.csv (else auto-download)")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--name", default=None, help="run name (default: lstm_attn_ctx or lstm_attn_noctx)")
    p.add_argument("--max-samples", type=int, default=200_000, help="0 = use all ~1M rows")
    p.add_argument("--max-len", type=int, default=40, help="max tokens in the comment")
    p.add_argument("--max-ctx-len", type=int, default=40, help="max tokens kept from the context")
    p.add_argument("--emb-dim", type=int, default=128)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--attn-dim", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=2, help="early stopping on val F1")
    p.add_argument("--no-context", action="store_true", help="ablation without context")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    name = args.name or ("lstm_attn_noctx" if args.no_context else "lstm_attn_ctx")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Device: {device} | run: {name}")

    sets, vocab, _ = prepare_data(args.csv, args.data_dir, args.max_samples or None, args.seed,
                                  args.max_len, args.max_ctx_len)
    train_dl = DataLoader(sets["train"], batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(sets["val"], batch_size=args.batch_size * 2)

    model = ContextAttentionLSTM(len(vocab), args.emb_dim, args.hidden, args.attn_dim,
                                 args.num_layers, args.dropout, not args.no_context).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=1)
    loss_fn = nn.BCEWithLogitsLoss()

    cfg = {**vars(args), "use_context": not args.no_context, "name": name}
    ckpt_path = os.path.join(args.out_dir, f"{name}.pt")
    history = {"train_loss": [], "val_f1": [], "val_auc": [], "val_acc": []}
    best_f1, bad, best_thr = -1.0, 0, 0.5

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for step, (x, xl, c, cl, y) in enumerate(train_dl, 1):
            x, xl, c, cl, y = (t.to(device) for t in (x, xl, c, cl, y))
            opt.zero_grad()
            loss = loss_fn(model(x, xl, c, cl), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * y.size(0)
            n += y.size(0)
            if step % 100 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_dl)} loss {total / n:.4f}", end="\r")

        y_val, p_val = predict_loader(model, val_dl, device)
        m = compute_metrics(y_val, p_val)
        sched.step(m["f1"])
        history["train_loss"].append(total / n)
        history["val_f1"].append(m["f1"])
        history["val_auc"].append(m["roc_auc"])
        history["val_acc"].append(m["accuracy"])
        print(f"epoch {epoch}/{args.epochs} | train loss {total / n:.4f} | val acc {m['accuracy']:.4f} "
              f"F1 {m['f1']:.4f} AUC {m['roc_auc']:.4f} | {time.time() - t0:.0f}s" + " " * 10)

        if m["f1"] > best_f1:
            best_f1, bad = m["f1"], 0
            best_thr = best_threshold(y_val, p_val)
            save_checkpoint(ckpt_path, model, vocab, cfg, best_thr, history)
            print(f"  saved best model -> {ckpt_path} (decision threshold {best_thr:.2f})")
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stopping.")
                break

    with open(os.path.join(args.out_dir, f"{name}_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(history["train_loss"], marker="o"); ax[0].set_title("Training loss"); ax[0].set_xlabel("epoch")
    ax[1].plot(history["val_f1"], marker="o", label="F1")
    ax[1].plot(history["val_auc"], marker="o", label="ROC-AUC")
    ax[1].set_title("Validation"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, f"{name}_curves.png"), dpi=130)
    print(f"Best val F1 {best_f1:.4f}. Next: python evaluate.py --name {name}")


if __name__ == "__main__":
    main()
