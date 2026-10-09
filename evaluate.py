"""Evaluate a trained model on the held-out test split and draw attention heatmaps.

    python evaluate.py                      # evaluates outputs/lstm_attn_ctx.pt
    python evaluate.py --name lstm_attn_noctx
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.data import prepare_data, tokenize
from src.utils import compute_metrics, get_device, load_checkpoint, predict_loader


def plot_attention(model, vocab, df, dataset, device, path, n=8):
    """Heatmap of attention weights on the first n correctly-flagged sarcastic test comments."""
    idx = np.where(df["label"].values == 1)[0][:60]
    rows = []
    model.eval()
    with torch.no_grad():
        for i in idx:
            x, xl, c, cl, y = (t.unsqueeze(0).to(device) if t.dim() else t.view(1).to(device)
                               for t in dataset[i])
            logit, attn = model(x, xl, c, cl, return_attention=True)
            p = torch.sigmoid(logit).item()
            toks = tokenize(df["comment"].iloc[i])[: int(xl.item())]
            if p > 0.5 and toks:
                rows.append((toks, attn[0, : len(toks)].cpu().numpy(), p))
            if len(rows) == n:
                break
    if not rows:
        return
    fig, axes = plt.subplots(len(rows), 1, figsize=(11, 0.9 * len(rows) + 0.5))
    axes = np.atleast_1d(axes)
    for ax, (toks, w, p) in zip(axes, rows):
        ax.imshow(w[None, :], aspect="auto", cmap="Reds", vmin=0, vmax=max(w.max(), 1e-6))
        ax.set_xticks(range(len(toks)))
        ax.set_xticklabels(toks, rotation=45, ha="right", fontsize=8)
        ax.set_yticks([0]); ax.set_yticklabels([f"p={p:.2f}"], fontsize=8)
    fig.suptitle("Attention on sarcastic comments (darker = more attention)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="lstm_attn_ctx")
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()

    device = get_device()
    model, vocab, cfg, thr, _ = load_checkpoint(os.path.join(args.out_dir, f"{args.name}.pt"), device)
    sets, _, frames = prepare_data(cfg["csv"], cfg["data_dir"], cfg["max_samples"] or None, cfg["seed"],
                                   cfg["max_len"], cfg["max_ctx_len"], vocab=vocab)
    dl = DataLoader(sets["test"], batch_size=512)
    y, p = predict_loader(model, dl, device)

    default_m = compute_metrics(y, p, 0.5)
    tuned_m = compute_metrics(y, p, thr)
    print(f"\nTEST @0.50 : " + "  ".join(f"{k}={v:.4f}" for k, v in default_m.items()))
    print(f"TEST @{thr:.2f} : " + "  ".join(f"{k}={v:.4f}" for k, v in tuned_m.items()))
    print("\n" + classification_report(y, (p >= thr).astype(int), target_names=["not sarcastic", "sarcastic"], digits=4))

    cm = confusion_matrix(y, (p >= thr).astype(int))
    ConfusionMatrixDisplay(cm, display_labels=["not sarcastic", "sarcastic"]).plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion matrix ({args.name})")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"{args.name}_confusion.png"), dpi=130)
    plt.close()

    plot_attention(model, vocab, frames["test"], sets["test"], device,
                   os.path.join(args.out_dir, f"{args.name}_attention.png"))

    with open(os.path.join(args.out_dir, f"{args.name}_test_metrics.json"), "w") as f:
        json.dump({"threshold": thr, "at_0.5": default_m, "at_tuned": tuned_m}, f, indent=2)
    print(f"Saved metrics + plots in {args.out_dir}/")


if __name__ == "__main__":
    main()
