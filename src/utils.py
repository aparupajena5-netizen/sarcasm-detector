"""Shared helpers: seeding, device, metrics, checkpoint I/O, plots."""
from __future__ import annotations

import os
import random

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_metrics(y_true, probs, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    pred = (probs >= threshold).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
    }
    try:
        out["roc_auc"] = roc_auc_score(y_true, probs)
    except ValueError:  # only one class present
        out["roc_auc"] = float("nan")
    return out


def best_threshold(y_true, probs) -> float:
    """Threshold that maximises F1 on the validation set."""
    best_t, best_f = 0.5, -1.0
    for t in np.linspace(0.2, 0.8, 61):
        f = f1_score(y_true, (np.asarray(probs) >= t).astype(int), zero_division=0)
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t


def save_checkpoint(path: str, model, vocab, cfg: dict, threshold: float, history: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "vocab": vocab.itos, "cfg": cfg,
                "threshold": threshold, "history": history}, path)


def load_checkpoint(path: str, device):
    from .model import ContextAttentionLSTM
    from .data import Vocab
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    vocab = Vocab(ckpt["vocab"])
    model = ContextAttentionLSTM(
        len(vocab), cfg["emb_dim"], cfg["hidden"], cfg["attn_dim"], cfg["num_layers"],
        cfg["dropout"], cfg["use_context"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, vocab, cfg, ckpt["threshold"], ckpt.get("history", {})


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    probs, ys = [], []
    for x, xl, c, cl, y in loader:
        logit = model(x.to(device), xl.to(device), c.to(device), cl.to(device))
        probs.append(torch.sigmoid(logit).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(probs)
