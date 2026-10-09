"""Single-message inference with attention read-out."""
from __future__ import annotations

import os
from typing import List, Optional

import torch

from .data import tokenize
from .utils import get_device, load_checkpoint


class SarcasmPredictor:
    def __init__(self, ckpt_path: str = "outputs/lstm_attn_ctx.pt"):
        if not os.path.exists(ckpt_path):
            raise SystemExit(f"Checkpoint {ckpt_path} not found. Train first:  python train.py")
        self.device = get_device()
        self.model, self.vocab, self.cfg, self.threshold, _ = load_checkpoint(ckpt_path, self.device)

    @torch.no_grad()
    def predict(self, comment: str, context: Optional[str] = "", top_k: int = 3) -> dict:
        x, xl = self.vocab.encode(tokenize(comment), self.cfg["max_len"])
        c, cl = self.vocab.encode(tokenize(context or ""), self.cfg["max_ctx_len"], keep_last=True)
        t = lambda a: torch.as_tensor(a).unsqueeze(0).to(self.device)
        logit, attn = self.model(t(x), t(xl), t(c), t(cl), return_attention=True)
        prob = torch.sigmoid(logit).item()
        toks: List[str] = tokenize(comment)[: self.cfg["max_len"]]
        w = attn[0, : len(toks)].cpu().tolist()
        top = sorted(zip(toks, w), key=lambda z: -z[1])[:top_k]
        return {"prob": prob, "flag": prob >= self.threshold,
                "tokens": toks, "weights": w, "top_words": [(a, round(b, 3)) for a, b in top]}
