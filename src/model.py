"""BiLSTM + context-conditioned attention for contextual sarcasm detection.

comment  --emb--> BiLSTM --> h_1..h_T
context  --emb--> BiLSTM --> mean-pool --> c          (shared encoder weights)

attention over comment tokens, conditioned on the context vector:
    e_t = v^T tanh(W_h h_t + W_c c + b)      a = softmax(e) (padding masked)
    s   = sum_t a_t h_t

classifier input: [s, c, s*c, |s-c|]  ->  MLP  ->  1 logit (sarcastic / not)
The attention weights show *which words* triggered the flag.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ContextAttentionLSTM(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 128, hidden: int = 128,
                 attn_dim: int = 64, num_layers: int = 1, dropout: float = 0.3,
                 use_context: bool = True):
        super().__init__()
        self.use_context = use_context
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.emb_drop = nn.Dropout(dropout)
        self.encoder = nn.LSTM(emb_dim, hidden, num_layers=num_layers, batch_first=True,
                               bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        d = 2 * hidden
        self.w_h = nn.Linear(d, attn_dim, bias=True)
        self.w_c = nn.Linear(d, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)
        self.classifier = nn.Sequential(
            nn.Linear(4 * d, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    def encode(self, ids: torch.Tensor, lengths: torch.Tensor):
        """Returns (outputs [B,T,2H], mask [B,T]). Empty sequences are treated as length 1."""
        lengths = lengths.clamp(min=1)
        emb = self.emb_drop(self.embedding(ids))
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.encoder(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=ids.size(1))
        mask = torch.arange(ids.size(1), device=ids.device)[None, :] < lengths[:, None]
        return out, mask

    def forward(self, x, x_len, c, c_len, return_attention: bool = False):
        h, mask = self.encode(x, x_len)                       # [B,T,D]
        if self.use_context:
            hc, cmask = self.encode(c, c_len)
            m = cmask.unsqueeze(-1).float()
            ctx = (hc * m).sum(1) / m.sum(1).clamp(min=1.0)   # [B,D]
        else:
            ctx = torch.zeros(h.size(0), h.size(2), device=h.device)

        scores = self.v(torch.tanh(self.w_h(h) + self.w_c(ctx).unsqueeze(1))).squeeze(-1)  # [B,T]
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)                       # [B,T]
        s = torch.bmm(attn.unsqueeze(1), h).squeeze(1)         # [B,D]

        feats = torch.cat([s, ctx, s * ctx, (s - ctx).abs()], dim=-1)
        logit = self.classifier(feats).squeeze(-1)
        return (logit, attn) if return_attention else logit
