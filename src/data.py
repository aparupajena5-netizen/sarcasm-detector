"""Data loading, live-chat-style text cleaning, vocabulary and dataset utilities.

Primary dataset (Kaggle): "Sarcasm on Reddit" -> danofer/sarcasm
    file: train-balanced-sarcasm.csv  (~1M rows, balanced)
    columns used: label, comment, parent_comment
The parent comment is the *context* the sarcastic reply responds to, which is
what makes the detection "contextual". In a live stream, the context is the
previous chat message(s).
"""
from __future__ import annotations

import glob
import os
import re
from collections import Counter
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset

PAD, UNK = "<pad>", "<unk>"
PAD_ID, UNK_ID = 0, 1

KAGGLE_HANDLE = "danofer/sarcasm"
KAGGLE_FILE = "train-balanced-sarcasm.csv"

# ----------------------------------------------------------------------------
# Text cleaning: tuned for short, noisy, fast live-stream chat
# ----------------------------------------------------------------------------
_URL = re.compile(r"https?://\S+|www\.\S+")
_USER = re.compile(r"(?<!\w)(@\w+|u/\w+|/u/\w+)")
_REPEAT = re.compile(r"(.)\1{3,}")            # "soooooo" -> "sooo"
_SLASH_S = re.compile(r"(?:^|\s)/s\b")        # explicit sarcasm tag = label leakage
_TOKEN = re.compile(r"<url>|<user>|[a-z0-9']+|[!?]+|\.{2,}|[^\sa-z0-9]")


def clean_text(text: object) -> str:
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = _SLASH_S.sub(" ", t)
    t = _URL.sub(" <url> ", t)
    t = _USER.sub(" <user> ", t)
    t = _REPEAT.sub(r"\1\1\1", t)
    return t


def tokenize(text: object) -> List[str]:
    return _TOKEN.findall(clean_text(text))


# ----------------------------------------------------------------------------
# Getting the data
# ----------------------------------------------------------------------------
def find_csv(root: str) -> Optional[str]:
    hits = glob.glob(os.path.join(root, "**", KAGGLE_FILE), recursive=True)
    return hits[0] if hits else None


def get_csv_path(csv: Optional[str], data_dir: str) -> str:
    """Return a path to the CSV: user supplied > local data/ > download via kagglehub."""
    if csv:
        if not os.path.exists(csv):
            raise FileNotFoundError(csv)
        return csv
    local = find_csv(data_dir)
    if local:
        return local
    try:
        import kagglehub
    except ImportError as e:  # pragma: no cover
        raise SystemExit("kagglehub missing. Run: pip install -r requirements.txt") from e
    print(f"Downloading Kaggle dataset '{KAGGLE_HANDLE}' (first run only) ...")
    try:
        folder = kagglehub.dataset_download(KAGGLE_HANDLE)
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            f"Kaggle download failed: {e}\n"
            "Fix: set up Kaggle credentials (see README) OR download the file manually from\n"
            f"https://www.kaggle.com/datasets/{KAGGLE_HANDLE} and put {KAGGLE_FILE} in the data/ folder."
        )
    path = find_csv(folder)
    if not path:
        raise SystemExit(f"{KAGGLE_FILE} not found inside {folder}")
    return path


def load_dataframe(csv_path: str, max_samples: Optional[int], seed: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=lambda c: c in {"label", "comment", "parent_comment"})
    missing = {"label", "comment"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV must contain columns {missing}. Optional column: parent_comment")
    if "parent_comment" not in df.columns:
        df["parent_comment"] = ""
    df = df.dropna(subset=["comment"]).copy()
    df["parent_comment"] = df["parent_comment"].fillna("")
    df["label"] = df["label"].astype(int)
    df = df.drop_duplicates(subset=["comment", "parent_comment"])
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed)
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Vocabulary
# ----------------------------------------------------------------------------
class Vocab:
    def __init__(self, itos: List[str]):
        self.itos = itos
        self.stoi = {w: i for i, w in enumerate(itos)}

    @classmethod
    def build(cls, token_lists, min_freq: int = 2, max_size: int = 30000) -> "Vocab":
        counter: Counter = Counter()
        for toks in token_lists:
            counter.update(toks)
        words = [w for w, c in counter.most_common(max_size) if c >= min_freq]
        return cls([PAD, UNK] + words)

    def encode(self, tokens: List[str], max_len: int, keep_last: bool = False) -> Tuple[np.ndarray, int]:
        """Encode + pad. `keep_last=True` keeps the END of long texts (best for context)."""
        if len(tokens) > max_len:
            tokens = tokens[-max_len:] if keep_last else tokens[:max_len]
        ids = [self.stoi.get(t, UNK_ID) for t in tokens]
        n = len(ids)
        ids = ids + [PAD_ID] * (max_len - n)
        return np.asarray(ids, dtype=np.int64), n

    def __len__(self) -> int:
        return len(self.itos)


def encode_frame(df: pd.DataFrame, vocab: Vocab, max_len: int, max_ctx_len: int):
    n = len(df)
    x = np.zeros((n, max_len), dtype=np.int64)
    xl = np.zeros(n, dtype=np.int64)
    c = np.zeros((n, max_ctx_len), dtype=np.int64)
    cl = np.zeros(n, dtype=np.int64)
    for i, (com, par) in enumerate(zip(df["comment"].values, df["parent_comment"].values)):
        x[i], xl[i] = vocab.encode(tokenize(com), max_len)
        c[i], cl[i] = vocab.encode(tokenize(par), max_ctx_len, keep_last=True)
    y = df["label"].values.astype(np.float32)
    return x, xl, c, cl, y


class SarcasmDataset(Dataset):
    def __init__(self, x, xl, c, cl, y):
        self.x, self.xl, self.c, self.cl, self.y = (
            torch.from_numpy(x), torch.from_numpy(xl),
            torch.from_numpy(c), torch.from_numpy(cl), torch.from_numpy(y),
        )

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.xl[i], self.c[i], self.cl[i], self.y[i]


# ----------------------------------------------------------------------------
# One entry point used by train.py and evaluate.py (deterministic given the args)
# ----------------------------------------------------------------------------
def split_frames(df: pd.DataFrame, seed: int):
    train_df, tmp = train_test_split(df, test_size=0.2, random_state=seed, stratify=df["label"])
    val_df, test_df = train_test_split(tmp, test_size=0.5, random_state=seed, stratify=tmp["label"])
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def prepare_data(csv: Optional[str], data_dir: str, max_samples: Optional[int], seed: int,
                 max_len: int, max_ctx_len: int, vocab: Optional[Vocab] = None,
                 min_freq: int = 2, max_vocab: int = 30000):
    path = get_csv_path(csv, data_dir)
    print(f"Reading {path}")
    df = load_dataframe(path, max_samples, seed)
    train_df, val_df, test_df = split_frames(df, seed)
    if vocab is None:  # vocabulary is built from TRAIN ONLY (no leakage)
        vocab = Vocab.build(
            (tokenize(t) for t in list(train_df["comment"]) + list(train_df["parent_comment"])),
            min_freq=min_freq, max_size=max_vocab,
        )
    sets = {name: SarcasmDataset(*encode_frame(part, vocab, max_len, max_ctx_len))
            for name, part in (("train", train_df), ("val", val_df), ("test", test_df))}
    print(f"Samples  train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}  "
          f"vocab={len(vocab):,}  sarcastic={df['label'].mean():.1%}")
    return sets, vocab, {"train": train_df, "val": val_df, "test": test_df}
