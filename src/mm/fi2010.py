"""Load FI-2010 and split it into clean stock-day segments.

Layout of a FI-2010 file: 149 text rows (features) x N columns (snapshots). Rows 0-39 are the raw
10-level book: for level i (0-based) row 4i = ask price, 4i+1 = ask size, 4i+2 = bid price,
4i+3 = bid size. The remaining rows are engineered features and labels, which we do not need.

Inside a file the snapshots of all stocks and days are simply concatenated:
  * test files (one day): stock 1, stock 2, ..., stock 5
  * the training file (days 1-7): stock 1 day 1..7, stock 2 day 1..7, ...
A boundary between two segments is found where the book is completely rebuilt: no
(side, price, size) pair survives from one snapshot to the next. Within a trading day 10 events can
never remove all 20 resting pairs, so the rule is exact (checked: no step inside a day has fewer
than 5 surviving pairs, while all 34 + 4x4 boundaries have at most 2).
"""
import numpy as np
import pandas as pd

from .config import (FI_FILES, FI_PRICE_SCALE, FI_SIZE_SCALE, FI_STOCKS, PROCESSED, RAW_FI)

LEVELS = 10


def read_raw_book(path, n_rows=40):
    """Read the first `n_rows` feature rows of a FI-2010 text file -> array (n_snapshots, n_rows)."""
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n_rows:
                break
            rows.append(np.array(line.split(), dtype=np.float64))
    return np.vstack(rows).T


def surviving_pairs(B):
    """Number of (side, price, size) pairs present in both snapshot t-1 and t (for t >= 1)."""
    ap, av = B[:, 0:40:4], B[:, 1:40:4]
    bp, bv = B[:, 2:40:4], B[:, 3:40:4]

    def key(p, v):
        return np.round(p * 1e6).astype(np.int64) * 10_000_000 + np.round(v * 1e6).astype(np.int64)

    out = np.zeros(len(B) - 1, dtype=int)
    for K in (key(ap, av), key(bp, bv)):
        prev, cur = K[:-1], K[1:]
        # for every pair in prev, is it anywhere in cur?
        out += (prev[:, :, None] == cur[:, None, :]).any(axis=2).sum(axis=1)
    return out


def segment_file(B, days):
    """Return list of (start, end, stock, day) for one file."""
    surv = surviving_pairs(B)
    cuts = np.flatnonzero(surv <= 2) + 1
    n_seg = len(FI_STOCKS) * len(days)
    if len(cuts) != n_seg - 1:
        raise ValueError(f"expected {n_seg - 1} boundaries, found {len(cuts)}")
    bounds = np.r_[0, cuts, len(B)]
    segs = []
    for j in range(n_seg):
        stock = FI_STOCKS[j // len(days)]   # stock-major order
        day = days[j % len(days)]
        segs.append((bounds[j], bounds[j + 1], stock, day))
    return segs


def book_frame(B):
    """Turn raw DecPre book rows into a DataFrame in EUR and shares."""
    cols = {}
    for i in range(LEVELS):
        cols[f"ap{i+1}"] = B[:, 4 * i] * FI_PRICE_SCALE
        cols[f"av{i+1}"] = B[:, 4 * i + 1] * FI_SIZE_SCALE
        cols[f"bp{i+1}"] = B[:, 4 * i + 2] * FI_PRICE_SCALE
        cols[f"bv{i+1}"] = B[:, 4 * i + 3] * FI_SIZE_SCALE
    df = pd.DataFrame(cols)
    px = [c for c in df if c[1] == "p"]
    df[px] = df[px].round(2)
    sz = [c for c in df if c[1] == "v"]
    df[sz] = df[sz].round(0)
    return df


def build_fi2010_book(cache=True):
    """All 10 days x 5 stocks as one DataFrame with stock, day, seg and within-segment index t."""
    out_path = PROCESSED / "fi2010_book.parquet"
    if cache and out_path.exists():
        return pd.read_parquet(out_path)
    parts = []
    for fname, days in FI_FILES:
        B = read_raw_book(RAW_FI / fname)
        for a, b, stock, day in segment_file(B, days):
            seg = book_frame(B[a:b])
            seg.insert(0, "t", np.arange(b - a))
            seg.insert(0, "day", day)
            seg.insert(0, "stock", stock)
            parts.append(seg)
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["stock", "day", "t"], kind="stable").reset_index(drop=True)
    df["seg"] = df["stock"] + "_d" + df["day"].astype(str).str.zfill(2)
    n = df.groupby("seg")["t"].transform("size")
    df["tod"] = df["t"] / n          # position within the trading day (event clock), 0..1
    df.to_parquet(out_path)
    return df
