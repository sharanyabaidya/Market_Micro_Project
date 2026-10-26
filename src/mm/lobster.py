"""Load LOBSTER sample files (NASDAQ, 2012-06-21, 10 levels).

Message file columns: time (s after midnight), type, order id, size, price (USD*1e4), direction.
  type 1 = new limit order, 2 = partial cancel, 3 = full delete, 4 = visible execution,
  5 = hidden execution, 7 = trading halt.
  direction: -1 = sell limit order, +1 = buy limit order. An execution of a sell limit order is a
  buyer-initiated trade, so the trade sign is d = -direction.
Order-book file: after each message, for each level i: ask price, ask size, bid price, bid size.
Empty levels are filled with prices +/-9999999999 and size 0.

Outputs:
  * snapshot panel sampled every LOB_BIN messages (same layout as FI-2010, so the same measures
    and event-study code apply), with per-bin message aggregates: market buy/sell volume and
    limit-order additions / cancellations near the top of the book on each side;
  * trade-level table for the adverse-selection analysis (H4).
"""
import numpy as np
import pandas as pd

from .config import LOB_BIN, LOB_DATE, LOB_END, LOB_START, LOB_TICKERS, PROCESSED, RAW_LOB

LEVELS = 10


def _paths(ticker):
    stem = f"{ticker}_{LOB_DATE}_34200000_57600000"
    return RAW_LOB / f"{stem}_message_{LEVELS}.csv", RAW_LOB / f"{stem}_orderbook_{LEVELS}.csv"


def load_raw(ticker):
    mpath, opath = _paths(ticker)
    msg = pd.read_csv(mpath, header=None, usecols=range(6),
                      names=["time", "type", "oid", "size", "price", "dir"])
    ob = pd.read_csv(opath, header=None).to_numpy(dtype=np.float64)
    assert len(msg) == len(ob)
    cols = {}
    for i in range(LEVELS):
        ap, av, bp, bv = ob[:, 4 * i], ob[:, 4 * i + 1], ob[:, 4 * i + 2], ob[:, 4 * i + 3]
        ap = np.where(np.abs(ap) >= 9e9, np.nan, ap / 1e4)
        bp = np.where(np.abs(bp) >= 9e9, np.nan, bp / 1e4)
        cols[f"ap{i+1}"], cols[f"av{i+1}"] = ap, av
        cols[f"bp{i+1}"], cols[f"bv{i+1}"] = bp, bv
    book = pd.DataFrame(cols)
    msg["price"] = msg["price"] / 1e4
    return msg, book


def trade_sign(msg, book):
    """Aggressor sign per message (+1 buyer-initiated, -1 seller-initiated, 0 = not a signable trade).

    * Visible executions (type 4): exact. Execution of a sell limit order (dir = -1) is a buy, so
      sign = -dir. Checked: 100% agree with the side of the mid-price on which the trade printed.
    * Hidden executions (type 5): the direction field is NOT reliable (in INTC/MSFT ~70% execute at the
      mid-price and the field agrees with the price side only ~15% of the time). They are signed with the
      quote rule (price above the prior mid = buy, below = sell); executions exactly at the mid are left
      unsigned and excluded from all signed-flow measures.
    """
    typ = msg["type"].to_numpy()
    mid_prev = ((book["ap1"] + book["bp1"]) / 2).shift(1).to_numpy()
    s = np.zeros(len(msg), dtype=int)
    vis = typ == 4
    s[vis] = -msg["dir"].to_numpy()[vis]
    hid = typ == 5
    s[hid] = np.sign(np.nan_to_num(msg["price"].to_numpy()[hid] - mid_prev[hid])).astype(int)
    return s


def _message_flags(msg, book):
    """Classify every message using the book *before* it arrived (previous row)."""
    prev = book.shift(1)
    typ, d, size, price = msg["type"].to_numpy(), msg["dir"].to_numpy(), msg["size"].to_numpy(), msg["price"].to_numpy()
    near_bid = price >= prev["bp3"].to_numpy()      # at or above the 3rd bid level
    near_ask = price <= prev["ap3"].to_numpy()
    is_add, is_can = typ == 1, np.isin(typ, (2, 3))
    is_exe = np.isin(typ, (4, 5))
    sgn = trade_sign(msg, book)
    f = pd.DataFrame(index=msg.index)
    f["buy_vol"] = np.where(sgn == 1, size, 0)                # buyer-initiated volume
    f["sell_vol"] = np.where(sgn == -1, size, 0)              # seller-initiated volume
    f["add_bid"] = np.where(is_add & (d == 1) & near_bid, size, 0)
    f["add_ask"] = np.where(is_add & (d == -1) & near_ask, size, 0)
    f["can_bid"] = np.where(is_can & (d == 1) & near_bid, size, 0)
    f["can_ask"] = np.where(is_can & (d == -1) & near_ask, size, 0)
    f["n_trades"] = is_exe.astype(int)
    return f


def build_lobster_panel(cache=True):
    out_path = PROCESSED / "lobster_book.parquet"
    if cache and out_path.exists():
        return pd.read_parquet(out_path)
    parts = []
    for tk in LOB_TICKERS:
        msg, book = load_raw(tk)
        flags = _message_flags(msg, book)
        keep = (msg["time"] >= LOB_START) & (msg["time"] <= LOB_END) & (msg["type"] != 7)
        msg, book, flags = msg[keep].reset_index(drop=True), book[keep].reset_index(drop=True), flags[keep].reset_index(drop=True)
        b = np.arange(len(msg)) // LOB_BIN
        agg = flags.groupby(b).sum()
        last = pd.Series(np.arange(len(msg))).groupby(b).max().to_numpy()
        snap = book.iloc[last].reset_index(drop=True)
        snap = pd.concat([snap, agg.reset_index(drop=True)], axis=1)
        snap.insert(0, "time", msg["time"].to_numpy()[last])
        snap.insert(0, "t", np.arange(len(snap)))
        snap.insert(0, "day", 1)
        snap.insert(0, "stock", tk)
        snap["seg"] = tk
        snap["tod"] = (snap["time"] - LOB_START) / (LOB_END - LOB_START)
        snap = snap.dropna(subset=["ap1", "bp1", "ap3", "bp3"]).reset_index(drop=True)
        snap["t"] = np.arange(len(snap))
        parts.append(snap)
    df = pd.concat(parts, ignore_index=True)
    # 30-minute block id: used as the clustering unit for standard errors (one day only)
    df["block"] = df["stock"] + "_" + ((df["time"] - LOB_START) // 1800).astype(int).astype(str)
    df.to_parquet(out_path)
    return df


def build_trades(horizons=(1, 5, 10, 30, 60, 120, 300), cache=True):
    """One row per market order (executions with the same timestamp and side are merged).

    Effective half-spread  = d (P - M0) / M0
    Price impact (h)       = d (M_h - M0) / M0          -- adverse selection (permanent part)
    Realised half-spread(h)= d (P - M_h) / M0           -- what the liquidity provider keeps
    all in basis points; M0 = mid just before the trade, M_h = mid h seconds later.
    """
    out_path = PROCESSED / "lobster_trades.parquet"
    if cache and out_path.exists():
        return pd.read_parquet(out_path)
    parts = []
    for tk in LOB_TICKERS:
        msg, book = load_raw(tk)
        mid = ((book["ap1"] + book["bp1"]) / 2).to_numpy()
        spr = (book["ap1"] - book["bp1"]).to_numpy()
        times = msg["time"].to_numpy()
        exe = msg["type"].isin([4, 5]).to_numpy()
        idx = np.flatnonzero(exe)
        tr = msg.iloc[idx].copy()
        tr["d"] = trade_sign(msg, book)[idx]
        tr["mid_pre"] = np.r_[np.nan, mid][idx]       # book row before the message
        tr["spread_pre"] = np.r_[np.nan, spr][idx]
        tr["notional"] = tr["price"] * tr["size"]
        tr = tr[tr["d"] != 0]                                # unsignable hidden executions at the mid
        g = tr.groupby(["time", "d"], sort=True)
        t = g.agg(size=("size", "sum"), notional=("notional", "sum"), mid_pre=("mid_pre", "first"),
                  spread_pre=("spread_pre", "first"), hidden=("type", lambda x: (x == 5).mean())).reset_index()
        t["price"] = t["notional"] / t["size"]
        for h in horizons:
            j = np.searchsorted(times, t["time"].to_numpy() + h, side="right") - 1
            t[f"mid_{h}s"] = mid[j]
        t = t[(t["time"] >= LOB_START) & (t["time"] <= LOB_END - max(horizons))]
        t = t[np.isfinite(t["mid_pre"])]
        m0 = t["mid_pre"]
        t["eff_hs_bps"] = 1e4 * t["d"] * (t["price"] - m0) / m0
        for h in horizons:
            t[f"impact_{h}s_bps"] = 1e4 * t["d"] * (t[f"mid_{h}s"] - m0) / m0
            t[f"realized_{h}s_bps"] = 1e4 * t["d"] * (t["price"] - t[f"mid_{h}s"]) / m0
        t.insert(0, "stock", tk)
        parts.append(t)
    df = pd.concat(parts, ignore_index=True)
    df["block"] = df["stock"] + "_" + ((df["time"] - LOB_START) // 1800).astype(int).astype(str)
    df.to_parquet(out_path)
    return df
