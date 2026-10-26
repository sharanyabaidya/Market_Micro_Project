"""Liquidity and order-flow measures computed from 10-level book snapshots.

Works for both datasets: the frame needs columns ap1..ap10, av1..av10, bp1..bp10, bv1..bv10 and a
segment id `seg` (one continuous stock-day). All flow measures are computed within a segment.

Measures (proposal Section 6.1 plus additions):
  mid, spread, relative spread       M = (A+B)/2,  S = A-B,  RS = S/M
  depth over K levels                D_K = sum_i<=K (bid + ask sizes)
  depth imbalance                    (Dbid - Dask)/(Dbid + Dask)  -- a *stock* of resting orders
  order-flow imbalance (OFI)         Cont, Kukanov & Stoikov (2014): net flow at the best quotes
  multi-level OFI (levels 1-3)       Xu, Gould & Howison (2019)
OFI is a *flow* measure: positive = buying pressure (bids added / asks removed or hit),
negative = selling pressure. This is what the proposal's "order-flow imbalance" refers to.
"""
import numpy as np
import pandas as pd

LEVELS = 10


def level_ofi(bp, bv, ap, av):
    """CKS order-flow contribution at one level between consecutive snapshots (arrays per segment)."""
    bp0, bv0, ap0, av0 = bp[:-1], bv[:-1], ap[:-1], av[:-1]
    bp1, bv1, ap1, av1 = bp[1:], bv[1:], ap[1:], av[1:]
    e_b = np.where(bp1 >= bp0, bv1, 0.0) - np.where(bp1 <= bp0, bv0, 0.0)
    e_a = np.where(ap1 <= ap0, av1, 0.0) - np.where(ap1 >= ap0, av0, 0.0)
    e = e_b - e_a
    e = np.where(np.isfinite(e), e, 0.0)
    return np.r_[0.0, e]


def add_measures(df, tick):
    df = df.copy()
    df["mid"] = (df["ap1"] + df["bp1"]) / 2
    df["spread"] = df["ap1"] - df["bp1"]
    df["spread_ticks"] = np.round(df["spread"] / tick)
    df["rel_spread_bps"] = 1e4 * df["spread"] / df["mid"]
    for K in (1, 3, 10):
        df[f"dbid{K}"] = df[[f"bv{i}" for i in range(1, K + 1)]].sum(axis=1)
        df[f"dask{K}"] = df[[f"av{i}" for i in range(1, K + 1)]].sum(axis=1)
        tot = df[f"dbid{K}"] + df[f"dask{K}"]
        df[f"imb{K}"] = (df[f"dbid{K}"] - df[f"dask{K}"]) / tot.where(tot > 0)
    df["logmid"] = np.log(df["mid"])

    ofi1 = np.zeros(len(df))
    ofi3 = np.zeros(len(df))
    ret = np.zeros(len(df))
    for _, idx in df.groupby("seg", sort=False).indices.items():
        g = df.iloc[idx]
        o = [level_ofi(g[f"bp{i}"].to_numpy(), g[f"bv{i}"].to_numpy(),
                       g[f"ap{i}"].to_numpy(), g[f"av{i}"].to_numpy()) for i in (1, 2, 3)]
        ofi1[idx] = o[0]
        ofi3[idx] = o[0] + o[1] + o[2]
        lm = g["logmid"].to_numpy()
        ret[idx] = np.r_[0.0, 1e4 * np.diff(lm)]
    df["ofi1"], df["ofi3"], df["ret_bps"] = ofi1, ofi3, ret

    # scale-free versions: divide by the segment's typical depth
    seg = df.groupby("seg")
    d1 = seg["dbid1"].transform("mean") / 2 + seg["dask1"].transform("mean") / 2
    d3 = (df["dbid3"] + df["dask3"]) / 2
    d3med = d3.groupby(df["seg"]).transform("median")
    df["ofi1n"] = df["ofi1"] / d1
    df["ofi3n"] = df["ofi3"] / d3med
    df["dbid3n"] = df["dbid3"] / d3med
    df["dask3n"] = df["dask3"] / d3med
    df["dbid1n"] = df["dbid1"] / d1
    df["dask1n"] = df["dask1"] / d1
    d10med = ((df["dbid10"] + df["dask10"]) / 2).groupby(df["seg"]).transform("median")
    df["dbid10n"] = df["dbid10"] / d10med
    df["dask10n"] = df["dask10"] / d10med
    # best quote improves on a side (a new, more aggressive limit order or a refill after a sweep)
    df["bid_up"] = (df.groupby("seg")["bp1"].diff() > 0).astype(float)
    df["ask_dn"] = (df.groupby("seg")["ap1"].diff() < 0).astype(float)
    df["cum_ofi1n"] = df.groupby("seg")["ofi1n"].cumsum()
    df["cum_ofi3n"] = df.groupby("seg")["ofi3n"].cumsum()
    # rolling realised volatility of mid returns (20 snapshots)
    df["rvol20"] = (df.groupby("seg")["ret_bps"]
                    .transform(lambda x: x.rolling(20, min_periods=10).std()))
    return df


def summary_by_stock(df, extra=()):
    cols = ["mid", "spread_ticks", "rel_spread_bps", "dbid1", "dask1", "dbid10", "dask10",
            "imb1", "imb10", "ofi1n", "rvol20", *extra]
    out = df.groupby("stock")[cols].mean()
    out.insert(0, "snapshots", df.groupby("stock").size())
    out.insert(1, "segments", df.groupby("stock")["seg"].nunique())
    return out
