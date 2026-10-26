"""Shock detection, event windows and matched control windows.

Time is measured in snapshots ("steps"); one step = 10 order-book events in both datasets.

Shock (proposal 6.2): the order-flow imbalance accumulated over W steps,
    S_t = sum_{j=t-W+1..t} flow_j,
falls in the lower q-tail (sell shock, s=-1) or upper q-tail (buy shock, s=+1) of that stock's
distribution. tau=0 is the last step of the shock window; tau=-W is the state just before it.
Events are declustered (no two events within `min_gap` steps in the same segment) and every
window [-pre, +post] stays inside one stock-day.

All outcome paths are *sign-adjusted* by s so that buy and sell shocks can be pooled:
    mid_bps   = s * (log M_tau - log M_-W) * 1e4    (>0 = price moved in the shock direction)
    cofi      = s * (cumOFI_tau - cumOFI_-W)       (>0 = flow in the shock direction)
    dep_press = depth (3 levels) on the side being hit (bid for a sell shock), / segment median
    dep_opp   = depth on the other side
    imb_press = s * depth imbalance (3 levels)     (>0 = book leans the way of the pressure)

Controls (needed because of regression to the mean): for each event, the k nearest non-event
moments in the same stock-day, with ordinary flow (|S| below the stock's median), matched either
  'pre'   on the book state at tau=-W  -> "what would have happened without the shock"
  'state' on the book state at tau=0   -> "same depleted book, but reached without a flow shock"
The 'state' comparison isolates mechanical refilling of a thin book from a response to the flow.
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------------------------
# shock detection
# ---------------------------------------------------------------------------------------------
def rolling_flow(df, col, W):
    return df.groupby("seg")[col].transform(lambda x: x.rolling(W, min_periods=W).sum())


def detect_shocks(df, flow_col, W=5, q=0.01, min_gap=100, pre=50, post=100):
    S = rolling_flow(df, flow_col, W)
    lo = S.groupby(df["stock"]).transform(lambda x: x.quantile(q))
    hi = S.groupby(df["stock"]).transform(lambda x: x.quantile(1 - q))
    sign = np.where(S <= lo, -1, np.where(S >= hi, 1, 0))
    seg_start = df.groupby("seg")["t"].transform("min").to_numpy()
    seg_len = df.groupby("seg")["t"].transform("size").to_numpy()
    t = df["t"].to_numpy()
    ok = (t - (pre + W) >= 0) & (t + post < seg_len)
    cand = np.flatnonzero((sign != 0) & ok)
    seg = df["seg"].to_numpy()
    keep, last_seg, last_t = [], None, -10**9
    for i in cand:
        if seg[i] != last_seg:
            last_seg, last_t = seg[i], -10**9
        if t[i] - last_t >= min_gap:
            keep.append(i)
            last_t = t[i]
    keep = np.array(keep, dtype=int)
    ev = pd.DataFrame({
        "idx": keep,
        "seg": seg[keep],
        "stock": df["stock"].to_numpy()[keep],
        "day": df["day"].to_numpy()[keep],
        "t": t[keep],
        "s": sign[keep],
        "shock_size": np.abs(S.to_numpy()[keep]),
        "tod": df["tod"].to_numpy()[keep],
    })
    if "block" in df:
        ev["block"] = df["block"].to_numpy()[keep]
    ev.attrs.update(flow_col=flow_col, W=W, q=q, min_gap=min_gap, pre=pre, post=post)
    return ev, S


# ---------------------------------------------------------------------------------------------
# event-time paths
# ---------------------------------------------------------------------------------------------
def paths(df, idx, s, taus, W, ofi_col="cum_ofi1n"):
    """Sign-adjusted paths for windows centred at rows `idx` (array) with directions `s`."""
    idx = np.asarray(idx)[:, None]
    s = np.asarray(s)[:, None]
    T = idx + np.asarray(taus)[None, :]
    base = idx - W
    lm = df["logmid"].to_numpy()
    cofi = df[ofi_col].to_numpy()
    bid3, ask3 = df["dbid3n"].to_numpy(), df["dask3n"].to_numpy()
    bid1, ask1 = df["dbid1n"].to_numpy(), df["dask1n"].to_numpy()
    sell = s == -1
    out = {
        "mid_bps": s * (lm[T] - lm[base]) * 1e4,
        "cofi": s * (cofi[T] - cofi[base]),
        "spread_ticks": df["spread_ticks"].to_numpy()[T],
        "rel_spread_bps": df["rel_spread_bps"].to_numpy()[T],
        "dep_press": np.where(sell, bid3[T], ask3[T]),
        "dep_opp": np.where(sell, ask3[T], bid3[T]),
        "dep1_press": np.where(sell, bid1[T], ask1[T]),
        "dep_total": (bid3[T] + ask3[T]) / 2,
        "dep10_press": np.where(sell, df["dbid10n"].to_numpy()[T], df["dask10n"].to_numpy()[T]),
        "improve_press": np.where(sell, df["bid_up"].to_numpy()[T], df["ask_dn"].to_numpy()[T]),
        "imb_press": s * df["imb3"].to_numpy()[T],
    }
    if "buy_vol" in df:  # LOBSTER message aggregates (per step)
        bv, sv = df["buy_vol"].to_numpy(), df["sell_vol"].to_numpy()
        vol = df["buy_vol"].to_numpy() + df["sell_vol"].to_numpy()
        vbar = df.groupby("seg")["buy_vol"].transform("mean").to_numpy() + df.groupby("seg")["sell_vol"].transform("mean").to_numpy()
        out["volume"] = vol[T] / vbar[T]
        out["mo_press"] = np.where(sell, sv[T], bv[T]) / vbar[T]          # market orders in shock direction
        ab, aa, cb, ca = (df[c].to_numpy() for c in ("add_bid", "add_ask", "can_bid", "can_ask"))
        dbar = df.groupby("seg")["dbid1"].transform("mean").to_numpy()
        out["net_add_press"] = np.where(sell, ab[T] - cb[T], aa[T] - ca[T]) / dbar[T]  # limit supply on the side being hit
        out["net_add_opp"] = np.where(sell, aa[T] - ca[T], ab[T] - cb[T]) / dbar[T]
    return out


# ---------------------------------------------------------------------------------------------
# matched controls
# ---------------------------------------------------------------------------------------------
def _features(df, rows, s, mode):
    sp = df["spread_ticks"].to_numpy()[rows]
    b3, a3 = df["dbid3n"].to_numpy()[rows], df["dask3n"].to_numpy()[rows]
    b1, a1 = df["dbid1n"].to_numpy()[rows], df["dask1n"].to_numpy()[rows]
    imb = df["imb3"].to_numpy()[rows]
    tod = df["tod"].to_numpy()[rows]
    rv = df["rvol20"].to_numpy()[rows]
    if mode == "pre":
        F = np.c_[sp, np.log1p(b3), np.log1p(a3), np.log1p(b1), np.log1p(a1), imb, rv, tod]
    else:
        sell = s == -1
        F = np.c_[sp, np.log1p(np.where(sell, b3, a3)), np.log1p(np.where(sell, a3, b3)),
                  np.log1p(np.where(sell, b1, a1)), np.log1p(np.where(sell, a1, b1)), s * imb, rv, tod]
    return np.nan_to_num(F)


def match_controls(df, ev, S, k=5, mode="pre", exclusion=None, step=2, seed=0):
    """Return array (n_events, k) of control centre rows (the control's tau=0)."""
    W, pre, post = ev.attrs["W"], ev.attrs["pre"], ev.attrs["post"]
    exclusion = exclusion or (min(post, 100) + W)   # keep controls clear of the main (100-step) window
    absS = np.abs(S.to_numpy())
    calm_thr = pd.Series(absS).groupby(df["stock"].to_numpy()).transform(lambda x: np.nanquantile(x, 0.5)).to_numpy()
    t = df["t"].to_numpy()
    seg_len = df.groupby("seg")["t"].transform("size").to_numpy()
    out = np.full((len(ev), k), -1, dtype=int)
    seg_rows = df.groupby("seg", sort=False).indices
    for seg, evs in ev.groupby("seg"):
        rows = seg_rows[seg]
        rows = rows[::step]
        rr = rows[(t[rows] - (pre + W) >= 0) & (t[rows] + post < seg_len[rows]) & (absS[rows] <= calm_thr[rows])]
        et = t[evs["idx"].to_numpy()]
        far = np.min(np.abs(t[rr][:, None] - et[None, :]), axis=1) > exclusion
        rr = rr[far]
        if len(rr) < k:
            continue
        for sgn in (-1, 1):
            sub = evs[evs["s"] == sgn]
            if sub.empty:
                continue
            lag = W if mode == "pre" else 0
            Fc = _features(df, rr - lag, np.full(len(rr), sgn), mode)
            Fe = _features(df, sub["idx"].to_numpy() - lag, sub["s"].to_numpy(), mode)
            mu, sd = Fc.mean(0), Fc.std(0) + 1e-9
            Zc, Ze = (Fc - mu) / sd, (Fe - mu) / sd
            D = ((Ze[:, None, :] - Zc[None, :, :]) ** 2).sum(-1)
            nn = np.argsort(D, axis=1)[:, :k]
            out[ev.index.get_indexer(sub.index)] = rr[nn]
    return out


def abnormal_paths(df, ev, ctrl, taus, ofi_col="cum_ofi1n"):
    """Event paths, mean control paths and their difference (dicts of arrays n_events x T)."""
    W = ev.attrs["W"]
    good = (ctrl >= 0).all(axis=1)
    e = ev[good]
    pe = paths(df, e["idx"].to_numpy(), e["s"].to_numpy(), taus, W, ofi_col)
    c = ctrl[good]
    k = c.shape[1]
    pc_flat = paths(df, c.ravel(), np.repeat(e["s"].to_numpy(), k), taus, W, ofi_col)
    pc = {v: a.reshape(len(e), k, -1).mean(axis=1) for v, a in pc_flat.items()}
    ab = {v: pe[v] - pc[v] for v in pe}
    return e.reset_index(drop=True), pe, pc, ab
