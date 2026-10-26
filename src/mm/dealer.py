"""Tests related to dealer theory on LOBSTER (where trades and order IDs are observed).

IMPORTANT - what can and cannot be measured
-------------------------------------------
LOBSTER does not identify traders. There is no observed dealer and no observed dealer inventory.
Every market order trades against resting limit orders, so the *aggregate* position change of all
passive liquidity providers is, by accounting identity, minus the aggregate signed market-order flow:

    passive-side flow per second  = seller-initiated volume - buyer-initiated volume
    "inventory proxy" I_w(t)      = rolling sum of that flow over the last w seconds (a rolling net
                                    position, not a cumulative inventory; standardised per stock)

Consequences:
  * any regression on I_w is also a regression on lagged signed order flow; a positive link between
    I_w and future passive-side flow is order-flow persistence, not a failure to "unwind";
  * the only test that separates an inventory mechanism from information is whether prices revert
    *holding future order flow fixed* (test D2b below). Inventory models predict beta > 0; pure
    adverse selection predicts beta = 0;
  * "orders at the best quote" counts orders, not competing dealers (one participant can post many).

Test catalogue (labels used in the report)
  D1  order-flow persistence      future passive-side flow on I_w                  (descriptive)
  D2a price continuation/reversal future return on I_w, past return               (descriptive)
  D2b inventory reversal          as D2a + future flow, volatility, spread, depth  inventory: beta > 0
  D3  spread and imbalance        spread on |I_w| with control variants            AM: beta > 0
  D4  order-book competition      spread on # orders at best / recent price
                                  improvements (proxies, not dealer counts)        Ho-Stoll: beta < 0
  D5  Huang-Stoll decomposition   exploratory, with sanity checks
  D6  liquidity supply by order age (age since the order ID was submitted)
D1/D2 (forward-window outcomes on backward-window regressors) use stock fixed effects: a placebo with
shuffled order flow shows that 30-minute-block fixed effects create a spurious negative coefficient
(fe_placebo). D3/D4 (point-in-time outcomes) use stock x 30-minute fixed effects. SEs are always
clustered by stock x 30-minute block (60 clusters).
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import LOB_END, LOB_START, LOB_TICK, LOB_TICKERS, PROCESSED
from .lobster import load_raw, trade_sign
from .stats import cluster_mean, ols_cluster, stars

AGE_BINS = [-np.inf, 1.0, 60.0, np.inf]
AGE_LABELS = ["young (<1s)", "middle (1-60s)", "old (>60s)"]


# ---------------------------------------------------------------------------------------------
# one pass through the message file: order tracking
# ---------------------------------------------------------------------------------------------
def track_orders(msg, book, sample_idx):
    """Replay the message file.

    counts : number of live visible orders at best bid / best ask after each sampled message.
             Orders resting from before 09:30 are not in the file and are not counted (we drop the
             first 15 minutes anyway).
    execs  : one row per *visible* execution (type 4) with the executed order's age = execution time
             minus the time its current order ID was submitted (NaN if the ID was never submitted in the
             file, i.e. a pre-opening order). A repriced order that LOBSTER records as delete + new
             submission gets a new ID, so its age restarts.
    """
    t = msg["time"].to_numpy()
    typ = msg["type"].to_numpy()
    oid = msg["oid"].to_numpy()
    size = msg["size"].to_numpy()
    price = np.round(msg["price"].to_numpy(), 4)
    dirn = msg["dir"].to_numpy()
    bb = np.round(book["bp1"].to_numpy(), 4)
    ba = np.round(book["ap1"].to_numpy(), 4)

    live = {}          # oid -> [remaining, price, dir, submit_time]
    cnt = {}           # (dir, price) -> number of live orders
    counts = np.zeros((len(sample_idx), 2), dtype=np.int32)
    ex_rows = []
    k = 0
    n_s = len(sample_idx)
    for i in range(len(t)):
        ty = typ[i]
        o = oid[i]
        if ty == 1:
            live[o] = [size[i], price[i], dirn[i], t[i]]
            key = (dirn[i], price[i])
            cnt[key] = cnt.get(key, 0) + 1
        elif ty == 2:
            rec = live.get(o)
            if rec is not None:
                rec[0] -= size[i]
        elif ty == 3 or ty == 4:
            rec = live.get(o)
            if ty == 4:
                age = t[i] - rec[3] if rec is not None else np.nan
                ex_rows.append((t[i], o, size[i], price[i], dirn[i], age))
            if rec is not None:
                if ty == 4:
                    rec[0] -= size[i]
                if ty == 3 or rec[0] <= 0:
                    key = (rec[2], rec[1])
                    cnt[key] = cnt.get(key, 1) - 1
                    del live[o]
        while k < n_s and sample_idx[k] == i:
            counts[k, 0] = cnt.get((1, bb[i]), 0)
            counts[k, 1] = cnt.get((-1, ba[i]), 0)
            k += 1
    execs = pd.DataFrame(ex_rows, columns=["time", "oid", "size", "price", "dir", "age"])
    return counts, execs


def build_second_panel(cache=True):
    """1-second panel per stock (09:45-15:45) and the visible-execution table with order ages."""
    p1, p2 = PROCESSED / "lobster_1s.parquet", PROCESSED / "lobster_execs.parquet"
    if cache and p1.exists() and p2.exists():
        return pd.read_parquet(p1), pd.read_parquet(p2)
    secs, exs = [], []
    for tk in LOB_TICKERS:
        msg, book = load_raw(tk)
        t = msg["time"].to_numpy()
        grid = np.arange(int(LOB_START), int(LOB_END))
        last = np.searchsorted(t, grid + 1, side="left") - 1         # last message before end of second
        last = np.clip(last, 0, len(t) - 1)
        counts, execs = track_orders(msg, book, np.unique(last))
        cmap = dict(zip(np.unique(last), counts))
        cc = np.array([cmap[j] for j in last])
        b = book.iloc[last].reset_index(drop=True)
        s = pd.DataFrame({"stock": tk, "sec": grid})
        s["mid"] = ((b["ap1"] + b["bp1"]) / 2).to_numpy()
        s["spread_bps"] = (1e4 * (b["ap1"] - b["bp1"]) / s["mid"]).to_numpy()
        s["spread_ticks"] = np.round(((b["ap1"] - b["bp1"]) / LOB_TICK).to_numpy())
        s["depth1"] = (b["av1"] + b["bv1"]).to_numpy()
        s["depth3"] = (b[["av1", "av2", "av3", "bv1", "bv2", "bv3"]].sum(axis=1)).to_numpy()
        s["n_best"] = cc.mean(axis=1)
        s["n_bid_best"], s["n_ask_best"] = cc[:, 0], cc[:, 1]
        sgn = trade_sign(msg, book)
        sec_of = np.floor(t).astype(int)
        size = msg["size"].to_numpy()
        buy = pd.Series(np.where(sgn == 1, size, 0)).groupby(sec_of).sum()
        sell = pd.Series(np.where(sgn == -1, size, 0)).groupby(sec_of).sum()
        s["buy_vol"] = buy.reindex(grid, fill_value=0).to_numpy()
        s["sell_vol"] = sell.reindex(grid, fill_value=0).to_numpy()
        # price improvements: new limit orders priced inside the spread (better than the prior best quote)
        prev_bb, prev_ba = book["bp1"].shift(1).to_numpy(), book["ap1"].shift(1).to_numpy()
        typ, d, px = msg["type"].to_numpy(), msg["dir"].to_numpy(), msg["price"].to_numpy()
        imp = (typ == 1) & (((d == 1) & (px > prev_bb + 1e-9)) | ((d == -1) & (px < prev_ba - 1e-9)))
        s["n_improve"] = pd.Series(imp.astype(int)).groupby(sec_of).sum().reindex(grid, fill_value=0).to_numpy()
        secs.append(s)
        execs.insert(0, "stock", tk)
        exs.append(execs)
    S = pd.concat(secs, ignore_index=True)
    S["block"] = S["stock"] + "_" + ((S["sec"] - LOB_START) // 1800).astype(int).astype(str)
    E = pd.concat(exs, ignore_index=True)
    S.to_parquet(p1)
    E.to_parquet(p2)
    return S, E


# ---------------------------------------------------------------------------------------------
# D1-D4 regressions on the 1-second panel
# ---------------------------------------------------------------------------------------------
def add_inventory(S, w):
    S = S.copy()
    S["lp_flow"] = S["sell_vol"] - S["buy_vol"]              # passive side buys when market orders sell
    S["logmid"] = np.log(S["mid"])
    g = S.groupby("stock")
    S["inv"] = g["lp_flow"].transform(lambda x: x.rolling(w, min_periods=w).sum())
    S["inv_z"] = S.groupby("stock")["inv"].transform(lambda x: (x - x.mean()) / x.std())
    S["abs_inv_z"] = S["inv_z"].abs()
    S["past_ret_bps"] = 1e4 * g["logmid"].diff(w)
    S["abs_past_ret_bps"] = S["past_ret_bps"].abs()
    S["ret1"] = 1e4 * g["logmid"].diff()
    S["rvol60"] = S.groupby("stock")["ret1"].transform(lambda x: x.rolling(60, min_periods=30).std())
    S["log_depth1"] = np.log(S["depth1"])
    S["log_n_best"] = np.log(S["n_best"].clip(lower=0.5))
    S["vol60"] = g["buy_vol"].transform(lambda x: x.rolling(60, min_periods=1).sum()) + \
        g["sell_vol"].transform(lambda x: x.rolling(60, min_periods=1).sum())
    S["log_vol60"] = np.log1p(S["vol60"] / S.groupby("stock")["vol60"].transform("mean"))
    S["improve60"] = g["n_improve"].transform(lambda x: x.rolling(60, min_periods=1).sum())
    S["log_improve60"] = np.log1p(S["improve60"])
    return S


def _row(test, m, var, **extra):
    b, se, p = m.params[var], m.bse[var], m.pvalues[var]
    return dict(test=test, beta=b, se=se, ci_low=b - 1.96 * se, ci_high=b + 1.96 * se, t=b / se, p=p,
                n=int(m.nobs), clusters=extra.pop("clusters", np.nan), sig=stars(p), **extra)


def inventory_tests(S0, windows=(60, 300, 900), horizons=(10, 60, 300)):
    rows = []
    for w in windows:
        S = add_inventory(S0, w)
        g = S.groupby("stock")
        for h in horizons:
            S[f"fut_ret_{h}"] = 1e4 * (g["logmid"].shift(-h) - S["logmid"])
            fut_flow = g["lp_flow"].transform(lambda x: x[::-1].rolling(h, min_periods=h).sum()[::-1].shift(-1))
            S[f"fut_flow_{h}"] = fut_flow / g["lp_flow"].transform(lambda x: x.abs().mean() * h)
            sub = S[(S["sec"] - LOB_START) % h == 0]           # non-overlapping forward windows
            G = sub["block"].nunique()
            specs = (
                ("D1 order-flow persistence: future passive flow on I_w", f"fut_flow_{h}", ["inv_z"],
                 "descriptive (inventory unwinding would give < 0)"),
                ("D2a future return on I_w | past return", f"fut_ret_{h}", ["inv_z", "past_ret_bps"],
                 "descriptive"),
                ("D2b future return on I_w | past return, future flow, vol, spread, depth", f"fut_ret_{h}",
                 ["inv_z", "past_ret_bps", f"fut_flow_{h}", "rvol60", "spread_bps", "log_depth1"],
                 "> 0 if inventory-driven reversal; 0 if information"),
            )
            for label, y, x, pred in specs:
                # stock FE only: fixed effects for short time cells (e.g. 30-min blocks) bias regressions of a
                # forward window sum on a backward window sum (see fe_placebo); clustering stays by block
                m = ols_cluster(sub, y, x, fe="stock", cluster="block")
                rows.append(_row(label, m, "inv_z", window_s=w, horizon_s=h, prediction=pred, clusters=G))
    return pd.DataFrame(rows)


def fe_placebo(S0, cases=((60, 10), (300, 60), (900, 10), (900, 60), (900, 300)), seed=0):
    """D1 on real vs randomly permuted (within stock) order flow, with block FE vs stock FE.
    A non-zero placebo coefficient means the specification itself manufactures the result."""
    rng = np.random.default_rng(seed)
    P = S0.copy()
    for _, g in P.groupby("stock"):
        perm = rng.permutation(len(g))
        P.loc[g.index, ["buy_vol", "sell_vol"]] = g[["buy_vol", "sell_vol"]].to_numpy()[perm]
    rows = []
    for w, h in cases:
        for data_lab, data in (("real flow", S0), ("shuffled flow (placebo)", P)):
            S = add_inventory(data, w)
            g = S.groupby("stock")
            ff = g["lp_flow"].transform(lambda x: x[::-1].rolling(h, min_periods=h).sum()[::-1].shift(-1))
            S["ff"] = ff / g["lp_flow"].transform(lambda x: x.abs().mean() * h)
            sub = S[(S["sec"] - LOB_START) % h == 0]
            for fe in ("block", "stock"):
                m = ols_cluster(sub, "ff", ["inv_z"], fe=fe, cluster="block")
                rows.append(_row("D1 future passive flow on I_w", m, "inv_z", window_s=w, horizon_s=h, data=data_lab,
                                 fixed_effects="stock x 30 min" if fe == "block" else "stock",
                                 clusters=sub["block"].nunique()))
    return pd.DataFrame(rows)


def spread_inventory_tests(S0, windows=(60, 300, 900)):
    """D3: spread vs |I_w|, contemporaneous and 10 s ahead, with progressively richer controls."""
    rows = []
    for w in windows:
        S = add_inventory(S0, w)
        S["spread_fwd10"] = S.groupby("stock")["spread_bps"].shift(-10)
        sub = S[(S["sec"] - LOB_START) % 10 == 0]
        G = sub["block"].nunique()
        variants = (
            ("no controls", []),
            ("+ volatility", ["rvol60"]),
            ("+ volatility, depth", ["rvol60", "log_depth1"]),
            ("+ volatility, depth, trading volume, |past return|", ["rvol60", "log_depth1", "log_vol60", "abs_past_ret_bps"]),
        )
        for y, timing in (("spread_bps", "same second"), ("spread_fwd10", "10 s later")):
            for lab, ctrl in variants:
                m = ols_cluster(sub, y, ["abs_inv_z", *ctrl], fe="block", cluster="block")
                rows.append(_row("D3 spread (bps) on |I_w| (SD units)", m, "abs_inv_z", window_s=w, spread_timing=timing,
                                 controls=lab, prediction="> 0 (Amihud-Mendelson)", clusters=G))
    return pd.DataFrame(rows)


def competition_tests(S0):
    """D4: order-book competition proxies. Neither is a count of competing dealers."""
    S = add_inventory(S0, 300)
    S["spread_fwd10"] = S.groupby("stock")["spread_bps"].shift(-10)
    sub = S[(S["sec"] - LOB_START) % 10 == 0]
    rows = []
    G = sub["block"].nunique()
    for lab, ctrl in (("+ volatility", ["rvol60"]), ("+ volatility, depth", ["rvol60", "log_depth1"])):
        m = ols_cluster(sub, "spread_bps", ["log_n_best", *ctrl], fe="block", cluster="block")
        rows.append(_row("D4a spread on log(# orders at best quotes)", m, "log_n_best", sample="all stocks",
                         controls=lab, clusters=G))
    for stk, g in sub.groupby("stock"):
        for lab, gg in (("all seconds", g), ("spread > 1 tick", g[g["spread_ticks"] > 1])):
            if len(gg) < 200:
                continue
            m = ols_cluster(gg, "spread_bps", ["log_n_best", "rvol60", "log_depth1"], fe="block", cluster="block")
            rows.append(_row("D4a spread on log(# orders at best quotes)", m, "log_n_best", sample=f"{stk}, {lab}",
                             controls="+ volatility, depth", clusters=gg["block"].nunique()))
    # price improvements in the last 60 s -> spread 10 s later, holding today's spread fixed
    m = ols_cluster(sub, "spread_fwd10", ["log_improve60", "spread_bps", "rvol60", "log_depth1"], fe="block", cluster="block")
    rows.append(_row("D4b spread 10 s later on log(1 + inside-spread quotes, last 60 s)", m, "log_improve60",
                     sample="all stocks", controls="current spread, volatility, depth", clusters=G))
    by = S.groupby("stock").agg(orders_at_best_bid=("n_bid_best", "mean"), orders_at_best_ask=("n_ask_best", "mean"),
                                spread_bps=("spread_bps", "mean"), share_one_tick=("spread_ticks", lambda x: (x == 1).mean()),
                                improvements_per_min=("n_improve", lambda x: 60 * x.mean()),
                                best_depth=("depth1", "mean")).reset_index()
    return pd.DataFrame(rows), by


def price_response(trades, horizons=(1, 5, 10, 30, 60, 120, 300)):
    """Average signed mid-price move after a market order, SEs clustered by stock x 30 min."""
    rows = []
    for grp, g in trades.groupby("shock"):
        for h in horizons:
            r = cluster_mean(g[f"impact_{h}s_bps"], g["block"])
            rows.append(dict(period="shock" if grp == 1 else "normal", horizon_s=h, impact_bps=r["mean"],
                             se=r["se"], ci_low=r["mean"] - 1.96 * r["se"], ci_high=r["mean"] + 1.96 * r["se"],
                             n=r["n"], clusters=r["clusters"]))
    return pd.DataFrame(rows)


def price_reversal_test(trades):
    rows = []
    for per, g in trades.groupby("shock"):
        for a, b in ((10, 120), (10, 300), (60, 300)):
            r = cluster_mean(g[f"impact_{b}s_bps"] - g[f"impact_{a}s_bps"], g["block"])
            base = g[f"impact_{a}s_bps"].mean()
            rows.append(dict(period="shock" if per == 1 else "normal", from_s=a, to_s=b, change_bps=r["mean"],
                             se=r["se"], ci_low=r["mean"] - 1.96 * r["se"], ci_high=r["mean"] + 1.96 * r["se"],
                             t=r["t"], p=r["p"], impact_at_from_bps=base, share_reversed=-r["mean"] / base,
                             n=r["n"], clusters=r["clusters"]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# D5 Huang & Stoll (1997) decomposition - exploratory
# ---------------------------------------------------------------------------------------------
def _hs_fit(dM, Q1, Q2, S1, S2, pi, blocks):
    X = pd.DataFrame({"X1": S1 / 2 * Q1, "X2": -(1 - 2 * pi) * S2 / 2 * Q2})
    m = sm.OLS(dM, X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(blocks)[0]})
    ab, a = m.params["X1"], m.params["X2"]
    V = m.cov_params()
    se_inv = np.sqrt(V.loc["X1", "X1"] + V.loc["X2", "X2"] - 2 * V.loc["X1", "X2"])
    return dict(adverse_selection=a, se_as=m.bse["X2"], inventory=ab - a, se_inv=se_inv, order_processing=1 - ab)


def huang_stoll(trades):
    """Huang & Stoll (1997), trade-flow model (two-step version):
        dM_t = (alpha+beta) S/2 Q_{t-1} - alpha (1-2 pi) S/2 Q_{t-2} + e_t,
    M_t = mid just before trade t, Q = +/-1 trade sign, pi = P(Q_t != Q_{t-1}).
    HS estimate pi jointly by GMM and use the *traded* spread S (a constant). Variants here:
      A 'quoted S_{t-1}' : time-varying quoted spread before each trade (original stage-2 version)
      B 'traded S'       : constant S = 2 x mean effective half-spread of the stock (closer to HS)
      C as B, visible-only trades (hidden executions removed)
    Sanity check: in the basic HS model dP_t = (S/2) Q_t + (alpha+beta-1)(S/2) Q_{t-1} + e_t, the
    coefficient on Q_t is the traded half-spread; compare it with the measured effective half-spread.
    The 'shock' subsample is selected on one-sided order flow (low pi), which violates the model's
    assumption of a stable trade-sign process, so normal-vs-shock comparisons are exploratory only.
    """
    rows, checks = [], []
    for variant in ("A quoted S_t-1", "B traded S", "C traded S, visible trades"):
        T = trades if not variant.startswith("C") else trades[trades["hidden"] == 0]
        for stk, g in T.sort_values(["stock", "time"]).groupby("stock"):
            g = g.reset_index(drop=True)
            Q = g["d"].to_numpy().astype(float)
            M = g["mid_pre"].to_numpy()
            P = g["price"].to_numpy()
            eff_half = np.nanmean(Q * (P - M))                     # $ traded half-spread
            Sq = g["spread_pre"].to_numpy()
            S = Sq if variant.startswith("A") else np.full(len(g), 2 * eff_half)
            dM = np.r_[np.nan, np.diff(M)]
            Q1, Q2 = np.r_[np.nan, Q[:-1]], np.r_[np.nan, np.nan, Q[:-2]]
            S1, S2 = np.r_[np.nan, S[:-1]], np.r_[np.nan, np.nan, S[:-2]]
            shock_prev = np.r_[0, g["shock"].to_numpy()[:-1]]
            for period, mask in (("all", np.ones(len(g), bool)), ("normal", shock_prev == 0), ("shock", shock_prev == 1)):
                sel = mask & np.isfinite(dM) & np.isfinite(Q2) & np.isfinite(S2)
                if sel.sum() < 200:
                    continue
                pi = np.mean(Q1[sel] != Q2[sel])
                r = _hs_fit(dM[sel], Q1[sel], Q2[sel], S1[sel], S2[sel], pi, g["block"].to_numpy()[sel])
                rows.append(dict(variant=variant, stock=stk, period=period, n=int(sel.sum()), pi_reversal=pi, **r))
            if variant.startswith("B"):
                dP = np.r_[np.nan, np.diff(P)]
                d = pd.DataFrame({"dP": dP, "Q": Q, "Q1": Q1}).dropna()
                mb = sm.OLS(d["dP"], d[["Q", "Q1"]]).fit()
                checks.append(dict(stock=stk, implied_half_spread_cents=100 * mb.params["Q"],
                                   effective_half_spread_cents=100 * eff_half,
                                   quoted_half_spread_cents=100 * np.nanmean(Sq) / 2,
                                   implied_alpha_plus_beta=1 + mb.params["Q1"] / mb.params["Q"]))
    return pd.DataFrame(rows), pd.DataFrame(checks)


# ---------------------------------------------------------------------------------------------
# D6 liquidity supply by age of the executed resting order
# ---------------------------------------------------------------------------------------------
def provider_types(execs, panel, events, W, K1, horizons=(5, 60)):
    """Visible executions only. Age = execution time - submission time of the order's current ID.
    Unmatched executions (order submitted before 09:30) are a separate 'pre-open (untraced)' class."""
    E = execs.copy()
    E["d"] = -E["dir"]
    out = []
    for tk, g in E.groupby("stock"):
        msg, book = load_raw(tk)
        t = msg["time"].to_numpy()
        mid = ((book["ap1"] + book["bp1"]) / 2).to_numpy()
        j0 = np.searchsorted(t, g["time"].to_numpy(), side="left") - 1
        g = g.assign(mid_pre=mid[np.clip(j0, 0, None)])
        for h in horizons:
            j = np.searchsorted(t, g["time"].to_numpy() + h, side="right") - 1
            g[f"mid_{h}s"] = mid[j]
        out.append(g)
    E = pd.concat(out, ignore_index=True)
    E = E[(E["time"] >= LOB_START) & (E["time"] <= LOB_END - max(horizons))].copy()
    for h in horizons:
        E[f"realized_{h}s_bps"] = 1e4 * E["d"] * (E["price"] - E[f"mid_{h}s"]) / E["mid_pre"]
        E[f"impact_{h}s_bps"] = 1e4 * E["d"] * (E[f"mid_{h}s"] - E["mid_pre"]) / E["mid_pre"]
    E["shock"] = 0
    E["near"] = False
    times = panel["time"].to_numpy()
    for _, r in events.iterrows():
        m_s = E["stock"] == r["stock"]
        near = m_s & (E["time"] >= times[r["idx"] - 50]) & (E["time"] <= times[r["idx"] + 100])
        E.loc[near, "near"] = True
        ins = m_s & (E["time"] >= times[r["idx"] - W]) & (E["time"] <= times[r["idx"] + K1]) & (E["d"] == r["s"])
        E.loc[ins, "shock"] = 1
    E = E[(E["shock"] == 1) | (~E["near"])].copy()
    E["period"] = np.where(E["shock"] == 1, "shock", "normal")
    E["block"] = E["stock"] + "_" + ((E["time"] - LOB_START) // 1800).astype(int).astype(str)
    E["log_size"] = np.log(E["size"])

    def classify(age, bins, labels):
        c = pd.cut(age, bins, labels=labels).astype(object)
        return np.where(age.isna(), "pre-open (untraced)", c)

    E["age_class"] = classify(E["age"], AGE_BINS, AGE_LABELS)
    # matching quality
    match = (E.assign(matched=E["age"].notna())
             .groupby(["stock", "period"])
             .apply(lambda x: pd.Series({"executions": len(x), "matched_share": x["matched"].mean(),
                                         "matched_volume_share": np.average(x["matched"], weights=x["size"])}),
                    include_groups=False).reset_index())
    # volume shares by order age. Computed PER STOCK and then averaged with equal stock weights: pooled share
    # volume would be dominated by INTC/MSFT (low price, huge share counts). Threshold and untraced sensitivity.
    share_rows = []
    for thr in (0.1, 1.0, 10.0):
        cls = classify(E["age"], [-np.inf, thr, 60.0, np.inf], [f"young (<{thr:g}s)", f"middle ({thr:g}-60s)", "old (>60s)"])
        for incl in (True, False):
            sub = E.assign(c=cls)
            if not incl:
                sub = sub[sub["age"].notna()]
            sh = (sub.groupby(["stock", "period", "c"])["size"].sum()
                  / sub.groupby(["stock", "period"])["size"].sum()).rename("share").reset_index()
            sh = sh.pivot_table(index=["period", "c"], columns="stock", values="share", fill_value=0.0)
            for (per, c), r in sh.iterrows():
                share_rows.append(dict(young_threshold_s=thr, untraced_included=incl, period=per, age_class=c,
                                       volume_share_equal_weight=r.mean(),
                                       **{f"share_{k}": v for k, v in r.items()}))
    shares = pd.DataFrame(share_rows)

    def vw(x):
        return pd.Series({"executions": len(x), **{c: np.average(x[c], weights=x["size"]) for c in
                          [f"realized_{h}s_bps" for h in horizons] + [f"impact_{h}s_bps" for h in horizons]}})
    perf_s = E.groupby(["stock", "age_class", "period"]).apply(vw, include_groups=False).reset_index()
    perf = (perf_s.groupby(["age_class", "period"])
            .agg(stocks=("stock", "nunique"), executions=("executions", "sum"),
                 **{c: (c, "mean") for c in perf_s.columns if c.startswith(("realized", "impact"))}).reset_index())
    E["young"] = (E["age_class"] == AGE_LABELS[0]).astype(int)
    E["shock_x_young"] = E["shock"] * E["young"]
    regs = []
    for h in horizons:
        m = ols_cluster(E, f"realized_{h}s_bps", ["shock", "young", "shock_x_young", "log_size"], fe="stock", cluster="block")
        for v in ("shock", "young", "shock_x_young"):
            regs.append(_row(f"realized_{h}s_bps", m, v, term=v, clusters=E["block"].nunique()))
    # does the young-order share fall in shocks? cluster-robust test on execution-level volume shares
    E["young_vol"] = E["young"] * E["size"]
    blk = E.groupby(["block", "period"]).agg(young=("young_vol", "sum"), vol=("size", "sum")).reset_index()
    blk["share"] = blk["young"] / blk["vol"]
    piv = blk.pivot(index="block", columns="period", values="share").dropna()
    diff = cluster_mean(piv["shock"] - piv["normal"], piv.index)
    regs.append(dict(test="young-order volume share, shock minus normal (paired by block)", beta=diff["mean"],
                     se=diff["se"], ci_low=diff["mean"] - 1.96 * diff["se"], ci_high=diff["mean"] + 1.96 * diff["se"],
                     t=diff["t"], p=diff["p"], n=diff["n"], clusters=diff["clusters"], sig=stars(diff["p"]), term="share"))
    return match, shares, perf, pd.DataFrame(regs), perf_s
