"""Hypothesis tests H1-H4, robustness grid and figures.

Timeline of one event (steps of 10 book events):

   pre window          shock window          provision window        recovery window
 [-50 ....... -W)      [-W ......... 0]      (0 ......... K1]        (K1 ......... K2]
 baseline state        flow is extreme       liquidity supply LP     outcomes (prices, spread, depth)

LP and outcomes never overlap, so a link between them is not built in mechanically.
"""
import numpy as np
import pandas as pd
from scipy import stats

from .events import abnormal_paths, detect_shocks, match_controls, paths
from .stats import cluster_mean, holm, ols_cluster, path_ci, stars

TAUS = np.arange(-50, 301)
K1, K2 = 10, 100


def col(tau):
    return int(tau - TAUS[0])


# ---------------------------------------------------------------------------------------------
# event study
# ---------------------------------------------------------------------------------------------
def run_event_study(df, flow_col="ofi1n", W=5, q=0.01, min_gap=100, k=5, ofi_cum="cum_ofi1n"):
    ev, S = detect_shocks(df, flow_col, W=W, q=q, min_gap=min_gap, pre=50, post=int(TAUS[-1]))
    res = {"ev_all": ev, "S": S, "W": W}
    for mode in ("pre", "state"):
        ctrl = match_controls(df, ev, S, k=k, mode=mode)
        e, pe, pc, ab = abnormal_paths(df, ev, ctrl, TAUS, ofi_col=ofi_cum)
        res[mode] = dict(ev=e, ev_path=pe, ctrl_path=pc, abn=ab, ctrl_idx=ctrl)
    return res


def did(res, mode, var, tau, ref):
    """Difference-in-differences per event: (event[tau]-event[ref]) - (control[tau]-control[ref])."""
    r = res[mode]
    a = r["ev_path"][var][:, col(tau)] - r["ev_path"][var][:, col(ref)]
    b = r["ctrl_path"][var][:, col(tau)] - r["ctrl_path"][var][:, col(ref)]
    return a - b


def pre_mean(P, var):
    return np.nanmean(P[var][:, col(-50):col(-6)], axis=1)


def event_level(res, cluster="seg", k1=None, k2=None):
    """One row per event: shock characteristics, liquidity provision (LP) and recovery outcomes."""
    K1 = k1 or globals()["K1"]
    K2 = k2 or globals()["K2"]
    W = res["W"]
    r = res["pre"]
    e = r["ev"].copy()
    P, C = r["ev_path"], r["ctrl_path"]
    e["cluster"] = e[cluster]
    # --- pre-shock state (controls in regressions)
    e["pre_spread_bps"] = pre_mean(P, "rel_spread_bps")
    e["pre_depth"] = pre_mean(P, "dep_total")
    e["log_shock"] = np.log(e["shock_size"])
    e["buy"] = (e["s"] == 1).astype(int)
    # --- the shock itself (H1): change from tau=-W to tau=0, net of the matched control change
    e["impact_bps"] = P["mid_bps"][:, col(0)]
    for v in ("rel_spread_bps", "dep_press", "dep1_press", "dep10_press", "dep_total", "imb_press"):
        e[f"h1_{v}"] = did(res, "pre", v, 0, -W)
    # quote improvements on the hit side in (0, K1]: someone steps in with a better price
    Ps, Cs = res["state"]["ev_path"], res["state"]["ctrl_path"]
    e["lp_improve"] = P["improve_press"][:, col(1):col(K1) + 1].sum(axis=1)
    e["lp_improve_did_state"] = (Ps["improve_press"][:, col(1):col(K1) + 1].sum(1)
                                 - Cs["improve_press"][:, col(1):col(K1) + 1].sum(1))
    # --- liquidity provision in (0, K1]
    e["lp_flow"] = -(P["cofi"][:, col(K1)] - P["cofi"][:, col(0)])               # contrarian OFI
    e["lp_flow_abn"] = e["lp_flow"] + (C["cofi"][:, col(K1)] - C["cofi"][:, col(0)])
    e["lp_refill"] = P["dep_press"][:, col(K1)] - P["dep_press"][:, col(0)]       # hit side refilled
    e["lp_refill_did_state"] = did(res, "state", "dep_press", K1, 0)
    e["lp_flow_did_state"] = -did(res, "state", "cofi", K1, 0)
    if "net_add_press" in P:
        e["lp_msg"] = P["net_add_press"][:, col(1):col(K1) + 1].sum(axis=1)       # LOBSTER messages
        S_ = res["state"]
        e["lp_msg_did_state"] = (S_["ev_path"]["net_add_press"][:, col(1):col(K1) + 1].sum(1)
                                 - S_["ctrl_path"]["net_add_press"][:, col(1):col(K1) + 1].sum(1))
    # --- recovery in (K1, K2]
    e["reversal_bps"] = -(P["mid_bps"][:, col(K2)] - P["mid_bps"][:, col(K1)])
    e["reversal_abn_bps"] = e["reversal_bps"] + (C["mid_bps"][:, col(K2)] - C["mid_bps"][:, col(K1)])
    e["post_drift_bps"] = P["mid_bps"][:, col(K2)] - P["mid_bps"][:, col(0)]      # continuation after shock
    e["post_drift_abn_bps"] = did(res, "pre", "mid_bps", K2, 0)
    e["post_drift300_abn_bps"] = did(res, "pre", "mid_bps", 300, 0)
    e["reversal_100_300_bps"] = -did(res, "pre", "mid_bps", 300, 100)  # >0 = prices move back between steps 100 and 300
    e["spread_gap_K2"] = P["rel_spread_bps"][:, col(K2)] - e["pre_spread_bps"]
    e["depth_gap_K2"] = P["dep_press"][:, col(K2)] - pre_mean(P, "dep_press")
    # time until spread back at/below and hit-side depth back at/above pre-shock level (censored)
    sp_ok = P["rel_spread_bps"][:, col(1):] <= e["pre_spread_bps"].to_numpy()[:, None] + 1e-9
    dp_ok = P["dep_press"][:, col(1):] >= pre_mean(P, "dep_press")[:, None]
    both = sp_ok & dp_ok
    first = np.where(both.any(1), both.argmax(1) + 1, TAUS[-1] + 1)
    e["t_recover"] = first
    e["recovered"] = both.any(1).astype(int)
    return e


# ---------------------------------------------------------------------------------------------
# hypothesis tables
# ---------------------------------------------------------------------------------------------
def hypothesis_table(e, res, dataset):
    rows = []

    def add(h, label, x, note="", role="primary"):
        r = cluster_mean(x, e["cluster"])
        crit = stats.t.ppf(0.975, max(r["clusters"] - 1, 1))
        rows.append(dict(dataset=dataset, hypothesis=h, role=role, measure=label, mean=r["mean"], se=r["se"],
                         ci_low=r["mean"] - crit * r["se"], ci_high=r["mean"] + crit * r["se"],
                         t=r["t"], p=r["p"], n=r["n"], clusters=r["clusters"], expected=note))

    add("H1", "Δ relative spread (bps), shock vs control", e["h1_rel_spread_bps"], "> 0")
    add("H1", "Δ depth on hit side, 3 lvls (× median)", e["h1_dep_press"], "< 0", "secondary")
    add("H1", "Δ depth on hit side, best level (× mean)", e["h1_dep1_press"], "< 0")
    add("H1", "Δ depth on hit side, 10 lvls (× median)", e["h1_dep10_press"], "< 0", "secondary")
    add("H1", "Δ book imbalance toward pressure", e["h1_imb_press"], "> 0")
    add("H2", "Contrarian order flow (0,K1] vs same-state control", e["lp_flow_did_state"], "> 0")
    add("H2", "Hit-side refill (0,K1] vs same-state control", e["lp_refill_did_state"], "> 0")
    add("H2", "Hit-side quote improvements (0,K1] vs control (count)", e["lp_improve_did_state"], "> 0")
    if "lp_msg_did_state" in e:
        add("H2", "Net limit orders added on hit side (0,K1] vs control", e["lp_msg_did_state"], "> 0", "secondary")
    add("H3/H4", "Price drift after shock (0,K2] vs control, bps", e["post_drift_abn_bps"], "< 0 if prices revert")
    add("H3/H4", "Price drift after shock (0,300] vs control, bps", e["post_drift300_abn_bps"], "< 0 if prices revert", "secondary")
    add("H3/H4", "Late reversal (100,300] vs control, bps", e["reversal_100_300_bps"], "> 0 if prices revert", "secondary")
    t = pd.DataFrame(rows)
    # Holm correction across ALL rows of this dataset's table (conservative family)
    t["p_holm"] = holm(t["p"].to_numpy())
    t["sig_holm"] = t["p_holm"].map(stars)
    return t


REG_CONTROLS = ["log_shock", "rvol_pre", "pre_spread_bps", "pre_depth", "tod", "buy"]


def h3_regressions(e, df, res, fe, cluster, lp_vars=("lp_flow", "lp_refill")):
    """Recovery_i = b0 + b1 LP_i + controls + FE ; LP standardised (per 1 SD)."""
    e = e.copy()
    W = res["W"]
    e["rvol_pre"] = df["rvol20"].to_numpy()[e["idx"].to_numpy() - W]
    out = []
    models = {}
    for lp in lp_vars:
        if lp not in e:
            continue
        e[f"{lp}_z"] = (e[lp] - e[lp].mean()) / e[lp].std()
        for y in ("reversal_bps", "spread_gap_K2", "depth_gap_K2", "t_recover"):
            m = ols_cluster(e, y, [f"{lp}_z", *REG_CONTROLS], fe=fe, cluster=cluster)
            models[(lp, y)] = m
            b, se, p = m.params[f"{lp}_z"], m.bse[f"{lp}_z"], m.pvalues[f"{lp}_z"]
            out.append(dict(LP=lp, outcome=y, beta_per_sd=b, se=se, t=b / se, p=p,
                            n=int(m.nobs), r2=m.rsquared, sig=stars(p)))
    return pd.DataFrame(out), models, e


def tercile_paths(e, res, var, lp="lp_flow"):
    """Abnormal (control-adjusted) path of `var` for weak / middle / strong LP terciles."""
    ab = res["pre"]["abn"][var]
    q = pd.qcut(e[lp], 3, labels=["weak LP", "middle", "strong LP"])
    out = {}
    for lab in ["weak LP", "strong LP"]:
        m = (q == lab).to_numpy()
        out[lab] = path_ci(ab[m], e["cluster"].to_numpy()[m])
    return out


# ---------------------------------------------------------------------------------------------
# H4 with LOBSTER trades
# ---------------------------------------------------------------------------------------------
def tag_trades(trades, panel, e, W, drop_near=True):
    """Mark trades that hit liquidity providers during a shock: inside [tau=-W, tau=K1] of an event
    and in the direction of the shock. 'normal' = trades far from any event window.
    drop_near=False keeps every trade (needed when the trade sequence matters, e.g. Huang-Stoll)."""
    tr = trades.copy()
    tr["shock"] = 0
    tr["near_event"] = False
    tr["severity"] = np.nan
    times = panel["time"].to_numpy()
    for _, r in e.iterrows():
        t0, t1 = times[r["idx"] - W], times[r["idx"] + K1]
        t_far0, t_far1 = times[r["idx"] - 50], times[r["idx"] + K2]
        m_stock = tr["stock"] == r["stock"]
        near = m_stock & (tr["time"] >= t_far0) & (tr["time"] <= t_far1)
        tr.loc[near, "near_event"] = True
        inside = m_stock & (tr["time"] >= t0) & (tr["time"] <= t1) & (tr["d"] == r["s"])
        tr.loc[inside, "shock"] = 1
        tr.loc[inside, "severity"] = r["log_shock"]
    if drop_near:
        tr = tr[(tr["shock"] == 1) | (~tr["near_event"])].copy()
    tr["log_size"] = np.log(tr["size"])
    return tr


def h4_table(tr, horizons=(1, 5, 30, 60), sample="all signed trades"):
    """Shock-trade minus normal-trade difference in effective spread, price impact and realised spread
    (bps), from OLS with stock fixed effects and log trade size; SEs clustered by stock x 30 min.
    Holm adjustment across the rows of this table."""
    rows = []
    G = tr["block"].nunique()
    for h in (0, *horizons):
        for y in ((f"impact_{h}s_bps", f"realized_{h}s_bps") if h else ("eff_hs_bps",)):
            m = ols_cluster(tr, y, ["shock", "log_size"], fe="stock", cluster="block")
            b, se, p = m.params["shock"], m.bse["shock"], m.pvalues["shock"]
            rows.append(dict(sample=sample, horizon_s=h,
                             measure="effective half-spread" if h == 0 else y.split("_")[0],
                             normal_mean_bps=tr.loc[tr["shock"] == 0, y].mean(),
                             shock_minus_normal_bps=b, se=se, ci_low=b - 1.96 * se, ci_high=b + 1.96 * se,
                             t=b / se, p=p, n_shock=int(tr["shock"].sum()), n_normal=int((tr["shock"] == 0).sum()),
                             clusters=G))
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t["p"].to_numpy())
    t["sig_holm"] = t["p_holm"].map(stars)
    return t


# ---------------------------------------------------------------------------------------------
# robustness grid
# ---------------------------------------------------------------------------------------------
def robustness(df, dataset, cluster, fe, specs):
    rows = []
    for sp in specs:
        sp = dict(sp)
        k1, k2 = sp.pop("k1", None), sp.pop("k2", None)
        res = run_event_study(df, **sp)
        e = event_level(res, cluster=cluster, k1=k1, k2=k2)
        sp.update(k1=k1 or K1, k2=k2 or K2)
        if len(e) < 30:
            continue
        reg, _, _ = h3_regressions(e, df, res, fe=fe, cluster="cluster", lp_vars=("lp_flow",))
        rv = reg.set_index("outcome").loc["reversal_bps"]
        f = lambda x: cluster_mean(x, e["cluster"])
        rows.append(dict(
            dataset=dataset, signal=sp.get("flow_col"), W=sp.get("W"), q=sp.get("q"), K1=sp["k1"], K2=sp["k2"],
            events=len(e),
            impact_bps=f(e["impact_bps"])["mean"],
            H1_spread=f(e["h1_rel_spread_bps"])["mean"], H1_spread_t=f(e["h1_rel_spread_bps"])["t"],
            H1_depth=f(e["h1_dep1_press"])["mean"], H1_depth_t=f(e["h1_dep1_press"])["t"],
            H1_depth10=f(e["h1_dep10_press"])["mean"], H1_depth10_t=f(e["h1_dep10_press"])["t"],
            H2_improve=f(e["lp_improve_did_state"])["mean"], H2_improve_t=f(e["lp_improve_did_state"])["t"],
            H2_flow=f(e["lp_flow_did_state"])["mean"], H2_flow_t=f(e["lp_flow_did_state"])["t"],
            H2_refill=f(e["lp_refill_did_state"])["mean"], H2_refill_t=f(e["lp_refill_did_state"])["t"],
            drift_bps=f(e["post_drift_abn_bps"])["mean"], drift_t=f(e["post_drift_abn_bps"])["t"],
            H3_beta=rv["beta_per_sd"], H3_t=rv["t"],
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------------
COLORS = {"event": "#2a6fdb", "control": "#9aa3ad", "abn": "#d1495b", "weak": "#e09f3e", "strong": "#2a9d8f"}


def plot_event_study(res, dataset, variables, W, cluster="seg", path=None):
    import matplotlib.pyplot as plt
    r = res["pre"]
    g = r["ev"][cluster].to_numpy()
    n = len(variables)
    fig, axes = plt.subplots(int(np.ceil(n / 3)), 3, figsize=(15, 3.6 * int(np.ceil(n / 3))), squeeze=False)
    for ax, (v, title) in zip(axes.flat, variables):
        m, lo, hi = path_ci(r["ev_path"][v], g)
        mc, _, _ = path_ci(r["ctrl_path"][v], g)
        ax.fill_between(TAUS, lo, hi, color=COLORS["event"], alpha=0.18, lw=0)
        ax.plot(TAUS, m, color=COLORS["event"], lw=1.8, label="shock events")
        ax.plot(TAUS, mc, color=COLORS["control"], lw=1.6, ls="--", label="matched controls")
        ax.axvspan(-W, 0, color="#f2c14e", alpha=0.18, lw=0)
        ax.axvline(K1, color="k", lw=0.6, ls=":")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("steps relative to shock end (1 step = 10 book events)")
        ax.grid(alpha=0.25)
    for ax in axes.flat[n:]:
        ax.axis("off")
    axes.flat[0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"{dataset}: event study around order-flow shocks (sign-adjusted, 95% CI)", fontsize=13)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130, bbox_inches="tight")
    return fig


def plot_terciles(e, res, dataset, variables, path=None):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(variables), figsize=(5 * len(variables), 3.6), squeeze=False)
    for ax, (v, title) in zip(axes.flat, variables):
        tp = tercile_paths(e, res, v)
        for lab, c in (("weak LP", COLORS["weak"]), ("strong LP", COLORS["strong"])):
            m, lo, hi = tp[lab]
            ax.fill_between(TAUS, lo, hi, color=c, alpha=0.18, lw=0)
            ax.plot(TAUS, m, color=c, lw=1.8, label=lab)
        ax.axvspan(0, K1, color="#bbb", alpha=0.2, lw=0)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("steps relative to shock end")
        ax.grid(alpha=0.25)
    axes.flat[0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"{dataset}: control-adjusted paths, strongest vs weakest contrarian provision (tercile of LP in (0,{K1}])", fontsize=12)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------------------------
# shock types: where do prices revert, and where does provision matter?
# ---------------------------------------------------------------------------------------------
def shock_types(e, df, res, fe, dataset):
    """Split shocks by ex-ante characteristics and re-estimate the key quantities in each group.
      * severity tercile (size of the flow shock);
      * LOBSTER only: trade-driven vs quote-driven (share of the shock window's market-order volume,
        above vs below the median), i.e. aggressive liquidity demand vs cancellations / new quotes.
    """
    e = e.copy()
    groups = {"severity": pd.qcut(e["shock_size"], 3, labels=["mild", "medium", "severe"]).astype(str)}
    P = res["pre"]["ev_path"]
    if "mo_press" in P:
        # aggressive (market-order) volume in the shock direction during the shock window, relative to the
        # stock's mean per-step volume. Known at tau=0 (ex ante for post-shock outcomes). It measures how much
        # of the shock came with aggressive trading; it does NOT identify information vs liquidity motives.
        W = res["W"]
        mo = P["mo_press"][:, col(-W + 1):col(0) + 1].sum(axis=1)
        groups["aggressive volume"] = np.where(mo >= np.median(mo), "high aggressive volume", "low aggressive volume")
    rows = []
    for gname, lab in groups.items():
        e["_g"] = lab
        for val, g in e.groupby("_g"):
            f = lambda x: cluster_mean(x, g["cluster"])
            reg, _, _ = h3_regressions(g, df, res, fe=fe, cluster="cluster", lp_vars=("lp_refill", "lp_flow"))
            rv = reg.set_index(["LP", "outcome"])
            rows.append(dict(dataset=dataset, split=gname, group=val, events=len(g),
                             impact_bps=f(g["impact_bps"])["mean"],
                             drift_abn_bps=f(g["post_drift_abn_bps"])["mean"], drift_t=f(g["post_drift_abn_bps"])["t"],
                             reversal_bps=f(g["reversal_abn_bps"])["mean"], reversal_t=f(g["reversal_abn_bps"])["t"],
                             H2_refill_t=f(g["lp_refill_did_state"])["t"],
                             H3_refill_beta=rv.loc[("lp_refill", "reversal_bps"), "beta_per_sd"],
                             H3_refill_t=rv.loc[("lp_refill", "reversal_bps"), "t"],
                             H3_flow_beta=rv.loc[("lp_flow", "reversal_bps"), "beta_per_sd"],
                             H3_flow_t=rv.loc[("lp_flow", "reversal_bps"), "t"]))
    return pd.DataFrame(rows)


def shock_type_diagnostics(e, res):
    """How the ex-ante split variables relate to the shock-defining OFI (LOBSTER and FI-2010)."""
    P = res["pre"]["ev_path"]
    W = res["W"]
    out = dict(events=len(e))
    if "mo_press" in P:
        mo = P["mo_press"][:, col(-W + 1):col(0) + 1].sum(axis=1)
        out["corr(aggressive volume, |OFI shock|)"] = np.corrcoef(mo, e["shock_size"])[0, 1]
        out["corr(aggressive volume, price impact)"] = np.corrcoef(mo, e["impact_bps"])[0, 1]
        hi = mo >= np.median(mo)
        out["mean |OFI| high-volume group"] = e["shock_size"][hi].mean()
        out["mean |OFI| low-volume group"] = e["shock_size"][~hi].mean()
    out["corr(|OFI shock|, price impact)"] = np.corrcoef(e["shock_size"], e["impact_bps"])[0, 1]
    return pd.DataFrame([out])


def cluster_sensitivity(e, measures, cluster_sets):
    """Same event-level means under alternative clustering schemes (t uses G-1 df)."""
    rows = []
    for label, col_ in measures:
        for cname, groups in cluster_sets.items():
            r = cluster_mean(e[col_], groups)
            crit = stats.t.ppf(0.975, max(r["clusters"] - 1, 1))
            rows.append(dict(measure=label, clustering=cname, clusters=r["clusters"], mean=r["mean"], se=r["se"],
                             ci_low=r["mean"] - crit * r["se"], ci_high=r["mean"] + crit * r["se"], t=r["t"], p=r["p"]))
    return pd.DataFrame(rows)


def event_clock_response(df, res, horizons=(5, 10, 30, 60, 120, 300, 600)):
    """LOBSTER: control-adjusted, sign-adjusted mid-price move from tau=-W to tau=0 + h seconds (clock time),
    for shock events and their pre-matched controls. Reconciles the event-clock study with trade-level
    clock-time price responses."""
    W = res["W"]
    r = res["pre"]
    e = r["ev"]
    ctrl = r["ctrl_idx"][(r["ctrl_idx"] >= 0).all(axis=1)] if len(r["ctrl_idx"]) != len(e) else r["ctrl_idx"]
    times, lm, stock = df["time"].to_numpy(), df["logmid"].to_numpy(), df["stock"].to_numpy()
    bounds = df.groupby("stock").indices

    def move(idx, h):
        st = stock[idx]
        out = np.full(len(idx), np.nan)
        for s_ in np.unique(st):
            rows = bounds[s_]
            m = st == s_
            tt = times[rows]
            j = np.searchsorted(tt, times[idx[m]] + h, side="right") - 1
            ok = times[idx[m]] + h <= tt[-1]
            v = (lm[rows[j]] - lm[idx[m] - W]) * 1e4
            out[np.flatnonzero(m)] = np.where(ok, v, np.nan)
        return out

    idx = e["idx"].to_numpy()
    s = e["s"].to_numpy()
    rows = []
    for h in (0, *horizons):
        ev_m = s * move(idx, h)
        k = ctrl.shape[1]
        c_m = s * np.nanmean(move(ctrl.ravel(), h).reshape(len(idx), k) * 1.0, axis=1)
        ab = ev_m - c_m
        cm = cluster_mean(ab, e["block"])
        rows.append(dict(horizon_s=h, event_bps=np.nanmean(ev_m), control_bps=np.nanmean(c_m),
                         abnormal_bps=cm["mean"], se=cm["se"], ci_low=cm["mean"] - 1.96 * cm["se"],
                         ci_high=cm["mean"] + 1.96 * cm["se"], n=cm["n"], clusters=cm["clusters"]))
    out = pd.DataFrame(rows)
    peak = out.loc[out["abnormal_bps"].idxmax()]
    out["share_of_peak_remaining"] = out["abnormal_bps"] / peak["abnormal_bps"]
    return out
