"""Run the full analysis and write every final table and figure.

Primary specification (declared before the robustness grid):
    shock signal = level-1 OFI, W = 5 steps, q = 1% tails, min gap 100 steps,
    provision window (0, 10], recovery window (10, 100], 5 matched controls.
Roles used in the tables:
    primary      the tests the conclusions rest on
    secondary    supporting measures / longer horizons
    exploratory  dealer-theory proxies (D1-D6) and shock-type splits
Old CSV/PNG outputs are removed first; the stage-2 baseline is kept in results/archive/.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mm import analysis as A
from mm import dealer as D
from mm.config import FIGS, FI_TICK, LOB_TICK, TABLES
from mm.fi2010 import build_fi2010_book
from mm.lobster import build_lobster_panel, build_trades
from mm.measures import add_measures, summary_by_stock

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 70)

for f in list(TABLES.glob("*.csv")) + list(TABLES.glob("*.parquet")) + list(FIGS.glob("*.png")):
    f.unlink()


def save(t, name):
    t.to_csv(TABLES / f"{name}.csv", index=False)
    print(f"\n### {name}\n", t.round(4).to_string(index=False))


EV_VARS = [("mid_bps", "Mid-price move in shock direction (bps)"),
           ("cofi", "Cumulative OFI (× mean best-level depth)"),
           ("rel_spread_bps", "Relative spread (bps)"),
           ("dep1_press", "Best-level depth, side being hit (× mean)"),
           ("dep_press", "3-level depth, side being hit (× median)"),
           ("dep_opp", "3-level depth, opposite side (× median)")]
LOB_EXTRA = [("mo_press", "Market-order volume in shock direction (× mean)"),
             ("net_add_press", "Net limit orders added, side being hit (× depth)"),
             ("net_add_opp", "Net limit orders added, opposite side (× depth)")]

# ---------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------
fi = add_measures(build_fi2010_book(), FI_TICK)
lob = add_measures(build_lobster_panel(cache=False), LOB_TICK)      # rebuilt: corrected trade signing
trades = build_trades(cache=False)
S1, E1 = D.build_second_panel(cache=False)

s1 = summary_by_stock(fi).reset_index(); s1.insert(0, "dataset", "FI-2010")
s2 = summary_by_stock(lob, extra=("buy_vol", "sell_vol")).reset_index(); s2.insert(0, "dataset", "LOBSTER")
save(pd.concat([s1, s2], ignore_index=True), "01_descriptives")

# ---------------------------------------------------------------------------------------------
# primary event studies: H1-H4
# ---------------------------------------------------------------------------------------------
out = {}
for name, df, cl, fe in (("FI-2010", fi, "seg", "seg"), ("LOBSTER", lob, "block", "stock")):
    res = A.run_event_study(df, flow_col="ofi1n", W=5, q=0.01, min_gap=100)
    e = A.event_level(res, cluster=cl)
    out[name] = (df, res, e)
    print(f"\n{name}: {len(e)} events, {e['cluster'].nunique()} clusters")
    save(A.hypothesis_table(e, res, name), f"02_hypotheses_{name}")
    lp_vars = ("lp_flow", "lp_refill", "lp_msg") if name == "LOBSTER" else ("lp_flow", "lp_refill")
    reg, _, _ = A.h3_regressions(e, df, res, fe=fe, cluster="cluster", lp_vars=lp_vars)
    reg.insert(0, "dataset", name)
    reg["p_holm"] = A.holm(reg["p"].to_numpy())
    save(reg, f"03_h3_regressions_{name}")
    rows = []
    for stk, g in e.groupby("stock"):
        rows.append(dict(dataset=name, stock=stk, events=len(g), impact_bps=g["impact_bps"].mean(),
                         h1_spread_bps=g["h1_rel_spread_bps"].mean(), h1_best_depth=g["h1_dep1_press"].mean(),
                         h2_contrarian_flow=g["lp_flow_did_state"].mean(), h2_refill=g["lp_refill_did_state"].mean(),
                         drift_abn_bps=g["post_drift_abn_bps"].mean(), share_recovered=g["recovered"].mean(),
                         median_t_recover=g["t_recover"].median()))
    save(pd.DataFrame(rows), f"04_by_stock_{name}")
    vars_ = EV_VARS + (LOB_EXTRA if name == "LOBSTER" else [])
    A.plot_event_study(res, name, vars_, W=5, cluster=cl, path=FIGS / f"fig1_event_study_{name}.png")
    plt.close("all")
    A.plot_terciles(e, res, name, [("mid_bps", "Abnormal price move (bps)"),
                                   ("rel_spread_bps", "Abnormal relative spread (bps)"),
                                   ("dep_press", "Abnormal hit-side depth")],
                    path=FIGS / f"supp_lp_terciles_{name}.png")
    plt.close("all")
    e.to_parquet(TABLES / f"events_{name}.parquet")

# ---- inference sensitivity -----------------------------------------------------------------
df, res, e = out["FI-2010"]
meas = [("price impact (bps)", "impact_bps"), ("H1 best-level depth", "h1_dep1_press"),
        ("H1 spread (bps)", "h1_rel_spread_bps"), ("H2 contrarian flow", "lp_flow_did_state"),
        ("H2 refill", "lp_refill_did_state"), ("drift (0,100] (bps)", "post_drift_abn_bps")]
cs_fi = A.cluster_sensitivity(e, meas, {"stock-day (50)": e["seg"], "day (10)": e["day"], "stock (5)": e["stock"]})
cs_fi.insert(0, "dataset", "FI-2010")
df, res, e = out["LOBSTER"]
tb = e["block"].str.split("_").str[1]
cs_lob = A.cluster_sensitivity(e, meas, {"stock x 30min (60)": e["block"], "30-min block, all stocks (12)": tb,
                                         "stock (5)": e["stock"]})
cs_lob.insert(0, "dataset", "LOBSTER")
save(pd.concat([cs_fi, cs_lob], ignore_index=True), "05_cluster_sensitivity")

# ---------------------------------------------------------------------------------------------
# H4: adverse selection of shock trades (LOBSTER)
# ---------------------------------------------------------------------------------------------
df, res, e = out["LOBSTER"]
tr = A.tag_trades(trades, df, e, W=5)
h4 = pd.concat([A.h4_table(tr), A.h4_table(tr[tr["hidden"] == 0], sample="visible-only trades")], ignore_index=True)
save(h4, "06_h4_adverse_selection_LOBSTER")
df_fi, res_fi, e_fi = out["FI-2010"]
e_fi = e_fi.assign(severity_q=pd.qcut(e_fi["shock_size"], 5, labels=[1, 2, 3, 4, 5]))
t = e_fi.groupby("severity_q", observed=True).agg(events=("idx", "size"), impact_bps=("impact_bps", "mean"),
                                                  drift_abn_bps=("post_drift_abn_bps", "mean")).reset_index()
save(t, "07_h4_proxy_by_severity_FI-2010")

# ---- price response: trades (clock time) and events (clock time) ---------------------------
pr = D.price_response(tr)
save(pr, "08a_price_response_trades")
save(D.price_reversal_test(tr), "08b_price_reversal_test_trades")
ecr = A.event_clock_response(df, res)
save(ecr, "08c_price_response_events_clock_time")

# ---------------------------------------------------------------------------------------------
# robustness grid (secondary) and earlier depth-imbalance definition
# ---------------------------------------------------------------------------------------------
specs = []
for flow in ("ofi1n", "ofi3n"):
    for W in (3, 5, 10):
        for q in (0.005, 0.01, 0.025, 0.05):
            if flow == "ofi3n" and W != 5:
                continue
            specs.append(dict(flow_col=flow, W=W, q=q, min_gap=100,
                              ofi_cum="cum_ofi1n" if flow == "ofi1n" else "cum_ofi3n"))
for k1 in (5, 10, 20):
    for k2 in (50, 100, 300):
        if (k1, k2) != (10, 100):
            specs.append(dict(flow_col="ofi1n", W=5, q=0.01, min_gap=100, k1=k1, k2=k2))
rob = pd.concat([A.robustness(out["FI-2010"][0], "FI-2010", "seg", "seg", specs),
                 A.robustness(out["LOBSTER"][0], "LOBSTER", "block", "stock", specs)], ignore_index=True)
rob.insert(0, "primary_spec", (rob["signal"] == "ofi1n") & (rob["W"] == 5) & (rob["q"] == 0.01)
           & (rob["K1"] == 10) & (rob["K2"] == 100))
save(rob, "09_robustness_grid")
legacy = []
for name, cl, fe in (("FI-2010", "seg", "seg"), ("LOBSTER", "block", "stock")):
    legacy.append(A.robustness(out[name][0], name, cl, fe, [dict(flow_col="imb10", W=1, q=0.05, min_gap=100)]))
save(pd.concat(legacy, ignore_index=True), "09b_depth_imbalance_definition")

# ---------------------------------------------------------------------------------------------
# exploratory: shock types
# ---------------------------------------------------------------------------------------------
st = pd.concat([A.shock_types(out["FI-2010"][2], out["FI-2010"][0], out["FI-2010"][1], "seg", "FI-2010"),
                A.shock_types(out["LOBSTER"][2], out["LOBSTER"][0], out["LOBSTER"][1], "stock", "LOBSTER")],
               ignore_index=True)
save(st, "10_shock_types")
dg = pd.concat([A.shock_type_diagnostics(out[n][2], out[n][1]).assign(dataset=n) for n in out], ignore_index=True)
save(dg, "10b_shock_type_diagnostics")

# ---------------------------------------------------------------------------------------------
# exploratory: dealer-theory proxies (LOBSTER)
# ---------------------------------------------------------------------------------------------
save(D.inventory_tests(S1), "11_D1_D2_inventory_proxy")
save(D.fe_placebo(S1), "11b_D1_fixed_effect_placebo")
save(D.spread_inventory_tests(S1), "12_D3_spread_vs_imbalance")
comp, comp_by = D.competition_tests(S1)
save(comp, "13_D4_competition_proxies")
save(comp_by, "13b_order_book_by_stock")
tr_all = A.tag_trades(trades, df, e, W=5, drop_near=False)
hs, hs_chk = D.huang_stoll(tr_all)
save(hs, "14_D5_huang_stoll_exploratory")
save(hs_chk, "14b_D5_sanity_check")
match, shares, perf, regs, perf_s = D.provider_types(E1, df, e, W=5, K1=A.K1)
save(match, "15_D6_order_id_matching")
save(shares, "15b_D6_volume_share_by_order_age")
save(perf, "15c_D6_performance_by_order_age")
save(perf_s, "15e_D6_performance_by_order_age_and_stock")
save(regs, "15d_D6_regressions")

# ---------------------------------------------------------------------------------------------
# final figures
# ---------------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
ax = axes[0]
for per, c in (("normal", "#2a6fdb"), ("shock", "#d1495b")):
    g = pr[pr.period == per]
    ax.errorbar(g.horizon_s, g.impact_bps, yerr=[g.impact_bps - g.ci_low, g.ci_high - g.impact_bps],
                color=c, marker="o", capsize=3, label=f"{per} trades")
ax.set_xscale("log"); ax.set_xlabel("seconds after the market order (log)"); ax.set_ylabel("signed mid move (bps)")
ax.set_title("(a) Every market order: price impact over time"); ax.grid(alpha=0.3); ax.legend(frameon=False)
ax = axes[1]
g = ecr[ecr.horizon_s > 0]
ax.errorbar(g.horizon_s, g.abnormal_bps, yerr=[g.abnormal_bps - g.ci_low, g.ci_high - g.abnormal_bps],
            color="#d1495b", marker="o", capsize=3)
ax.axhline(0, color="k", lw=0.6)
ax.set_xscale("log"); ax.set_xlabel("seconds after the shock ends (log)"); ax.set_ylabel("abnormal mid move vs controls (bps)")
ax.set_title("(b) Extreme OFI shocks: control-adjusted price path"); ax.grid(alpha=0.3)
fig.suptitle("LOBSTER: how much of the price impact reverts? (95% CIs clustered by stock × 30 min)", fontsize=12)
fig.tight_layout(); fig.savefig(FIGS / "fig2_price_response_LOBSTER.png", dpi=130); plt.close(fig)

sh = shares[(shares.young_threshold_s == 1.0) & (shares.untraced_included)]
pv = sh.pivot(index="age_class", columns="period", values="volume_share_equal_weight")
order = ["young (<1s)", "middle (1-60s)", "old (>60s)", "pre-open (untraced)"]
pv = pv.reindex([o for o in order if o in pv.index])
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
x = np.arange(len(pv))
for i, (per, c) in enumerate((("normal", "#2a6fdb"), ("shock", "#d1495b"))):
    axes[0].bar(x + (i - 0.5) * 0.38, pv[per], width=0.38, color=c, label=per)
axes[0].set_xticks(x, pv.index, rotation=10); axes[0].set_ylabel("share of passive volume (avg of 5 stocks)")
axes[0].set_title("(a) Who is on the passive side: share by order age"); axes[0].legend(frameon=False)
pf = perf.pivot(index="age_class", columns="period", values="realized_5s_bps").reindex(pv.index)
for i, (per, c) in enumerate((("normal", "#2a6fdb"), ("shock", "#d1495b"))):
    axes[1].bar(x + (i - 0.5) * 0.38, pf[per], width=0.38, color=c, label=per)
axes[1].axhline(0, color="k", lw=0.6)
axes[1].set_xticks(x, pf.index, rotation=10); axes[1].set_ylabel("realised spread, 5 s (bps, avg of 5 stocks)")
axes[1].set_title("(b) What passive orders earn (negative = loss)")
for a in axes:
    a.grid(alpha=0.3, axis="y")
fig.suptitle("LOBSTER: liquidity supply by age of the executed resting order (order age ≠ trader type)", fontsize=12)
fig.tight_layout(); fig.savefig(FIGS / "fig3_liquidity_supply_by_order_age_LOBSTER.png", dpi=130); plt.close(fig)

print("\nDone. Tables in", TABLES, "figures in", FIGS)
