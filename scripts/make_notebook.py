"""Build notebooks/analysis.ipynb: the narrative over the FINAL verified outputs of scripts/run_all.py.

The notebook re-runs only the cheap data checks (stock-day segmentation, trade signing) and reads every
table and figure produced by the pipeline, so its numbers cannot drift from the report.
    .venv/bin/python scripts/run_all.py
    .venv/bin/python scripts/make_notebook.py
    .venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
C = []


def md(s):
    C.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    C.append(nbf.v4.new_code_cell(s.strip()))


md("""
# Dealer-like liquidity provision around extreme order-flow shocks

**Question.** When order flow becomes extremely one-sided, who supplies liquidity, how fast does the book
recover, and do prices revert, as dealer inventory models predict, or stay put, as adverse selection predicts?

This notebook is the narrative companion to `REPORT.md`. All estimates come from `scripts/run_all.py`
(tables in `results/tables/`, figures in `results/figures/`).

| Role | Meaning |
|---|---|
| **primary** | the tests the conclusions rest on (declared specification below) |
| **secondary** | supporting measures, longer horizons, robustness grid |
| **exploratory** | dealer-theory proxies D1–D6 and shock-type splits |
""")

code("""
import sys, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from IPython.display import display, Image
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 70)
from mm.config import FIGS, TABLES, RAW_FI
from mm.fi2010 import read_raw_book, surviving_pairs

def T(name, cols=None, rnd=3):
    d = pd.read_csv(TABLES / f"{name}.csv")
    return (d[cols] if cols else d).round(rnd)

def F(name):
    display(Image(filename=str(FIGS / name)))
""")

md("""
## 1. Data and validation checks

| | FI-2010 | LOBSTER (free sample) |
|---|---|---|
| Market | Nasdaq Helsinki, 5 stocks × 10 days (June 2010) | NASDAQ: AAPL, AMZN, GOOG, INTC, MSFT, 21 June 2012 |
| Content | 10-level book, one snapshot per 10 events | 10-level book plus every message (orders, cancels, executions, order IDs) |

**Check 1: FI-2010 must be split into stock-days.** The files stack stocks and days. A day boundary is a
snapshot where *no* (side, price, size) pair survives from the previous one.
""")
code("""
B = read_raw_book(RAW_FI / "Train_Dst_NoAuction_DecPre_CF_7.txt")
surv = surviving_pairs(B)
print("histogram of surviving pairs (0..20):", np.bincount(surv, minlength=21))
print("boundaries found:", int((surv <= 2).sum()), " expected 5 stocks x 7 days - 1 = 34")
del B
""")
md("""
Exactly 34 breaks, and no within-day step has fewer than 5 surviving pairs, so the split is exact (50 stock-days in total).

**Check 2: LOBSTER trade signs.**
* Visible executions are signed by the executed order's side; they match the price side of the mid 100% of the time.
* Hidden executions' direction field is unreliable: in INTC and MSFT ~70% print at the mid and the field agrees
  with the price side only ~15% of the time. They are therefore signed by the quote rule, and those exactly at
  the mid are dropped (see `VALIDATION_LOG.md`).
""")
code("""
display(T("01_descriptives"))
""")

md("""
## 2. Primary design

* **Shock:** order-flow imbalance (Cont, Kukanov & Stoikov 2014) summed over 5 steps is in the stock's 1% tail.
  Events are ≥ 100 steps apart and windows stay inside a stock-day. One step = 10 book events, which is
  32–161 s per 100 steps in LOBSTER, depending on the stock.
* **Controls:** 5 calm moments from the same stock-day, matched on spread, depth, imbalance, volatility and time
  of day. Two kinds:
  * matched on the state *before* the shock, to measure its effect;
  * matched on the *depleted* book after it, so that mechanical refilling is not counted as provision.
* **Windows:** shock [−5, 0], provision (0, 10], recovery (10, 100]; long horizon 300 steps.
* **Inference:** SEs clustered by stock-day (FI-2010, 50 clusters) or stock × 30 min (LOBSTER, 60 clusters).
  CIs use t with G − 1 df. `p_holm` = Holm adjustment across all rows of a table.
""")
code('F("fig1_event_study_FI-2010.png")')
code('F("fig1_event_study_LOBSTER.png")')

md("## 3. Hypotheses H1–H4 (primary)")
code("""
cols = ["role", "hypothesis", "measure", "mean", "ci_low", "ci_high", "p", "p_holm", "n", "clusters"]
display(T("02_hypotheses_FI-2010", cols)); display(T("02_hypotheses_LOBSTER", cols))
""")
md("""
**Sensitivity of inference to the clustering unit.** Only five stocks underlie each dataset, so stock-level
clustering (4 df) is the harshest check. Effects that survive it are the most defensible.
""")
code("""
display(T("05_cluster_sensitivity"))
display(T("04_by_stock_FI-2010")); display(T("04_by_stock_LOBSTER"))
""")

md("""
## 4. H3: provision and recovery (secondary)

Per event: $Recovery_i=\\beta_0+\\beta_1 LP_i+\\gamma' controls_i+FE+\\varepsilon_i$, with LP measured in
(0, 10] and recovery in (10, 100]. LP is standardised, so β is per 1 SD. `t_recover` overlaps the provision
window and is descriptive only.
""")
code("""
display(T("03_h3_regressions_FI-2010")); display(T("03_h3_regressions_LOBSTER"))
""")

md("""
## 5. H4: adverse selection, and how much of the price impact reverts

Shock trades are trades in the shock direction during [−5, 10]; normal trades are trades far from any event.
Differences come from OLS with stock fixed effects and log size. The table repeats the estimates on visible-only trades.
""")
code("""
display(T("06_h4_adverse_selection_LOBSTER", ["sample", "horizon_s", "measure", "normal_mean_bps",
          "shock_minus_normal_bps", "ci_low", "ci_high", "p_holm", "n_shock", "n_normal"]))
""")
code('F("fig2_price_response_LOBSTER.png")')
code("""
display(T("08b_price_reversal_test_trades")); display(T("08c_price_response_events_clock_time"))
""")
md("""
**Two different objects.**
* For the *average market order*, about one third of the 60-second impact reverts by 5 minutes.
* For *extreme order-flow episodes*, the control-adjusted price move peaks at about 3.5 bps after 2 minutes, and
  about 85% of that peak is still present after 10 minutes.

So shocks are followed by largely persistent price changes, and only ordinary trades show partial reversal.
""")

md("## 6. Robustness grid (secondary): 24 specifications per dataset")
code("""
rob = T("09_robustness_grid")
display(rob[rob["primary_spec"]])
summary = []
for ds, g in rob.groupby("dataset"):
    for c, tc in (("impact_bps", None), ("H1_spread", "H1_spread_t"), ("H2_flow", "H2_flow_t"),
                  ("H2_improve", "H2_improve_t"), ("drift_bps", "drift_t"), ("H3_beta", "H3_t")):
        summary.append(dict(dataset=ds, quantity=c, positive=f"{(g[c] > 0).sum()}/{len(g)}",
                            significant_5pct=None if tc is None else f"{(g[tc].abs() > 1.96).sum()}/{len(g)}"))
display(pd.DataFrame(summary))
display(T("09b_depth_imbalance_definition"))
""")

md("""
## 7. Exploratory: shock types

Splits defined at τ = 0 (ex ante for post-shock outcomes):
* severity tercile;
* LOBSTER: high vs low aggressive (market-order) volume during the shock window.

The volume split correlates *negatively* with OFI size, so it describes how the shock was built (trades vs
quote changes). It does not identify information vs liquidity motives.
""")
code("""
display(T("10_shock_types")); display(T("10b_shock_type_diagnostics"))
""")

md("""
## 8. Exploratory: dealer-theory proxies (LOBSTER)

There are no trader IDs, so the "inventory proxy" is minus aggregate signed market-order flow over a rolling
window. It is not any dealer's inventory.

| Test | Measures | Interpretation |
|---|---|---|
| D1 | future passive flow on I_w | order-flow persistence (not unwinding) |
| D2b | future return on I_w, *given future flow* | > 0 would indicate inventory-driven reversal |
| D3 | spread on \\|I_w\\| | Amihud–Mendelson (spread rises with imbalance) |
| D4 | spread on # orders at best / recent price improvements | order-book competition proxies, not dealer counts |
| D5 | Huang–Stoll decomposition | fails its sanity check, so descriptive only |
| D6 | passive volume and realised spreads by order age | order age ≠ trader type |

Fixed-effect choice matters (D1 placebo): block fixed effects manufacture negative coefficients on shuffled data.
""")
code("""
display(T("11b_D1_fixed_effect_placebo", ["window_s", "horizon_s", "data", "fixed_effects", "beta", "t"]))
display(T("11_D1_D2_inventory_proxy", ["test", "window_s", "horizon_s", "beta", "ci_low", "ci_high", "p"]))
display(T("12_D3_spread_vs_imbalance", ["window_s", "spread_timing", "controls", "beta", "ci_low", "ci_high", "p"]))
display(T("13_D4_competition_proxies", ["test", "sample", "controls", "beta", "ci_low", "ci_high", "p"]))
display(T("13b_order_book_by_stock"))
display(T("14b_D5_sanity_check"))
display(T("14_D5_huang_stoll_exploratory"))
""")
code('F("fig3_liquidity_supply_by_order_age_LOBSTER.png")')
code("""
display(T("15_D6_order_id_matching"))
s = T("15b_D6_volume_share_by_order_age")
display(s[s["untraced_included"]])
display(T("15c_D6_performance_by_order_age")); display(T("15d_D6_regressions"))
""")

md("""
## 9. Conclusions

See `REPORT.md` §6 for the hypothesis classification and `VALIDATION_LOG.md` for every audit performed.
""")

nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
(ROOT / "notebooks").mkdir(exist_ok=True)
nbf.write(nb, ROOT / "notebooks" / "analysis.ipynb")
print("written", ROOT / "notebooks" / "analysis.ipynb")
