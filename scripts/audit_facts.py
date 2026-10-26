"""Audit step 1: verify definitions and data facts behind the dealer tests (prints only)."""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from mm.config import LOB_END, LOB_START, LOB_TICKERS
from mm.dealer import build_second_panel
from mm.lobster import build_lobster_panel, build_trades, load_raw

pd.set_option("display.width", 220)

print("=== A1 sign convention: executed sell limit order (dir=-1) should trade at/above mid (buyer-initiated)")
for tk in LOB_TICKERS:
    msg, book = load_raw(tk)
    mid_prev = ((book["ap1"] + book["bp1"]) / 2).shift(1)
    ex = msg["type"] == 4
    d = -msg.loc[ex, "dir"]
    above = np.sign(msg.loc[ex, "price"] - mid_prev[ex])
    print(f"  {tk}: share with sign(price - mid) == trade sign: {(above == d).mean():.4f}  "
          f"(ties {(above == 0).mean():.4f});  hidden execs share of all execs: "
          f"{(msg['type'] == 5).sum() / msg['type'].isin([4, 5]).sum():.3f}")
    # hidden executions: does the direction field agree with price vs mid?
    hx = msg["type"] == 5
    dh = -msg.loc[hx, "dir"]
    ah = np.sign(msg.loc[hx, "price"] - mid_prev[hx])
    print(f"        hidden execs: sign agreement {(ah == dh).mean():.3f}, at mid {(ah == 0).mean():.3f}")

print("\n=== A2 order IDs: re-used IDs, partial fills, unmatched executions inside 09:45-15:45")
for tk in LOB_TICKERS:
    msg, _ = load_raw(tk)
    sub = msg[msg["type"] == 1]
    reused = sub["oid"].duplicated().sum()
    ex = msg[(msg["type"] == 4)]
    win = ex[(ex["time"] >= LOB_START) & (ex["time"] <= LOB_END)]
    known = win["oid"].isin(set(sub["oid"]))
    multi = ex["oid"].value_counts()
    print(f"  {tk}: re-used submit IDs {reused}; execs in window {len(win)}, matched to a submission "
          f"{known.mean():.4f} (by volume {np.average(known, weights=win['size']):.4f}); "
          f"orders executed in >1 piece {(multi > 1).mean():.3f}; "
          f"min submit time of matched {sub['time'].min():.1f}")

print("\n=== A3 seconds spanned by 100 / 300 event-clock steps in LOBSTER (median per stock)")
P = build_lobster_panel()
for tk, g in P.groupby("stock"):
    t = g["time"].to_numpy()
    print(f"  {tk}: 10 steps {np.median(t[10:] - t[:-10]):6.1f}s  100 steps {np.median(t[100:] - t[:-100]):6.1f}s  "
          f"300 steps {np.median(t[300:] - t[:-300]):6.1f}s")

print("\n=== A4 1-second panel: what the inventory proxy is")
S, E = build_second_panel()
S["lp_flow"] = S["sell_vol"] - S["buy_vol"]
for tk, g in S.groupby("stock"):
    f = g["lp_flow"]
    ac = [f.autocorr(l) for l in (1, 5, 10, 60)]
    print(f"  {tk}: autocorr of signed market-order flow at lags 1,5,10,60s: {np.round(ac, 3)}; "
          f"share of seconds with no trade {(g['buy_vol'] + g['sell_vol'] == 0).mean():.2f}")

print("\n=== A5 trade table: hidden share, spread in ticks at trade")
T = build_trades()
T["spread_ticks"] = np.round(T["spread_pre"] / 0.01)
print(T.groupby("stock").agg(trades=("size", "size"), hidden_share=("hidden", "mean"),
                             one_tick_share=("spread_ticks", lambda x: (x == 1).mean()),
                             med_spread_ticks=("spread_ticks", "median")).round(3))
