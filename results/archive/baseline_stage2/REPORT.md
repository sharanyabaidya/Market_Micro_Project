# Results: the role of dealer-like liquidity providers around temporary liquidity shocks

*All numbers come from `scripts/run_all.py` (tables in `results/tables/`, figures in `results/figures/`).
Every analysis is in `notebooks/analysis.ipynb`.*

## Summary

Liquidity providers in a limit order book act like the dealers of Amihud–Mendelson and Ho–Stoll only weakly, and
mostly at longer horizons.

* **H1, liquidity deterioration.** Large order-flow shocks drain liquidity on the side being hit, and spreads widen
  briefly. Liquidity then refills within about 10 steps (100 book events).
* **H2, contrarian provision.** The refill is not contrarian, dealer-like provision:
  * order flow keeps pushing in the shock's direction;
  * there are fewer price-improving quotes on the side being hit than in matched calm periods;
  * fast, short-lived orders supply a *smaller* share of volume during shocks.
* **H4, adverse selection.** Providers who trade against shocks lose money. Their realised spread falls by about
  1.1 bps, and they are not compensated by wider spreads.
* **Inventory model (D-tests).** The inventory story shows up only slowly. About 37% of a trade's price impact
  reverts between 1 and 5 minutes, and the inventory share of the spread rises during shocks in the large-tick
  stocks. The direct inventory tests D1–D3 do not support the inventory model at short horizons.

Overall, the data fit a market where liquidity supply is fast and information-sensitive. Pure inventory
management, the classic dealer role, plays only a secondary and slow part.

## 1. Data and design

| | FI-2010 | LOBSTER |
|---|---|---|
| Market | Nasdaq Helsinki, 5 stocks, 10 days (June 2010) | NASDAQ, AAPL AMZN GOOG INTC MSFT, 1 day (21 June 2012) |
| Content | 10-level book, 1 snapshot per 10 events (394,337) | 10-level book plus every message (orders, cancels, executions, order IDs) |
| Shocks | 1,269 in 50 stock-days | 657 in 60 stock × 30-min blocks |

* **Shock:** order-flow imbalance (OFI; Cont, Kukanov & Stoikov 2014) summed over 5 steps falls in the stock's 1%
  tail. At the shock, market-order volume in the shock direction is about 6–7× normal (LOBSTER), so these are
  episodes of real liquidity demand.
* **Matched controls:** 5 calm moments per shock from the same stock-day.
  * Matched on the book state *before* the shock, to measure its effect.
  * Matched on the *depleted* book after it, so that normal refilling of a thin book is not counted as
    contrarian provision.
* **Windows:** shock [−5, 0]; provision (0, 10]; recovery (10, 100]; long horizon up to 300 steps
  (one step = 10 book events; in LOBSTER, 100 steps ≈ 40–60 s).
* **Inference:** standard errors clustered by stock-day (FI-2010) or stock × 30 min (LOBSTER); Holm-adjusted
  p-values within each hypothesis table.

## 2. Proposal hypotheses H1–H4

| | FI-2010 | LOBSTER | Verdict |
|---|---|---|---|
| **H1 deterioration** | Best-level depth on hit side −0.56× its mean (t = −12.3); 3-level −0.11 (t = −4.2); 10-level ≈ 0; book imbalance toward pressure +0.15 (t = 8.4); spread +0.39 bps (n.s.) | Spread +0.75 bps, about +18% (t = 4.7), back to normal in about 10 steps; 10-level depth −0.08 (t = −3.0); imbalance +0.12 (t = 4.9) | **Supported.** Liquidity is drained close to the best price, not deep in the book. |
| **H2 contrarian provision** | Flow against the shock in (0,10] vs same-state control −0.56 (t = −3.6); refill +0.02 (n.s.); quote improvements −0.07 (t = −2.1) | Contrarian flow −1.00 (t = −4.4); quote improvements on hit side −0.21 per window (t = −3.1); net limit orders added −0.15 (t = −1.8) | **Rejected.** Less contrarian provision than in comparable calm moments, robust across the whole grid. |
| **H3 provision → stabilisation** | 1 SD more hit-side refill → +1.1 bps reversal (p = 0.08); with a 20-step provision window +2.5 bps (t = 4.9) | 1 SD more limit-order supply → book recovers 19 steps faster (t = −3.7, partly mechanical); more refill → narrower spread (−0.15 bps, p = 0.02) but *less* price reversal (−0.42 bps, p = 0.02) | **Mixed.** Provision speeds up liquidity recovery; its link to price recovery appears only in FI-2010 and only for mild shocks. |
| **H4 adverse selection** | Price keeps drifting +5.0 bps in the shock direction over (0,100] (t = 6.1); +3.4 bps at 300 steps (t = 2.8) | Shock trades: 5 s impact +1.24 bps (t = 5.7), realised spread −1.12 bps (t = −6.0), still significant at 60 s; effective spread only +0.12 bps | **Supported.** Providers lose on shock trades and are barely compensated. |

**Shock types** (table 10):
* **Trade-driven shocks** (mostly market orders) are informational: drift +2.1 bps (t = 5.7) and no reversal.
* **Quote-driven shocks** (mostly cancellations and new quotes) leave no significant lasting price effect
  (+0.4 bps, t = 1.3). These are the "temporary liquidity shocks" the proposal had in mind.
* **In FI-2010, contrarian flow predicts price reversal only for mild shocks** (β = 2.8 bps per SD, t = 2.6).
  For severe shocks it has no effect (t = −0.1).

## 3. Direct tests of dealer theory (LOBSTER)

All passive liquidity providers together are treated as "the dealer". Their inventory change is the mirror image of
market-order flow.

| Test | Theory predicts | Result | Verdict |
|---|---|---|---|
| **D1** inventory mean reversion | Future provider flow opposite to inventory (β < 0) | β = +0.10 (t = 3.6) at 10 s for 60 s inventory; ≈ 0 at longer windows and horizons | **Not supported.** Market-order flow is persistent (order splitting), so providers keep accumulating in the short run. |
| **D2** price reversal from inventory | Holding the past return fixed, prices move back toward the dealer (β > 0) | β = −0.14 bps (t = −4.3) at 10 s and −0.41 bps (t = −2.4) at 60 s for 60 s inventory; n.s. elsewhere | **Not supported at short horizons.** Short-run continuation looks like information. |
| **Price response after a trade** | Inventory: impact partly reverts; information: permanent | Impact peaks at 10–60 s, then falls back about 37% by 300 s (normal t = −3.0; shock trades −38%, t = −2.1) | **Partly supported at 1–5 minutes.** A slow temporary component exists. |
| **D3** spread vs \|inventory\| | Wider spread when inventory is far from target (β > 0) | β = −0.07 bps (t = −2.8) for 60 s inventory; n.s. for 300/900 s | **Not supported.** |
| **D4** competition | More competing quotes → narrower spread (β < 0) | Positive in AAPL, AMZN, GOOG (t = 3.7–9.3); ≈ 0 in INTC and MSFT | **Not testable with order counts.** In small-tick stocks, competition shows up as price improvement (a new level with one order), not as a longer queue. Number of orders ≠ number of dealers. |
| **D5** Huang–Stoll decomposition | Inventory share > 0; adverse-selection share rises in stress | Inventory share 0.18–0.31 in normal times; in shocks it rises to 0.42 for INTC and MSFT (vs 0.18 and 0.29, se ≈ 0.05); order processing ≈ 0.8 | **Inventory component present and stronger in stress in large-tick stocks.** The model fits poorly overall (negative adverse-selection estimates), so read the comparisons, not the levels. |
| **D6** who supplies liquidity | BDM: fast traders supply liquidity but pull back under stress | Volume share of young orders (<1 s) falls from 22% to 16% in shocks; all age groups lose about 1.1 bps more in shocks; young orders earn +0.26 bps more in general (t = 3.6), with no different shock penalty | **Supported.** Fast liquidity supply retreats in stress, leaving slower orders to absorb the shock. |

## 4. How the results relate to the papers

* **Amihud & Mendelson (1980); Ho & Stoll (1983).** The inventory mechanism exists, but it is slow and is
  outweighed by information in the first minute:
  * D2 and the event studies show continuation, not reversal, up to about 1 minute.
  * D1 shows no short-run unwinding.
  * D3 shows no spread widening with inventory.
  * Only at 1–5 minutes does part of the price impact revert (about 37%), and the Huang–Stoll inventory share
    rises during shocks in INTC and MSFT.

  The classic dealer role is therefore present but secondary in a modern limit order book.
* **Ho & Stoll on competition.** Competition cannot be measured by queue length. The positive spread–queue relation
  in small-tick stocks is a tick-size effect: competition happens through price improvement.
* **Biais & Weill (2009).** Fits the liquidity part well: after a shock the hit side is drained, spreads spike, and
  the book refills gradually, faster when more limit orders arrive. The price part fits only for quote-driven and
  mild shocks. Trade-driven and severe shocks move prices permanently.
* **Biais, Declerck & Moinas (2016).** Consistent with their finding that fast traders supply liquidity in normal
  times but reduce it when order flow is one-sided. In stress, slower and older orders carry a larger share and
  lose money.

## 5. Robustness (`results/tables/08_robustness.csv`, 48 specifications, 24 per dataset)

The grid covers:
* two shock signals: OFI at level 1 or over 3 levels;
* W ∈ {3, 5, 10} and tail q ∈ {0.5%, 1%, 2.5%, 5%};
* provision window K1 ∈ {5, 10, 20} and recovery horizon K2 ∈ {50, 100, 300};
* depth over 1, 3 and 10 levels.

Findings:
* **Price impact** is positive everywhere.
* **H2 contrarian flow** is negative in every specification.
* **Hit-side quote improvements** are negative in all LOBSTER specifications. In FI-2010 they are negative or n.s.
  in most specifications, but positive in two wide-tail ones (5% tails with W = 3 or with 3-level OFI).
* **Post-shock drift** is positive in almost every specification.
* **The H1 spread result** is strong in LOBSTER for level-1 OFI.
* **H3 in FI-2010** strengthens with a longer provision window (K1 = 20) and with milder shocks.

**The earlier notebook's definition** (10-level depth imbalance, 5% tails; table 09) picks events where the price
moves slightly *against* the supposed pressure, so it does not capture liquidity shocks.

## 6. Limitations

* **Dealers are not identified.** The "dealer" is the aggregate of passive providers. Order age is a proxy for
  trader speed, not trader identity.
* **Event clock.** FI-2010 has no timestamps, so horizons are in book events. LOBSTER clock-time results cover only
  one day and five stocks.
* **Huang–Stoll fit.** The decomposition was built for dealer markets and fits limit order books poorly.
* **Shock type.** Shocks are defined from microstructure data, and part of them carry information. The split into
  trade-driven and quote-driven shocks is a first step toward separating the two.

## 7. Next steps

1. More LOBSTER days via institutional access: day fixed effects, intraday seasonality, market-wide shocks (closer
   to the aggregate shock in Biais–Weill).
2. A competition measure based on price improvement rather than queue length: the number of distinct price levels
   created inside the spread per minute.
3. Track individual liquidity-providing orders through their life around shocks (submitted → executed or
   cancelled), using the order-ID replay already implemented in `src/mm/dealer.py`.
