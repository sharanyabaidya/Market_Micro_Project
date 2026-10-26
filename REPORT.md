# The Role of Dealer-Like Liquidity Providers in Absorbing Temporary Liquidity Shocks

*Final empirical report. Every number is produced by `scripts/run_all.py` (tables in `results/tables/`, figures
in `results/figures/`). The audit trail is in `VALIDATION_LOG.md`, and the narrative with all tables is in
`notebooks/analysis.ipynb`.*

---

## 1. Research question

When order flow in a limit order book becomes extremely one-sided:

1. Does liquidity deteriorate?
2. Do liquidity providers lean against the flow, as dealers are supposed to?
3. Does stronger provision bring faster recovery of prices and liquidity?
4. What adverse-selection cost do providers bear?

A limit order book has no designated dealers, and neither dataset identifies traders. The project therefore
studies **dealer-like behaviour**: what passive liquidity supply does around extreme order-flow events. It does
not observe any dealer's actions.

## 2. Theoretical framework (instructor-suggested papers)

| Paper | Mechanism | Empirical prediction |
|---|---|---|
| Amihud & Mendelson (1980) | A monopolist dealer sets quotes as a function of inventory; quotes fall when inventory is long; the spread widens as inventory leaves its target | Temporary price pressure that reverts as inventory unwinds; spread rises with inventory imbalance |
| Ho & Stoll (1983) | Competing dealers shade quotes according to inventory; the best quote comes from the dealer with the most extreme position; competition narrows the spread | Narrower spreads with more competing dealers; quote shading after trades |
| Biais & Weill (2009) | After a liquidity shock, traders with limited market access re-enter gradually; the book thins and then refills | Liquidity drained by the shock and recovering gradually; price pressure recovering as the book refills |
| Biais, Declerck & Moinas (2016) | Liquidity is supplied by different trader types (fast/proprietary vs slow); fast suppliers reduce provision when flow is one-sided | The composition of liquidity supply shifts under stress; those who absorb stress bear losses |

## 3. Data

| | FI-2010 (Ntakaris et al. 2018) | LOBSTER free sample files |
|---|---|---|
| Market | Nasdaq Helsinki: KESBV, OUT1V, SAMPO, RTRKS, WRT1V | NASDAQ: AAPL, AMZN, GOOG, INTC, MSFT |
| Period | 10 trading days, June 2010 (10:30–18:00) | 21 June 2012 (09:45–15:45 used) |
| Content | 10-level book, one snapshot per 10 events, 394,337 snapshots; no timestamps, no trades | 10-level book and every message: submissions, cancels, executions, order IDs, trade direction |
| Unit of analysis | 50 stock-days | 5 stock-days; 60 stock × 30-minute blocks |

**Data preparation** (all verified, see `VALIDATION_LOG.md`):
* FI-2010 files concatenate stocks and days. They are split exactly into 50 stock-days by detecting snapshots
  where no (side, price, size) pair survives.
* LOBSTER visible executions are signed exactly. Hidden executions are signed with the quote rule; those exactly
  at the mid are dropped.

## 4. Methodology

**Primary specification.**
* *Shock:* order-flow imbalance (Cont, Kukanov & Stoikov 2014), summed over 5 steps, falls in the stock's 1% tail.
  One step = 10 book events; in LOBSTER, 100 steps are 32–161 s depending on the stock.
* *Declustering:* events at least 100 steps apart, and every window inside one stock-day.
  This gives 1,269 FI-2010 and 657 LOBSTER events.
* *Matched controls:* 5 calm moments from the same stock-day, matched on spread, 1- and 3-level depth,
  imbalance, volatility and time of day. Two matches:
  * on the state **before** the shock, to measure the shock's effect;
  * on the **depleted** book after it, so that mechanical refilling is not counted as contrarian provision.
* *Windows:* shock [−5, 0]; provision (0, 10]; recovery (10, 100]; long horizon (0, 300].
  Provision and recovery windows do not overlap.
* *Estimates:* event minus control, as differences-in-differences.

**Adverse selection (LOBSTER, trade level).** For every market order, the effective half-spread, the price impact
and the realised half-spread at 1–300 s are computed. *Shock trades* are those in the shock direction during
[−5, 10]; *normal trades* are those far from any event.

**Inference.**
* SEs are clustered by stock-day (FI-2010, 50 clusters) or stock × 30 min (LOBSTER, 60 clusters), with t(G − 1)
  confidence intervals.
* p-values are Holm-adjusted across all rows of each hypothesis table; statements of significance below use the
  **Holm-adjusted** p unless marked "unadjusted".
* Because only five stocks underlie each dataset, every key estimate is also shown with day, market-wide time-block
  and stock-level clustering (table 05), with per-stock signs (table 04), and across a 24-specification
  robustness grid per dataset (table 09).

## 5. Results

### 5.1 Primary findings

**(a) Liquidity deteriorates on the side being hit (H1).**

| | FI-2010 | LOBSTER |
|---|---|---|
| Best-level depth on hit side (× mean) | −0.56 [−0.66, −0.47] | 0.00 [−0.24, 0.23] |
| 3-level / 10-level depth | −0.11 / 0.00 | −0.17 / −0.08 [−0.14, −0.03] |
| Book imbalance toward pressure | +0.15 [0.12, 0.19] | +0.12 [0.07, 0.17] |
| Relative spread (bps) | +0.39 [−0.13, 0.92], n.s. | +0.75 [0.44, 1.07], p_holm < 0.001 |

The FI-2010 depth result has the same sign in 5/5 stocks and survives stock-level clustering. The LOBSTER spread
effect is positive in 3/5 stocks and positive in 21/24 robustness specifications, but not significant with
stock-level clustering (p = 0.16).

**(b) No contrarian provision beyond normal book dynamics; if anything, less (H2).** Compared with calm moments
that have the *same depleted book*:
* order flow over the next 10 steps keeps going in the shock direction:
  * FI-2010: −0.56 [−0.87, −0.24], p_holm = 0.006;
  * LOBSTER: −1.00 [−1.45, −0.55], p_holm < 0.001;
* the hit side does not refill faster (FI-2010 +0.02, LOBSTER +0.05, both n.s.);
* LOBSTER shows *fewer* price-improving quotes on the hit side (−0.21 per window, p_holm = 0.023).

The contrarian-flow result is negative in 24/24 grid specifications in both datasets and in 4/5 (FI-2010) and
5/5 (LOBSTER) stocks. With only 5 stock clusters, p = 0.05 (FI-2010) and 0.11 (LOBSTER).

**(c) Price changes after shocks largely persist.**
* *Event clock:* control-adjusted drift over (0, 100] steps is +5.0 bps [3.3, 6.6] in FI-2010 and +1.3 bps
  [0.8, 1.7] in LOBSTER, positive in every stock, and robust to stock-level clustering. Over (100, 300] there is
  no significant reversal (FI-2010 +1.6 [−0.6, 3.8]; LOBSTER +0.1).
* *Clock time (LOBSTER, fig. 2b):* the control-adjusted price move after a shock peaks at 3.5 bps around 2
  minutes, and **85% of that peak remains after 10 minutes**.
* *Contrast with ordinary trades (fig. 2a):* for normal market orders, about **one third of the 60-second impact
  reverts by 5 minutes** (−0.52 bps, unadjusted p = 0.006). For shock trades the decline from 10 s to 300 s is
  36% but not significant (p = 0.06).

So the dealer-model prediction of reversal holds partially for ordinary trades but not for extreme order-flow
episodes.

**(d) Providers who absorb shocks are adversely selected (H4, LOBSTER).** Shock trades vs normal trades
(2,688 vs 21,896 market orders):

| Horizon | Price impact | Realised spread |
|---|---|---|
| 5 s | +1.23 bps [0.80, 1.66] | −1.14 bps [−1.51, −0.78] (p_holm < 0.001) |
| 30 s | — | −1.03 bps (p_holm = 0.012) |
| 60 s | — | −0.83 bps (p_holm = 0.057) |

The effective half-spread is not significantly higher: +0.09 bps [−0.03, 0.21]. Liquidity providers on the other
side of shock trades lose about 1 bp per trade relative to normal times and are not compensated through wider
spreads. Results are unchanged on visible-only trades.

### 5.2 Secondary findings

* **H3: provision and recovery.** No provision measure predicts price reversal or spread/depth recovery after
  Holm correction in either dataset:
  * best FI-2010 estimate: hit-side refill → +1.06 bps reversal per SD, unadjusted p = 0.08;
  * in LOBSTER, more refill goes with *less* reversal (−0.42 bps, unadjusted p = 0.02, p_holm = 0.22);
  * more limit-order supply shortens the time until the book is back to normal (−18.8 steps per SD,
    p_holm = 0.003), but that measure overlaps the provision window and is descriptive;
  * in the grid, the FI-2010 coefficient is positive in 19/24 specifications and significant in 7 (strongest with
    a 20-step provision window).

  **Mixed / specification-dependent.**
* **Shock severity (FI-2010 proxy for H4):** post-shock drift is positive in all severity quintiles; no monotone pattern.
* **Depth-imbalance definition used in the earlier notebook (table 09b):** its events show a price move *against*
  the supposed pressure (−1.2 bps in FI-2010), confirming that depth imbalance does not identify liquidity shocks.

### 5.3 Exploratory analyses (not used for the main conclusions)

**Shock types (table 10).** Splits are fixed at τ = 0, so they are ex ante:
* *Aggressive volume (LOBSTER).* Shocks with high aggressive (market-order) volume show persistent drift
  (+2.2 bps, t = 6.0); shocks with low aggressive volume show none (+0.3 bps, n.s.). The split correlates
  *negatively* with OFI size (−0.13), so it describes how the shock was built (trades vs quote changes). It does
  **not** separate information from liquidity motives.
* *Severity (FI-2010).* In mild shocks, contrarian flow predicts reversal (β = 2.8 bps per SD, unadjusted
  t = 2.6); in severe shocks it does not. This is suggestive only, given the number of splits.

**Dealer-theory proxies (LOBSTER, tables 11–15).**

| Test | Proxy | Result | Assessment |
|---|---|---|---|
| D1 inventory unwinding | I_w = −(aggregate signed market-order flow over w s) | +0.10 [0.05, 0.16] for 60 s windows at 10 s; ≈ 0 otherwise | Measures **order-flow persistence**, not unwinding. Block fixed effects produce spurious negative coefficients (placebo, table 11b), so stock fixed effects are used. |
| D2b inventory-driven reversal | future return on I_w given future flow, volatility, spread, depth | −0.04 [−0.09, 0.01] at 10 s; positive but n.s. at 60–300 s (best unadjusted p = 0.06) | **Inconclusive** |
| D3 spread vs imbalance (AM) | spread on \|I_w\| | Sign depends on controls; with volume and \|return\| controls ≈ 0 (e.g. +0.02 [−0.03, 0.07]) | **Not supported** |
| D4 competition (HS) | # orders at best quotes; recent inside-spread quotes | Order count: +0.58 bps per log unit (wider, not narrower), only in small-tick stocks; ≈ 0 in INTC/MSFT, which are at one tick 99% of the time. Price improvements: −0.05 [−0.12, 0.01] | **Not testable.** Order counts are not dealer counts, and queue length is confounded by tick size. |
| D5 Huang–Stoll decomposition | trade-flow model with reversal probability | Fails sanity check (implied traded half-spread is 55–65% of measured); negative adverse-selection shares for INTC/MSFT in all variants | **Descriptive only.** Not a valid decomposition here. |
| D6 who supplies liquidity (BDM) | age of executed resting order since its current ID was submitted | Young (<1 s) share of passive volume: 30% → 17% in shocks (paired block difference −0.105 [−0.148, −0.061]); old and pre-open orders: 16% → 33%; every age class earns ≈ 1.1–1.5 bps less in shocks; no differential penalty for young orders | **Consistent with BDM.** Order age is not trader identity. |

## 6. Hypothesis classification

| Hypothesis | Classification | Basis |
|---|---|---|
| **H1** shocks deteriorate liquidity | **Supported** for hit-side depth (FI-2010) and imbalance (both); **mixed** for spreads | Spread widening clear in LOBSTER under the primary clustering but not robust to stock-level clustering; absent in FI-2010 |
| **H2** contrarian liquidity provision | **Contradicted** | Less contrarian flow than in matched depleted-book moments; no faster refill; fewer price improvements (LOBSTER). Robust in the grid and in per-stock signs; borderline with only 5 stock clusters |
| **H3** stronger provision → faster stabilisation | **Mixed / specification-dependent** | No coefficient survives multiple-testing correction; some support in FI-2010 for mild shocks and longer provision windows |
| **H4** provision under stress is adversely selected | **Supported** (LOBSTER) | Higher price impact and lower realised spreads for shock trades, not offset by wider effective spreads |

| Paper | Proxy used here | Verdict | Principal limitation |
|---|---|---|---|
| Amihud–Mendelson | Aggregate passive position (minus signed market-order flow); price path after trades and shocks; spread vs \|position\| | **Not directly testable.** Proxies show partial reversal for ordinary trades but persistent prices after shocks, and no spread–imbalance link | No individual inventories; the aggregate position is identical to lagged order flow |
| Ho–Stoll | Orders at best quote; inside-spread quote submissions | **Not directly testable** | Orders ≠ dealers; tick size dominates queue length |
| Biais–Weill | Event study of depth, spread and prices around OFI shocks | **Partly supported.** Liquidity is drained and refills within tens of steps; prices do not recover | Shocks defined from microstructure data; one LOBSTER day |
| Biais–Declerck–Moinas | Executed-order age (young vs old); realised spreads by age | **Consistent.** Fast-quote supply falls under stress; older orders absorb more and lose | Order age is a proxy; repricing restarts an order's age; untraced pre-open orders |

## 7. Robustness

The grid (table 09, 24 specifications per dataset) varies:
* the shock signal: OFI at level 1 or over 3 levels;
* the shock window W ∈ {3, 5, 10} and tail q ∈ {0.5, 1, 2.5, 5}%;
* the provision window K1 ∈ {5, 10, 20} and recovery horizon K2 ∈ {50, 100, 300}.

| Quantity | FI-2010 | LOBSTER |
|---|---|---|
| Price impact positive | 24/24 | 24/24 |
| Contrarian flow negative (significant) | 24/24 (20) | 24/24 (20) |
| Post-shock drift positive (significant) | 24/24 (24) | 22/24 (21) |
| Spread effect positive (significant) | 22/24 (9) | 21/24 (21) |
| Hit-side quote improvements negative | 20/24 (2 specs significantly positive) | 24/24 |
| H3 reversal coefficient positive (significant) | 19/24 (7) | 17/24 (3) |

Stable: price impact, contrarian-flow deficit and post-shock drift. Specification-dependent: spread widening in
FI-2010, quote improvements in FI-2010, and H3.

## 8. Limitations

1. **No trader identities.** "Dealer", "inventory" and "provider type" are behavioural proxies; order age restarts
   when an order is repriced.
2. **Few independent units.** Five stocks per dataset; LOBSTER covers a single day. Stock-level inference has
   4 degrees of freedom, and some H2 and spread results are borderline at that level.
3. **FI-2010 lacks timestamps and trades.** Horizons are in book events, and H4 can only be tested in LOBSTER.
4. **Shock identification.** Shocks are defined from microstructure variables and mix information and liquidity
   motives; the exploratory splits do not resolve this.
5. **Huang–Stoll.** The decomposition was built for dealer markets and fails basic checks in these data.

## 9. Conclusion

Around extreme order-flow events, passive liquidity in these limit order books does **not** behave like the
inventory-managing dealer of Amihud–Mendelson or Ho–Stoll:
* order flow persists rather than being leaned against;
* the hit side refills only at its normal pace, with fewer price improvements than in calm moments;
* the resulting price change is largely permanent over the following minutes;
* those who absorb shock trades are adversely selected and not compensated by wider spreads.

Liquidity itself recovers quickly, as in Biais–Weill. The composition of supply shifts away from very young
(fast, re-priced) quotes toward older resting orders, as Biais–Declerck–Moinas would predict. The data establish
how replenishment, order-flow persistence, price adjustment and adverse-selection exposure behave around these
events. They cannot establish what any individual dealer does.

**Highest-value next step:** obtain multi-day LOBSTER data (or any data with participant or account flags) to:
* replace the 5-stock, 1-day inference with day-level clustering;
* test whether *identified* market makers' inventories mean-revert and shade quotes (D1–D3 done properly).
