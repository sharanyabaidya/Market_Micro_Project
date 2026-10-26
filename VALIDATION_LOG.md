# Validation log

Each entry records what was checked, the outcome, any change made, and what remains uncertain.
The pre-audit ("stage 2") outputs are preserved in `results/archive/baseline_stage2/` (tables, figures, log,
notebook, report).

Outcomes: **PASS** = no problem found. **FIXED** = problem found and corrected. **LIMITATION** = cannot be fixed
with these data and is disclosed. **RECLASSIFIED** = the result stands, but its interpretation or role changed.

---

## Phase 1 — Reproducibility

| # | Check | Outcome |
|---|---|---|
| 1.1 | `scripts/run_all.py` reruns from source without errors | **PASS** (exit 0) |
| 1.2 | Rerun reproduces the stage-2 outputs | **PASS**: all 22 stage-2 CSV tables identical to the archived ones (numerical tolerance 1e-9) |
| 1.3 | Notebook executes end to end | **PASS** (stage 2: 32 cells, 0 errors; final: 29 cells, 0 errors) |
| 1.4 | Sample sizes in report match outputs | **PASS**: 1,269 FI-2010 events (50 clusters), 657 LOBSTER events (60 clusters) |
| 1.5 | Stale estimates in report | **FIXED**: stage-2 report errors listed below; report fully rewritten from the final tables |

Stage-2 report statements found to be wrong or overstated:

| Statement in stage-2 report | Problem |
|---|---|
| Order-ID match rate "96%" | 87–99% of volume depending on the stock in the analysis window; only 63% of AAPL's shock-period volume |
| "100 steps ≈ 40–60 s" in LOBSTER | Actually 32–161 s (median), depending on the stock |
| "37% of impact reverts" applied to shocks | That was a trade-level figure with unclustered SEs. For shock *episodes*, 85% of the peak persists at 10 minutes |
| Effective spread "+0.12 bps, p = 0.04" for shock trades | After correct signing: +0.09 [−0.03, 0.21], n.s. |
| D1/D3/D5 described as supporting or contradicting dealer theory | Proxies not identified (see Phase 2) |
| "Quote-driven shocks = temporary liquidity shocks" | Not justified (see Phase 3) |

## Phase 2 — Dealer-theory tests

### A. Trade signing and the inventory proxy

| # | Check | Outcome |
|---|---|---|
| A1 | Sign of visible executions (type 4): execution of a sell limit order = buyer-initiated | **PASS**: 99.99–100% of executions print on the predicted side of the prior mid |
| A2 | Sign of hidden executions (type 5) | **FIXED.** The direction field is unreliable. INTC/MSFT: 68–75% print exactly at the mid, and the field agrees with the price side in only 13–16% of cases (AAPL/AMZN/GOOG: 83–89%). Hidden executions (11–34% of all executions) had entered the trade table, the 1-second order-flow panel and the market-volume measures with these signs. They are now signed by the quote rule (`lobster.trade_sign`), and executions exactly at the mid are dropped. A visible-only robustness version of H4 is reported (table 06) and gives the same results. |
| A3 | What the "inventory proxy" is | **RECLASSIFIED.** I_w is a *rolling* w-second sum of (seller-initiated − buyer-initiated) volume, standardised within stock; it is neither cumulative nor reset. By accounting identity it equals minus aggregate signed market-order flow, so it is not any dealer's inventory. |
| A4 | Does the D1 dependent variable measure unwinding? | **RECLASSIFIED.** Future aggregate passive flow on I_w is the autocorrelation of signed order flow, now labelled "order-flow persistence". The positive stage-2 coefficient means persistence, not "failure to unwind". |
| A5 | Fixed-effect bias in D1/D2 | **FIXED.** A placebo with order flow shuffled within each stock (table 11b) shows that stock × 30-min fixed effects manufacture significant negative coefficients when a forward window sum is regressed on a backward window sum (e.g. −0.10, t = −3.5 to −4.0 for 15-min windows). The stage-2 "inventory mean reversion" signs at long windows were partly this artefact. D1/D2 now use stock fixed effects, whose placebo coefficients are ≈ 0 (\|t\| ≤ 1.43). |
| A6 | Controls for contemporaneous flow, volatility, spread and depth | **FIXED.** Added D2b: future return on I_w controlling for past return, *future* order flow, volatility, spread and depth. This is the only version that separates inventory-driven reversal from information. Result: no significant effect at any window or horizon. |
| A7 | Overlapping windows | **PASS / disclosed.** Forward windows are sampled every h seconds, so they do not overlap; backward windows overlap when w > h, which clustering by 30-min block absorbs. |

### B. Spread vs inventory imbalance (D3)

| # | Check | Outcome |
|---|---|---|
| B1 | Units and timing | **PASS.** Spread in bps, \|I_w\| in SD units; contemporaneous and 10 s later both reported |
| B2 | Time-of-day control | **FIXED.** Stage 2 entered time of day as a *linear* bucket number. Now stock × 30-min fixed effects (appropriate here because the outcome is a point-in-time level, not a window sum) |
| B3 | Sensitivity to controls | **RECLASSIFIED.** The coefficient is ≈ 0 with no controls, negative with volatility/depth, and ≈ 0 with volume and \|return\| added. No specification shows the Amihud–Mendelson sign significantly. Reported as **not supported**. |

### C. Competition (D4)

| # | Check | Outcome |
|---|---|---|
| C1 | Interpretation of order counts | **RECLASSIFIED.** Labelled an *order-book competition proxy*; one participant can post many orders |
| C2 | Positive spread–order-count association | **Explained.** Present only in the small-tick stocks (AAPL/AMZN/GOOG). INTC/MSFT sit at one tick 99% of the time and show ≈ 0. Unchanged when restricted to spreads > 1 tick, and with block fixed effects and depth controls. With wide spreads, competition shows up as price improvement (a new level with one order), so queue length is not a competition measure. |
| C3 | Price improvements as an alternative | **Added (D4b).** Inside-spread submissions over the last 60 s vs the spread 10 s later, holding the current spread fixed: −0.05 [−0.12, 0.01], n.s. Endogeneity disclosed |
| C4 | Ho–Stoll classification | **Not directly testable** with these data |

### D. Huang–Stoll decomposition (D5)

| # | Check | Outcome |
|---|---|---|
| D1 | Equation | **PASS.** ΔM_t = (α+β)(S/2)Q_{t−1} − α(1−2π)(S/2)Q_{t−2} + e_t matches the HS (1997) trade-flow model; M is the mid before each trade, and Q is the corrected trade sign |
| D2 | Spread definition | **FIXED / variant added.** Stage 2 used the time-varying *quoted* spread; HS use a constant *traded* spread. Variants: A quoted S_{t−1}; B traded S (2 × mean effective half-spread); C as B on visible trades only |
| D3 | π estimation | Disclosed: two-step (π from sample sign reversals) rather than joint GMM |
| D4 | Sanity check | **FAILED.** The basic HS model's implied traded half-spread is only 55–65% of the measured effective half-spread (table 14b). Adverse-selection shares are negative for INTC/MSFT in all three variants, which violates α ≥ 0 |
| D5 | Shock subsample | Selected on one-sided order flow (π ≈ 0.06–0.14 vs ≈ 0.2), which violates the model's assumed stable trade-sign process |
| — | Conclusion | **RECLASSIFIED as exploratory / descriptive only.** Realised spread and price impact (table 06) are the primary adverse-selection evidence. |

### E. Order age and liquidity-provider types (D6)

| # | Check | Outcome |
|---|---|---|
| E1 | Age definition | **PASS / disclosed.** Age = execution time − submission time of the order's *current* ID. LOBSTER IDs are never reused. Delete + same-time same-side resubmission pairs (3% of deletions in AAPL, 16% in INTC) suggest repricing creates new IDs, so age restarts on repricing |
| E2 | Partial fills | **PASS.** 12–24% of executed orders fill in several pieces; each piece is an execution row with the same age origin; shares are volume-weighted |
| E3 | Unmatched executions | **FIXED.** Stage 2 put untraced executions into "old (>60s)" (by filling the age with a large number). They are pre-opening orders (all matched submissions start at 09:30:00) and are now a separate **"pre-open (untraced)"** class. Matching by volume in the analysis window: normal 93–98%; shock 63% (AAPL), 82–83% (AMZN, GOOG), 98–99% (INTC, MSFT). Results are shown with and without untraced orders |
| E4 | Denominators | **FIXED.** Stage 2 pooled share volume across stocks, so INTC/MSFT (low price, huge share counts) dominated. Shares and realised spreads are now computed per stock and averaged with equal stock weights |
| E5 | Threshold sensitivity | **PASS.** The young share falls in shocks for thresholds of 0.1 s (19% → 10%), 1 s (30% → 17%) and 10 s (59% → 42%) |
| E6 | Inference | **PASS.** Execution regressions cluster by stock × 30 min (60 clusters); the young-share change is tested with a paired block-level comparison (59 blocks): −0.105 [−0.148, −0.061] |
| E7 | Labels | **FIXED.** "Young / old" are order-age classes, not HFT / dealer identities, in all text and figures |

## Phase 3 — Shock classification and price recovery

| # | Check | Outcome |
|---|---|---|
| 3.1 | Severity split ex ante? | **PASS.** Uses \|OFI\| at τ = 0, the shock-defining variable; terciles of 423 (FI-2010) and 219 (LOBSTER) events |
| 3.2 | "Trade-driven / quote-driven" indicator | **RECLASSIFIED.** It is the sum over the shock window of market-order volume in the shock direction relative to mean volume (not a share of OFI, as stage 2 implied). It is known at τ = 0, and the groups have 329/328 events. It correlates *negatively* with \|OFI\| (−0.13). Renamed "high / low aggressive volume"; it is **not** an information-vs-liquidity classification |
| 3.3 | Reconcile "no reversal" with trade-level reversal | **FIXED.** Added the event-level *clock-time* response (table 08c). Shock episodes: the abnormal move peaks at 3.5 bps (~120 s), and 85% remains at 600 s. Ordinary trades: 33% of the 60-s impact reverts by 300 s (p = 0.006). Shock trades: 36% decline from 10 s to 300 s, n.s. (p = 0.06). The report now states horizon and fraction explicitly |
| 3.4 | Horizon change and sample size | **Recorded.** Extending windows to 300 steps reduced events 1,300 → 1,269 (FI-2010) and 662 → 657 (LOBSTER), because windows must fit in the trading day. Control exclusion kept at the 100-step main window, so cluster counts are unchanged (50/60). Main estimates moved by at most 0.2 of their SE (FI-2010 drift 4.83 → 4.99 bps; contrarian flow −0.55 → −0.56) |

## Phase 4 — Inference

| # | Check | Outcome |
|---|---|---|
| 4.1 | Cluster counts | FI-2010 50 stock-days; LOBSTER 60 stock × 30-min blocks; execution-level D6 60 blocks |
| 4.2 | Dependence across blocks / stocks | **Sensitivity added (table 05).** FI-2010: day (10) and stock (5) clustering. LOBSTER: market-wide 30-min blocks (12) and stock (5). Survive all: price impact, FI-2010 best-level depth, post-shock drift. Borderline with 5 stock clusters: H2 contrarian flow (p = 0.05 FI-2010, 0.11 LOBSTER). Not robust: LOBSTER spread effect (p = 0.16) |
| 4.3 | Few clusters | **LIMITATION.** With 5 stocks, stock-clustered t-tests have 4 df. Per-stock signs (table 04) are reported alongside. A wild cluster bootstrap would be the next step |
| 4.4 | Confidence intervals | **FIXED.** All primary tables now report CIs (t with G−1 df for event-level means; normal for regressions) |
| 4.5 | Multiple testing | **FIXED.** Holm adjustment across all rows of each hypothesis table, the H3 tables and the H4 table. The report states adjusted p unless marked "unadjusted" |
| 4.6 | Primary vs grid | **FIXED.** Primary specification declared in `run_all.py` and flagged (`primary_spec`) in table 09. The grid is summarised as sign/significance counts and used for stability, not as extra evidence |

## Remaining uncertainties

1. Hidden-execution signs by the quote rule may still misclassify some inside-spread prints (visible-only results agree).
2. Dependence across the 5 LOBSTER stocks on a single day cannot be fully addressed.
3. The FI-2010 stock labels OUT1V and RTRKS (similar price levels) follow the paper's table order and cannot be
   independently verified; no result depends on which is which.
4. Time-of-day effects in FI-2010 are controlled only through matching on position within the day (no timestamps).
