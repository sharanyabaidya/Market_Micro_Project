# Presentation outline (10 slides, ~15 minutes)

Each slide lists **what it claims**, **what to show**, and **what not to say**.

---

### 1. Question and motivation
**Claim:** Dealers are supposed to absorb temporary order imbalances. Do limit-order-book liquidity providers do
the same when order flow becomes extremely one-sided?
**Show:** The mechanism chain *shock → provision → absorption → stabilisation*, and the four hypotheses H1–H4.
**Avoid:** Promising to identify dealers.

### 2. Theory: four papers, four predictions
**Claim:** Each paper gives a testable prediction.
* Amihud–Mendelson: inventory shading, so temporary price pressure.
* Ho–Stoll: competition narrows spreads.
* Biais–Weill: liquidity drains, then refills.
* Biais–Declerck–Moinas: the composition of suppliers shifts under stress.

**Show:** The prediction table from REPORT §2.

### 3. Data and two data-quality fixes
**Claim:** Two public datasets complement each other:
* FI-2010: 5 stocks × 10 days, book only;
* LOBSTER: 5 stocks × 1 day, every message.

Both needed cleaning before use.
**Show:**
* FI-2010 files stack stocks and days; split exactly into 50 stock-days (surviving-pairs histogram from notebook §1);
* LOBSTER hidden executions are mis-signed in the raw field; re-signed with the quote rule.

**Avoid:** Spending more than one minute here.

### 4. Design
**Claim:** Shocks are 1%-tail order-flow imbalance over 50 events. Each is compared with matched calm moments,
including moments with the *same depleted book*. This removes mechanical mean reversion.
**Show:** A timeline diagram: shock [−5, 0], provision (0, 10], recovery (10, 100]; and the two control types.

### 5. H1: liquidity is drained on the side being hit
**Claim:**
* best-level depth on the hit side falls by ~56% of its mean (FI-2010, all 5 stocks);
* the book tilts toward the pressure;
* spreads widen in LOBSTER (+0.75 bps), but not robustly across stocks.

**Show:** `results/figures/fig1_event_study_FI-2010.png` (depth and spread panels) and the H1 rows of table 02.

### 6. H2: no contrarian provision
**Claim:** After the shock, compared with the same depleted book in calm times:
* order flow keeps going in the shock direction;
* the hit side refills no faster;
* in LOBSTER, there are *fewer* price-improving quotes.

**Show:** `results/figures/fig1_event_study_LOBSTER.png` (market-order volume, net limit orders, cumulative OFI)
and the H2 rows with CIs.
**Avoid:** Overstating precision. With only 5 stocks, stock-level p ≈ 0.05–0.11. Say so, and point to 24/24
specifications with the same sign.

### 7. Prices after shocks: persistent, not reverting
**Claim:**
* Ordinary trades: about a third of the impact reverts within 5 minutes.
* Shock episodes: about 85% of the peak price move is still there after 10 minutes.

**Show:** `results/figures/fig2_price_response_LOBSTER.png` (both panels side by side).
**Avoid:** Calling it "permanent". Say "persistent over 10 minutes".

### 8. H4: those who absorb shocks pay for it
**Claim:** Liquidity providers on the other side of shock trades:
* lose ~1.1 bps more in realised spread at 5 s;
* face ~1.2 bps more price impact;
* receive no wider effective spread.

**Show:** Table 06 (5 s and 60 s rows, all trades and visible-only).

### 9. Who supplies liquidity under stress (exploratory)
**Claim:**
* The share of passive volume from very young (< 1 s) orders drops from 30% to 17%.
* Older and pre-open orders absorb more.
* Everyone loses about the same.

This is consistent with Biais–Declerck–Moinas.
**Show:** `results/figures/fig3_liquidity_supply_by_order_age_LOBSTER.png`.
**Avoid:** "HFTs withdraw". Order age is not trader identity.

### 10. Verdicts, limits, next step
**Claim:**
* H1 supported (depth, imbalance), spread mixed;
* H2 contradicted;
* H3 mixed;
* H4 supported;
* Amihud–Mendelson and Ho–Stoll are not directly testable without trader IDs;
* Biais–Weill partly supported; Biais–Declerck–Moinas consistent.

**Show:** The classification table from REPORT §6; one line on the next step (multi-day data with participant flags).

---

**Backup slides:**
* robustness summary (REPORT §7 table);
* cluster sensitivity (table 05);
* D1 fixed-effect placebo (table 11b), to show the audit discipline;
* Huang–Stoll sanity check (table 14b), to explain why it is not used.
