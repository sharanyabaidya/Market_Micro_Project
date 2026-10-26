# The Role of Dealer-Like Liquidity Providers in Absorbing Temporary Liquidity Shocks

This is an empirical market-microstructure study of extreme order-flow shocks in two limit order books. It asks
four questions:
1. Does liquidity deteriorate during a shock?
2. Do liquidity providers lean against the flow, as dealers are supposed to?
3. Do prices and liquidity recover, and does stronger provision speed that up?
4. Who supplies liquidity under stress, and at what adverse-selection cost?

| Document | Contents |
|---|---|
| [REPORT.md](REPORT.md) | Final report: question, literature, data, method, results, hypothesis verdicts, limitations |
| [VALIDATION_LOG.md](VALIDATION_LOG.md) | Every audit performed, what failed, what was fixed |
| [notebooks/analysis.ipynb](notebooks/analysis.ipynb) | Executed narrative notebook over the final outputs |
| [PRESENTATION_OUTLINE.md](PRESENTATION_OUTLINE.md) | 10-slide outline with the figure to use on each slide |
| `Market_Micro_Project_Proposal.pdf` | Original proposal |

## Data

The raw data (~1.5 GB) is not stored in the repository. It is downloaded into `data/raw/` by
`scripts/download_data.sh`.

| Dataset | Content | Source |
|---|---|---|
| FI-2010 (NoAuction, DecPre) | 5 Helsinki stocks × 10 days (June 2010), 10-level book, 394,337 snapshots (1 per 10 events) | Mirror in the DeepLOB repository (Train_CF_7 = days 1–7, Test_CF_7/8/9 = days 8/9/10) |
| LOBSTER sample files | AAPL, AMZN, GOOG, INTC, MSFT, 21 June 2012, 10-level book and message file | Official LOBSTER samples via a public Hugging Face copy (lobsterdata.com now gates samples behind a form) |

## Reproduce (Linux / WSL, Python 3.12)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
bash scripts/download_data.sh                 # raw data -> data/raw/
.venv/bin/python scripts/run_all.py           # all tables + figures (~2 min); rebuilds data/processed/
.venv/bin/python scripts/make_notebook.py     # writes notebooks/analysis.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb
.venv/bin/python scripts/audit_facts.py       # optional: prints the data-validation facts behind VALIDATION_LOG.md
```

## Pipeline

```
data/raw  ──► src/mm/fi2010.py   split FI-2010 into 50 stock-days, convert to EUR / shares
          ──► src/mm/lobster.py  LOBSTER snapshots every 10 messages; trade signing; per-trade table
                │
                ▼
src/mm/measures.py   spreads, depth (1/3/10 levels), imbalance, order-flow imbalance (OFI), volatility
src/mm/events.py     shock detection (OFI 1% tails), event windows, matched control windows
src/mm/analysis.py   H1-H4 tests, H3 regressions, H4 trade analysis, price response, robustness grid,
                     cluster sensitivity, shock types, figures
src/mm/dealer.py     exploratory dealer-theory proxies D1-D6 (order-ID replay of the message files)
src/mm/stats.py      clustered means, OLS with fixed effects, Holm correction
                │
                ▼
scripts/run_all.py   ──► results/tables/*.csv, results/figures/*.png
```

**Primary specification:** level-1 OFI, 5-step shock window, 1% tails, provision window (0, 10], recovery
window (10, 100], 5 matched controls. Standard errors are clustered by stock-day (FI-2010) or stock × 30 min
(LOBSTER), with Holm-adjusted p-values.

## Outputs

| Tables | Content | Role |
|---|---|---|
| 01 | descriptives | — |
| 02 | H1–H4 event tests (means, CIs, Holm p) | primary |
| 03 | H3 regressions | secondary |
| 04 | per-stock results | primary (sign consistency) |
| 05 | inference under alternative clustering | primary (sensitivity) |
| 06 | H4 adverse selection: all trades and visible-only | primary |
| 07 | severity quintiles (FI-2010) | secondary |
| 08a–c | price response after trades and after shocks (clock time) | primary |
| 09, 09b | robustness grid (24 specs per dataset); earlier depth-imbalance definition | secondary |
| 10, 10b | shock types and diagnostics | exploratory |
| 11–15e | dealer-theory proxies D1–D6, including a fixed-effect placebo and Huang–Stoll sanity check | exploratory |

| Figures | Content |
|---|---|
| fig1 | event studies (FI-2010, LOBSTER) |
| fig2 | price response after trades vs after shocks |
| fig3 | liquidity supply by order age |
| supp_* | provision terciles (supplementary) |

`results/archive/baseline_stage2/` keeps the pre-audit outputs for comparison.

`marketmicro.ipynb` is the earliest exploratory notebook, kept for reference only.
