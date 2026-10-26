"""Paths and constants shared by every module."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_FI = ROOT / "data" / "raw" / "fi2010"
RAW_LOB = ROOT / "data" / "raw" / "lobster"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
TABLES = RESULTS / "tables"

for _p in (PROCESSED, FIGS, TABLES):
    _p.mkdir(parents=True, exist_ok=True)

# ---- FI-2010 -------------------------------------------------------------------------------
# Files and the trading days they contain (anchored CV: Train_CF_7 = days 1-7, Test_CF_k = day k+1).
FI_FILES = [
    ("Train_Dst_NoAuction_DecPre_CF_7.txt", [1, 2, 3, 4, 5, 6, 7]),
    ("Test_Dst_NoAuction_DecPre_CF_7.txt", [8]),
    ("Test_Dst_NoAuction_DecPre_CF_8.txt", [9]),
    ("Test_Dst_NoAuction_DecPre_CF_9.txt", [10]),
]
# Order of stocks inside every file (Ntakaris et al. 2018, Table 1; confirmed by price levels).
FI_STOCKS = ["KESBV", "OUT1V", "SAMPO", "RTRKS", "WRT1V"]
FI_PRICE_SCALE = 100.0       # DecPre price * 100 = EUR
FI_SIZE_SCALE = 1e5          # DecPre size * 1e5 = shares
FI_TICK = 0.01               # EUR
EVENTS_PER_ROW_FI = 10       # every FI-2010 row is one snapshot per 10 order-book events

# ---- LOBSTER ---------------------------------------------------------------------------------
LOB_TICKERS = ["AAPL", "AMZN", "GOOG", "INTC", "MSFT"]
LOB_DATE = "2012-06-21"
LOB_TICK = 0.01              # USD
LOB_START = 9.75 * 3600      # 09:45, skip opening 15 min
LOB_END = 15.75 * 3600       # 15:45, skip closing 15 min
LOB_BIN = 10                 # messages per snapshot (mimics FI-2010's 10-event sampling)
