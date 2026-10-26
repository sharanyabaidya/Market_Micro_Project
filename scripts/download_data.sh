#!/usr/bin/env bash
# Downloads all raw data used by the project into data/raw/.
#
# 1) FI-2010 (Ntakaris et al. 2018), NoAuction / DecPre normalisation: all 10 days, 5 Helsinki stocks.
#    Mirror from the DeepLOB repo (Zhang, Zohren & Roberts 2019).
#    Train_Dst_NoAuction_DecPre_CF_7 = days 1-7, Test_..._CF_7/8/9 = days 8/9/10.
# 2) LOBSTER free sample files (official samples, 2012-06-21, 10 levels) for AAPL AMZN GOOG INTC MSFT.
#    lobsterdata.com now gates samples behind a request form, so we use a public Hugging Face copy.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p data/raw/fi2010 data/raw/lobster

# ---- FI-2010 ----
if [ ! -f data/raw/fi2010/Test_Dst_NoAuction_DecPre_CF_9.txt ]; then
  curl -fsSL -o data/raw/fi2010/data.zip \
    https://raw.githubusercontent.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/master/data/data.zip
  python3 -c "import zipfile; zipfile.ZipFile('data/raw/fi2010/data.zip').extractall('data/raw/fi2010')"
fi

# ---- LOBSTER samples ----
BASE="https://huggingface.co/datasets/totalorganfailure/lobster-data/resolve/main"
for t in AAPL AMZN GOOG INTC MSFT; do
  for kind in message orderbook; do
    f="${t}_2012-06-21_34200000_57600000_${kind}_10.csv"
    [ -s "data/raw/lobster/$f" ] && continue
    echo "downloading $f"
    curl -fsSL -o "data/raw/lobster/$f.part" "$BASE/LOBSTER_SampleFile_${t}_2012-06-21_10/$f"
    mv "data/raw/lobster/$f.part" "data/raw/lobster/$f"
  done
done
[ -s data/raw/lobster/LOBSTER_SampleFiles_ReadMe.txt ] || \
  curl -fsSL -o data/raw/lobster/LOBSTER_SampleFiles_ReadMe.txt \
    "$BASE/LOBSTER_SampleFile_GOOG_2012-06-21_10/LOBSTER_SampleFiles_ReadMe.txt"
rm -f data/raw/lobster/*.zip
ls -la data/raw/fi2010 data/raw/lobster
