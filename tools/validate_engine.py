"""Full self-test (about 5 minutes): re-computes the 25-Sep-2026 calls on this server and compares them with the
reference calls shipped in static/. Every call must match - this proves the Python packages behave identically here.

Usage:  .venv/bin/python tools/validate_engine.py
"""
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from n5ai import engine  # noqa: E402
from n5ai.data_update import PriceStore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
t0 = time.time()
store = PriceStore(ROOT / "data").load()
prices, bench, uni = store.prices, store.bench, store.universe
res = engine.run(prices, bench, uni, ROOT / "model", asof=pd.Timestamp("2026-09-25"), workers=2)
D = res["decisions"].set_index(["symbol", "tf"]).sort_index()
S = pd.read_pickle(ROOT / "static" / "seed_decisions_20260925.pkl").set_index(["symbol", "tf"]).sort_index()
common = D.index.intersection(S.index)
bad_type = int((D.loc[common, "type"] != S.loc[common, "type"]).sum())
worst = 0.0
for c in ("conf_buy", "conf_short", "pct_buy", "pct_short", "entry", "target", "stop"):
    a, b = D.loc[common, c].astype(float), S.loc[common, c].astype(float)
    if c in ("entry", "target", "stop"):          # compare as % of the price (unchanged by later splits/bonuses)
        a, b = a / D.loc[common, "close"].astype(float), b / S.loc[common, "close"].astype(float)
    both = a.notna() & b.notna()
    if both.any():
        worst = max(worst, float(np.max(np.abs(a[both] - b[both]))))
print(f"calls compared: {len(common)} of {len(S)} | different calls: {bad_type} | largest numeric difference: {worst:.2e}")
ok = len(common) == len(S) and bad_type == 0 and worst < 1e-6
print("SELF-TEST PASSED - identical to the reference calls of 25-Sep-2026" if ok else
      "SELF-TEST FAILED - check that requirements.txt versions are installed")
print(f"{time.time() - t0:.0f}s")
sys.exit(0 if ok else 1)
