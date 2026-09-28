"""Quick self-checks of the parsers, the trade evaluation and the corporate-action logic (no network needed)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from n5ai import data_update as du  # noqa: E402
from n5ai import tracker as tk  # noqa: E402

ok = True


def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name)
    ok &= bool(cond)


# 1. NSE index close file (format of ind_close_all_DDMMYYYY.csv)
raw = (b'Index Name,Index Date,Open Index Value,High Index Value,Low Index Value,Closing Index Value,Points Change,'
       b'Change(%),Volume,Turnover (Rs. Cr.),P/E,P/B,Div Yield\n'
       b'Nifty 50,25-09-2026,23035.00,23162.70,23020.95,23140.50,77.40,0.34,242720711,21000.5,20.1,3.4,1.3\n'
       b'Nifty Next 50,25-09-2026,60000,60500,59800,60200,10,0.1,1000,500,20,3,1\n'
       b'Nifty 500,25-09-2026,21000,21100,20900,21050,10,0.1,1000,500,20,3,1\n'
       b'Nifty Midcap 150,25-09-2026,-,-,-,19500.25,10,0.1,1000,500,20,3,1\n'
       b'Nifty Smallcap 250,25-09-2026,15000,15100,14900,15050,10,0.1,1000,500,20,3,1\n')
x = du.read_ind_close(raw)
check("index file parsed (5 indices)", len(x) == 5 and abs(x["NIFTY50"][3] - 23140.50) < 1e-9)
check("missing open/high/low fall back to close", x["MIDCAP150"][0] == 19500.25)

# 2. trade evaluation
idx = pd.bdate_range("2026-09-01", periods=15)
def frame(o, h, l, c):
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c}, index=idx[:len(o)])
base = dict(entry_limit=101.0, target=106.0, stop=97.0, side=1, max_days=10, close=100.0)
sd = idx[0]
df = frame([100, 100.5, 101, 104], [100, 101, 103, 107], [100, 99.5, 100, 103], [100, 100.8, 102, 106.5])
u = tk._eval_trade(pd.Series(base), df, sd, 1.0)
check("target hit on day 3 at the target price", u["status"] == "Target hit" and u["exit_price"] == 106.0
      and u["days_held"] == 3 and abs(u["result_pct"] - 100 * (106 / 100.5 - 1 - 0.003)) < 1e-9)
df = frame([100, 100.5, 99, 96], [100, 101, 99.5, 97], [100, 99.5, 98, 95], [100, 100, 98.5, 95.5])
u = tk._eval_trade(pd.Series(base), df, sd, 1.0)
check("gap below the stop exits at the open", u["status"] == "Stop hit" and u["exit_price"] == 96 and u["outcome"] == "Loss")
df = frame([100, 102], [100, 103], [100, 101], [100, 102])
u = tk._eval_trade(pd.Series(base), df, sd, 1.0)
check("open above the entry limit = not triggered", u["status"] == "Not triggered")
df = frame([100] + [100.2] * 10, [100] + [101] * 10, [100] + [99] * 10, [100] + [100.9] * 10)
u = tk._eval_trade(pd.Series(base), df, sd, 1.0)
check("time exit after 10 sessions", u["status"] == "Time exit" and u["days_held"] == 10 and u["exit_price"] == 100.9)
df = frame([100, 100.2, 100.4], [100, 101, 101], [100, 99, 99], [100, 100.5, 100.6])
u = tk._eval_trade(pd.Series(base), df, sd, 1.0)
check("still running -> Open with mark-to-market", u["status"] == "Open" and u["days_held"] == 2)
df = frame([200, 201, 203, 208], [200, 202, 206, 214], [200, 199, 200, 206], [200, 201.6, 204, 213])
u = tk._eval_trade(pd.Series(base), df, sd, 2.0)
check("levels follow a 1:1 bonus (factor 2)", u["status"] == "Target hit" and abs(u["exit_price"] - 106.0) < 1e-9)
short = dict(entry_limit=99.0, target=94.0, stop=103.0, side=-1, max_days=10, close=100.0)
df = frame([100, 99.5, 97, 95], [100, 100, 98, 95.5], [100, 98, 96, 93.5], [100, 98.2, 96.5, 94])
u = tk._eval_trade(pd.Series(short), df, sd, 1.0)
check("short trade target", u["status"] == "Target hit" and u["exit_price"] == 94.0 and u["outcome"] == "Win")

# 3. avoid evaluation vs the average stock
closes = pd.DataFrame({"A": np.linspace(100, 90, 15), "B": np.linspace(100, 110, 15), "C": np.linspace(100, 105, 15)},
                      index=idx)
rec = pd.Series(dict(max_days=10, close=100.0))
dfA = pd.DataFrame({"close": closes["A"]})
u = tk._eval_avoid(rec, dfA, closes, idx[0], 1.0)
check("avoid call on a laggard is correct", u["status"] == "Correct" and u["excess_pct"] < 0)
dfB = pd.DataFrame({"close": closes["B"]})
u = tk._eval_avoid(rec, dfB, closes, idx[0], 1.0)
check("avoid call on a leader is wrong", u["status"] == "Wrong")

# 4. corporate action from PREV_CLOSE
class S:  # minimal stand-in for PriceStore
    prices = {"XYZ": pd.DataFrame({c: [100.0, 102.0] for c in ("open", "high", "low", "close")} |
                                  {"volume": [1000.0, 1000.0], "deliv_qty": [500.0, 500.0], "trades": [10.0, 10.0],
                                   "series": ["EQ", "EQ"], "deliv_per": [50.0, 50.0], "turnover_cr": [0.01, 0.01]},
                                  index=pd.DatetimeIndex(["2026-09-24", "2026-09-25"], name="date"))}
    ca_new = []
    ca_run = []
b = pd.DataFrame({"SERIES": ["EQ"], "PREV_CLOSE": [51.0], "OPEN_PRICE": [51.5], "HIGH_PRICE": [52.0], "LOW_PRICE": [50.5],
                  "CLOSE_PRICE": [51.8], "TTL_TRD_QNTY": [2500.0], "DELIV_QTY": [1000.0], "NO_OF_TRADES": [30.0]},
                 index=["XYZ"])
st = du.apply_bhav(S, pd.Timestamp("2026-09-28"), b)
d = S.prices["XYZ"]
check("1:1 split detected and history halved", st["ca"] == 1 and abs(d["close"].iloc[1] - 51.0) < 1e-9
      and abs(d["volume"].iloc[0] - 2000.0) < 1e-9 and len(d) == 3 and d.index[-1] == pd.Timestamp("2026-09-28"))
# 5. holiday copies on the mirror: a file named for 14-Sep that holds 11-Sep data is treated as "not available"
rows = "\n".join(f"SYM{i}, EQ, 11-Sep-2026, 100, 100, 101, 99, 100, 100.5, 100, 1000, 1, 10, 500, 50" for i in range(600))
raw = ("SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, LAST_PRICE, CLOSE_PRICE, AVG_PRICE, "
       "TTL_TRD_QNTY, TURNOVER_LACS, NO_OF_TRADES, DELIV_QTY, DELIV_PER\n" + rows).encode()
try:
    du.validate_bhav_for(pd.Timestamp("2026-09-14"))(raw)
    stale = False
except du.StaleFile:
    stale = True
check("stale holiday copy detected", stale)
try:
    du.validate_bhav_for(pd.Timestamp("2026-09-11"))(raw)
    good = True
except Exception:
    good = False
check("same file accepted for its own date", good)
print("ALL PASS" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
