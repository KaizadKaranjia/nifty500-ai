"""Recommendation history (SQLite) and the success rate of the calls.

Every Buy / Short / Avoid call is stored with its date, price levels and AI confidence. Each run re-checks the
open calls against the prices that followed:

  BUY / SHORT  - paper trade exactly as in the research back-test: enter at the NEXT session's open only if it
                 is within the entry limit (and not beyond the stop); then target / stop / time exit
                 (2 weeks = 10, 3 months = 60, 1 year = 250 sessions). Result after 0.3% costs.
                 Success = the trade made money. "Target hit" is also reported, because that is exactly what
                 the AI confidence % predicts.
  AVOID        - after the same number of sessions, did the stock do worse than the average Nifty 500 stock?
                 Success = yes (it underperformed).

A call that repeats while the first one is still running is not counted twice (repeat_count goes up).
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from .common import COST, TF_RULES, call_kind

log = logging.getLogger("n5ai.tracker")

SCHEMA = """
CREATE TABLE IF NOT EXISTS recommendations (
    rec_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_date   TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    company       TEXT,
    timeframe     TEXT NOT NULL,
    kind          TEXT NOT NULL,
    call_type     TEXT NOT NULL,
    side          INTEGER NOT NULL,
    close         REAL,
    atr           REAL,
    entry_limit   REAL,
    target        REAL,
    stop          REAL,
    max_days      INTEGER,
    ai_conf       REAL,
    win_prob      REAL,
    exp_ret       REAL,
    base_hit      REAL,
    model_rank    REAL,
    evidence      TEXT,
    status        TEXT NOT NULL DEFAULT 'Pending',
    outcome       TEXT,
    note          TEXT,
    entry_date    TEXT,
    entry_price   REAL,
    exit_date     TEXT,
    exit_price    REAL,
    result_pct    REAL,
    days_held     INTEGER,
    last_date     TEXT,
    last_price    REAL,
    stock_ret_pct REAL,
    mkt_ret_pct   REAL,
    excess_pct    REAL,
    repeat_count  INTEGER NOT NULL DEFAULT 0,
    last_seen     TEXT,
    created_at    TEXT DEFAULT (datetime('now', 'localtime')),
    updated_at    TEXT,
    UNIQUE (signal_date, symbol, timeframe, kind)
);
CREATE INDEX IF NOT EXISTS ix_rec_status ON recommendations (status);
CREATE INDEX IF NOT EXISTS ix_rec_symbol ON recommendations (symbol, timeframe, kind);
CREATE TABLE IF NOT EXISTS daily_calls (
    call_date   TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    timeframe   TEXT NOT NULL,
    call_type   TEXT,
    side        INTEGER,
    conf_buy    REAL,
    conf_short  REAL,
    win_buy     REAL,
    exp_ret_buy REAL,
    pct_buy     REAL,
    pct_short   REAL,
    close       REAL,
    atr         REAL,
    PRIMARY KEY (call_date, symbol, timeframe)
);
CREATE TABLE IF NOT EXISTS runs (
    run_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    started    TEXT,
    finished   TEXT,
    data_date  TEXT,
    status     TEXT,
    message    TEXT
);
CREATE TABLE IF NOT EXISTS success_daily (
    as_of_date           TEXT PRIMARY KEY,
    calls                INTEGER,
    completed            INTEGER,
    running              INTEGER,
    not_triggered        INTEGER,
    success_rate         REAL,
    mature               INTEGER,
    success_rate_mature  REAL,
    buy_completed        INTEGER,
    buy_success_rate     REAL,
    buy_target_hit_rate  REAL,
    buy_avg_ai_conf      REAL,
    buy_avg_result_pct   REAL,
    avoid_completed      INTEGER,
    avoid_success_rate   REAL,
    avoid_avg_excess_pct REAL,
    short_completed      INTEGER,
    short_success_rate   REAL,
    recorded_at          TEXT
);
"""
SUCCESS_DAILY_COLS = ("as_of_date", "calls", "completed", "running", "not_triggered", "success_rate", "mature",
                      "success_rate_mature", "buy_completed", "buy_success_rate", "buy_target_hit_rate",
                      "buy_avg_ai_conf", "buy_avg_result_pct", "avoid_completed", "avoid_success_rate",
                      "avoid_avg_excess_pct", "short_completed", "short_success_rate", "recorded_at")
OPEN_STATES = ("Pending", "Open")
CLOSED_STATES = ("Target hit", "Stop hit", "Time exit", "Correct", "Wrong")


class Tracker:
    def __init__(self, db_path: Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(self.path)
        self.con.row_factory = sqlite3.Row
        self.con.executescript(SCHEMA)
        self.con.commit()

    def close(self):
        self.con.close()

    # ---- runs --------------------------------------------------------------------------------------------------
    def run_start(self) -> int:
        cur = self.con.execute("INSERT INTO runs (started, status) VALUES (datetime('now','localtime'), 'running')")
        self.con.commit()
        return int(cur.lastrowid)

    def run_end(self, run_id: int, status: str, data_date=None, message: str = ""):
        self.con.execute("UPDATE runs SET finished = datetime('now','localtime'), status = ?, data_date = ?, message = ? "
                         "WHERE run_id = ?", (status, None if data_date is None else str(pd.Timestamp(data_date).date()),
                                              message[:2000], run_id))
        self.con.commit()

    # ---- logging calls -------------------------------------------------------------------------------------------
    def n_recommendations(self) -> int:
        return int(self.con.execute("SELECT COUNT(*) FROM recommendations").fetchone()[0])

    def has_calls_for(self, date) -> bool:
        d = str(pd.Timestamp(date).date())
        return self.con.execute("SELECT 1 FROM daily_calls WHERE call_date = ? LIMIT 1", (d,)).fetchone() is not None

    def log_calls(self, D: pd.DataFrame, date, companies: dict) -> dict:
        """Store the day's calls. Returns counts of new and repeated recommendations."""
        d = str(pd.Timestamp(date).date())
        new = rep = 0
        daily = []
        for _, x in D.iterrows():
            if not bool(x.get("eligible", False)) or x.get("type") == "Not rated":
                continue
            daily.append((d, x["symbol"], x["tf"], x["type"], int(x["side"]), _f(x.get("conf_buy")),
                          _f(x.get("conf_short")), _f(x.get("win_buy")), _f(x.get("exp_ret_buy")), _f(x.get("pct_buy")),
                          _f(x.get("pct_short")), _f(x.get("close")), _f(x.get("atr"))))
            kind = call_kind(x["type"])
            if kind is None:
                continue
            row = self.con.execute("SELECT rec_id, last_seen FROM recommendations WHERE symbol = ? AND timeframe = ? "
                                   "AND kind = ? AND status IN ('Pending','Open') ORDER BY signal_date DESC LIMIT 1",
                                   (x["symbol"], x["tf"], kind)).fetchone()
            if row is not None:
                if (row["last_seen"] or "") < d:
                    self.con.execute("UPDATE recommendations SET repeat_count = repeat_count + 1, last_seen = ?, "
                                     "updated_at = datetime('now','localtime') WHERE rec_id = ?", (d, row["rec_id"]))
                    rep += 1
                continue
            side = int(x["side"])
            if kind == "BUY":
                conf, win, exp, base, rank = (x.get("conf_buy"), x.get("win_buy"), x.get("exp_ret_buy"),
                                              x.get("base_hit_buy"), x.get("pct_buy"))
            elif kind == "SHORT":
                conf, win, exp, base, rank = (x.get("conf_short"), x.get("win_short"), x.get("exp_ret_short"),
                                              x.get("base_hit_short"), x.get("pct_short"))
            else:
                conf, win, exp, base, rank = (None, None, -abs(_f(x.get("model_ex_short")) or 0.0), None,
                                              x.get("pct_short"))
            cur = self.con.execute(
                "INSERT OR IGNORE INTO recommendations (signal_date, symbol, company, timeframe, kind, call_type, side, "
                "close, atr, entry_limit, target, stop, max_days, ai_conf, win_prob, exp_ret, base_hit, model_rank, "
                "evidence, status, last_seen, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'Pending',?,"
                "datetime('now','localtime'))",
                (d, x["symbol"], companies.get(x["symbol"], ""), x["tf"], kind, x["type"], side, _f(x.get("close")),
                 _f(x.get("atr")), _f(x.get("entry")) if side else None, _f(x.get("target")) if side else None,
                 _f(x.get("stop")) if side else None, int(TF_RULES[x["tf"]][4]), _f(conf), _f(win), _f(exp), _f(base),
                 _f(rank), evidence_short(x), d))
            new += cur.rowcount
        self.con.executemany("INSERT OR REPLACE INTO daily_calls VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", daily)
        self.con.commit()
        log.info("calls for %s logged: %d new recommendations, %d repeats of calls still running", d, new, rep)
        return {"new": new, "repeat": rep}

    # ---- evaluation ------------------------------------------------------------------------------------------
    def evaluate(self, prices: dict, universe: pd.DataFrame, ca: pd.DataFrame | None = None) -> dict:
        """Re-check every Pending/Open call against the latest prices."""
        recs = pd.read_sql_query("SELECT * FROM recommendations WHERE status IN ('Pending','Open')", self.con)
        if recs.empty:
            return {"checked": 0}
        ca = ca if ca is not None else pd.DataFrame(columns=["symbol", "ex_date", "factor"])
        if len(ca):
            ca = ca.assign(ex_date=pd.to_datetime(ca["ex_date"]))
        closes = None
        if (recs["kind"] == "AVOID").any():
            ok = set(universe["symbol"])
            closes = pd.DataFrame({s: d["close"] for s, d in prices.items() if s in ok and len(d)}).sort_index()
        counts: dict = {}
        for _, r in recs.iterrows():
            df = prices.get(r["symbol"])
            if df is None or df.empty:
                continue
            sd = pd.Timestamp(r["signal_date"])
            cf = float(np.prod(ca.loc[(ca["symbol"] == r["symbol"]) & (ca["ex_date"] > sd), "factor"].astype(float)))
            if r["kind"] == "AVOID":
                upd = _eval_avoid(r, df, closes, sd, cf)
            else:
                upd = _eval_trade(r, df, sd, cf)
            if upd:
                upd["updated_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                cols = ", ".join(f"{k} = ?" for k in upd)
                self.con.execute(f"UPDATE recommendations SET {cols} WHERE rec_id = ?", (*upd.values(), int(r["rec_id"])))
                counts[upd.get("status", r["status"])] = counts.get(upd.get("status", r["status"]), 0) + 1
        self.con.commit()
        log.info("evaluated %d running calls: %s", len(recs), counts)
        return counts

    # ---- reporting -----------------------------------------------------------------------------------------------
    def table(self) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT * FROM recommendations ORDER BY signal_date DESC, "
            "CASE kind WHEN 'BUY' THEN 0 WHEN 'SHORT' THEN 1 ELSE 2 END, "
            "CASE timeframe WHEN 'ST' THEN 0 WHEN 'MT' THEN 1 ELSE 2 END, "
            "CASE call_type WHEN 'Strong Buy' THEN 0 ELSE 1 END, symbol", self.con)

    def export_csv(self, path: Path):
        self.table().to_csv(path, index=False)

    def summary(self) -> dict:
        T = self.table()
        return summarize(T)

    # ---- daily success-rate history ---------------------------------------------------------------------------
    def record_success(self, date) -> dict:
        """Store today's headline success numbers (one row per session date) - the daily history shown in APEX."""
        hd = self.summary()["headline"]
        if not hd:
            return {}
        b, a, s = hd.get("buy", {}), hd.get("avoid", {}), hd.get("short", {})
        row = {"as_of_date": str(pd.Timestamp(date).date()), "calls": hd["calls"], "completed": hd["closed"],
               "running": hd["running"], "not_triggered": hd["not_triggered"], "success_rate": _f(hd["success_rate"]),
               "mature": hd["mature"], "success_rate_mature": _f(hd["success_rate_mature"]),
               "buy_completed": b.get("closed", 0), "buy_success_rate": _f(b.get("success_rate")),
               "buy_target_hit_rate": _f(b.get("target_hit_rate")), "buy_avg_ai_conf": _f(b.get("avg_ai_conf")),
               "buy_avg_result_pct": _f(b.get("avg_result_pct")), "avoid_completed": a.get("closed", 0),
               "avoid_success_rate": _f(a.get("success_rate")), "avoid_avg_excess_pct": _f(a.get("avg_excess_pct")),
               "short_completed": s.get("closed", 0), "short_success_rate": _f(s.get("success_rate")),
               "recorded_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
        self.con.execute(f"INSERT OR REPLACE INTO success_daily ({', '.join(SUCCESS_DAILY_COLS)}) VALUES "
                         f"({', '.join('?' * len(SUCCESS_DAILY_COLS))})", [row[c] for c in SUCCESS_DAILY_COLS])
        self.con.commit()
        return row

    def success_history(self) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM success_daily ORDER BY as_of_date", self.con)

    def call_dates(self) -> list[str]:
        return [r[0] for r in self.con.execute("SELECT DISTINCT call_date FROM daily_calls ORDER BY call_date")]


# ------------------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------------------
def _f(v):
    try:
        v = float(v)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def evidence_short(x) -> str:
    parts = []
    if x.get("long_model") and x.get("side", 0) > 0:
        parts.append(f"AI model top {max(1, round(100 * (1 - float(x['pct_buy']))))}%")
    if x.get("avoid_model") and x.get("side", 0) <= 0:
        parts.append(f"AI model laggard flag (short-side top {max(1, round(100 * (1 - float(x['pct_short']))))}%)")
    for key, sign in (("pos_signals", "+"), ("neg_signals", "-")):
        v = x.get(key)
        if isinstance(v, list):
            parts += [f"{sign} {p[0]}" for p in v[:2]]
    return "; ".join(parts)[:500]


def _eval_trade(r, df: pd.DataFrame, sd: pd.Timestamp, cf: float) -> dict:
    side, hold = int(r["side"]), int(r["max_days"])
    bars = df[df.index > sd]
    if bars.empty:
        return {}
    lim, tgt, stp = float(r["entry_limit"]) * cf, float(r["target"]) * cf, float(r["stop"]) * cf
    o0 = float(bars["open"].iloc[0])
    taken = (o0 <= lim and o0 > stp) if side > 0 else (o0 >= lim and o0 < stp)
    entry_date = bars.index[0]
    if not taken:
        why = ("opened above the entry limit" if (side > 0 and o0 > lim) or (side < 0 and o0 < lim)
               else "opened beyond the stop loss")
        return {"status": "Not triggered", "outcome": None, "note": f"{entry_date:%d-%b-%Y} {why} "
                f"(open {o0 / cf:,.2f} vs limit {r['entry_limit']:,.2f})", "entry_date": str(entry_date.date()),
                "last_date": str(bars.index[-1].date()), "last_price": float(bars["close"].iloc[-1]) / cf}
    entry = o0
    path = bars.iloc[:hold]
    exit_px = exit_d = None
    status = None
    for k, (d, o, h, lo, c) in enumerate(zip(path.index, path["open"], path["high"], path["low"], path["close"])):
        if side > 0:
            if k > 0 and o <= stp:
                exit_px, exit_d, status = o, d, "Stop hit"
            elif k > 0 and o >= tgt:
                exit_px, exit_d, status = o, d, "Target hit"
            elif lo <= stp:
                exit_px, exit_d, status = stp, d, "Stop hit"
            elif h >= tgt:
                exit_px, exit_d, status = tgt, d, "Target hit"
        else:
            if k > 0 and o >= stp:
                exit_px, exit_d, status = o, d, "Stop hit"
            elif k > 0 and o <= tgt:
                exit_px, exit_d, status = o, d, "Target hit"
            elif h >= stp:
                exit_px, exit_d, status = stp, d, "Stop hit"
            elif lo <= tgt:
                exit_px, exit_d, status = tgt, d, "Target hit"
        if status is None and k == hold - 1:
            exit_px, exit_d, status = c, d, "Time exit"
        if status is not None:
            days = k + 1
            break
    last_c = float(bars["close"].iloc[min(len(bars), hold) - 1] if status is None else exit_px)
    base = {"entry_date": str(entry_date.date()), "entry_price": entry / cf,
            "last_date": str((exit_d if exit_d is not None else path.index[-1]).date()), "last_price": last_c / cf}
    if status is None:                                     # still running
        gross = (last_c / entry - 1) if side > 0 else (1 - last_c / entry)
        return {**base, "status": "Open", "result_pct": 100 * (gross - COST), "days_held": len(path),
                "note": f"running: day {len(path)} of {hold}"}
    gross = (exit_px / entry - 1) if side > 0 else (1 - exit_px / entry)
    res = 100 * (gross - COST)
    return {**base, "status": status, "exit_date": str(exit_d.date()), "exit_price": exit_px / cf, "result_pct": res,
            "days_held": days, "outcome": "Win" if res > 0 else "Loss", "note": None}


def _eval_avoid(r, df: pd.DataFrame, closes: pd.DataFrame, sd: pd.Timestamp, cf: float) -> dict:
    hold = int(r["max_days"])
    bars = df[df.index > sd]
    if bars.empty or closes is None or sd not in closes.index:
        return {}
    end_i = min(len(bars), hold) - 1
    end_d = bars.index[end_i]
    c0 = float(r["close"]) * cf
    c1 = float(bars["close"].iloc[end_i])
    stock = c1 / c0 - 1
    a, b = closes.loc[sd], closes.loc[end_d] if end_d in closes.index else None
    if b is None:
        return {}
    m = (b / a - 1).replace([np.inf, -np.inf], np.nan).dropna()
    mkt = float(m.mean()) if len(m) else np.nan
    excess = stock - mkt
    upd = {"last_date": str(end_d.date()), "last_price": c1 / cf, "stock_ret_pct": 100 * stock, "mkt_ret_pct": 100 * mkt,
           "excess_pct": 100 * excess, "days_held": end_i + 1}
    if len(bars) >= hold:
        right = excess < 0
        upd.update(status="Correct" if right else "Wrong", outcome="Win" if right else "Loss",
                   exit_date=str(end_d.date()),
                   note=f"stock {100 * stock:+.1f}% vs average stock {100 * mkt:+.1f}% over {hold} sessions")
    else:
        upd.update(status="Open", note=f"running: day {end_i + 1} of {hold}; so far {100 * excess:+.1f}% vs average")
    return upd


def _rate(num, den):
    return (num / den) if den else np.nan


def summarize(T: pd.DataFrame) -> dict:
    """Headline numbers and breakdown tables for the Excel sheet."""
    if T.empty:
        return {"table": T, "headline": {}, "by_group": pd.DataFrame(), "by_month": pd.DataFrame()}
    T = T.copy()
    T["closed"] = T["status"].isin(CLOSED_STATES)
    T["win"] = T["outcome"].eq("Win")
    T["tgt"] = T["status"].eq("Target hit")
    # a "finished group" = all calls of one date, timeframe and kind are done -> no early-close bias
    running = T["status"].isin(OPEN_STATES)
    open_groups = set(map(tuple, T.loc[running, ["signal_date", "timeframe", "kind"]].drop_duplicates().to_numpy()))
    T["mature"] = T["closed"] & np.array([(a, b, c) not in open_groups for a, b, c in
                                          zip(T["signal_date"], T["timeframe"], T["kind"])], dtype=bool)
    tf_lab = {"ST": "2 weeks", "MT": "3 months", "LT": "1 year"}

    def block(g: pd.DataFrame) -> dict:
        c = g[g["closed"]]
        m = g[g["mature"]]
        trades = c[c["kind"] != "AVOID"]
        return {"calls": len(g), "closed": len(c), "running": int(g["status"].isin(OPEN_STATES).sum()),
                "mature": len(m), "success_rate_mature": _rate(m["win"].sum(), len(m)),
                "not_triggered": int(g["status"].eq("Not triggered").sum()),
                "success_rate": _rate(c["win"].sum(), len(c)),
                "target_hit_rate": _rate(trades["tgt"].sum(), len(trades)) if len(trades) else np.nan,
                "avg_ai_conf": trades["ai_conf"].mean() if len(trades) else np.nan,
                "avg_result_pct": trades["result_pct"].mean() if len(trades) else np.nan,
                "avg_excess_pct": c.loc[c["kind"] == "AVOID", "excess_pct"].mean() if (c["kind"] == "AVOID").any()
                else np.nan}

    head = block(T)
    head["buy"] = block(T[T["kind"] == "BUY"])
    head["avoid"] = block(T[T["kind"] == "AVOID"])
    head["short"] = block(T[T["kind"] == "SHORT"])
    head["first_date"], head["last_date"] = T["signal_date"].min(), T["signal_date"].max()
    rows = []
    for (kind, tf, typ), g in T.groupby(["kind", "timeframe", "call_type"], sort=False):
        rows.append({"Call": typ, "Timeframe": tf_lab.get(tf, tf), "kind": kind, "tf": tf, **block(g)})
    order = {"BUY": 0, "SHORT": 1, "AVOID": 2}
    G = pd.DataFrame(rows)
    G = G.assign(o1=G["kind"].map(order), o2=G["tf"].map({"ST": 0, "MT": 1, "LT": 2})).sort_values(["o1", "o2", "Call"])
    T["month"] = pd.to_datetime(T["signal_date"]).dt.to_period("M").astype(str)
    mrows = [{"Month": m, **block(g)} for m, g in T.groupby("month")]
    return {"table": T, "headline": head, "by_group": G.drop(columns=["o1", "o2"]), "by_month": pd.DataFrame(mrows)}
