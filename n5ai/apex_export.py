"""Files for the Oracle APEX 'AI Research' popup.

Written after the Excel of a daily run (the GitHub job publishes them on its 'results' branch; the APEX package
AIR_PKG downloads and loads them). Compact JSON: {"columns": [...], "rows": [[...], ...]} per table.

  signals.json   every Nifty 500 stock x 3 timeframes - the 25 columns of the 'Signals' sheet (+ rank, tf, side)
  calls.json     every tracked Buy / Short / Avoid call with its current status and result (success-rate base)
  success.json   headline success rates, breakdown by call type / timeframe and by month, daily history

Percentages are written in percent units (63.4 = 63.4%), prices in rupees rounded to 2 decimals.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from .common import TF_RULES

SCHEMA = 1
TF_LABEL = {"ST": "2 weeks", "MT": "3 months", "LT": "1 year"}


def _num(v, nd=4):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    r = round(f, nd)
    return int(r) if nd == 0 else r


def _pct(v, nd=2):
    f = _num(v, 8)
    return None if f is None else round(100.0 * f, nd)


def _int(v):
    f = _num(v, 6)
    return None if f is None else int(round(f))


def _txt(v, max_bytes=3500):
    if v is None:
        return None
    if isinstance(v, float) and not math.isfinite(v):
        return None
    s = str(v)
    if s.lower() in ("nan", "none", "nat"):
        return None
    b = s.encode("utf-8")
    if len(b) > max_bytes:                           # Oracle VARCHAR2(4000) is a byte limit
        s = b[:max_bytes].decode("utf-8", "ignore")
    return s


def _date(v):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return None
    try:
        return str(pd.Timestamp(v).date())
    except (ValueError, TypeError):
        return None


# column name, converter  (order = order of the JSON arrays; the APEX loader reads them by position)
SIGNAL_COLUMNS = [
    ("rank", _int), ("symbol", _txt), ("tf", _txt), ("company", _txt), ("sector", _txt), ("timeframe", _txt),
    ("ai_recommendation", _txt), ("recommendation_type", _txt), ("conf_buy_pct", None), ("conf_short_pct", None),
    ("chance_profit_pct", None), ("exp_return_pct", None), ("edge_pts", None), ("current_price", None),
    ("atr14", None), ("entry_price", None), ("target_price", None), ("stop_loss", None),
    ("potential_profit_pct", None), ("potential_loss_pct", None), ("max_days", _int), ("typical_days", _int),
    ("current_trend", _txt), ("evidence", _txt), ("chart_patterns", _txt), ("results_check", _txt),
    ("remarks", _txt), ("side", _int)]

CALL_COLUMNS = [
    "signal_date", "symbol", "company", "tf", "timeframe", "kind", "call_type", "side", "close", "atr", "entry_limit",
    "target", "stop", "max_days", "ai_conf_pct", "win_prob_pct", "exp_ret_pct", "base_hit_pct", "status", "outcome",
    "note", "entry_date", "entry_price", "exit_date", "exit_price", "result_pct", "days_held", "last_date",
    "last_price", "stock_ret_pct", "mkt_ret_pct", "excess_pct", "repeat_count", "last_seen", "evidence"]

GROUP_COLUMNS = ["grp_type", "grp_key", "call_label", "kind", "tf", "timeframe", "calls", "completed", "running",
                 "not_triggered", "success_rate_pct", "target_hit_rate_pct", "avg_ai_conf_pct", "avg_result_pct",
                 "avg_excess_pct", "mature", "success_rate_mature_pct", "sort_order"]

DAILY_COLUMNS = ["as_of_date", "calls", "completed", "running", "not_triggered", "success_rate_pct", "mature",
                 "success_rate_mature_pct", "buy_completed", "buy_success_rate_pct", "buy_target_hit_rate_pct",
                 "buy_avg_ai_conf_pct", "buy_avg_result_pct", "avoid_completed", "avoid_success_rate_pct",
                 "avoid_avg_excess_pct", "short_completed", "short_success_rate_pct"]


def signal_rows(rows: list[dict]) -> list[list]:
    out = []
    for r in rows:
        v = {"rank": r.get("rank"), "symbol": r.get("symbol"), "tf": r.get("tf"), "company": r.get("company"),
             "sector": r.get("sector"), "timeframe": r.get("timeframe"), "ai_recommendation": r.get("ai_recommendation"),
             "recommendation_type": r.get("recommendation_type"), "conf_buy_pct": _pct(r.get("conf_buy")),
             "conf_short_pct": _pct(r.get("conf_short")), "chance_profit_pct": _pct(r.get("chance_profit")),
             "exp_return_pct": _pct(r.get("exp_return"), 3), "edge_pts": _num(r.get("edge_pts"), 2),
             "current_price": _num(r.get("current_price"), 2), "atr14": _num(r.get("atr14"), 2),
             "entry_price": _num(r.get("entry_price"), 2), "target_price": _num(r.get("target_price"), 2),
             "stop_loss": _num(r.get("stop_loss"), 2), "potential_profit_pct": _pct(r.get("potential_profit")),
             "potential_loss_pct": _pct(r.get("potential_loss")), "max_days": r.get("max_days"),
             "typical_days": r.get("typical_days"), "current_trend": r.get("current_trend"),
             "evidence": r.get("evidence"), "chart_patterns": r.get("chart_patterns"),
             "results_check": r.get("results_check"), "remarks": r.get("remarks"), "side": r.get("side")}
        row = []
        for name, conv in SIGNAL_COLUMNS:
            x = v[name]
            row.append(conv(x) if conv else x)
        out.append(row)
    return out


def call_rows(T: pd.DataFrame) -> list[list]:
    out = []
    for _, x in T.iterrows():
        tf = x.get("timeframe")
        out.append([_date(x.get("signal_date")), _txt(x.get("symbol"), 40), _txt(x.get("company"), 200), _txt(tf, 4),
                    TF_LABEL.get(tf, tf), _txt(x.get("kind"), 10), _txt(x.get("call_type"), 60), _int(x.get("side")),
                    _num(x.get("close"), 2), _num(x.get("atr"), 2), _num(x.get("entry_limit"), 2),
                    _num(x.get("target"), 2), _num(x.get("stop"), 2), _int(x.get("max_days")),
                    _pct(x.get("ai_conf")), _pct(x.get("win_prob")), _pct(x.get("exp_ret"), 3),
                    _pct(x.get("base_hit")), _txt(x.get("status"), 30), _txt(x.get("outcome"), 10),
                    _txt(x.get("note"), 1000), _date(x.get("entry_date")), _num(x.get("entry_price"), 2),
                    _date(x.get("exit_date")), _num(x.get("exit_price"), 2), _num(x.get("result_pct"), 3),
                    _int(x.get("days_held")), _date(x.get("last_date")), _num(x.get("last_price"), 2),
                    _num(x.get("stock_ret_pct"), 3), _num(x.get("mkt_ret_pct"), 3), _num(x.get("excess_pct"), 3),
                    _int(x.get("repeat_count")), _date(x.get("last_seen")), _txt(x.get("evidence"), 1000)])
    return out


def _block_row(grp_type, key, label, kind, tf, blk: dict, order: int) -> list:
    return [grp_type, key, label, kind, tf, TF_LABEL.get(tf, tf) if tf else None, _int(blk.get("calls")),
            _int(blk.get("closed")), _int(blk.get("running")), _int(blk.get("not_triggered")),
            _pct(blk.get("success_rate"), 1), _pct(blk.get("target_hit_rate"), 1), _pct(blk.get("avg_ai_conf"), 1),
            _num(blk.get("avg_result_pct"), 3), _num(blk.get("avg_excess_pct"), 3), _int(blk.get("mature")),
            _pct(blk.get("success_rate_mature"), 1), order]


def success_rows(summary: dict) -> list[list]:
    hd = summary.get("headline") or {}
    rows = []
    if not hd:
        return rows
    rows.append(_block_row("OVERALL", "ALL", "All tracked calls", None, None, hd, 1))
    for k, (kind, lab) in enumerate((("BUY", "Buy calls"), ("AVOID", "Avoid / Reduce calls"), ("SHORT", "Short sells")),
                                    2):
        blk = hd.get(kind.lower()) or {}
        if blk.get("calls"):
            rows.append(_block_row("KIND", kind, lab, kind, None, blk, k))
    G = summary.get("by_group")
    if G is not None and len(G):
        for k, (_, g) in enumerate(G.iterrows(), 10):
            rows.append(_block_row("GROUP", f"{g['kind']}|{g['tf']}|{g['Call']}", g["Call"], g["kind"], g["tf"],
                                   g.to_dict(), k))
    Mo = summary.get("by_month")
    if Mo is not None and len(Mo):
        for k, (_, g) in enumerate(Mo.iterrows(), 500):
            rows.append(_block_row("MONTH", g["Month"], g["Month"], None, None, g.to_dict(), k))
    return rows


def daily_rows(H: pd.DataFrame) -> list[list]:
    out = []
    for _, x in H.iterrows():
        out.append([_date(x["as_of_date"]), _int(x["calls"]), _int(x["completed"]), _int(x["running"]),
                    _int(x["not_triggered"]), _pct(x["success_rate"], 1), _int(x["mature"]),
                    _pct(x["success_rate_mature"], 1), _int(x["buy_completed"]), _pct(x["buy_success_rate"], 1),
                    _pct(x["buy_target_hit_rate"], 1), _pct(x["buy_avg_ai_conf"], 1),
                    _num(x["buy_avg_result_pct"], 3), _int(x["avoid_completed"]), _pct(x["avoid_success_rate"], 1),
                    _num(x["avoid_avg_excess_pct"], 3), _int(x["short_completed"]), _pct(x["short_success_rate"], 1)])
    return out


def _dump(path: Path, obj: dict):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    tmp.replace(path)


def write(out_dir: Path, session_date, signals: list[dict], tracker, generated_utc: str) -> dict:
    """Write signals.json, calls.json and success.json; returns counts and the headline for status.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sd = str(pd.Timestamp(session_date).date())
    head = {"schema": SCHEMA, "session_date": sd, "generated_utc": generated_utc}
    s_rows = signal_rows(signals)
    _dump(out_dir / "signals.json", {**head, "kind": "signals", "count": len(s_rows),
                                     "columns": [c for c, _ in SIGNAL_COLUMNS], "rows": s_rows})
    T = tracker.table()
    c_rows = call_rows(T)
    _dump(out_dir / "calls.json", {**head, "kind": "calls", "count": len(c_rows), "columns": CALL_COLUMNS,
                                   "rows": c_rows})
    summ = tracker.summary()
    g_rows = success_rows(summ)
    d_rows = daily_rows(tracker.success_history())
    rules = [[tf, lab, sm, tm, cm, hold] for tf, (lab, sm, tm, cm, hold) in TF_RULES.items()]
    _dump(out_dir / "success.json", {**head, "kind": "success", "count": len(g_rows), "columns": GROUP_COLUMNS,
                                     "rows": g_rows, "daily_count": len(d_rows), "daily_columns": DAILY_COLUMNS,
                                     "daily_rows": d_rows,
                                     "rules_columns": ["tf", "label", "stop_atr", "target_atr", "no_chase_atr",
                                                       "max_days"], "rules": rules})
    hd = summ.get("headline") or {}
    headline = {"success_rate_pct": _pct(hd.get("success_rate"), 1), "completed": _int(hd.get("closed")),
                "success_rate_mature_pct": _pct(hd.get("success_rate_mature"), 1), "mature": _int(hd.get("mature")),
                "running": _int(hd.get("running")), "calls": _int(hd.get("calls")),
                "first_date": hd.get("first_date"), "last_date": hd.get("last_date")}
    return {"counts": {"signals": len(s_rows), "calls": len(c_rows), "success": len(g_rows), "daily": len(d_rows)},
            "headline": headline}
