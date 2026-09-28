"""Daily engine: indicators + chart patterns + signals for every stock, cross-sectional ranks, market
breadth, AI model scores and the final Buy / Avoid / Hold call per stock and timeframe.

It is the v2 research pipeline (v2_build panel stage + v2_model scoring + v2_decide rules) restricted to
the latest session, so a daily run gives exactly the calls the research workbook would give.
"""
from __future__ import annotations

import json
import logging
import re
import time
import traceback
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from . import features as vf
from . import patterns as vp
from . import signals as vs
from .common import TF_RULES

log = logging.getLogger("n5ai.engine")

PATTERN_DECAY = 10.0
SIG_KEEP_BARS = 15           # signal events kept per stock (the longest "active" window is 10 sessions)
EV_KEEP_BARS = 60            # chart-pattern events kept per stock (charts, active-pattern sheet)
CORE_EXTRA = ("mom_12_1", "excess_ret_63", "excess_ret_21", "dist_52w_high", "atr_pct_14", "vol_ratio_20",
              "deliv_pct_z60", "log_turnover_20", "mansfield_rs", "dist_sma200", "dist_sma50", "ta_RSI",
              "hv_60", "beta_60", "deliv_pct_5_minus_20", "updown_vol_50", "linreg_slope_100", "idio_vol_60",
              "maxdd_252", "ta_ADX")
XS_RANK_COLS = ("ret_21", "ret_63", "ret_126", "mom_12_1", "excess_ret_63", "dist_52w_high", "atr_pct_14",
                "vol_ratio_20", "deliv_pct_z60", "log_turnover_20", "mansfield_rs", "dist_sma200", "ta_RSI", "hv_60",
                "beta_60", "deliv_pct_5_minus_20", "updown_vol_50", "linreg_slope_100", "idio_vol_60", "maxdd_252",
                "ta_ADX", "ret_1")
# fields of the current row that the report needs besides the model features
REPORT_FIELDS = ("_close", "_atr14", "dist_ema20", "ta_SUPERTREND_trend", "dist_sma50", "slope_sma50", "dist_sma200",
                 "slope_sma200", "ta_RSI", "dist_52w_high", "deliv_pct_5_minus_20", "atr_pct_14", "log_turnover_20")


def pat_id(name: str) -> str:
    return "pat_" + re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:60]


# ------------------------------------------------------------------------------------------------------------
# per-stock work (runs in worker processes)
# ------------------------------------------------------------------------------------------------------------
_W: dict = {}


def _init_worker(n50: pd.DataFrame, vocab_ids: dict):
    warnings.filterwarnings("ignore")
    _W["n50"], _W["vocab_ids"] = n50, vocab_ids


def core_frame(sym: str, F: pd.DataFrame) -> pd.DataFrame:
    c = F["_close"].astype(float)
    core = pd.DataFrame({
        "symbol": sym, "close": c, "ret_1": F["ret_1"], "ret_21": F["ret_21"], "ret_63": F["ret_63"],
        "ret_126": F["ret_126"], "ret_189": 100 * (c / c.shift(189) - 1), "ret_252": F["ret_252"],
        "above50": (c > F["_sma50"]).astype(np.float32), "above200": (c > F["_sma200"]).astype(np.float32),
        "nh": F["new_52w_high"], "nl": F["new_52w_low"],
        "adv": (F["ret_1"] > 0).astype(np.float32), "dec": (F["ret_1"] < 0).astype(np.float32),
        "rsi70": (F["ta_RSI"] > 70).astype(np.float32), "rsi30": (F["ta_RSI"] < 30).astype(np.float32),
        "turn": F["_turnover_cr_20"], "atr": F["_atr14"],
    }, index=F.index)
    core["bars"] = np.arange(len(core))
    for col in CORE_EXTRA:
        core[col] = F[col]
    return core


def pattern_features_last(E: pd.DataFrame, n: int, vocab_ids: dict) -> dict:
    """Pattern features of the LAST bar, identical to the research panel (decaying 'age' + 5/20-bar counts)."""
    out = {v: 0.0 for v in vocab_ids.values()}
    agg = {"bull": E.iloc[0:0], "bear": E.iloc[0:0], "bull_classic": E.iloc[0:0], "bear_classic": E.iloc[0:0]}
    if not E.empty:
        for nm, grp in E.groupby("pattern"):
            if nm in vocab_ids:
                age = (n - 1) - int(grp["t"].max())
                out[vocab_ids[nm]] = float(np.exp(-age / PATTERN_DECAY)) if age <= 30 else 0.0
        classic = ~E["family"].isin(["Range contraction", "Divergence", "Reversal bars", "Volume patterns"])
        agg = {"bull": E[E["dir"] > 0], "bear": E[E["dir"] < 0], "bull_classic": E[(E["dir"] > 0) & classic],
               "bear_classic": E[(E["dir"] < 0) & classic]}
    for key, sub in agg.items():
        t = sub["t"].to_numpy() if len(sub) else np.array([], dtype=int)
        out[f"patagg_{key}_5"] = float((t >= n - 5).sum())
        out[f"patagg_{key}_20"] = float((t >= n - 20).sum())
    out["patagg_net_20"] = out["patagg_bull_20"] - out["patagg_bear_20"]
    return out


def _stock_job(args):
    sym, df = args
    try:
        F = vf.compute_features(df, _W["n50"])
        E = vp.detect_patterns(df, F)
        n = len(F)
        core = core_frame(sym, F)
        num = F.iloc[-1]
        last = {k: num[k] for k in F.columns}
        pf = pattern_features_last(E, n, _W["vocab_ids"])
        lo = max(0, n - SIG_KEEP_BARS)
        sig = []
        for name, fam, d, m in vs.signals(F):
            for t in np.flatnonzero(m[lo:]) + lo:
                sig.append((int(t), F.index[t], sym, name, fam, int(d)))
        ev = E[E["t"] >= n - EV_KEEP_BARS].copy() if not E.empty else E
        if not ev.empty:
            ev["symbol"] = sym
        return sym, core, last, pf, sig, ev, None
    except Exception:  # pragma: no cover - reported to the caller
        return sym, None, None, None, None, None, traceback.format_exc()


# ------------------------------------------------------------------------------------------------------------
# cross-sectional + market features (main process)
# ------------------------------------------------------------------------------------------------------------
def ema_series(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def market_features(core: pd.DataFrame, bench: dict) -> pd.DataFrame:
    g = core.groupby("date")
    M = pd.DataFrame({
        "mkt_pct_above50": g["above50"].mean(), "mkt_pct_above200": g["above200"].mean(),
        "mkt_nh_nl": g["nh"].mean() - g["nl"].mean(),
        "mkt_ad_ratio": g["adv"].sum() / (g["adv"].sum() + g["dec"].sum()).replace(0, np.nan),
        "mkt_median_ret21": g["ret_21"].median(), "mkt_dispersion21": g["ret_21"].std(),
        "mkt_pct_rsi70": g["rsi70"].mean(), "mkt_pct_rsi30": g["rsi30"].mean(),
        "mkt_n_stocks": g.size(),
    }).sort_index()
    rana = 1000 * (2 * M["mkt_ad_ratio"] - 1)
    M["mkt_ad_10"] = M["mkt_ad_ratio"].rolling(10).mean()
    M["mkt_mcclellan"] = ema_series(rana, 19) - ema_series(rana, 39)
    M["mkt_pct_above50_chg10"] = M["mkt_pct_above50"] - M["mkt_pct_above50"].shift(10)
    n50 = bench["NIFTY50"]["close"].reindex(M.index).ffill()
    for k in (1, 5, 21, 63, 126):
        M[f"n50_ret_{k}"] = 100 * (n50 / n50.shift(k) - 1)
    s50, s200 = n50.rolling(50).mean(), n50.rolling(200).mean()
    M["n50_dist_sma50"], M["n50_dist_sma200"] = 100 * (n50 / s50 - 1), 100 * (n50 / s200 - 1)
    d = n50.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    M["n50_rsi14"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    M["n50_hv20"] = np.log(n50).diff().rolling(20).std() * np.sqrt(252) * 100
    M["n50_dd252"] = 100 * (n50 / n50.rolling(252, min_periods=100).max() - 1)
    for nm, tag in (("SMALLCAP250", "sc"), ("MIDCAP150", "mc")):
        x = bench[nm]["close"].reindex(M.index).ffill()
        for k in (21, 63):
            M[f"{tag}_vs_n50_{k}"] = 100 * ((x / x.shift(k)) - (n50 / n50.shift(k)))
    M["regime"] = np.select([(n50 > s50) & (n50 > s200), (n50 < s50) & (n50 < s200)], [1.0, -1.0], 0.0)
    M = M.astype(np.float32)
    M["n50_close"] = n50.astype(float)              # for the report only (not a model feature)
    return M


def cross_sectional_last(core_last: pd.DataFrame) -> pd.DataFrame:
    """Same cross-sectional features as the research panel, for the rows of one date."""
    c = core_last.copy()
    w = {"ret_63": 0.4, "ret_126": 0.2, "ret_189": 0.2, "ret_252": 0.2}
    num = sum(v * c[k].fillna(0) for k, v in w.items())
    den = sum(v * c[k].notna() for k, v in w.items())
    cnt = sum(c[k].notna().astype(int) for k in w)
    c["rs_raw"] = (num / den.replace(0, np.nan)).where(cnt >= 3)
    X = pd.DataFrame({"symbol": c["symbol"].to_numpy()}, index=c.index)
    X["xs_rs_rating"] = c["rs_raw"].rank(pct=True).mul(99).round().clip(1, 99)
    for col in XS_RANK_COLS:
        X[f"xs_rank_{col}"] = c[col].rank(pct=True)
    gs = c.groupby("industry")
    sec21, sec63 = gs["ret_21"].transform("median"), gs["ret_63"].transform("median")
    X["sec_ret21_med"], X["sec_ret63_med"] = sec21, sec63
    X["rel_sec_ret21"], X["rel_sec_ret63"] = c["ret_21"] - sec21, c["ret_63"] - sec63
    X["sec_breadth50"] = gs["above50"].transform("mean")
    X["sec_size"] = gs["symbol"].transform("count")
    sec_tab = c.groupby("industry")["ret_63"].median()
    X["sec_rs_rank"] = c["industry"].map(sec_tab.rank(pct=True)).to_numpy()
    X["eligible"] = c["eligible"].to_numpy()
    X["industry"] = c["industry"].to_numpy()
    X["rs_raw"] = c["rs_raw"].to_numpy()
    num_cols = [k for k in X.columns if k not in ("symbol", "industry", "eligible")]
    X[num_cols] = X[num_cols].astype(np.float32)
    return X.set_index("symbol")


# ------------------------------------------------------------------------------------------------------------
# calls
# ------------------------------------------------------------------------------------------------------------
def decide(cur: pd.DataFrame, calib: dict, regime_now: int, A: pd.DataFrame) -> pd.DataFrame:
    """Port of v2_decide.main(): model rank bucket + proven signals -> call, confidence and trade levels."""
    edges, top, net_thr, window = calib["edges"], calib["top"], calib["net_thr"], calib["signal_window"]
    tabs, sigs, days = calib["tabs"][regime_now], calib["signals"][regime_now], calib["days"]
    el = cur["eligible"].astype(bool)
    rows = []
    for tf, (lab, sm, tm, cm, hold) in TF_RULES.items():
        qL = cur[f"p_{tf}L"].where(el).rank(pct=True)
        qS = cur[f"p_{tf}S"].where(el).rank(pct=True)
        TL, TS = tabs[(tf, "L")]["table"], tabs[(tf, "S")]["table"]
        bL, bS = tabs[(tf, "L")]["base"], tabs[(tf, "S")]["base"]
        Aw = A[A["age"] < window[tf]]
        by_sym = {s: g for s, g in Aw.groupby("symbol")}
        for i, x in cur.iterrows():
            sym = x["symbol"]
            r = {"symbol": sym, "tf": tf, "timeframe": lab, "close": float(x["_close"]), "atr": float(x["_atr14"]),
                 "hold": hold, "sm": sm, "tm": tm, "cm": cm, "eligible": bool(x["eligible"]),
                 "base_hit_buy": bL["y"], "base_hit_short": bS["y"], "base_ret_buy": bL["ret"],
                 "base_ret_short": bS["ret"], "base_win_buy": bL["win"], "evidence": ""}
            if not x["eligible"] or not np.isfinite(qL[i]):
                r.update(rec="Not rated", type="Not rated", side=0)
                rows.append(r)
                continue
            kL = int(np.clip(np.searchsorted(edges, qL[i], side="right") - 1, 0, 5))
            kS = int(np.clip(np.searchsorted(edges, qS[i], side="right") - 1, 0, 5))
            a, b = TL.iloc[kL], TS.iloc[kS]
            pos, neg = [], []
            g = by_sym.get(sym)
            if g is not None:
                for _, e in g.iterrows():
                    s_ = sigs[tf].get((e["pattern"], int(e["dir"])))
                    if s_ is None:
                        continue
                    eff, ex, hx, t, src, verdict = s_
                    item = (e["pattern"], int(e["dir"]), ex, hx, t, verdict, e["date"])
                    (pos if eff > 0 else neg).append(item)
            pos = sorted({p[0]: p for p in pos}.values(), key=lambda z: -z[2])
            neg = sorted({p[0]: p for p in neg}.values(), key=lambda z: -z[2])
            net_r = sum(p[2] for p in pos) - sum(p[2] for p in neg)
            net_h = sum(p[3] for p in pos) - sum(p[3] for p in neg)
            net_r = float(np.clip(net_r, -3 * net_thr[tf], 3 * net_thr[tf]))
            net_h = float(np.clip(net_h, -0.10, 0.10))
            conf_buy = float(np.clip(a["y"] + net_h, 0.01, 0.99))
            exp_buy = float(a["ret"] + net_r)
            win_buy = float(np.clip(a["win"] + 0.5 * net_h, 0.01, 0.99))
            s_y, s_w, s_r = float(b["y"]), float(b["win"]), float(b["ret"])
            r.update(pct_buy=float(qL[i]), pct_short=float(qS[i]), bucket_buy=a["label"], bucket_short=b["label"],
                     conf_buy=conf_buy, conf_short=s_y, win_buy=win_buy, win_short=s_w,
                     exp_ret_buy=exp_buy, exp_ret_short=s_r,
                     edge_buy=conf_buy - bL["y"], edge_short=s_y - bS["y"],
                     model_ex_buy=float(a["ex_ret"]), model_ex_short=float(b["ex_ret"]),
                     pos_signals=pos, neg_signals=neg, net_evidence=net_r)
            long_strong = kL == 5 and a["edge_ok"]
            long_model = kL in top and a["edge_ok"]
            avoid_model = kS in top and b["edge_ok"]
            short_ok = kS in top and b["edge_ok"] and b["abs_ok"] and tf != "LT"
            sig_pos, sig_neg = net_r >= net_thr[tf], net_r <= -net_thr[tf]
            if long_strong and net_r >= 0 and not avoid_model:
                rec, typ, side = "Buy", "Strong Buy", 1
            elif ((long_model and net_r >= 0) or (sig_pos and kL > 0)) and not avoid_model:
                rec, typ, side = ("Accumulate", "Accumulate (buy on dips)", 1) if tf == "LT" else ("Buy", "Buy", 1)
            elif short_ok and net_r <= 0 and not long_model:
                rec, typ, side = "Sell", ("Strong Short Sell" if kS == 5 else "Short Sell"), -1
            elif (avoid_model and net_r <= 0) or (sig_neg and not long_model):
                rec, typ, side = (("Reduce", "Reduce / Avoid", 0) if tf == "LT"
                                  else ("Avoid", "Avoid (likely underperformer)", 0))
            elif (avoid_model and sig_pos) or (long_model and sig_neg):
                rec, typ, side = "Hold", "Hold (mixed signals)", 0
            else:
                rec, typ, side = "Hold", "Hold / No fresh trade", 0
            r.update(rec=rec, type=typ, side=side, avoid_model=bool(avoid_model), long_model=bool(long_model),
                     med_days=days[(tf, "L" if side >= 0 else "S")])
            c, at = r["close"], r["atr"]
            if side > 0:
                r.update(entry=c + cm * at, target=c + tm * at, stop=c - sm * at)
            elif side < 0:
                r.update(entry=c - cm * at, target=c - tm * at, stop=c + sm * at)
            rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------------------
# public entry point
# ------------------------------------------------------------------------------------------------------------
def load_model(model_dir: Path) -> dict:
    mdl = {"feats": json.load(open(model_dir / "features.json")),
           "industries": json.load(open(model_dir / "industries.json")),
           "vocab": pd.read_csv(model_dir / "pattern_vocab.csv").iloc[:, 0].astype(str).tolist(),
           "calib": pd.read_pickle(model_dir / "calibration.pkl"), "models": {}}
    for tag in ("STL", "STS", "MTL", "MTS", "LTL", "LTS"):
        mdl["models"][tag] = pd.read_pickle(model_dir / f"final_abs_{tag}.pkl")
    return mdl


def run(prices: dict, bench: dict, universe: pd.DataFrame, model_dir: Path, asof: pd.Timestamp | None = None,
        workers: int = 2) -> dict:
    """Compute the calls for the latest session in `prices` (or for `asof`)."""
    t0 = time.time()
    mdl = load_model(model_dir)
    if asof is not None:
        prices = {s: d[d.index <= asof] for s, d in prices.items()}
        bench = {k: v[v.index <= asof] for k, v in bench.items()}
    uni = universe[universe["status"] == "ok"]
    industry = dict(zip(uni["symbol"], uni["industry"]))
    syms = [s for s in uni["symbol"] if s in prices and len(prices[s]) >= 60]
    vocab_ids = {nm: pat_id(nm) for nm in mdl["vocab"]}
    jobs = [(s, prices[s]) for s in syms]
    cores, lasts, pfs, sig_rows, evs, errors = [], {}, {}, [], [], {}
    log.info("engine: %d stocks, %d worker(s)", len(jobs), workers)
    with Pool(max(1, workers), initializer=_init_worker, initargs=(bench["NIFTY50"], vocab_ids)) as pool:
        for k, (sym, core, last, pf, sig, ev, err) in enumerate(pool.imap_unordered(_stock_job, jobs, chunksize=4), 1):
            if err:
                errors[sym] = err
                log.warning("stock %s failed:\n%s", sym, err)
                continue
            cores.append(core)
            lasts[sym], pfs[sym] = last, pf
            sig_rows.extend(sig)
            if ev is not None and not ev.empty:
                evs.append(ev)
            if k % 100 == 0:
                log.info("  %d/%d stocks  %.0fs", k, len(jobs), time.time() - t0)
    core = pd.concat(cores)
    core.index.name = "date"
    core = core.reset_index()
    core["industry"] = core["symbol"].map(industry).fillna("OTHER")
    core["eligible"] = (core["bars"] >= 209) & (core["turn"] >= 10) & core["atr"].notna()
    M = market_features(core, bench)
    last_date = M.index.max()
    X = cross_sectional_last(core[core["date"] == last_date])
    # ---- current rows: last-bar features + pattern features + cross-sectional + market ----------------------
    rows = []
    for sym in X.index:
        r = dict(lasts[sym])
        r.update(pfs[sym])
        r.update(X.loc[sym].to_dict())
        r["symbol"], r["date"] = sym, last_date
        rows.append(r)
    cur = pd.DataFrame(rows)
    mrow = M.loc[last_date]
    for col in M.columns:
        if col != "n50_close":
            cur[col] = mrow[col]
    cur["eligible"] = cur["eligible"].fillna(False).astype(bool)
    cur["industry_code"] = pd.Categorical(cur["industry"], categories=mdl["industries"]).codes.astype(np.float32)
    feats = mdl["feats"]
    missing = [f for f in feats if f not in cur.columns]
    if missing:
        log.warning("%d model features missing (set to NaN): %s", len(missing), missing[:10])
        for f in missing:
            cur[f] = np.nan
    Xc = cur[feats].to_numpy(np.float32)
    for tag, m in mdl["models"].items():
        cur[f"p_{tag}"] = m.predict_proba(Xc)[:, 1]
    # ---- active events (chart patterns + signals) ---------------------------------------------------------
    dates = M.index
    since = dates[-max(mdl["calib"]["signal_window"].values())]
    EV = pd.concat(evs, ignore_index=True) if evs else pd.DataFrame(
        columns=["t", "pattern", "family", "dir", "scale", "level", "target", "stop", "date", "close", "symbol"])
    SE = pd.DataFrame(sig_rows, columns=["t", "date", "symbol", "pattern", "family", "dir"])
    A = pd.concat([EV.loc[EV["date"] >= since, ["date", "symbol", "pattern", "dir"]],
                   SE.loc[SE["date"] >= since, ["date", "symbol", "pattern", "dir"]]], ignore_index=True)
    pos = {d: i for i, d in enumerate(dates)}
    A["age"] = len(dates) - 1 - A["date"].map(pos)
    regime_now = int(M["regime"].iloc[-1])
    D = decide(cur, mdl["calib"], regime_now, A)
    elig = dict(zip(cur["symbol"], cur["eligible"]))
    if not EV.empty:
        EV["eligible"] = EV["symbol"].map(elig).fillna(False)
    log.info("engine done for %s in %.0fs: regime %+d, %d stocks rated, %d failed", last_date.date(),
             time.time() - t0, regime_now, int(cur["eligible"].sum()), len(errors))
    return {"date": last_date, "decisions": D, "current": cur, "market": M, "events": EV, "signal_events": SE,
            "regime": regime_now, "calib": mdl["calib"], "errors": errors, "prices": prices, "bench": bench,
            "seconds": time.time() - t0}
