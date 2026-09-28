#!/usr/bin/env python3
"""Nifty 500 AI signals - daily job.

Run once a day after the NSE close. Each run:
  1. downloads every new trading session (NSE bhavcopy + index closes; GitHub mirrors as fallback) and adjusts the
     price history for splits / bonuses / demergers,
  2. re-checks all earlier calls against the new prices (target / stop / time exit; avoid calls vs the average stock),
  3. computes today's calls for all Nifty 500 stocks (same AI model + proven-signal rules as the research workbook),
  4. stores the new calls with their date in data/recommendations.db,
  5. writes output/Nifty500_AI_Signals_<date>.xlsx - the last sheet 'Success Rate' shows how the calls have done.

Usage (run from any folder):
  python nifty500_daily.py                    normal daily run
  python nifty500_daily.py --status           print the success rate and exit
  python nifty500_daily.py --report-only      rebuild the Excel for the latest session without downloading
  python nifty500_daily.py --no-download      use only data already on disk (or files dropped in data/incoming)
  python nifty500_daily.py --asof 2026-10-01  compute and log the calls for a missed past session
  python nifty500_daily.py --force            recompute today's calls even if they already exist
  python nifty500_daily.py --download-only    only bring the price history up to date (used by the GitHub job)
  python nifty500_daily.py --no-report        store the calls but skip the Excel (catch-up of missed sessions)
  python nifty500_daily.py --apex-export DIR  also write the JSON files for the APEX 'AI Research' popup to DIR

Exit codes: 0 = OK, 1 = failed (see logs/), 2 = another run is still active.
Research / paper-trading tool - not investment advice.
"""
from __future__ import annotations

import argparse
import configparser
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from n5ai import apex_export  # noqa: E402
from n5ai import charts as vc  # noqa: E402
from n5ai import data_update as du  # noqa: E402
from n5ai import engine, report  # noqa: E402
from n5ai import tracker as tk  # noqa: E402

log = logging.getLogger("n5ai")
SEED_FILE = "seed_decisions_20260925.pkl"
SEED_DATE = pd.Timestamp("2026-09-25")
DEFAULTS = {
    "paths": {"data_dir": "data", "output_dir": "output", "log_dir": "logs", "model_dir": "model", "static_dir": "static"},
    "run": {"workers": "2", "keep_reports_days": "0", "keep_run_bundles": "30", "charts_st": "8", "charts_mt": "3",
            "charts_lt": "3"},
    "sources": {
        "nse_bhav_url": "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{ddmmyyyy}.csv",
        "mirror_bhav_url": "https://raw.githubusercontent.com/tilak999/NSE-Data-bank/main/data/"
                           "sec_bhavdata_full_{ddmmyyyy}.csv",
        "nse_index_url": "https://nsearchives.nseindia.com/content/indices/ind_close_all_{ddmmyyyy}.csv",
        "mirror_index_url": "https://raw.githubusercontent.com/Chaudharyrohit2506/nse-daily-ohlcv/main/data/chatgpt/"
                            "index/{file}",
        "constituents_url": "https://niftyindices.com/IndexConstituent/ind_nifty500list.csv",
        "constituents_url2": "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
        "history_url": "https://raw.githubusercontent.com/kaushikghoshindia/NSE500Tracker/main/data/stocks/"
                       "{symbol_lower}.csv",
        "http_timeout": "30", "http_retries": "3", "proxy": "", "refresh_days": "7"},
    "research": {"valid_until": "2026-11-15",
                 "note": "Latest quarterly results (Q1 FY27) were checked by web search on 27-Sep-2026 - context only."},
}


# ------------------------------------------------------------------------------------------------------------------
def load_config(path: Path) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser(interpolation=None)
    cfg.read_dict(DEFAULTS)
    if path.exists():
        cfg.read(path)
    return cfg


def resolve(cfg, key: str) -> Path:
    p = Path(cfg["paths"][key]).expanduser()
    return p if p.is_absolute() else BASE / p


def setup_logging(log_dir: Path, verbose: bool):
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    fh = logging.FileHandler(log_dir / f"nifty500_daily_{ist_now():%Y%m%d}.log")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.handlers = [fh, sh]
    for noisy in ("urllib3", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def ist_now() -> pd.Timestamp:
    """India time without depending on the server's time zone setting."""
    return pd.Timestamp.now(tz="UTC").tz_localize(None) + pd.Timedelta(hours=5, minutes=30)


class RunLock:
    """Only one run at a time (Linux flock / Windows msvcrt); the lock is released automatically if a run dies."""

    def __init__(self, path: Path):
        self.path, self.fh = path, None

    def __enter__(self):
        self.fh = open(self.path, "a+")
        try:
            if os.name == "nt":
                import msvcrt
                self.fh.seek(0)
                msvcrt.locking(self.fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.fh.close()
            raise SystemExit(2)
        return self

    def __exit__(self, *exc):
        try:
            if os.name == "nt":
                import msvcrt
                self.fh.seek(0)
                msvcrt.locking(self.fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.fh, fcntl.LOCK_UN)
        finally:
            self.fh.close()


# ------------------------------------------------------------------------------------------------------------------
def market_lines(M: pd.DataFrame, bench: dict) -> list[str]:
    m = M.iloc[-1]
    c = bench["NIFTY50"]["close"].reindex(M.index).ffill()
    last = float(c.iloc[-1])

    def chg(k):
        return float(c.iloc[-1] / c.iloc[-1 - k] - 1) if len(c) > k else float("nan")

    lines = [f"Nifty 50 closed at {last:,.2f} ({100 * chg(1):+.2f}% on the day, {100 * chg(21):+.1f}% over one month, "
             f"{100 * chg(250):+.1f}% over one year); {float(m['n50_dist_sma50']):+.1f}% from its 50-DMA and "
             f"{float(m['n50_dist_sma200']):+.1f}% from its 200-DMA.",
             f"Breadth: {100 * float(m['mkt_pct_above50']):.0f}% of Nifty 500 stocks above their 50-DMA "
             f"({100 * float(m['mkt_pct_above50_chg10']):+.0f} pts in 10 sessions) and "
             f"{100 * float(m['mkt_pct_above200']):.0f}% above their 200-DMA; {100 * float(m['mkt_ad_ratio']):.0f}% of "
             f"stocks rose on the day; 52-week highs minus lows: {100 * float(m['mkt_nh_nl']):+.1f}% of stocks.",
             f"Mid caps (Nifty Midcap 150) vs Nifty 50 over one month: {float(m['mc_vs_n50_21']):+.1f} pts; small caps "
             f"(Nifty Smallcap 250): {float(m['sc_vs_n50_21']):+.1f} pts."]
    return lines


def data_checks(store, new_sessions: list, last: pd.Timestamp) -> list[str]:
    """Plain-language notes on this run's data for the 'Read Me' sheet."""
    out = []
    if new_sessions:
        out.append("Sessions added in this run: " + ", ".join(f"{d:%d-%b-%Y}" for d in new_sessions) + ".")
    else:
        out.append(f"No new NSE session in this run; latest session on file: {last:%d-%b-%Y}.")
    for sym, ex, f in store.ca_run:
        out.append(f"Corporate action: {sym} price history adjusted x {f:.4f} (ex-date {ex}).")
    if store.meta.get("bench_filled"):
        out.append("Index closes not yet available for " + ", ".join(store.meta["bench_filled"])
                   + " - last close carried forward; replaced automatically on a later run.")
    ok = store.universe.loc[store.universe["status"] == "ok", "symbol"]
    miss = [s for s in ok if s in store.prices and len(store.prices[s]) and store.prices[s].index.max() < last]
    if miss:
        out.append(f"{len(miss)} Nifty 500 stock(s) did not trade on {last:%d-%b-%Y} (not rated today): "
                   + ", ".join(miss[:15]) + (" ..." if len(miss) > 15 else "") + ".")
    out += store.notes
    return out


def make_charts(res: dict, universe: pd.DataFrame, out_dir: Path, cfg) -> list[Path]:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    D, EV, prices = res["decisions"], res["events"], res["prices"]
    comp = dict(zip(universe["symbol"], universe["company"]))
    lab = {"ST": "Short term (2 weeks)", "MT": "Mid term (3 months)", "LT": "Long term (1 year)"}
    rank = {"Strong Buy": 0, "Buy": 1, "Accumulate (buy on dips)": 1}
    files, k = [], 0
    for tf, key in (("ST", "charts_st"), ("MT", "charts_mt"), ("LT", "charts_lt")):
        S = D[(D.tf == tf) & (D.side > 0)].copy()
        S["tr"] = S["type"].map(rank).fillna(2)
        S = S.sort_values(["tr", "exp_ret_buy", "pct_buy"], ascending=[True, False, False]).head(int(cfg["run"][key]))
        for _, d in S.iterrows():
            sym = d["symbol"]
            k += 1
            ev = EV[EV["symbol"] == sym] if len(EV) else EV
            title = (f"{lab[tf]}: {d['type']}  |  buy confidence {100 * d['conf_buy']:.0f}% (average stock "
                     f"{100 * d['base_hit_buy']:.0f}%)  |  expected {100 * d['exp_ret_buy']:+.1f}% per trade")
            why = []
            if d.get("long_model"):
                why.append(f"AI model rank: top {max(1, round(100 * (1 - d['pct_buy'])))}% for this horizon")
            for p_ in (d.get("pos_signals") or [])[:2]:
                why.append(f"proven setup: {p_[0]} ({p_[6]:%d-%b})")
            if why:
                title += "\nWhy: " + "; ".join(why)
            png = out_dir / f"{k:02d}_{tf}_{sym.replace('&', 'and')}.png"
            try:
                vc.pick_chart(sym, comp.get(sym, sym), prices[sym], ev,
                              {"entry": d["entry"], "target": d["target"], "stop": d["stop"]}, title, png)
                files.append(png)
            except Exception as e:  # a chart must never stop the run
                log.warning("chart for %s failed: %s", sym, e)
    return files


def research_for(asof: pd.Timestamp, static_dir: Path, cfg) -> tuple[dict, str]:
    p = static_dir / "research.json"
    until = pd.Timestamp(cfg["research"]["valid_until"])
    if not p.exists() or asof > until:
        return {}, ""
    return json.load(open(p)), cfg["research"]["note"]


def bundle_path(data_dir: Path, d: pd.Timestamp) -> Path:
    return data_dir / "runs" / f"run_{d:%Y%m%d}.pkl"


def save_bundle(path: Path, res: dict):
    keep_cols = ["symbol", "date", "eligible", "industry", "xs_rs_rating"] + list(engine.REPORT_FIELDS) + \
        [c for c in res["current"].columns if c.startswith("p_")]
    b = {k: res[k] for k in ("date", "decisions", "market", "regime")}
    b["current"] = res["current"][[c for c in keep_cols if c in res["current"].columns]].copy()
    b["events"] = res["events"]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(b, path)


def prune(folder: Path, pattern: str, keep_days: int = 0, keep_n: int = 0):
    files = sorted(folder.glob(pattern))
    if keep_n and len(files) > keep_n:
        for f in files[:-keep_n]:
            f.unlink(missing_ok=True)
    if keep_days:
        cutoff = time.time() - keep_days * 86400
        for f in folder.glob(pattern):
            if f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)


def print_status(trk: tk.Tracker):
    s = trk.summary()
    hd = s["headline"]
    if not hd:
        print("No calls tracked yet.")
        return
    f = report.fmt_pct
    print(f"Calls tracked since {hd['first_date']} (latest {hd['last_date']}): {hd['calls']} "
          f"| completed {hd['closed']} | running {hd['running']} | not triggered {hd['not_triggered']}")
    print(f"OVERALL SUCCESS RATE: {f(hd['success_rate'])} of {hd['closed']} completed calls"
          + (f" | finished groups only: {f(hd['success_rate_mature'])} of {hd['mature']}" if hd.get("mature") else ""))
    b, a = hd["buy"], hd["avoid"]
    print(f"Buy calls: made money {f(b['success_rate'])}, hit target {f(b['target_hit_rate'])} "
          f"(AI confidence said {f(b['avg_ai_conf'])}), avg result {b['avg_result_pct']:+.2f}% per trade"
          if b.get("closed") else "Buy calls: none completed yet")
    print(f"Avoid calls: correct {f(a['success_rate'])} of {a['closed']}" if a.get("closed")
          else "Avoid calls: none completed yet")
    G = s["by_group"]
    if len(G):
        show = G[["Call", "Timeframe", "calls", "closed", "running", "success_rate", "target_hit_rate", "avg_ai_conf"]]
        print(show.to_string(index=False, float_format=lambda v: f"{100 * v:.1f}%"))


# ------------------------------------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Nifty 500 AI signals - daily job")
    ap.add_argument("--config", default=str(BASE / "config.ini"))
    ap.add_argument("--no-download", action="store_true", help="do not download; use data on disk / data/incoming")
    ap.add_argument("--report-only", action="store_true", help="rebuild the Excel for the latest session only")
    ap.add_argument("--asof", help="compute the calls for this past session (YYYY-MM-DD)")
    ap.add_argument("--force", action="store_true", help="recompute the calls even if they exist")
    ap.add_argument("--no-log", action="store_true", help="do not store the calls in the history (testing)")
    ap.add_argument("--refresh-universe", action="store_true", help="download the Nifty 500 list now")
    ap.add_argument("--status", action="store_true", help="print the success rate and exit")
    ap.add_argument("--workers", type=int, help="parallel processes (default from config.ini)")
    ap.add_argument("--download-only", action="store_true", help="only update the price history, then stop")
    ap.add_argument("--no-report", action="store_true", help="store the calls but do not write the Excel")
    ap.add_argument("--apex-export", help="write the JSON files for the APEX 'AI Research' popup to this folder")
    ap.add_argument("--today", help=argparse.SUPPRESS)                     # testing: pretend the date is this
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    data_dir, out_dir = resolve(cfg, "data_dir"), resolve(cfg, "output_dir")
    model_dir, static_dir = resolve(cfg, "model_dir"), resolve(cfg, "static_dir")
    setup_logging(resolve(cfg, "log_dir"), args.verbose)
    data_dir.mkdir(parents=True, exist_ok=True)
    trk = tk.Tracker(data_dir / "recommendations.db")
    if args.status:
        print_status(trk)
        return 0
    try:
        lock = RunLock(data_dir / ".run.lock").__enter__()
    except SystemExit:
        log.error("another run is still active (lock file %s) - exiting", data_dir / ".run.lock")
        return 2
    run_id = trk.run_start()
    t0 = time.time()
    data_date = None
    try:
        now = ist_now()
        today = pd.Timestamp(args.today) if args.today else now.normalize()
        log.info("=== Nifty 500 AI daily run - India time %s ===", now.strftime("%d-%b-%Y %H:%M"))
        store = du.PriceStore(data_dir).load()
        scfg = dict(cfg["sources"])
        fetcher = du.Fetcher(timeout=int(scfg["http_timeout"]), retries=int(scfg["http_retries"]), proxy=scfg["proxy"],
                             offline=args.no_download)
        companies = dict(zip(store.universe["symbol"], store.universe["company"]))
        # ---- 0. first run: start the history with the 25-Sep-2026 calls (first out-of-sample calls of the model) -----
        if (trk.n_recommendations() == 0 and (static_dir / SEED_FILE).exists() and not args.no_log
                and store.last_date() >= SEED_DATE):
            trk.log_calls(pd.read_pickle(static_dir / SEED_FILE), SEED_DATE, companies)
            log.info("history started with the %s calls (the first calls of the frozen model)", SEED_DATE.date())
        # ---- 1. data -------------------------------------------------------------------------------------------
        new_sessions = []
        if not args.report_only:
            sig_before = store.signature()
            du.apply_symbol_changes(store, data_dir / "symbol_changes.csv")
            du.apply_manual_cas(store, data_dir / "manual_corporate_actions.csv")
            if not args.no_download:
                du.refresh_universe(store, fetcher, scfg, today, force=args.refresh_universe)
            new_sessions = du.update_prices(store, fetcher, scfg, today)
            du.update_indices(store, fetcher, scfg, new_sessions)
            du.refresh_universe_stats(store)
            store.save(full=bool(new_sessions) or store.signature() != sig_before)
            companies = dict(zip(store.universe["symbol"], store.universe["company"]))
        last = store.last_date()
        if args.download_only:
            msg = (f"download only: {len(new_sessions)} new session(s); latest session on file {last.date()}; "
                   f"{time.time() - t0:.0f}s")
            log.info("DONE - %s", msg)
            trk.run_end(run_id, "ok", last, msg)
            return 0
        data_date = pd.Timestamp(args.asof) if args.asof else last
        checks = data_checks(store, new_sessions, last)
        if data_date > last:
            raise ValueError(f"--asof {data_date.date()} is after the last session on disk ({last.date()})")
        # ---- 2. re-check earlier calls ---------------------------------------------------------------------------
        # (for a past session only the prices known on that day, so repeats of still-running calls are judged as then)
        ca = pd.read_csv(store.p_ca) if store.p_ca.exists() else None
        prices_then = (store.prices if data_date >= last else
                       {s: d[d.index <= data_date] for s, d in store.prices.items()})
        trk.evaluate(prices_then, store.universe, ca)
        # ---- 3. today's calls ------------------------------------------------------------------------------------
        bpath = bundle_path(data_dir, data_date)
        logged = trk.has_calls_for(data_date)
        xlsx = out_dir / f"Nifty500_AI_Signals_{data_date:%Y%m%d}.xlsx"
        if (not new_sessions and not args.force and not args.report_only and not args.asof and logged
                and bpath.exists() and xlsx.exists()):
            trk.export_csv(out_dir / "recommendation_history.csv")
            msg = f"nothing new since the last run (latest session {data_date.date()}); report {xlsx.name} unchanged"
            log.info("DONE - %s", msg)
            trk.run_end(run_id, "ok", data_date, msg)
            return 0
        if args.force or not bpath.exists() or (not logged and not args.no_log):
            workers = args.workers or int(cfg["run"]["workers"])
            res = engine.run(store.prices, store.bench, store.universe, model_dir,
                             asof=data_date if data_date < last else None, workers=workers)
            if res["date"] != data_date:
                log.warning("engine used session %s (no data for %s)", res["date"].date(), data_date.date())
                data_date = res["date"]
            save_bundle(bundle_path(data_dir, data_date), res)
            if not args.no_log and (args.force or not trk.has_calls_for(data_date)):
                trk.log_calls(res["decisions"], data_date, companies)
            if res["errors"]:
                log.warning("%d stocks failed in the engine: %s", len(res["errors"]), list(res["errors"])[:10])
        else:
            log.info("calls for %s already computed - reusing them", data_date.date())
            res = pd.read_pickle(bpath)
            res["prices"] = {s: d[d.index <= data_date] for s, d in store.prices.items()}
            res["bench"] = store.bench
        trk.evaluate(prices_then, store.universe, ca)       # (a later run brings past sessions up to date)
        if not args.no_log and data_date == last:
            trk.record_success(data_date)                  # one row per session: the daily success-rate history
        if args.no_report:
            hd = trk.summary()["headline"]
            msg = (f"session {data_date.date()}: calls stored (no Excel requested); success rate "
                   f"{report.fmt_pct(hd.get('success_rate', float('nan')))} of {hd.get('closed', 0)} completed calls; "
                   f"{time.time() - t0:.0f}s")
            log.info("DONE - %s", msg)
            trk.run_end(run_id, "ok", data_date, msg)
            return 0
        # ---- 4. Excel ----------------------------------------------------------------------------------------------
        chart_files = make_charts(res, store.universe, data_dir / "charts", cfg)
        research, rnote = research_for(data_date, static_dir, cfg)
        calib = pd.read_pickle(model_dir / "calibration.pkl")
        ctx = {"asof": data_date, "decisions": res["decisions"], "current": res["current"], "universe": store.universe,
               "market": res["market"], "events": res["events"], "regime": res["regime"], "calib": calib,
               "static_dir": static_dir, "charts": chart_files, "track": trk.summary(), "research": research,
               "research_note": rnote, "feats": json.load(open(model_dir / "features.json")),
               "market_lines": market_lines(res["market"], store.bench), "data_checks": checks,
               "data_note": ("Data: NSE bhavcopy and index closes (nsearchives.nseindia.com; GitHub mirrors "
                             "tilak999/NSE-Data-bank and Chaudharyrohit2506/nse-daily-ohlcv as fallback), history from "
                             "kaushikghoshindia/NSE500Tracker. Splits, bonuses and demergers are adjusted automatically "
                             "from NSE's adjusted previous close. Research tool - not investment advice.")}
        xlsx = out_dir / f"Nifty500_AI_Signals_{data_date:%Y%m%d}.xlsx"
        log.info("writing %s", xlsx)
        report.build_workbook(ctx, xlsx)
        shutil.copy2(xlsx, out_dir / "Nifty500_AI_Signals_latest.xlsx")
        trk.export_csv(out_dir / "recommendation_history.csv")
        if args.apex_export:
            exp_dir = Path(args.apex_export)
            gen = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
            man = apex_export.write(exp_dir, data_date, ctx.get("signals_rows") or [], trk, gen)
            man.update(session_date=str(data_date.date()), generated_utc=gen, excel=xlsx.name,
                       regime=int(res["regime"]))
            json.dump(man, open(exp_dir / "manifest.json", "w"), indent=1)
            log.info("APEX files written to %s: %s", exp_dir, man["counts"])
        prune(out_dir, "Nifty500_AI_Signals_2*.xlsx", keep_days=int(cfg["run"]["keep_reports_days"]))
        prune(data_dir / "runs", "run_*.pkl", keep_n=int(cfg["run"]["keep_run_bundles"]))
        hd = trk.summary()["headline"]
        msg = (f"session {data_date.date()}: {len(new_sessions)} new session(s) downloaded; report {xlsx.name}; "
               f"success rate {report.fmt_pct(hd.get('success_rate', float('nan')))} of {hd.get('closed', 0)} completed "
               f"calls; {time.time() - t0:.0f}s")
        log.info("DONE - %s", msg)
        trk.run_end(run_id, "ok", data_date, msg)
        return 0
    except Exception as e:
        log.exception("run failed: %s", e)
        trk.run_end(run_id, "failed", data_date, repr(e))
        return 1
    finally:
        lock.__exit__(None, None, None)
        trk.close()


if __name__ == "__main__":
    sys.exit(main())
