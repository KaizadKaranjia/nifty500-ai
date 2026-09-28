#!/usr/bin/env python3
"""Daily job as run by GitHub Actions (also runs on any PC/server for testing).

  1. brings the price history up to date (NSE files / GitHub data mirrors),
  2. computes and stores the calls of every session that has none yet (oldest first, so a missed day is caught up),
  3. writes the Excel and the JSON files for the Oracle APEX 'AI Research' popup of the latest session,
  4. prepares the folder that ci/state.py publishes on the 'results' branch.

Usage:
  python ci/gh_job.py                    normal daily run
  python ci/gh_job.py --mode force       publish the latest session again even if nothing is new
  python ci/gh_job.py --mode selftest    re-compute the 25-Sep-2026 calls and compare with the reference (5 min)

Writes 'changed=yes|no' and 'session=YYYY-MM-DD' to $GITHUB_OUTPUT (when present). Exit code 0 = OK, 1 = failed.
Research / paper-trading tool - not investment advice.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from n5ai.data_update import PriceStore  # noqa: E402

SEED_DATE = pd.Timestamp("2026-09-25")


def log(msg: str):
    print(f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M:%S}Z  {msg}", flush=True)


def gh_output(**kv):
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a") as fh:
            for k, v in kv.items():
                fh.write(f"{k}={v}\n")
    for k, v in kv.items():
        log(f"output {k}={v}")


def run_daily(args: list[str], workers: int) -> int:
    cmd = [sys.executable, str(ROOT / "nifty500_daily.py"), "--workers", str(workers)] + args
    log("running: " + " ".join(cmd[1:]))
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=ROOT)
    log(f"exit code {rc} after {time.time() - t0:.0f}s")
    return rc


def logged_dates(db: Path) -> set[str]:
    if not db.exists():
        return set()
    con = sqlite3.connect(db)
    try:
        return {r[0] for r in con.execute("SELECT DISTINCT call_date FROM daily_calls")}
    except sqlite3.Error:
        return set()
    finally:
        con.close()


def sessions_after(store: PriceStore, start: pd.Timestamp) -> list[pd.Timestamp]:
    ok = set(store.universe.loc[store.universe["status"] == "ok", "symbol"])
    idx = set()
    for s, d in store.prices.items():
        if s in ok and len(d):
            idx.update(d.index[d.index > start])
    return sorted(idx)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="normal", choices=["normal", "force", "selftest"])
    ap.add_argument("--results-dir", default=str(ROOT / "build" / "results"))
    ap.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 2)))
    ap.add_argument("--start", help="(testing) compute calls only for sessions after this date")
    args = ap.parse_args()
    t0 = time.time()
    data = ROOT / "data"

    if args.mode == "selftest":
        rc = subprocess.call([sys.executable, str(ROOT / "tools" / "validate_engine.py")], cwd=ROOT)
        gh_output(changed="no", session="")
        return rc

    # ---- 1. prices -------------------------------------------------------------------------------------------------
    if run_daily(["--download-only"], args.workers) != 0:
        log("ERROR: the price update failed - see the log above")
        gh_output(changed="no", session="")
        return 1
    store = PriceStore(data).load()
    latest = store.last_date()
    done = logged_dates(data / "recommendations.db")
    if args.start:
        start = pd.Timestamp(args.start)
    else:
        start = max(pd.Timestamp(max(done)), SEED_DATE) if done else SEED_DATE - pd.Timedelta(days=1)
    pending = [d for d in sessions_after(store, start) if str(d.date()) not in done]
    pub_file = data / "published.json"
    published = json.load(open(pub_file)).get("session_date") if pub_file.exists() else None
    log(f"latest session {latest.date()} | calls stored for {len(done)} session(s) | pending: "
        f"{[str(d.date()) for d in pending] or 'none'} | last published: {published}")
    if not pending and published == str(latest.date()) and args.mode != "force":
        log("nothing new - the latest session is already published")
        gh_output(changed="no", session=str(latest.date()))
        return 0

    # ---- 2. catch up missed sessions (oldest first, no Excel) --------------------------------------------------------
    for d in pending:
        if d >= latest:
            continue
        if run_daily(["--no-download", "--asof", str(d.date()), "--no-report"], args.workers) != 0:
            log(f"ERROR: the calls for {d.date()} could not be computed")
            gh_output(changed="no", session="")
            return 1

    # ---- 3. latest session: calls + Excel + APEX files ----------------------------------------------------------------
    res = Path(args.results_dir)
    if res.exists():
        shutil.rmtree(res)
    day_dir = res / "d" / str(latest.date())
    if run_daily(["--no-download", "--apex-export", str(day_dir)], args.workers) != 0:
        log("ERROR: the daily run failed - see the log above")
        gh_output(changed="no", session="")
        return 1
    man_path = day_dir / "manifest.json"
    if not man_path.exists():
        log("ERROR: the APEX files were not written")
        gh_output(changed="no", session="")
        return 1
    man = json.load(open(man_path))
    if man["session_date"] != str(latest.date()):
        log(f"ERROR: exported session {man['session_date']} is not the latest session {latest.date()}")
        gh_output(changed="no", session="")
        return 1

    # ---- 4. results folder (published on the 'results' branch by ci/state.py) -------------------------------------
    out = ROOT / "output"
    (res / "excel").mkdir(parents=True, exist_ok=True)
    xlsx = out / "Nifty500_AI_Signals_latest.xlsx"
    if xlsx.exists():
        shutil.copy2(xlsx, res / "excel" / "Nifty500_AI_Signals_latest.xlsx")
    (res / "backup").mkdir(parents=True, exist_ok=True)
    shutil.copy2(data / "recommendations.db", res / "backup" / "recommendations.db")
    if (out / "recommendation_history.csv").exists():
        shutil.copy2(out / "recommendation_history.csv", res / "backup" / "recommendation_history.csv")
    folder = f"d/{latest.date()}"
    status = {
        "schema": 1,
        "session_date": str(latest.date()),
        "generated_utc": man["generated_utc"],
        "folder": folder,
        "files": {k: f"{folder}/{k}.json" for k in ("signals", "calls", "success")},
        "counts": man["counts"],
        "headline": man["headline"],
        "regime": man.get("regime"),
        "excel": "excel/Nifty500_AI_Signals_latest.xlsx",
        "sessions_computed": [str(d.date()) for d in pending],
        "note": "Research / paper-trading output - not investment advice.",
    }
    json.dump(status, open(res / "status.json", "w"), indent=1)
    (res / "README.md").write_text(
        "# Nifty 500 AI - daily results\n\n"
        "Written automatically by the daily GitHub job (branch `results`, replaced on every run).\n\n"
        f"* Latest session: **{latest.date()}**\n"
        "* `status.json` - what the Oracle APEX loader reads first\n"
        "* `d/<date>/signals.json`, `calls.json`, `success.json` - data for the APEX 'AI Research' popup\n"
        "* `excel/Nifty500_AI_Signals_latest.xlsx` - the full workbook (download it from here)\n"
        "* `backup/recommendations.db` - copy of the call history\n\n"
        "Research / paper-trading output - not investment advice.\n", encoding="utf-8")
    log(f"results ready in {res} for session {latest.date()} (counts {man['counts']}) in {time.time() - t0:.0f}s")
    gh_output(changed="yes", session=str(latest.date()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
