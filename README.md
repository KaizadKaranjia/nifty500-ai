# Nifty 500 AI — daily signals and success rate

Runs every weekday evening on GitHub Actions (free for public repositories) and publishes the results for the
**AI RESEARCH** popup of the Oracle APEX app *Family Portfolio Tracker*.

What one run does:

1. Downloads the day's NSE bhavcopy and index closes (official NSE files; GitHub data mirrors as fallback) and adjusts
   the price history for splits, bonuses and demergers.
2. Re-checks every earlier call against the new prices (target / stop / time exit; avoid calls vs the average stock).
3. Computes the calls for all Nifty 500 stocks × 3 timeframes with the frozen AI model and the proven-signal rules.
4. Stores the calls with their date and records the day's success rate.
5. Publishes on the branch **`results`**: `status.json`, `d/<date>/signals.json`, `calls.json`, `success.json`,
   the full Excel (`excel/Nifty500_AI_Signals_latest.xlsx`) and a backup of the call history.

The APEX package `AIR_PKG` reads the `results` branch (automation every 2 hours; at most ~7 web-service calls a day).

## Folders

| Folder / file | What it is |
|---|---|
| `.github/workflows/daily.yml` | the schedule (Mon–Fri 20:00, 00:15, 03:00 and 08:00 IST) and the steps |
| `ci/gh_job.py`, `ci/state.py` | the GitHub job: catch-up of missed sessions, results folder, saved data |
| `nifty500_daily.py`, `n5ai/` | the engine (indicators, chart patterns, AI model scoring, tracker, Excel) |
| `model/`, `static/` | frozen AI models and research tables (trained on data to 25-Sep-2026) |
| `data/` | index history, Nifty 500 list, optional manual corrections |
| `nifty500_ai_prices_part1.zip`, `..._part2.zip` | price history up to 25-Sep-2026 — used once, on the first run |
| release **`state`** | saved data between runs (created automatically — please do not delete) |

## Manual run

*Actions* tab → *Nifty 500 AI daily* → *Run workflow* → mode:

* `normal` – the daily run (does nothing if the latest session is already published)
* `force` – publish the latest session again
* `selftest` – re-computes the 25-Sep-2026 calls and compares them with the reference (must say *SELF-TEST PASSED*)

The engine can also run on a PC/server — see `docs/SERVER_SETUP.md`.

Research / paper-trading tool for education — not investment advice; not a SEBI-registered adviser.
