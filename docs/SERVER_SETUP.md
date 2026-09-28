> **Running on a PC or server instead of GitHub:** this repository folder is the same as the folder `nifty500_ai` described below (unzip the two price zips into it the same way). The GitHub job itself needs none of this.

# Nifty 500 AI Signals — daily job

One command, once a day after the NSE close:

1. **Prices** – downloads every new trading session (NSE full bhavcopy with delivery data + index closes).
   Splits, bonuses, rights and demergers are adjusted automatically from NSE's adjusted previous close.
2. **Re-checks earlier calls** against the new prices.
3. **Today's calls** for every Nifty 500 stock × 3 timeframes — same AI model, chart-pattern engine and
   proven-signal rules as the research workbook of 25-Sep-2026 (tested: the daily engine reproduces the
   research pipeline's 1,500 calls exactly).
4. **Stores every Buy / Avoid call with its date** in `data/recommendations.db` (SQLite).
5. **Writes the Excel** `output/Nifty500_AI_Signals_<YYYYMMDD>.xlsx`. The **last sheet, "Success Rate"**,
   shows how the calls have done so far.

Research / paper-trading tool — not investment advice.

---

## How the success rate is measured

| Call | Tracked as | Success means |
|---|---|---|
| Strong Buy / Buy / Accumulate | Paper trade entered at the **next session's open**, only if the open is not above the *Entry (buy up to)* price and not below the stop. Exit at the target, the stop, or at the close of the last allowed session (10 / 60 / 250 sessions). 0.3% costs deducted. | The trade made money |
| Avoid / Reduce | Stock return over the same 10 / 60 / 250 sessions vs the average Nifty 500 stock | The stock did worse than average |
| Short Sell (rare) | Mirror of the buy trade | The trade made money |

- **Target hit rate** is also shown for buy calls, next to the average *AI confidence* those calls carried —
  that is the model's promise vs what happened.
- A call that repeats on later days while the first one is still running is counted **once**.
- "Not triggered" buys (the stock opened above the entry limit) are listed but not counted in the rate.
- Stops sit closer than targets, so losing trades finish first and the rate starts low. The sheet therefore also
  shows the rate for **finished groups only** (days whose calls have all completed) — the fair number to judge.
- The history starts with the 25-Sep-2026 calls — the first calls the frozen model makes on data it was not
  trained on, so the success rate is a genuine out-of-sample track record.

---

## Folder layout

```
nifty500_ai/
├── nifty500_daily.py      main script (run this)
├── run_daily.sh           cron wrapper (low CPU priority, logs to logs/cron.log)
├── run_daily.bat          same for Windows Task Scheduler
├── config.ini             settings (paths, workers, download sources, proxy)
├── requirements.txt       Python packages (exact tested versions)
├── n5ai/                  engine: indicators, chart patterns, signals, model scoring, tracker, Excel
├── model/                 frozen AI models + calibration (trained on data to 25-Sep-2026)
├── static/                research tables used in the Excel (report card, validation, results check)
├── data/
│   ├── prices.pkl         price history of all stocks (joined from prices_part1/2.pkl on the first run;
│   │                      updated daily; previous copy kept as .pkl.bak)
│   ├── bench.pkl          index history (Nifty 50 / 500 / Midcap 150 / Smallcap 250 / Next 50)
│   ├── universe.csv       Nifty 500 list (refreshed weekly)
│   ├── recommendations.db call history + success tracking (SQLite)
│   ├── corporate_actions.csv  every price adjustment applied
│   ├── manual_corporate_actions.csv / symbol_changes.csv   optional manual corrections
│   ├── bhav/              downloaded NSE files (gzip cache)
│   ├── incoming/          drop NSE files here by hand if downloads fail
│   └── runs/              daily engine snapshots (for --report-only)
├── output/                daily Excel + Nifty500_AI_Signals_latest.xlsx + recommendation_history.csv
├── logs/                  one log file per day + cron.log
└── tools/                 self-checks (unit_checks.py)
```

---

## Setup on the DEV server (Oracle Linux 8.10) — run as the **developer user**

### Step 1 — Copy and unzip

The package comes as three zip files (split to stay under the chat upload limit). Copy all three to your
home folder on the server (WinSCP), then:

```bash
mkdir -p /srv/projects/AITIMES
cd /srv/projects/AITIMES
unzip ~/nifty500_ai_daily_code.zip
unzip ~/nifty500_ai_prices_part1.zip
unzip ~/nifty500_ai_prices_part2.zip
cd /srv/projects/AITIMES/nifty500_ai
chmod +x run_daily.sh
ls data/            # must show prices_part1.pkl and prices_part2.pkl (joined automatically on the first run)
```

### Step 2 — Python 3.12 virtual environment

```bash
cd /srv/projects/AITIMES/nifty500_ai
/opt/py312-common/bin/python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

If that Python path is different on your server, check with `ls /opt/py312-common/bin/`.
All packages install from binary wheels (TA-Lib included) — no compiler or C library needed.

### Step 3 — Check internet access to the data sources

```bash
curl -sS -o /dev/null -w "NSE: %{http_code}\n" -A "Mozilla/5.0" \
  https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_25092026.csv
curl -sS -o /dev/null -w "GitHub mirror: %{http_code}\n" \
  https://raw.githubusercontent.com/tilak999/NSE-Data-bank/main/data/sec_bhavdata_full_25092026.csv
```

- `200` from NSE → best (files are ready by ~6–7 pm IST).
- NSE fails but GitHub gives `200` → still fine; the mirror is used automatically (updated ~11:30 pm IST).
- Both fail → set `proxy =` in `config.ini` if the server needs a proxy.

### Step 4 — Self-check and first run

```bash
cd /srv/projects/AITIMES/nifty500_ai
.venv/bin/python tools/unit_checks.py          # 10 seconds - must end with: ALL PASS
.venv/bin/python tools/validate_engine.py      # ~5 minutes - must end with: SELF-TEST PASSED
.venv/bin/python nifty500_daily.py             # first real run: ~5-8 minutes on 2 cores
```

`validate_engine.py` re-computes the 25-Sep-2026 calls on your server and compares them with the research
workbook — run it once after installing; it proves the installed packages give identical results.

Check the result:

```bash
grep -E "DONE|ERROR" logs/nifty500_daily_$(date +%Y%m%d).log | tail -3
ls -l output/
.venv/bin/python nifty500_daily.py --status
```

### Step 5 — Schedule it (cron, developer user)

```bash
crontab -e
```

Add these lines:

```cron
CRON_TZ=Asia/Kolkata
# Nifty 500 AI daily job - 7:45 pm IST, Monday to Friday
45 19 * * 1-5 /srv/projects/AITIMES/nifty500_ai/run_daily.sh
# safety net - 00:30 IST Tue-Sat: picks up the GitHub mirror if NSE was unreachable (does nothing if already done)
30 0 * * 2-6 /srv/projects/AITIMES/nifty500_ai/run_daily.sh
```

Verify with `crontab -l`. The next day check `tail -20 logs/cron.log` (look for `exit code 0`).
If your cron ignores `CRON_TZ` and the server clock is UTC, use `15 14 * * 1-5` and `0 19 * * 1-5` instead.

### Step 6 — Daily use

- Open `output/Nifty500_AI_Signals_latest.xlsx` (WinSCP / shared folder). The **last sheet "Success Rate"**
  has the headline success rate, a breakdown by call type and timeframe, and every call with its result.
- Quick check from the terminal: `.venv/bin/python nifty500_daily.py --status`
- Full history as CSV (for Excel / Oracle / APEX): `output/recommendation_history.csv`

---

## Alternative: Windows PC

1. Install Python 3.12 from python.org (tick *Add python.exe to PATH*).
2. Right-click each of the three zip files → *Extract All* → set the destination to `C:\` (not the suggested
   sub-folder). All three fill the same folder `C:\nifty500_ai`.
3. In *Command Prompt*:
   ```bat
   cd /d C:\nifty500_ai
   py -3.12 -m venv .venv
   .venv\Scripts\pip install -r requirements.txt
   .venv\Scripts\python tools\unit_checks.py
   .venv\Scripts\python nifty500_daily.py
   ```
4. *Task Scheduler* → *Create Task* → Triggers: *Weekly*, Mon–Fri, 7:45 PM → Actions: *Start a program*
   `C:\nifty500_ai\run_daily.bat`, *Start in* `C:\nifty500_ai` → Conditions: untick *Start only if on AC power*.
   The PC must be on at that time.

---

## Commands

| Command | What it does |
|---|---|
| `python nifty500_daily.py` | normal daily run |
| `python nifty500_daily.py --status` | print the success rate and exit |
| `python nifty500_daily.py --report-only` | rebuild the Excel for the latest session (no download) |
| `python nifty500_daily.py --no-download` | use only files already on disk or dropped in `data/incoming/` |
| `python nifty500_daily.py --asof 2026-10-01` | compute and store the calls for a missed session (oldest first) |
| `python nifty500_daily.py --force` | recompute today's calls even if they exist |
| `python nifty500_daily.py --refresh-universe` | download the Nifty 500 list now |
| `python nifty500_daily.py --workers 1` | use one CPU core |
| `python nifty500_daily.py --download-only` | only bring the price history up to date |
| `python nifty500_daily.py --no-report` | store the calls without writing the Excel (catch-up of a missed session) |
| `python nifty500_daily.py --apex-export DIR` | also write the JSON files for the APEX *AI Research* popup |

Exit codes: `0` OK · `1` failed (see `logs/`) · `2` another run is still active.

---

## Backup and rollback

Backup (run while no job is active, e.g. before changes):

```bash
cd /srv/projects/AITIMES
tar czf ~/nifty500_ai_data_backup_$(date +%Y%m%d).tgz nifty500_ai/data nifty500_ai/config.ini
```

- Every run that adds prices keeps the previous files as `data/prices.pkl.bak` and `data/bench.pkl.bak`.
  Roll back one run:
  ```bash
  cd /srv/projects/AITIMES/nifty500_ai
  cp data/prices.pkl.bak data/prices.pkl && cp data/bench.pkl.bak data/bench.pkl
  ```
- Full restore: `cd /srv/projects/AITIMES && tar xzf ~/nifty500_ai_data_backup_<date>.tgz`
- The call history is only in `data/recommendations.db` — include it in your regular backups.

---

## Troubleshooting

| Log message | Meaning / fix |
|---|---|
| `bhavcopy not published yet` | Normal before ~6 pm IST or on holidays. The 00:30 run catches up. |
| `market holiday` | No NSE file for that weekday and a later session exists — skipped. |
| `download error ... stopping here` | Network problem. Next run retries. To force it: download the files by hand (below). |
| `corporate action XYZ on <date>` | Split / bonus / demerger detected; history adjusted and logged in `data/corporate_actions.csv`. |
| `index closes not found ... carried forward` | Index file missing; last close used for now and replaced on a later run. |
| `could not download the Nifty 500 list` | niftyindices.com blocked or down; the current list is kept. |

**Manual files:** from nseindia.com → *All Reports* → *Equities*, download *Full Bhavcopy and Security
Deliverable data* (`sec_bhavdata_full_DDMMYYYY.csv`) and from *Indices* the *Index closing* file
(`ind_close_all_DDMMYYYY.csv`). Put both in `data/incoming/` and run `./run_daily.sh`.

**Symbol renamed by NSE:** add a line `OLDSYMBOL,NEWSYMBOL` to `data/symbol_changes.csv`
(first line of the file: `old_symbol,new_symbol`).

**Split / bonus / demerger missed or wrong:** the *Data checks* section on the Read Me sheet lists every
adjustment made in that run. To correct one, add a row to `data/manual_corporate_actions.csv`
(`symbol,ex_date,factor,note`; factor multiplies prices before the ex-date: bonus 1:1 = 0.5, split 10→2 = 0.2;
to undo a wrong automatic factor f add 1/f). It is applied once on the next run.

**New Nifty 500 members** (March / September reshuffles) get their history from the GitHub tracker
automatically; a stock is rated once it has 210 sessions and ≥ Rs 10 Cr average daily turnover.

---

## Model refresh

The AI model and its calibration are frozen as trained on data to 25-Sep-2026 (so the success rate
measures one fixed model honestly). Re-training every 6–12 months is recommended.
