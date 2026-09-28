"""Daily data update for the Nifty 500 price store.

Sources, tried in this order for every missing trading day:
  1. data/incoming/<file>        - a file you dropped there by hand (manual override)
  2. data/bhav/<file>.gz         - cache of files downloaded earlier
  3. NSE archives                - official bhavcopy / index close files (usually ready by ~6-7 pm IST)
  4. GitHub mirrors              - tilak999/NSE-Data-bank (bhavcopy), Chaudharyrohit2506/nse-daily-ohlcv (indices)

Corporate actions: on an ex-date NSE sets PREV_CLOSE in the bhavcopy to the adjusted base price. If it differs
from the last close in the store, the whole history of that stock is multiplied by PREV_CLOSE / last close
(volumes divided), so splits, bonuses, rights and demergers are handled without a separate feed.
"""
from __future__ import annotations

import gzip
import io
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .common import INDEX_NAMES

log = logging.getLogger("n5ai.data")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")
CA_TOL = 0.003                   # |PREV_CLOSE / last close - 1| above this = corporate action
MIRROR_INDEX_FILES = {"NIFTY50": "Nifty_50.csv", "NIFTY500": "Nifty_500.csv", "MIDCAP150": "Nifty_Midcap_150.csv",
                      "SMALLCAP250": "Nifty_Smallcap_250.csv", "NEXT50": "Nifty_Next_50.csv"}
PRICE_COLS = ["open", "high", "low", "close", "volume", "deliv_qty", "trades", "series", "deliv_per", "turnover_cr"]


class StaleFile(Exception):
    """The file exists but holds an earlier session (mirrors copy the last file on market holidays)."""


# ------------------------------------------------------------------------------------------------------------
# HTTP
# ------------------------------------------------------------------------------------------------------------
class Fetcher:
    """Small HTTP helper: browser-like headers, retries, optional proxy, NSE cookie warm-up."""

    def __init__(self, timeout: int = 30, retries: int = 3, proxy: str = "", offline: bool = False):
        self.timeout, self.retries, self.offline = timeout, retries, offline
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA, "Accept": "text/csv,application/octet-stream,*/*;q=0.8",
                               "Accept-Language": "en-US,en;q=0.9", "Referer": "https://www.nseindia.com/"})
        if proxy:
            self.s.proxies.update({"http": proxy, "https": proxy})
        self._nse_warm = False

    def get(self, url: str) -> tuple[str, bytes | None]:
        """Returns (status, content): status is 'ok', 'missing' (404) or 'error'."""
        if self.offline:
            return "error", None
        for attempt in range(self.retries):
            try:
                r = self.s.get(url, timeout=self.timeout)
                if r.status_code == 200 and len(r.content) > 50:
                    return "ok", r.content
                if r.status_code == 404:
                    return "missing", None
                if r.status_code in (401, 403) and "nseindia.com" in url and not self._nse_warm:
                    self._nse_warm = True
                    try:                                           # NSE sometimes wants its cookies first
                        self.s.get("https://www.nseindia.com/", timeout=self.timeout)
                    except requests.RequestException:
                        pass
                    continue
                log.debug("GET %s -> HTTP %s", url, r.status_code)
            except requests.RequestException as e:
                log.debug("GET %s failed: %s", url, e)
            time.sleep(2 * (attempt + 1))
        return "error", None


# ------------------------------------------------------------------------------------------------------------
# parsing
# ------------------------------------------------------------------------------------------------------------
def read_bhav(raw: bytes) -> pd.DataFrame:
    """NSE sec_bhavdata_full -> one row per symbol (EQ preferred over BE/BZ), numeric columns parsed."""
    b = pd.read_csv(io.BytesIO(raw), skipinitialspace=True, dtype=str)
    b.columns = [c.strip().upper() for c in b.columns]
    for c in b.columns:
        b[c] = b[c].astype(str).str.strip()
    b = b[b["SERIES"].isin(["EQ", "BE", "BZ"])].copy()
    num = ["PREV_CLOSE", "OPEN_PRICE", "HIGH_PRICE", "LOW_PRICE", "CLOSE_PRICE", "TTL_TRD_QNTY", "DELIV_QTY",
           "NO_OF_TRADES"]
    for c in num:
        b[c] = pd.to_numeric(b[c].str.replace(",", "", regex=False), errors="coerce")
    b["DATE1"] = pd.to_datetime(b["DATE1"], format="%d-%b-%Y", errors="coerce")
    prio = {"EQ": 0, "BE": 1, "BZ": 2}
    b = b.assign(_p=b["SERIES"].map(prio)).sort_values("_p").drop_duplicates("SYMBOL")
    return b.set_index("SYMBOL")


def read_ind_close(raw: bytes) -> dict:
    """NSE ind_close_all -> {bench key: (open, high, low, close, volume)} for the indices the model uses."""
    x = pd.read_csv(io.BytesIO(raw), dtype=str)
    x.columns = [c.strip().lower() for c in x.columns]

    def col(*keys):
        for c in x.columns:
            if all(k in c for k in keys):
                return c
        raise KeyError(keys)

    name_c = col("index name")
    cols = {"open": col("open"), "high": col("high"), "low": col("low"), "close": col("clos"), "volume": col("volume")}
    x[name_c] = x[name_c].str.strip().str.lower()
    out = {}
    for key, name in INDEX_NAMES.items():
        r = x[x[name_c] == name.lower()]
        if r.empty:
            continue
        v = {k: pd.to_numeric(str(r.iloc[0][c]).replace(",", "").strip(), errors="coerce") for k, c in cols.items()}
        if not np.isfinite(v["close"]) or v["close"] <= 0:
            continue
        for k in ("open", "high", "low"):
            if not np.isfinite(v[k]) or v[k] <= 0:
                v[k] = v["close"]
        out[key] = (v["open"], v["high"], v["low"], v["close"], v["volume"] if np.isfinite(v["volume"]) else 0.0)
    return out


def read_tracker_history(raw: bytes, start: str = "2016-06-01") -> pd.DataFrame | None:
    """History file of kaushikghoshindia/NSE500Tracker (split/bonus adjusted) -> store format."""
    d = pd.read_csv(io.BytesIO(raw))
    if "Date" not in d.columns:
        return None
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d = d[d["Date"] >= start].rename(columns={"Date": "date", "Open": "open", "High": "high", "Low": "low",
                                              "Close": "close", "Volume": "volume", "DLV_QTY": "deliv_qty",
                                              "TOTAL_TRADES": "trades", "Series": "series"})
    for c in ("open", "high", "low", "close", "volume", "deliv_qty", "trades"):
        d[c] = pd.to_numeric(d[c], errors="coerce").astype(float) if c in d.columns else np.nan
    if "series" not in d.columns:
        d["series"] = "EQ"
    d = d.dropna(subset=["close"])
    d = d[d["close"] > 0].drop_duplicates("date", keep="last").sort_values("date")
    for c in ("open", "high", "low"):
        d[c] = d[c].where(d[c] > 0, d["close"])
    d["high"] = d[["open", "high", "low", "close"]].max(axis=1)
    d["low"] = d[["open", "high", "low", "close"]].min(axis=1)
    d = d.set_index("date")
    d["deliv_per"] = 100.0 * d["deliv_qty"] / d["volume"].replace(0, np.nan)
    d["turnover_cr"] = d["close"] * d["volume"] / 1e7
    return d[PRICE_COLS] if len(d) else None


# ------------------------------------------------------------------------------------------------------------
# store
# ------------------------------------------------------------------------------------------------------------
class PriceStore:
    """prices.pkl (dict symbol -> DataFrame), bench.pkl (dict index -> DataFrame), universe.csv, store_meta.json."""

    def __init__(self, data_dir: Path):
        self.dir = Path(data_dir)
        self.p_prices, self.p_bench = self.dir / "prices.pkl", self.dir / "bench.pkl"
        self.p_uni, self.p_meta = self.dir / "universe.csv", self.dir / "store_meta.json"
        self.p_ca = self.dir / "corporate_actions.csv"
        self.cache = self.dir / "bhav"
        self.incoming = self.dir / "incoming"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.incoming.mkdir(parents=True, exist_ok=True)
        self.prices: dict = {}
        self.bench: dict = {}
        self.universe = pd.DataFrame()
        self.meta: dict = {}
        self.ca_new: list = []
        self.notes: list = []               # human-readable data checks for the Excel 'Read Me'
        self.ca_run: list = []              # corporate actions applied in this run

    def load(self):
        if not self.p_prices.exists():                  # first run: join the price-history parts shipped in 2 zips
            parts = sorted(self.dir.glob("prices_part*.pkl"))
            if not parts:
                raise FileNotFoundError(f"{self.p_prices} not found - unzip the price-history zip files into this folder")
            merged: dict = {}
            for p in parts:
                merged.update(pd.read_pickle(p))
            pd.to_pickle(merged, self.p_prices)
            for p in parts:
                p.unlink()
            log.info("price history assembled from %d parts (%d stocks)", len(parts), len(merged))
        self.prices = pd.read_pickle(self.p_prices)
        self.bench = pd.read_pickle(self.p_bench)
        self.universe = pd.read_csv(self.p_uni)
        self.meta = json.load(open(self.p_meta)) if self.p_meta.exists() else {}
        log.info("store loaded: %d stocks, last session %s", len(self.prices), self.last_date().date())
        return self

    def last_date(self) -> pd.Timestamp:
        ok = set(self.universe.loc[self.universe["status"] == "ok", "symbol"])
        lasts = [d.index.max() for s, d in self.prices.items() if s in ok and len(d)]
        return max(lasts) if lasts else pd.Timestamp("2016-01-01")

    def signature(self) -> tuple:
        """Cheap fingerprint of the price store (used to decide whether the big files must be rewritten)."""
        return (len(self.prices), sum(len(d) for d in self.prices.values()), tuple(sorted(self.prices)),
                tuple(self.universe["symbol"]), tuple(self.universe["status"]))

    def save(self, full: bool = True):
        """Atomic save; the previous price/index files are kept as *.pkl.bak for rollback."""
        if full:
            for path, obj in ((self.p_prices, self.prices), (self.p_bench, self.bench)):
                tmp = path.with_suffix(".tmp")
                pd.to_pickle(obj, tmp)
                if path.exists():
                    os.replace(path, path.with_suffix(".pkl.bak"))
                os.replace(tmp, path)
        self.universe.to_csv(self.p_uni, index=False)
        json.dump(self.meta, open(self.p_meta, "w"), indent=1, default=str)
        if self.ca_new:
            ca = pd.DataFrame(self.ca_new)
            if self.p_ca.exists():
                ca = pd.concat([pd.read_csv(self.p_ca), ca], ignore_index=True)
            ca.to_csv(self.p_ca, index=False)
            self.ca_new = []
        log.info("store saved%s", " (previous price files kept as .pkl.bak)" if full else " (metadata only)")

    # ---- sources -------------------------------------------------------------------------------------------
    def _cached(self, fname: str) -> bytes | None:
        p_in = self.incoming / fname
        if p_in.exists():
            return p_in.read_bytes()
        p_c = self.cache / (fname + ".gz")
        if p_c.exists():
            return gzip.decompress(p_c.read_bytes())
        return None

    def _cache_put(self, fname: str, raw: bytes):
        (self.cache / (fname + ".gz")).write_bytes(gzip.compress(raw))

    def fetch_file(self, fname: str, urls: list[str], fetcher: Fetcher, validate) -> tuple[str, bytes | None, str]:
        """(status, content, source) - status 'ok' / 'missing' (404 everywhere) / 'error'."""
        raw = self._cached(fname)
        statuses = []
        if raw is not None:
            try:
                validate(raw)
                return "ok", raw, "local"
            except StaleFile:
                statuses.append("missing")
            except Exception as e:  # corrupt cache -> ignore it
                log.warning("local copy of %s unusable (%s) - downloading again", fname, e)
        for url in urls:
            st, raw = fetcher.get(url)
            if st == "ok":
                try:
                    validate(raw)
                except StaleFile as e:
                    log.info("%s from %s holds an earlier session (%s) - treated as not available", fname,
                             url.split("/")[2], e)
                    statuses.append("missing")
                    continue
                except Exception as e:
                    log.warning("%s from %s failed validation: %s", fname, url.split("/")[2], e)
                    statuses.append("error")
                    continue
                self._cache_put(fname, raw)
                return "ok", raw, url.split("/")[2]
            statuses.append(st)
        # 'missing' = at least one source says the file does not exist (holiday or not yet published)
        return ("missing" if "missing" in statuses else "error"), None, ""


# ------------------------------------------------------------------------------------------------------------
# update steps
# ------------------------------------------------------------------------------------------------------------
def apply_factor(df: pd.DataFrame, f: float) -> pd.DataFrame:
    df = df.copy()
    for c in ("open", "high", "low", "close"):
        df[c] = df[c] * f
    for c in ("volume", "deliv_qty"):
        df[c] = df[c] / f
    return df


def bhav_row(r: pd.Series) -> list:
    o, h, lo, c = (float(r[k]) for k in ("OPEN_PRICE", "HIGH_PRICE", "LOW_PRICE", "CLOSE_PRICE"))
    o = o if o > 0 else c
    h = max(v for v in (o, h, lo, c) if v > 0)
    lo = min(v for v in (o, h, lo, c) if v > 0)
    vol = float(r["TTL_TRD_QNTY"]) if np.isfinite(r["TTL_TRD_QNTY"]) else 0.0
    dq = float(r["DELIV_QTY"]) if np.isfinite(r["DELIV_QTY"]) else np.nan
    tr = float(r["NO_OF_TRADES"]) if np.isfinite(r["NO_OF_TRADES"]) else np.nan
    dp = 100.0 * dq / vol if vol > 0 and np.isfinite(dq) else np.nan
    return [o, h, lo, c, vol, dq, tr, r["SERIES"], dp, c * vol / 1e7]


def apply_bhav(store: PriceStore, d: pd.Timestamp, b: pd.DataFrame, symbols=None) -> dict:
    """Append session d to every stock of the store (or `symbols`), adjusting history on corporate actions."""
    stats = {"updated": 0, "ca": 0}
    for sym in (symbols if symbols is not None else list(store.prices)):
        if sym not in b.index or sym not in store.prices:
            continue
        df = store.prices[sym]
        if len(df) and d <= df.index.max():
            continue
        r = b.loc[sym]
        if not (np.isfinite(r["CLOSE_PRICE"]) and r["CLOSE_PRICE"] > 0):
            continue
        if len(df):
            last_c, prev = float(df["close"].iloc[-1]), float(r["PREV_CLOSE"])
            if np.isfinite(prev) and prev > 0 and last_c > 0 and abs(prev / last_c - 1) > CA_TOL:
                f = prev / last_c
                df = apply_factor(df, f)
                store.ca_new.append({"symbol": sym, "ex_date": d.date().isoformat(), "factor": round(f, 8),
                                     "last_close_before": last_c, "nse_prev_close": prev, "source": "auto"})
                store.ca_run.append((sym, d.date().isoformat(), f))
                stats["ca"] += 1
                log.info("corporate action %s on %s: history x %.6f (prev close %.2f -> %.2f)", sym, d.date(), f,
                         last_c, prev)
        row = pd.DataFrame([bhav_row(r)], columns=PRICE_COLS, index=pd.DatetimeIndex([d], name=df.index.name))
        store.prices[sym] = pd.concat([df, row]) if len(df) else row
        stats["updated"] += 1
    return stats


def bhav_urls(cfg, d: pd.Timestamp) -> list[str]:
    k = d.strftime("%d%m%Y")
    return [u.format(ddmmyyyy=k) for u in (cfg["nse_bhav_url"], cfg["mirror_bhav_url"]) if u]


def index_urls(cfg, d: pd.Timestamp) -> list[str]:
    k = d.strftime("%d%m%Y")
    return [cfg["nse_index_url"].format(ddmmyyyy=k)] if cfg.get("nse_index_url") else []


def validate_bhav_for(d: pd.Timestamp):
    def _v(raw):
        b = read_bhav(raw)
        if len(b) < 500:
            raise ValueError(f"only {len(b)} rows")
        dd = b["DATE1"].dropna()
        if len(dd) and dd.mode().iloc[0] < d:
            raise StaleFile(f"file is for {dd.mode().iloc[0].date()}")
        if len(dd) and dd.mode().iloc[0] != d:
            raise ValueError(f"file is for {dd.mode().iloc[0].date()}, not {d.date()}")
    return _v


def update_prices(store: PriceStore, fetcher: Fetcher, cfg: dict, today: pd.Timestamp) -> list[pd.Timestamp]:
    """Download and apply every trading day after the store's last session up to `today`."""
    last = store.last_date()
    cand = [d for d in pd.bdate_range(last + pd.Timedelta(days=1), today)]
    if not cand:
        log.info("prices already up to date (last session %s)", last.date())
        return []
    got, status = {}, {}
    for d in cand:
        fname = f"sec_bhavdata_full_{d:%d%m%Y}.csv"
        st, raw, src = store.fetch_file(fname, bhav_urls(cfg, d), fetcher, validate_bhav_for(d))
        status[d] = st
        if st == "ok":
            got[d] = read_bhav(raw)
            log.info("bhavcopy %s: %d rows (source: %s)", d.date(), len(got[d]), src)
    applied = []
    for d in cand:
        if d in got:
            s = apply_bhav(store, d, got[d])
            applied.append(d)
            log.info("session %s applied: %d stocks updated, %d corporate actions", d.date(), s["updated"], s["ca"])
        elif any(x > d for x in got):
            if status[d] == "missing":
                log.info("%s: no bhavcopy anywhere and a later session exists -> market holiday", d.date())
            else:
                log.warning("%s: download error and a later session exists - stopping here so no gap is created; "
                            "the next run will retry", d.date())
                break
        else:
            log.info("%s: bhavcopy not published yet (status %s) - will retry on the next run", d.date(), status[d])
            break
    if applied:
        store.meta["last_price_update"] = pd.Timestamp.now().isoformat(timespec="seconds")
    return applied


def _mirror_indices(store: PriceStore, fetcher: Fetcher, cfg: dict) -> dict:
    out = {}
    if not cfg.get("mirror_index_url"):
        return out
    for key, f in MIRROR_INDEX_FILES.items():
        st, raw = fetcher.get(cfg["mirror_index_url"].format(file=f))
        if st != "ok":
            continue
        try:
            x = pd.read_csv(io.BytesIO(raw), parse_dates=["trade_date"]).set_index("trade_date").sort_index()
            out[key] = x[["open", "high", "low", "close", "volume"]].astype(float)
        except Exception as e:
            log.warning("mirror index file %s unreadable: %s", f, e)
    return out


def update_indices(store: PriceStore, fetcher: Fetcher, cfg: dict, sessions: list[pd.Timestamp]):
    """Index closes for new sessions (and for sessions filled earlier), NSE first, then the GitHub mirror."""
    filled_before = [pd.Timestamp(x) for x in store.meta.get("bench_filled", [])]
    todo = sorted(set(sessions) | set(filled_before))
    if not todo:
        return
    got: dict = {}
    for d in todo:
        fname = f"ind_close_all_{d:%d%m%Y}.csv"
        st, raw, src = store.fetch_file(fname, index_urls(cfg, d), fetcher, lambda r: read_ind_close(r)["NIFTY50"])
        if st == "ok":
            got[d] = read_ind_close(raw)
    need = [d for d in todo if d not in got or len(got[d]) < len(INDEX_NAMES)]
    mirror = _mirror_indices(store, fetcher, cfg) if need else {}
    still = []
    for d in todo:
        vals = dict(got.get(d, {}))
        for key, x in mirror.items():
            if key not in vals and d in x.index:
                vals[key] = tuple(float(v) for v in x.loc[d, ["open", "high", "low", "close", "volume"]])
        for key in INDEX_NAMES:
            b = store.bench.get(key)
            if b is None:
                continue
            if key in vals:
                b.loc[d] = list(vals[key])
            else:
                prev = b[b.index < d]
                if len(prev) and d not in b.index:
                    c = float(prev["close"].iloc[-1])
                    b.loc[d] = [c, c, c, c, 0.0]              # placeholder, replaced on a later run
                if key == "NIFTY50" or key in ("MIDCAP150", "SMALLCAP250"):
                    still.append(d)
            store.bench[key] = b.sort_index()
    still = sorted(set(still))
    store.meta["bench_filled"] = [x.date().isoformat() for x in still]
    if still:
        log.warning("index closes not found for %s - carried forward; will retry next run",
                    ", ".join(str(x.date()) for x in still))
    else:
        log.info("index closes updated for %d session(s)", len(todo))


def read_constituents(raw: bytes) -> pd.DataFrame:
    c = pd.read_csv(io.BytesIO(raw), dtype=str)
    c.columns = [x.strip() for x in c.columns]
    need = {"Company Name", "Industry", "Symbol"}
    if not need.issubset(c.columns) or len(c) < 450:
        raise ValueError("unexpected constituent file")
    return c


def refresh_universe(store: PriceStore, fetcher: Fetcher, cfg: dict, today: pd.Timestamp, force: bool = False):
    """Refresh the Nifty 500 list (weekly by default). New members get history from the GitHub tracker."""
    last = store.meta.get("universe_checked")
    days = int(cfg.get("refresh_days", 7))
    if not force and last and (today - pd.Timestamp(last)).days < days:
        return
    cons = None
    for url in (cfg.get("constituents_url"), cfg.get("constituents_url2")):
        if not url:
            continue
        st, raw = fetcher.get(url)
        if st == "ok":
            try:
                cons = read_constituents(raw)
                break
            except Exception as e:
                log.warning("constituent list from %s unusable: %s", url.split("/")[2], e)
    store.meta["universe_checked"] = today.date().isoformat()
    if cons is None:
        log.warning("could not download the Nifty 500 list - keeping the current universe")
        return
    U = store.universe.set_index("symbol")
    new_syms = [s.strip() for s in cons["Symbol"]]
    added = [s for s in new_syms if s not in U.index or U.at[s, "status"] != "ok"]
    removed = [s for s in U.index[U["status"] == "ok"] if s not in set(new_syms)]
    if not added and not removed:
        log.info("Nifty 500 list unchanged (%d stocks)", len(new_syms))
        return
    log.info("Nifty 500 list changed: +%d %s, -%d %s", len(added), added[:10], len(removed), removed[:10])
    store.notes.append(f"Nifty 500 list changed: added {', '.join(added) or 'none'}; removed {', '.join(removed) or 'none'}.")
    for s in removed:
        U.at[s, "status"] = "removed"
        U.at[s, "note"] = f"left the Nifty 500 (seen {today.date()}); prices still tracked for open calls"
    meta = cons.set_index(cons["Symbol"].str.strip())
    for s in added:
        if s not in store.prices or len(store.prices[s]) < 60:
            url = cfg.get("history_url", "").format(symbol_lower=s.lower(), symbol=s)
            st, raw = fetcher.get(url) if url else ("error", None)
            hist = read_tracker_history(raw) if st == "ok" else None
            if hist is None:
                log.warning("no history found for new member %s - it will be rated once it has 210 sessions", s)
                hist = pd.DataFrame(columns=PRICE_COLS, index=pd.DatetimeIndex([], name="date"))
            store.prices[s] = hist
            backfill_symbol(store, fetcher, cfg, s)
        U.loc[s, ["company", "industry", "status", "note"]] = [meta.at[s, "Company Name"], meta.at[s, "Industry"], "ok",
                                                              "added to the Nifty 500"]
    store.universe = U.reset_index()


def backfill_symbol(store: PriceStore, fetcher: Fetcher, cfg: dict, sym: str):
    """Fill the gap between a stock's last bar and the store's last session from (cached) bhavcopies."""
    df = store.prices[sym]
    start = df.index.max() + pd.Timedelta(days=1) if len(df) else store.last_date() - pd.Timedelta(days=400)
    for d in pd.bdate_range(start, store.last_date()):
        fname = f"sec_bhavdata_full_{d:%d%m%Y}.csv"
        st, raw, _ = store.fetch_file(fname, bhav_urls(cfg, d), fetcher, validate_bhav_for(d))
        if st == "ok":
            apply_bhav(store, d, read_bhav(raw), symbols=[sym])


def apply_manual_cas(store: PriceStore, path: Path):
    """data/manual_corporate_actions.csv (symbol,ex_date,factor,note): an adjustment the automatic check missed.
    factor multiplies all prices BEFORE ex_date (1:1 bonus = 0.5, split 10->2 = 0.2). Each row is applied once.
    To undo a wrong automatic adjustment with factor f, add a row with factor 1/f and the same ex_date."""
    if not path.exists():
        return
    m = pd.read_csv(path, comment="#", dtype=str, skipinitialspace=True)
    if not {"symbol", "ex_date", "factor"}.issubset(m.columns):
        log.warning("%s needs the columns symbol,ex_date,factor", path.name)
        return
    done = set(store.meta.get("manual_ca_applied", []))
    for _, r in m.dropna(subset=["symbol", "ex_date", "factor"]).iterrows():
        sym, ex, fac = r["symbol"].strip(), r["ex_date"].strip(), r["factor"].strip()
        key = f"{sym}|{ex}|{fac}"
        if key in done or sym not in store.prices:
            continue
        try:
            f, exd = float(fac), pd.Timestamp(ex)
        except ValueError:
            log.warning("manual corporate action row not understood: %s", key)
            continue
        df = store.prices[sym].copy()
        mask = df.index < exd
        for c in ("open", "high", "low", "close"):
            df.loc[mask, c] = df.loc[mask, c] * f
        for c in ("volume", "deliv_qty"):
            df.loc[mask, c] = df.loc[mask, c] / f
        store.prices[sym] = df
        store.ca_new.append({"symbol": sym, "ex_date": exd.date().isoformat(), "factor": round(f, 8),
                             "last_close_before": None, "nse_prev_close": None, "source": "manual"})
        store.ca_run.append((sym, exd.date().isoformat(), f))
        done.add(key)
        log.info("manual corporate action applied: %s ex %s factor %.6f", sym, exd.date(), f)
    store.meta["manual_ca_applied"] = sorted(done)


def apply_symbol_changes(store: PriceStore, path: Path):
    """data/symbol_changes.csv (old_symbol,new_symbol): carry a renamed stock's history to its new symbol."""
    if not path.exists():
        return
    ch = pd.read_csv(path, dtype=str).dropna()
    U = store.universe.set_index("symbol")
    for _, r in ch.iterrows():
        old, new = r["old_symbol"].strip(), r["new_symbol"].strip()
        if old in store.prices and new not in store.prices:
            store.prices[new] = store.prices.pop(old)
            if old in U.index:
                row = U.loc[old].copy()
                U = U.drop(index=old)
                U.loc[new] = row
                U.at[new, "note"] = f"renamed from {old}"
            log.info("symbol change applied: %s -> %s", old, new)
            store.notes.append(f"Symbol change applied: {old} -> {new}.")
    store.universe = U.reset_index().rename(columns={"index": "symbol"})


def refresh_universe_stats(store: PriceStore):
    U = store.universe.set_index("symbol")
    for s in U.index:
        d = store.prices.get(s)
        if d is not None and len(d):
            U.at[s, "first"], U.at[s, "last"] = d.index.min().date().isoformat(), d.index.max().date().isoformat()
            U.at[s, "bars"] = float(len(d))
    store.universe = U.reset_index()
