
"""v2 feature library - every indicator family we could compute from daily OHLCV + delivery data.

All features are causal (value at bar t uses bars <= t only) and scale-free
(percent distances, ratios, bounded oscillators), so they can be pooled across stocks.

Families
  1. Complete TA-Lib library (all 136 indicator functions of this build, default settings)
  2. Extra parameterisations of the popular ones (RSI 2..28, ROC 1..252, SMA/EMA 5..200, ...)
  3. Indicators outside TA-Lib: Ichimoku, Laguerre RSI, Connors RSI, Fisher, Schaff trend cycle,
     relative vigor index, Klinger, EMV, McGinley, ALMA, Alligator, Gann HiLo, Chandelier exit,
     volatility stop, Renko / three-line-break / Kagi / Heikin-Ashi states, Elder impulse,
     pivot points (classic, Camarilla, Fibonacci), volume profile (POC/VAH/VAL), anchored VWAPs,
     Hurst/variance ratio, volatility estimators (Parkinson, Garman-Klass, Rogers-Satchell,
     Yang-Zhang), ulcer index, choppiness, drawdowns, delivery-% analytics (NSE specific),
     relative strength vs Nifty (Mansfield, RRG ratio/momentum, beta, correlation),
     earnings-gap drift proxy, calendar effects
  4. All 61 TA-Lib candlestick patterns + tweezers, inside/outside bars, NR4/NR7/WR7, pin bars
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import talib
from talib import abstract

EPS = 1e-12
GROUPS = ["Momentum Indicators", "Overlap Studies", "Statistic Functions", "Volatility Indicators",
          "Volume Indicators", "Price Transform", "Cycle Indicators"]
SKIP = {"MAVP", "BETA", "CORREL"}          # need a second series -> computed against the Nifty below
PRICE_LEVEL = {"ACCBANDS", "BBANDS", "DEMA", "DONCHIAN", "EMA", "HMA", "HT_TRENDLINE", "KAMA", "KC", "MA", "MAMA",
               "MIDPOINT", "MIDPRICE", "RMA", "SAR", "SMA", "SUPERTREND", "T3", "TEMA", "TRIMA", "VWMA", "WMA",
               "ZLEMA", "LINEARREG", "LINEARREG_INTERCEPT", "TSF", "PERCENTILE", "AVGPRICE", "MEDPRICE",
               "TYPPRICE", "WCLPRICE", "VWAP", "HA"}
PRICE_DIFF = {"APO", "MACD", "MACDEXT", "MACDFIX", "MOM", "PLUS_DM", "MINUS_DM", "LINEARREG_SLOPE", "STDDEV", "ATR",
              "TRANGE", "AVGDEV", "ADR", "AO", "AC", "DPO", "QSTICK", "ERI", "HT_PHASOR"}
CUMULATIVE = {"AD", "OBV", "PVT", "WAD", "NVI", "PVI"}


def _f32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _div(a, b):
    a, b = _f32(a), _f32(b)
    out = np.full_like(a, np.nan)
    m = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > EPS)
    out[m] = a[m] / b[m]
    return out


def lag(x, n: int = 1) -> np.ndarray:
    """x shifted forward by n bars (NaN-filled at the start) - never wraps around like np.roll."""
    x = _f32(x)
    out = np.full_like(x, np.nan)
    if n < len(x):
        out[n:] = x[:len(x) - n]
    return out


def dist(c, x):
    """close relative to a price level, in percent."""
    return 100.0 * (_div(c, x) - 1.0)


def rma(x: np.ndarray, n: int) -> np.ndarray:
    """Wilder smoothing seeded with the SMA of the first n valid values (TA-Lib convention)."""
    x = _f32(x)
    out = np.full_like(x, np.nan)
    idx = np.where(np.isfinite(x))[0]
    if len(idx) < n:
        return out
    s = idx[0]
    # require a contiguous run for the seed
    seed_end = s + n
    if seed_end > len(x):
        return out
    out[seed_end - 1] = np.nanmean(x[s:seed_end])
    a = 1.0 / n
    prev = out[seed_end - 1]
    for i in range(seed_end, len(x)):
        v = x[i]
        if np.isfinite(v):
            prev = prev + a * (v - prev)
        out[i] = prev
    return out


def ema_np(x: np.ndarray, n: int) -> np.ndarray:
    x = _f32(x)
    out = np.full_like(x, np.nan)
    idx = np.where(np.isfinite(x))[0]
    if len(idx) < n:
        return out
    s = idx[0]
    if s + n > len(x):
        return out
    out[s + n - 1] = np.nanmean(x[s:s + n])
    a = 2.0 / (n + 1)
    prev = out[s + n - 1]
    for i in range(s + n, len(x)):
        v = x[i]
        if np.isfinite(v):
            prev = prev + a * (v - prev)
        out[i] = prev
    return out


# ----------------------------------------------------------------------------------------------
# 1. complete TA-Lib library
# ----------------------------------------------------------------------------------------------
def talib_block(o, h, l, c, v) -> dict:
    inp = {"open": o, "high": h, "low": l, "close": c, "volume": v}
    out = {}
    vol20 = pd.Series(v).rolling(20, min_periods=10).mean().to_numpy()
    volsum20 = pd.Series(v).rolling(20, min_periods=10).sum().to_numpy()
    for grp in GROUPS:
        for name in talib.get_function_groups()[grp]:
            if name in SKIP:
                continue
            fn = abstract.Function(name)
            try:
                res = fn(inp)
            except Exception:           # pragma: no cover - defensive
                continue
            outs = list(fn.info["output_names"])
            if not isinstance(res, list):
                res = [res]
            for oname, arr in zip(outs, res):
                arr = _f32(arr)
                key = f"ta_{name}" if len(outs) == 1 else f"ta_{name}_{oname}"
                if name == "SUPERTREND" and oname == "trend":
                    out[key] = arr
                elif name == "HA":
                    continue
                elif name == "SAREXT":
                    out[key + "_dir"] = np.sign(arr)
                    out[key] = dist(c, np.abs(arr))
                elif name in PRICE_LEVEL:
                    out[key] = dist(c, arr)
                elif name in PRICE_DIFF:
                    out[key] = 100.0 * _div(arr, c)
                elif name == "VAR":
                    out[key] = 100.0 * _div(np.sqrt(np.maximum(arr, 0)), c)
                elif name in ("AD", "OBV", "PVT"):
                    out[key + "_flow20"] = _div(arr - lag(arr, 20), volsum20)
                    out[key + "_flow20"][:20] = np.nan
                elif name == "WAD":
                    d20 = arr - lag(arr, 20)
                    d20[:20] = np.nan
                    out[key + "_chg20"] = 100.0 * _div(d20, c)
                elif name in ("NVI", "PVI"):
                    out[key + "_vs_ema255"] = dist(arr, ema_np(arr, 255))
                elif name == "ADOSC":
                    out[key] = _div(arr, vol20)
                elif name == "EFI":
                    out[key] = _div(arr, c * vol20) * 100.0
                elif name == "MARKETFI":
                    out[key] = np.log1p(np.maximum(_div(arr * vol20, c), 0))
                elif name == "LINEARREG_ANGLE":
                    out[key] = _f32(talib.LINEARREG_ANGLE(100.0 * np.log(c), timeperiod=14))
                elif name == "FRACTAL":
                    out[key] = arr / 100.0
                else:
                    out[key] = arr
            # HA handled separately below
    ha_o, ha_h, ha_l, ha_c = (_f32(x) for x in talib.HA(o, h, l, c))
    out["ha_body_pct"] = 100.0 * _div(ha_c - ha_o, ha_o)
    out["ha_bull"] = (ha_c > ha_o).astype(float)
    out["ha_no_lower_shadow"] = ((ha_c > ha_o) & (np.abs(np.minimum(ha_o, ha_c) - ha_l) <= 1e-9 * ha_c)).astype(float)
    out["ha_no_upper_shadow"] = ((ha_c < ha_o) & (np.abs(ha_h - np.maximum(ha_o, ha_c)) <= 1e-9 * ha_c)).astype(float)
    bull = ha_c > ha_o
    streak = np.zeros(len(c))
    for i in range(1, len(c)):
        streak[i] = (streak[i - 1] + 1 if bull[i] else 1) if bull[i] == bull[i - 1] else 1
    out["ha_streak"] = np.where(bull, streak, -streak)
    return out


# ----------------------------------------------------------------------------------------------
# 2./3. extra parameterisations and non-TA-Lib indicators
# ----------------------------------------------------------------------------------------------
def rolling_max(x, n, mp=None):
    return pd.Series(x).rolling(n, min_periods=mp or n).max().to_numpy()


def rolling_min(x, n, mp=None):
    return pd.Series(x).rolling(n, min_periods=mp or n).min().to_numpy()


def days_since_extreme(x: np.ndarray, n: int, kind: str) -> np.ndarray:
    s = pd.Series(x)
    r = s.rolling(n, min_periods=n)
    arg = r.apply(np.argmax if kind == "max" else np.argmin, raw=True)
    return (n - 1 - arg).to_numpy()


def laguerre_rsi(c, gamma=0.5):
    l0 = l1 = l2 = l3 = c[0]
    out = np.full(len(c), np.nan)
    for i in range(len(c)):
        p0, p1, p2 = l0, l1, l2
        l0 = (1 - gamma) * c[i] + gamma * p0
        l1 = -gamma * l0 + p0 + gamma * p1
        l2 = -gamma * l1 + p1 + gamma * p2
        l3 = -gamma * l2 + p2 + gamma * l3
        cu = max(l0 - l1, 0) + max(l1 - l2, 0) + max(l2 - l3, 0)
        cd = max(l1 - l0, 0) + max(l2 - l1, 0) + max(l3 - l2, 0)
        out[i] = 100.0 * cu / (cu + cd) if cu + cd > 0 else 50.0
    out[:20] = np.nan
    return out


def connors_rsi(c, n_rsi=3, n_streak=2, n_rank=100):
    r1 = _f32(talib.RSI(c, n_rsi))
    streak = np.zeros(len(c))
    for i in range(1, len(c)):
        if c[i] > c[i - 1]:
            streak[i] = streak[i - 1] + 1 if streak[i - 1] > 0 else 1
        elif c[i] < c[i - 1]:
            streak[i] = streak[i - 1] - 1 if streak[i - 1] < 0 else -1
    r2 = _f32(talib.RSI(streak, n_streak))
    ret = pd.Series(c).pct_change()
    rank = ret.rolling(n_rank + 1).apply(lambda w: 100.0 * (w[:-1] < w[-1]).mean(), raw=True).to_numpy()
    return (r1 + r2 + rank) / 3.0, streak


def fisher_transform(h, l, n=10):
    med = (h + l) / 2
    hh, ll = rolling_max(med, n), rolling_min(med, n)
    val = np.zeros(len(h))
    fish = np.full(len(h), np.nan)
    prev_v, prev_f = 0.0, 0.0
    for i in range(len(h)):
        if not np.isfinite(hh[i]) or hh[i] - ll[i] <= 0:
            continue
        x = 0.66 * ((med[i] - ll[i]) / (hh[i] - ll[i]) - 0.5) + 0.67 * prev_v
        x = min(max(x, -0.999), 0.999)
        f = 0.5 * np.log((1 + x) / (1 - x)) + 0.5 * prev_f
        val[i], fish[i] = x, f
        prev_v, prev_f = x, f
    return fish


def schaff_trend_cycle(c, cycle=10, fast=23, slow=50):
    macd = ema_np(c, fast) - ema_np(c, slow)
    s = pd.Series(macd)

    def stoch_pass(x):
        lo, hi = x.rolling(cycle, min_periods=cycle).min(), x.rolling(cycle, min_periods=cycle).max()
        k = 100 * (x - lo) / (hi - lo).replace(0, np.nan)
        k = k.ffill()
        out = np.full(len(x), np.nan)
        prev = np.nan
        for i, v in enumerate(k.to_numpy()):
            if np.isnan(v):
                continue
            prev = v if np.isnan(prev) else prev + 0.5 * (v - prev)
            out[i] = prev
        return pd.Series(out)
    return stoch_pass(stoch_pass(s)).to_numpy()


def relative_vigor_index(o, h, l, c, n=10):
    num = (c - o) + 2 * lag(c - o, 1) + 2 * lag(c - o, 2) + lag(c - o, 3)
    den = (h - l) + 2 * lag(h - l, 1) + 2 * lag(h - l, 2) + lag(h - l, 3)
    num[:3], den[:3] = np.nan, np.nan
    rvi = _div(pd.Series(num).rolling(n).mean().to_numpy(), pd.Series(den).rolling(n).mean().to_numpy())
    sig = (rvi + 2 * lag(rvi, 1) + 2 * lag(rvi, 2) + lag(rvi, 3)) / 6
    sig[:3] = np.nan
    return rvi, sig


def klinger(h, l, c, v, fast=34, slow=55, sig=13):
    tp = h + l + c
    trend = np.where(tp > lag(tp, 1), 1.0, -1.0)
    trend[0] = 1.0
    dm = h - l
    cm = np.zeros(len(c))
    for i in range(1, len(c)):
        cm[i] = (cm[i - 1] + dm[i]) if trend[i] == trend[i - 1] else (dm[i - 1] + dm[i])
    vf = v * np.abs(2 * (_div(dm, cm) - 1)) * trend * 100
    vf = np.nan_to_num(vf, nan=0.0, posinf=0.0, neginf=0.0)
    kvo = ema_np(vf, fast) - ema_np(vf, slow)
    return kvo, ema_np(np.nan_to_num(kvo), sig)


def mcginley(c, n=14):
    out = np.full(len(c), np.nan)
    md = c[0]
    for i in range(len(c)):
        md = md + (c[i] - md) / (n * (c[i] / md) ** 4) if md > 0 else c[i]
        out[i] = md
    out[:n] = np.nan
    return out


def alma(c, n=9, offset=0.85, sigma=6.0):
    m = offset * (n - 1)
    s = n / sigma
    w = np.exp(-((np.arange(n) - m) ** 2) / (2 * s * s))
    w /= w.sum()
    return pd.Series(c).rolling(n).apply(lambda x: float(np.dot(x, w)), raw=True).to_numpy()


def smma(x, n):
    return rma(x, n)


def volatility_stop(h, l, c, atr, mult=3.0):
    """Wilder's volatility system: trailing stop at mult*ATR from the extreme close; returns direction."""
    n = len(c)
    d = np.zeros(n)
    stop = np.full(n, np.nan)
    trend, ext = 1, c[0]
    for i in range(n):
        a = atr[i]
        if not np.isfinite(a):
            continue
        if trend == 1:
            ext = max(ext, c[i])
            st = ext - mult * a
            if c[i] < st:
                trend, ext, st = -1, c[i], c[i] + mult * a
        else:
            ext = min(ext, c[i])
            st = ext + mult * a
            if c[i] > st:
                trend, ext, st = 1, c[i], c[i] - mult * a
        d[i], stop[i] = trend, st
    return d, stop


def renko_state(c, box):
    """Causal close-based Renko with a per-bar box size; returns direction and bricks in the run."""
    n = len(c)
    direction = np.zeros(n)
    run = np.zeros(n)
    base = c[0]
    dirn, cnt = 0, 0
    for i in range(n):
        b = box[i]
        if not np.isfinite(b) or b <= 0:
            direction[i], run[i] = dirn, cnt
            continue
        if dirn >= 0:
            while c[i] >= base + b:
                base += b
                cnt = cnt + 1 if dirn == 1 else 1
                dirn = 1
            if c[i] <= base - 2 * b:
                k = int((base - b - c[i]) // b)
                base -= b * (k + 1)
                dirn, cnt = -1, k
        else:
            while c[i] <= base - b:
                base -= b
                cnt = cnt + 1 if dirn == -1 else 1
                dirn = -1
            if c[i] >= base + 2 * b:
                k = int((c[i] - base - b) // b)
                base += b * (k + 1)
                dirn, cnt = 1, k
        direction[i], run[i] = dirn, cnt
    return direction, run


def three_line_break(c, lines=3):
    n = len(c)
    out = np.zeros(n)
    blocks = []                                   # list of (low, high, dir)
    last = c[0]
    dirn = 0
    for i in range(1, n):
        if not blocks:
            if c[i] != last:
                dirn = 1 if c[i] > last else -1
                blocks.append((min(last, c[i]), max(last, c[i]), dirn))
            out[i] = dirn
            continue
        lo, hi, d = blocks[-1]
        recent = blocks[-lines:]
        if d == 1:
            if c[i] > hi:
                blocks.append((hi, c[i], 1))
            elif c[i] < min(b[0] for b in recent):
                blocks.append((c[i], lo, -1))
                dirn = -1
        else:
            if c[i] < lo:
                blocks.append((c[i], lo, -1))
            elif c[i] > max(b[1] for b in recent):
                blocks.append((hi, c[i], 1))
                dirn = 1
        dirn = blocks[-1][2]
        out[i] = dirn
        if len(blocks) > 50:
            blocks = blocks[-10:]
    return out


def kagi_state(c, rev_pct=4.0):
    """Kagi yang (1) / yin (-1): a line turns yang when it rises above the prior shoulder."""
    n = len(c)
    out = np.zeros(n)
    dirn, ext = 1, c[0]
    shoulder, waist = c[0], c[0]
    state = 1
    for i in range(n):
        if dirn == 1:
            if c[i] > ext:
                ext = c[i]
            elif c[i] <= ext * (1 - rev_pct / 100):
                shoulder, dirn, ext = ext, -1, c[i]
        else:
            if c[i] < ext:
                ext = c[i]
            elif c[i] >= ext * (1 + rev_pct / 100):
                waist, dirn, ext = ext, 1, c[i]
        if c[i] > shoulder:
            state = 1
        elif c[i] < waist:
            state = -1
        out[i] = state
    return out


def volume_profile(h, l, c, v, window, step=5, bins=40):
    """Rolling volume profile: point of control and 70% value area; distances in ATR-free percent."""
    n = len(c)
    poc = np.full(n, np.nan)
    vah = np.full(n, np.nan)
    val = np.full(n, np.nan)
    tp = (h + l + c) / 3.0
    for t in range(window - 1, n, step):
        s = slice(t - window + 1, t + 1)
        lo, hi = np.nanmin(l[s]), np.nanmax(h[s])
        if not (hi > lo):
            continue
        hist, edges = np.histogram(tp[s], bins=bins, range=(lo, hi), weights=np.nan_to_num(v[s]))
        if hist.sum() <= 0:
            continue
        k = int(np.argmax(hist))
        mids = (edges[:-1] + edges[1:]) / 2
        poc[t] = mids[k]
        # value area: expand from POC until 70% of volume
        tot, acc, lo_i, hi_i = hist.sum(), hist[k], k, k
        while acc < 0.7 * tot and (lo_i > 0 or hi_i < bins - 1):
            up = hist[hi_i + 1] if hi_i < bins - 1 else -1
            dn = hist[lo_i - 1] if lo_i > 0 else -1
            if up >= dn:
                hi_i += 1
                acc += hist[hi_i]
            else:
                lo_i -= 1
                acc += hist[lo_i]
        vah[t], val[t] = edges[hi_i + 1], edges[lo_i]
    f = lambda x: pd.Series(x).ffill(limit=step - 1).to_numpy()  # noqa: E731
    return f(poc), f(vah), f(val)


def anchored_vwap(tp, v, anchor_idx):
    pv = np.nancumsum(tp * v)
    cv = np.nancumsum(v)
    out = np.full(len(tp), np.nan)
    ok = anchor_idx >= 0
    a = anchor_idx[ok].astype(int)
    t = np.where(ok)[0]
    num = pv[t] - np.where(a > 0, pv[np.maximum(a - 1, 0)], 0.0)
    den = cv[t] - np.where(a > 0, cv[np.maximum(a - 1, 0)], 0.0)
    out[t] = _div(num, den)
    return out


def pivots_prev_period(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Classic / Camarilla / Fibonacci pivot levels of the previous week or month, aligned to each day."""
    key = df.index.to_period(freq)
    g = df.groupby(key).agg(high=("high", "max"), low=("low", "min"), close=("close", "last")).shift(1)
    H, L, C = g["high"], g["low"], g["close"]
    P = (H + L + C) / 3
    lv = pd.DataFrame({"P": P, "R1": 2 * P - L, "S1": 2 * P - H, "R2": P + (H - L), "S2": P - (H - L),
                       "CH4": C + 1.1 * (H - L) / 2, "CL4": C - 1.1 * (H - L) / 2,
                       "CH3": C + 1.1 * (H - L) / 4, "CL3": C - 1.1 * (H - L) / 4,
                       "FR1": P + 0.382 * (H - L), "FS1": P - 0.382 * (H - L)})
    return lv.reindex(key).set_axis(df.index)


# ----------------------------------------------------------------------------------------------
# 4. candlesticks
# ----------------------------------------------------------------------------------------------
def candle_block(o, h, l, c, atr) -> dict:
    out = {}
    for name in talib.get_function_groups()["Pattern Recognition"]:
        out[f"cdl_{name[3:].lower()}"] = _f32(getattr(talib, name)(o, h, l, c)) / 100.0
    rng = h - l
    body = np.abs(c - o)
    up_sh = h - np.maximum(o, c)
    lo_sh = np.minimum(o, c) - l
    ph, pl = lag(h, 1), lag(l, 1)
    ph[0], pl[0] = np.nan, np.nan
    tol = 0.1 * atr
    out["cdlx_tweezer_bottom"] = ((np.abs(l - pl) <= tol) & (lag(c, 1) < lag(o, 1)) & (c > o)).astype(float)
    out["cdlx_tweezer_top"] = ((np.abs(h - ph) <= tol) & (lag(c, 1) > lag(o, 1)) & (c < o)).astype(float)
    out["cdlx_inside_bar"] = ((h <= ph) & (l >= pl)).astype(float)
    outside = (h > ph) & (l < pl)
    out["cdlx_outside_bull"] = (outside & (c > lag(h, 1))).astype(float)
    out["cdlx_outside_bear"] = (outside & (c < lag(l, 1))).astype(float)
    out["cdlx_pin_bull"] = ((lo_sh >= 2 * np.maximum(body, 1e-9)) & (lo_sh >= 0.6 * rng) & (up_sh <= 0.2 * rng)).astype(float)
    out["cdlx_pin_bear"] = ((up_sh >= 2 * np.maximum(body, 1e-9)) & (up_sh >= 0.6 * rng) & (lo_sh <= 0.2 * rng)).astype(float)
    rs = pd.Series(rng)
    out["cdlx_nr4"] = (rs <= rs.rolling(4).min()).astype(float).to_numpy()
    out["cdlx_nr7"] = (rs <= rs.rolling(7).min()).astype(float).to_numpy()
    out["cdlx_wr7"] = (rs >= rs.rolling(7).max()).astype(float).to_numpy()
    out["cdlx_ibs"] = _div(c - l, rng)                     # internal bar strength
    out["cdlx_body_atr"] = _div(c - o, atr)
    out["cdlx_upper_shadow_atr"] = _div(up_sh, atr)
    out["cdlx_lower_shadow_atr"] = _div(lo_sh, atr)
    out["cdlx_range_atr"] = _div(rng, atr)
    return out


# ----------------------------------------------------------------------------------------------
# main entry
# ----------------------------------------------------------------------------------------------
def compute_features(df: pd.DataFrame, bench: pd.DataFrame) -> pd.DataFrame:
    """df: open, high, low, close, volume, deliv_qty, deliv_per, trades (adjusted); bench: Nifty 50 OHLC."""
    o, h, l, c, v = (df[x].to_numpy(float) for x in ("open", "high", "low", "close", "volume"))
    v = np.nan_to_num(v, nan=0.0)
    idx = df.index
    F: dict[str, np.ndarray] = {}
    F.update(talib_block(o, h, l, c, v))

    atr14 = _f32(talib.ATR(h, l, c, 14))
    atr50 = _f32(talib.ATR(h, l, c, 50))
    cs = pd.Series(c)
    ret1 = cs.pct_change().to_numpy()
    lr = np.log(cs).diff().to_numpy()

    # --- extra parameterisations -------------------------------------------------------------
    for n in (2, 3, 5, 7, 9, 21, 28):
        F[f"rsi_{n}"] = _f32(talib.RSI(c, n))
    for n in (1, 2, 3, 5, 21, 42, 63, 126, 189, 252):
        F[f"ret_{n}"] = 100.0 * (_div(c, lag(c, n)) - 1)
        F[f"ret_{n}"][:n] = np.nan
    F["mom_12_1"] = 100.0 * (_div(lag(c, 21), lag(c, 252)) - 1)
    F["mom_12_1"][:252] = np.nan
    F["mom_6_1"] = 100.0 * (_div(lag(c, 21), lag(c, 126)) - 1)
    F["mom_6_1"][:126] = np.nan
    sma = {n: _f32(talib.SMA(c, n)) for n in (5, 10, 20, 50, 100, 150, 200)}
    ema = {n: _f32(talib.EMA(c, n)) for n in (5, 8, 9, 10, 13, 20, 21, 26, 34, 50, 55, 89, 100, 200)}
    for n, x in sma.items():
        F[f"dist_sma{n}"] = dist(c, x)
        F[f"dist_sma{n}_atr"] = _div(c - x, atr14)
    for n, x in ema.items():
        F[f"dist_ema{n}"] = dist(c, x)
    for n, k in ((20, 5), (50, 10), (150, 20), (200, 20)):
        F[f"slope_sma{n}"] = 100.0 * (_div(sma[n], lag(sma[n], k)) - 1) / k
    F["ema9_gt_ema21"] = (ema[9] > ema[21]).astype(float)
    F["ema20_gt_sma50"] = (ema[20] > sma[50]).astype(float)
    F["sma50_gt_sma200"] = (sma[50] > sma[200]).astype(float)
    F["ma_stack_bull"] = ((c > ema[20]) & (ema[20] > sma[50]) & (sma[50] > sma[150]) & (sma[150] > sma[200])).astype(float)
    F["ma_stack_bear"] = ((c < ema[20]) & (ema[20] < sma[50]) & (sma[50] < sma[150]) & (sma[150] < sma[200])).astype(float)
    gc = pd.Series(np.sign(sma[50] - sma[200]))
    chg = gc.ne(gc.shift()).cumsum()
    F["days_since_ma50_200_cross"] = np.minimum(gc.groupby(chg).cumcount().to_numpy(), 500) * np.sign(gc.to_numpy())
    for n in (5, 14, 20, 50):
        F[f"atr_pct_{n}"] = 100.0 * _div(_f32(talib.ATR(h, l, c, n)), c)
    F["atr_ratio_5_50"] = _div(_f32(talib.ATR(h, l, c, 5)), atr50)
    F["atr_pctile_252"] = pd.Series(F["atr_pct_14"]).rolling(252, min_periods=120).rank(pct=True).to_numpy()
    for n in (7, 28):
        F[f"adx_{n}"] = _f32(talib.ADX(h, l, c, n))
    F["di_diff_14"] = _f32(talib.PLUS_DI(h, l, c, 14)) - _f32(talib.MINUS_DI(h, l, c, 14))
    for n in (20, 50):
        F[f"cci_{n}"] = _f32(talib.CCI(h, l, c, n))
    for n in (7, 21):
        F[f"mfi_{n}"] = _f32(talib.MFI(h, l, c, v, n))
    for n in (10, 28):
        F[f"willr_{n}"] = _f32(talib.WILLR(h, l, c, n))
    for fk, sk, sd in ((14, 3, 3), (21, 5, 5)):
        k, d = talib.STOCH(h, l, c, fk, sk, 0, sd, 0)
        F[f"stoch_k_{fk}"], F[f"stoch_d_{fk}"] = _f32(k), _f32(d)
    for n, dev in ((20, 2.0), (50, 2.0), (20, 1.0)):
        u, m, lo = (_f32(x) for x in talib.BBANDS(c, n, dev, dev, 0))
        F[f"bb_pctb_{n}_{int(dev)}"] = _div(c - lo, u - lo)
        F[f"bb_width_{n}_{int(dev)}"] = 100.0 * _div(u - lo, m)
    F["bb_width_pctile_120"] = pd.Series(F["bb_width_20_2"]).rolling(120, min_periods=60).rank(pct=True).to_numpy()
    ku, km, kl = (_f32(x) for x in talib.KC(h, l, c, 20, 20, 1.5))
    bu, bm, bl = (_f32(x) for x in talib.BBANDS(c, 20, 2.0, 2.0, 0))
    sq = (bu < ku) & (bl > kl)
    F["ttm_squeeze_on"] = sq.astype(float)
    sqs = pd.Series(sq.astype(int))
    F["ttm_squeeze_len"] = sqs.groupby(sqs.ne(sqs.shift()).cumsum()).cumcount().to_numpy() * sq
    F["kc_pos"] = _div(c - kl, ku - kl)
    for n in (10, 20, 55):
        dh, dl = rolling_max(h, n), rolling_min(l, n)
        F[f"donch_pos_{n}"] = _div(c - dl, dh - dl)
        F[f"donch_break_{n}"] = (c > lag(dh, 1)).astype(float) - (c < lag(dl, 1)).astype(float)
    for per, mult in ((7, 3.0), (11, 2.0), (20, 5.0)):
        st, tr = talib.SUPERTREND(h, l, c, per, mult)
        F[f"supertrend_{per}_{int(mult)}_dir"] = _f32(tr)
        F[f"supertrend_{per}_{int(mult)}_dist"] = dist(c, st)
    # --- Ichimoku (9/26/52) --------------------------------------------------------------------
    tenkan = (rolling_max(h, 9) + rolling_min(l, 9)) / 2
    kijun = (rolling_max(h, 26) + rolling_min(l, 26)) / 2
    span_a_now = (tenkan + kijun) / 2
    span_b_now = (rolling_max(h, 52) + rolling_min(l, 52)) / 2
    span_a = lag(span_a_now, 25)
    span_b = lag(span_b_now, 25)
    span_a[:25], span_b[:25] = np.nan, np.nan
    ctop, cbot = np.fmax(span_a, span_b), np.fmin(span_a, span_b)
    F["ichi_tk_diff_atr"] = _div(tenkan - kijun, atr14)
    F["ichi_close_kijun_atr"] = _div(c - kijun, atr14)
    F["ichi_above_cloud_atr"] = _div(c - ctop, atr14)
    F["ichi_below_cloud_atr"] = _div(cbot - c, atr14)
    F["ichi_in_cloud"] = ((c <= ctop) & (c >= cbot)).astype(float)
    F["ichi_cloud_thick_pct"] = 100.0 * _div(ctop - cbot, c)
    F["ichi_future_cloud_bull"] = (span_a_now > span_b_now).astype(float)
    chik = lag(c, 25)
    chik[:25] = np.nan
    F["ichi_chikou_vs_price"] = dist(c, chik)
    # --- other trend tools -----------------------------------------------------------------------
    F["hma_20_slope"] = 100.0 * (_div(_f32(talib.HMA(c, 20)), lag(_f32(talib.HMA(c, 20)), 1)) - 1)
    F["dist_mcginley14"] = dist(c, mcginley(c, 14))
    F["dist_alma9"] = dist(c, alma(c, 9))
    med = (h + l) / 2
    jaw, teeth, lips = (lag(smma(med, p), s) for p, s in ((13, 8), (8, 5), (5, 3)))
    F["alligator_bull"] = ((lips > teeth) & (teeth > jaw)).astype(float) - ((lips < teeth) & (teeth < jaw)).astype(float)
    F["alligator_spread_atr"] = _div(np.abs(lips - jaw), atr14)
    hi_avg, lo_avg = _f32(talib.SMA(h, 10)), _f32(talib.SMA(l, 10))
    gann = np.zeros(len(c))
    for i in range(1, len(c)):
        gann[i] = 1 if c[i] > hi_avg[i - 1] else (-1 if c[i] < lo_avg[i - 1] else gann[i - 1])
    F["gann_hilo_dir"] = gann
    atr22 = _f32(talib.ATR(h, l, c, 22))
    F["chandelier_long_dist_atr"] = _div(c - (rolling_max(h, 22) - 3 * atr22), atr14)
    F["chandelier_short_dist_atr"] = _div((rolling_min(l, 22) + 3 * atr22) - c, atr14)
    vs_dir, vs_stop = volatility_stop(h, l, c, atr14, 3.0)
    F["volstop_dir"], F["volstop_dist_atr"] = vs_dir, _div(np.abs(c - vs_stop), atr14)
    rd, rr = renko_state(c, atr14)
    F["renko_dir"], F["renko_run"] = rd, rr * rd
    F["tlb_dir"] = three_line_break(c)
    F["kagi_state"] = kagi_state(c, 4.0)
    macd_h = _f32(talib.MACD(c, 12, 26, 9)[2])
    F["elder_impulse"] = np.sign(ema[13] - lag(ema[13], 1)) + np.sign(macd_h - lag(macd_h, 1))
    lr_s = pd.Series(np.log(c))
    for n in (50, 100, 200):
        x = np.arange(n)
        xm = x - x.mean()
        slope = lr_s.rolling(n).apply(lambda y: float(np.dot(xm, y - y.mean()) / np.dot(xm, xm)), raw=True)
        corr = lr_s.rolling(n).corr(pd.Series(np.arange(len(c)), dtype=float))
        F[f"linreg_slope_{n}"] = 100.0 * slope.to_numpy()
        F[f"linreg_r2_{n}"] = (corr ** 2).to_numpy()
    # --- momentum extras --------------------------------------------------------------------------
    F["laguerre_rsi"] = laguerre_rsi(c, 0.5)
    crsi, streak = connors_rsi(c)
    F["connors_rsi"], F["updown_streak"] = crsi, streak
    F["fisher_10"] = fisher_transform(h, l, 10)
    F["schaff_tc"] = schaff_trend_cycle(c)
    rvi, rvs = relative_vigor_index(o, h, l, c)
    F["rvgi"], F["rvgi_minus_sig"] = rvi, rvi - rvs
    kvo, kvs = klinger(h, l, c, v)
    vol20 = pd.Series(v).rolling(20, min_periods=10).mean().to_numpy()
    F["klinger_norm"] = _div(kvo, vol20 * c) * 100
    F["klinger_minus_sig"] = _div(kvo - kvs, vol20 * c) * 100
    box = (h + l) / 2 - (lag(h, 1) + lag(l, 1)) / 2
    br = _div(v / 1e6, h - l)
    emv = _div(box, br)
    F["emv_14"] = pd.Series(emv).rolling(14).mean().to_numpy() / np.maximum(c, EPS) * 100
    F["kst"] = (pd.Series(100 * (_div(c, lag(c, 10)) - 1)).rolling(10).mean() * 1
                + pd.Series(100 * (_div(c, lag(c, 15)) - 1)).rolling(10).mean() * 2
                + pd.Series(100 * (_div(c, lag(c, 20)) - 1)).rolling(10).mean() * 3
                + pd.Series(100 * (_div(c, lag(c, 30)) - 1)).rolling(15).mean() * 4).to_numpy()
    F["kst_minus_sig"] = F["kst"] - pd.Series(F["kst"]).rolling(9).mean().to_numpy()
    F["price_pctile_252"] = pd.Series(c).rolling(252, min_periods=200).rank(pct=True).to_numpy()
    F["zscore_20"] = _div(c - sma[20], pd.Series(c).rolling(20).std().to_numpy())
    F["zscore_50"] = _div(c - sma[50], pd.Series(c).rolling(50).std().to_numpy())
    # --- 52-week / multi-year structure --------------------------------------------------------------
    hi252, lo252 = rolling_max(h, 252, 200), rolling_min(l, 252, 200)
    F["dist_52w_high"] = dist(c, hi252)
    F["dist_52w_low"] = dist(c, lo252)
    F["pos_52w_range"] = _div(c - lo252, hi252 - lo252)
    F["days_since_52w_high"] = days_since_extreme(h, 252, "max")
    F["days_since_52w_low"] = days_since_extreme(l, 252, "min")
    ath = np.fmax.accumulate(np.nan_to_num(h, nan=0))
    F["dist_multi_year_high"] = dist(c, ath)
    F["new_52w_high"] = (h >= hi252).astype(float)
    F["new_52w_low"] = (l <= lo252).astype(float)
    # --- volatility estimators -------------------------------------------------------------------------
    for n in (10, 20, 60, 252):
        F[f"hv_{n}"] = pd.Series(lr).rolling(n, min_periods=int(n * 0.8)).std().to_numpy() * np.sqrt(252) * 100
    F["hv_ratio_10_60"] = _div(F["hv_10"], F["hv_60"])
    hl = np.log(_div(h, l))
    co = np.log(_div(c, o))
    oc = np.log(_div(o, lag(c, 1)))
    F["vol_parkinson_20"] = np.sqrt(pd.Series(hl ** 2).rolling(20).mean().to_numpy() / (4 * np.log(2)) * 252) * 100
    gk = 0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2
    F["vol_garman_klass_20"] = np.sqrt(np.maximum(pd.Series(gk).rolling(20).mean().to_numpy(), 0) * 252) * 100
    rs_ = np.log(_div(h, c)) * np.log(_div(h, o)) + np.log(_div(l, c)) * np.log(_div(l, o))
    F["vol_rogers_satchell_20"] = np.sqrt(np.maximum(pd.Series(rs_).rolling(20).mean().to_numpy(), 0) * 252) * 100
    k_yz = 0.34 / (1.34 + 21 / 19)
    yz = (pd.Series(oc).rolling(20).var() + k_yz * pd.Series(co).rolling(20).var()
          + (1 - k_yz) * pd.Series(rs_).rolling(20).mean())
    F["vol_yang_zhang_20"] = np.sqrt(np.maximum(yz.to_numpy(), 0) * 252) * 100
    dd14 = 100 * (_div(c, rolling_max(c, 14)) - 1)
    F["ulcer_14"] = np.sqrt(pd.Series(dd14 ** 2).rolling(14).mean().to_numpy())
    tr = _f32(talib.TRANGE(h, l, c))
    F["chop_14"] = 100 * np.log10(_div(pd.Series(tr).rolling(14).sum().to_numpy(),
                                       rolling_max(h, 14) - rolling_min(l, 14))) / np.log10(14)
    for n in (60, 252):
        F[f"maxdd_{n}"] = 100 * (pd.Series(c).rolling(n, min_periods=int(n * 0.8))
                                 .apply(lambda w: (w / np.maximum.accumulate(w)).min() - 1, raw=True).to_numpy())
    r_s = pd.Series(ret1)
    F["skew_60"] = r_s.rolling(60).skew().to_numpy()
    F["kurt_60"] = r_s.rolling(60).kurt().to_numpy()
    F["autocorr1_60"] = r_s.rolling(60).corr(r_s.shift(1)).to_numpy()
    r10 = pd.Series(np.log(c)).diff(10)
    vr = _div(r10.rolling(120, min_periods=100).var().to_numpy(),
              10 * pd.Series(lr).rolling(120, min_periods=100).var().to_numpy())
    F["variance_ratio_10"] = vr
    F["hurst_vr"] = 0.5 + 0.5 * np.log(np.maximum(vr, 1e-6)) / np.log(10)
    gap = _div(o - lag(c, 1), atr14)
    gap[0] = np.nan
    F["gap_atr"] = gap
    F["gap_up_full"] = (l > lag(h, 1)).astype(float)
    F["gap_dn_full"] = (h < lag(l, 1)).astype(float)
    # --- volume & delivery -------------------------------------------------------------------------------
    vs = pd.Series(v)
    v50 = vs.rolling(50, min_periods=25).mean().to_numpy()
    F["vol_ratio_20"] = _div(v, vol20)
    F["vol_ratio_50"] = _div(v, v50)
    F["vol_dryup_10_50"] = _div(vs.rolling(10).mean().to_numpy(), v50)
    F["vol_z_20"] = _div(v - vol20, vs.rolling(20).std().to_numpy())
    upv = np.where(ret1 > 0, v, 0.0)
    dnv = np.where(ret1 < 0, v, 0.0)
    for n in (20, 50):
        F[f"updown_vol_{n}"] = _div(pd.Series(upv).rolling(n).sum().to_numpy(), pd.Series(dnv).rolling(n).sum().to_numpy())
    to = c * v / 1e7
    F["log_turnover_20"] = np.log10(np.maximum(pd.Series(to).rolling(20, min_periods=10).mean().to_numpy(), 1e-3))
    dp = df["deliv_per"].to_numpy(float)
    dq = np.nan_to_num(df["deliv_qty"].to_numpy(float), nan=np.nan)
    dps = pd.Series(dp)
    F["deliv_pct"] = dp
    F["deliv_pct_5"] = dps.rolling(5, min_periods=3).mean().to_numpy()
    F["deliv_pct_20"] = dps.rolling(20, min_periods=10).mean().to_numpy()
    F["deliv_pct_5_minus_20"] = F["deliv_pct_5"] - F["deliv_pct_20"]
    F["deliv_pct_z60"] = _div(dp - dps.rolling(60, min_periods=30).mean().to_numpy(), dps.rolling(60, min_periods=30).std().to_numpy())
    F["deliv_qty_ratio_20"] = _div(dq, pd.Series(dq).rolling(20, min_periods=10).mean().to_numpy())
    F["deliv_up_day_spike"] = ((F["deliv_qty_ratio_20"] >= 1.5) & (ret1 > 0)).astype(float)
    tr_n = df["trades"].to_numpy(float)
    F["trades_ratio_20"] = _div(tr_n, pd.Series(tr_n).rolling(20, min_periods=10).mean().to_numpy())
    qpt = _div(v, tr_n)
    F["qty_per_trade_ratio_20"] = _div(qpt, pd.Series(qpt).rolling(20, min_periods=10).mean().to_numpy())
    tp = (h + l + c) / 3
    for n in (20, 50):
        F[f"dist_vwap_{n}"] = dist(c, _div(pd.Series(tp * v).rolling(n).sum().to_numpy(), vs.rolling(n).sum().to_numpy()))
    ar = np.arange(len(c))
    a_hi = ar - np.nan_to_num(F["days_since_52w_high"], nan=-1)
    a_lo = ar - np.nan_to_num(F["days_since_52w_low"], nan=-1)
    a_hi[~np.isfinite(F["days_since_52w_high"])] = -1
    a_lo[~np.isfinite(F["days_since_52w_low"])] = -1
    F["dist_avwap_52w_high"] = dist(c, anchored_vwap(tp, v, a_hi))
    F["dist_avwap_52w_low"] = dist(c, anchored_vwap(tp, v, a_lo))
    years = idx.year.to_numpy()
    first_of_year = np.array([np.searchsorted(years, y) for y in years])
    F["dist_avwap_ytd"] = dist(c, anchored_vwap(tp, v, first_of_year))
    for w in (60, 120, 250):
        poc, vah, val = volume_profile(h, l, c, v, w)
        F[f"vp{w}_dist_poc"] = dist(c, poc)
        F[f"vp{w}_above_vah"] = _div(c - vah, atr14)
        F[f"vp{w}_below_val"] = _div(val - c, atr14)
    # --- pivot points (previous week / month) ----------------------------------------------------------
    for freq, tag in (("W-FRI", "w"), ("M", "m")):
        pv = pivots_prev_period(df[["high", "low", "close"]], freq)
        for col in ("P", "R1", "S1", "CH4", "CL4"):
            F[f"piv_{tag}_{col}_atr"] = _div(c - pv[col].to_numpy(float), atr14)
    # --- relative strength vs Nifty 50 ---------------------------------------------------------------------
    b = bench["close"].reindex(idx).ffill().to_numpy(float)
    rs_line = _div(c, b)
    rs_sma250 = pd.Series(rs_line).rolling(250, min_periods=200).mean().to_numpy()
    F["mansfield_rs"] = 100.0 * (_div(rs_line, rs_sma250) - 1)
    F["rs_line_new_high_252"] = (rs_line >= rolling_max(rs_line, 252, 200)).astype(float)
    for n in (21, 63, 126, 252):
        br_ = 100.0 * (_div(b, lag(b, n)) - 1)
        F[f"excess_ret_{n}"] = F[f"ret_{n}"] - br_
    rr = _div(rs_line, pd.Series(rs_line).rolling(50, min_periods=40).mean().to_numpy()) * 100
    F["rrg_ratio"] = rr
    F["rrg_momentum"] = _div(rr, lag(rr, 10)) * 100
    F["rrg_quadrant"] = np.select([(rr > 100) & (F["rrg_momentum"] > 100), (rr > 100), (F["rrg_momentum"] > 100)],
                                  [2, 1, -1], -2).astype(float)   # 2 leading, 1 weakening, -1 improving, -2 lagging
    bret = pd.Series(b).pct_change()
    for n in (60, 250):
        cov = pd.Series(ret1).rolling(n, min_periods=int(0.8 * n)).cov(bret)
        var = bret.rolling(n, min_periods=int(0.8 * n)).var()
        F[f"beta_{n}"] = (cov / var).to_numpy()
        F[f"corr_{n}"] = pd.Series(ret1).rolling(n, min_periods=int(0.8 * n)).corr(bret).to_numpy()
    resid = pd.Series(ret1) - F["beta_60"] * bret
    F["idio_vol_60"] = resid.rolling(60, min_periods=40).std().to_numpy() * np.sqrt(252) * 100
    # --- earnings-like gap drift proxy (results days show big gaps on heavy volume) -----------------------
    ev = (np.abs(np.nan_to_num(gap)) >= 1.5) & (np.nan_to_num(F["vol_ratio_50"]) >= 3)
    last_ev = -1
    since = np.full(len(c), np.nan)
    ev_dir = np.zeros(len(c))
    ev_ret = np.full(len(c), np.nan)
    ev_gap = np.zeros(len(c))
    cur_dir, cur_px, cur_gap = 0.0, np.nan, 0.0
    prev_c = lag(c, 1)
    for i in range(len(c)):
        if ev[i]:
            last_ev, cur_dir, cur_px, cur_gap = i, np.sign(gap[i]), prev_c[i], gap[i]
        if last_ev >= 0 and i - last_ev <= 60:
            since[i], ev_dir[i], ev_ret[i], ev_gap[i] = i - last_ev, cur_dir, 100 * (c[i] / cur_px - 1), cur_gap
    F["evgap_days_since"], F["evgap_dir"], F["evgap_ret_since"], F["evgap_size_atr"] = since, ev_dir, ev_ret, ev_gap
    # --- calendar --------------------------------------------------------------------------------------------
    F["cal_month"] = idx.month.to_numpy(float)
    F["cal_weekday"] = idx.weekday.to_numpy(float)
    F["cal_dom"] = idx.day.to_numpy(float)
    ym = idx.year * 100 + idx.month
    pos = pd.Series(1, index=idx).groupby(ym).cumsum().to_numpy()
    mend = (idx + pd.offsets.MonthEnd(0)).to_numpy().astype("datetime64[D]")
    F["cal_tdom"] = pos.astype(float)
    F["cal_tdays_to_mend"] = np.busday_count(idx.to_numpy().astype("datetime64[D]") + 1, mend + 1).astype(float)
    md = idx.month * 100 + idx.day
    F["cal_results_season"] = (((md >= 110) & (md <= 215)) | ((md >= 410) & (md <= 520))
                               | ((md >= 710) & (md <= 815)) | ((md >= 1010) & (md <= 1115))).astype(float)
    # --- candlesticks -----------------------------------------------------------------------------------------
    F.update(candle_block(o, h, l, c, atr14))

    out = pd.DataFrame(F, index=idx)
    out = out.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    # helper columns used downstream (not model features)
    out["_close"] = c
    out["_atr14"] = atr14
    out["_atr50"] = atr50
    out["_sma50"], out["_sma150"], out["_sma200"], out["_ema20"] = sma[50], sma[150], sma[200], ema[20]
    out["_hi252"], out["_lo252"] = hi252, lo252
    out["_turnover_cr_20"] = pd.Series(to).rolling(20, min_periods=10).mean().to_numpy()
    return out
