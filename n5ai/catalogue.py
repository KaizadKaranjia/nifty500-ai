"""Catalogue of every indicator, chart pattern, signal and candlestick used in v2 (for the workbook)."""
import pandas as pd
import talib

TALIB_DESC = {
    "AC": "Accelerator oscillator", "ADX": "Average directional index", "ADXR": "ADX rating", "AO": "Awesome oscillator",
    "APO": "Absolute price oscillator", "AROON": "Aroon up/down", "AROONOSC": "Aroon oscillator", "BOP": "Balance of power",
    "CCI": "Commodity channel index", "CMO": "Chande momentum oscillator", "CMOU": "Chande momentum (unsmoothed)",
    "COPPOCK": "Coppock curve", "DPO": "Detrended price oscillator", "DX": "Directional movement index",
    "ER": "Kaufman efficiency ratio", "ERI": "Elder ray (bull/bear power)", "FOSC": "Forecast oscillator",
    "FRACTAL": "Williams fractals", "IMI": "Intraday momentum index", "KDJ": "KDJ stochastic", "MACD": "MACD",
    "MACDEXT": "MACD (configurable MAs)", "MACDFIX": "MACD 12/26 fixed", "MFI": "Money flow index", "MINUS_DI": "-DI",
    "MINUS_DM": "-DM", "MOM": "Momentum", "PLUS_DI": "+DI", "PLUS_DM": "+DM", "PPO": "Percentage price oscillator",
    "QSTICK": "Qstick", "ROC": "Rate of change", "ROCP": "Rate of change (fraction)", "ROCR": "Rate of change ratio",
    "ROCR100": "Rate of change ratio x100", "RSI": "Relative strength index", "SMI": "Stochastic momentum index",
    "STOCH": "Slow stochastic", "STOCHF": "Fast stochastic", "STOCHRSI": "Stochastic RSI", "TRIX": "TRIX",
    "TSI": "True strength index", "ULTOSC": "Ultimate oscillator", "VHF": "Vertical horizontal filter",
    "VORTEX": "Vortex indicator", "WAD": "Williams accumulation/distribution", "WILLR": "Williams %R",
    "ACCBANDS": "Acceleration bands", "BBANDS": "Bollinger bands", "DEMA": "Double EMA", "DONCHIAN": "Donchian channel",
    "EMA": "Exponential MA", "HMA": "Hull MA", "HT_TRENDLINE": "Hilbert instantaneous trendline", "KAMA": "Kaufman adaptive MA",
    "KC": "Keltner channel", "MA": "Moving average", "MAMA": "MESA adaptive MA", "MIDPOINT": "Midpoint", "MIDPRICE": "Midprice",
    "RMA": "Wilder MA", "SAR": "Parabolic SAR", "SAREXT": "Parabolic SAR (extended)", "SMA": "Simple MA",
    "SUPERTREND": "Supertrend", "T3": "Tillson T3", "TEMA": "Triple EMA", "TRIMA": "Triangular MA", "VWMA": "Volume-weighted MA",
    "WMA": "Weighted MA", "ZLEMA": "Zero-lag EMA", "LINEARREG": "Linear regression", "LINEARREG_ANGLE": "Linear regression angle",
    "LINEARREG_INTERCEPT": "Linear regression intercept", "LINEARREG_SLOPE": "Linear regression slope",
    "PERCENTILE": "Rolling percentile", "PERCENTRANK": "Percent rank", "STDDEV": "Standard deviation", "TSF": "Time series forecast",
    "VAR": "Variance", "ADR": "Average daily range", "ATR": "Average true range", "CVI": "Chaikin volatility", "MASSI": "Mass index",
    "NATR": "Normalized ATR", "RVI": "Relative volatility index", "TRANGE": "True range", "AD": "Chaikin A/D line",
    "ADOSC": "Chaikin oscillator", "CMF": "Chaikin money flow", "EFI": "Elder force index", "MARKETFI": "Market facilitation index",
    "NVI": "Negative volume index", "OBV": "On-balance volume", "PVI": "Positive volume index", "PVO": "Percentage volume oscillator",
    "PVT": "Price-volume trend", "RVOL": "Relative volume", "VWAP": "VWAP (cumulative)", "AVGDEV": "Average deviation",
    "AVGPRICE": "Average price", "HA": "Heikin-Ashi", "MEDPRICE": "Median price", "TYPPRICE": "Typical price",
    "WCLPRICE": "Weighted close", "HT_DCPERIOD": "Hilbert dominant cycle period", "HT_DCPHASE": "Hilbert dominant cycle phase",
    "HT_PHASOR": "Hilbert phasor components", "HT_SINE": "Hilbert sine wave", "HT_TRENDMODE": "Hilbert trend vs cycle mode",
    "BETA": "Beta vs Nifty 50", "CORREL": "Correlation vs Nifty 50",
}

EXTRA = [
    ("Trend", "Supertrend 7/3, 11/2, 20/5", "direction and distance"),
    ("Trend", "Ichimoku cloud", "Tenkan/Kijun cross, price vs cloud, cloud thickness, future cloud, Chikou span"),
    ("Trend", "Moving-average set", "SMA 5-200 and EMA 5-200 distances, slopes, 9/21 & 50/200 crosses, MA stacking"),
    ("Trend", "Hull MA, McGinley dynamic, ALMA", "adaptive moving averages"),
    ("Trend", "Williams Alligator", "jaw/teeth/lips alignment and spread"),
    ("Trend", "Gann HiLo activator", "trend direction"),
    ("Trend", "Chandelier exit, Wilder volatility stop", "trailing-stop states"),
    ("Trend", "Linear regression channels 50/100/200", "slope and R-squared"),
    ("Trend", "Elder impulse system", "EMA-13 slope + MACD-histogram slope"),
    ("Price-only charts", "Renko (ATR box)", "brick direction and run length"),
    ("Price-only charts", "Three-line break", "direction"),
    ("Price-only charts", "Kagi (4% reversal)", "yang / yin state"),
    ("Price-only charts", "Heikin-Ashi", "colour, streak, shadowless candles"),
    ("Price-only charts", "Point & figure (2% x 3)", "double/triple top-bottom and ascending/descending signals"),
    ("Momentum", "RSI 2, 3, 5, 7, 9, 14, 21, 28", "several look-backs"),
    ("Momentum", "Laguerre RSI, Connors RSI", "fast mean-reversion oscillators"),
    ("Momentum", "Fisher transform, Schaff trend cycle, relative vigor index", "cycle oscillators"),
    ("Momentum", "Know Sure Thing (KST)", "and signal line"),
    ("Momentum", "Rate of change 1 to 252 days, 12-1 and 6-1 month momentum", "return features"),
    ("Volatility", "Bollinger %B and width (20/2, 50/2, 20/1), TTM squeeze", "squeeze state and length"),
    ("Volatility", "Historical volatility 10/20/60/252, Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang", "estimators"),
    ("Volatility", "ATR % (5/14/20/50), ATR percentile, ulcer index, choppiness index, drawdowns", ""),
    ("Statistics", "Skew, kurtosis, autocorrelation, variance ratio, Hurst exponent", "return behaviour"),
    ("Volume", "Klinger oscillator, ease of movement, rolling VWAP 20/50", ""),
    ("Volume", "Anchored VWAP from 52-week high, 52-week low and year start", ""),
    ("Volume", "Volume profile 60/120/250 days", "point of control, value-area high/low"),
    ("Volume", "Relative volume, volume dry-up, up/down volume ratio, volume z-score", ""),
    ("NSE delivery data", "Delivery %, 5 vs 20-day delivery, delivery z-score, delivery-quantity spikes", "India-specific"),
    ("NSE delivery data", "Trades count ratio, quantity per trade", "institutional footprint proxy"),
    ("Support / resistance", "Pivot points (weekly & monthly): classic, Camarilla, Fibonacci", "distance in ATR"),
    ("Support / resistance", "52-week high/low distance, range position, days since high/low, multi-year high", ""),
    ("Relative strength", "Mansfield RS, RS-line new highs, excess return vs Nifty 21-252 days", ""),
    ("Relative strength", "IBD-style RS rating (1-99) and cross-sectional ranks of 22 features", "computed daily across the Nifty 500"),
    ("Relative strength", "Relative rotation graph (RS-ratio, RS-momentum, quadrant)", ""),
    ("Relative strength", "Beta and correlation vs Nifty (60/250), idiosyncratic volatility", ""),
    ("Sector", "Sector median returns, stock vs sector, sector RS rank, sector breadth", ""),
    ("Market breadth", "% above 50/200 DMA, new highs - lows, advance/decline, McClellan oscillator, dispersion", ""),
    ("Market", "Nifty 50 returns, distance to 50/200 DMA, RSI, volatility, drawdown; small/mid vs large caps; regime", ""),
    ("Events", "Earnings-like gap (1.5 ATR on 3x volume): days since, direction, drift since", "post-results drift proxy"),
    ("Calendar", "Month, weekday, trading day of month, days to month end, results season", ""),
    ("Candlesticks (extra)", "Tweezer top/bottom, inside/outside bars, pin bars, NR4/NR7/WR7, internal bar strength", ""),
]


def catalogue(feature_cols: list[str], used: set[str], report: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = talib.get_function_groups()
    for grp in ("Overlap Studies", "Momentum Indicators", "Volatility Indicators", "Volume Indicators", "Statistic Functions",
                "Price Transform", "Cycle Indicators"):
        for fn in groups[grp]:
            if fn == "MAVP":
                continue
            cols = [c for c in feature_cols if c == f"ta_{fn}" or c.startswith(f"ta_{fn}_")]
            rows.append({"Type": "Indicator (TA-Lib)", "Family": grp.replace(" Functions", "").replace(" Indicators", ""),
                         "Name": f"{TALIB_DESC.get(fn, fn)} ({fn})", "Detail": "default settings; scale-free form",
                         "In AI model": "Yes" if any(c in used for c in cols) or fn in ("BETA", "CORREL") else "Kept for report"})
    for fam, name, detail in EXTRA:
        rows.append({"Type": "Indicator (additional)", "Family": fam, "Name": name, "Detail": detail, "In AI model": "Yes"})
    for fn in groups["Pattern Recognition"]:
        rows.append({"Type": "Candlestick pattern", "Family": "Candlesticks (TA-Lib)", "Name": fn[3:].title(),
                     "Detail": "bullish and bearish, raw and in trend context", "In AI model": "Yes"})
    rp = report[(report["level"] == "pattern")]
    for src, typ in (("Chart pattern", "Chart pattern"), ("Indicator signal", "Indicator signal")):
        sub = rp[rp["source"] == src].sort_values(["family", "pattern"])
        for _, r in sub.iterrows():
            rows.append({"Type": typ, "Family": r["family"], "Name": r["pattern"],
                         "Detail": ("bullish" if r["dir"] > 0 else "bearish") + f"; {int(r['n']):,} events 2017-26",
                         "In AI model": "Yes" if typ == "Chart pattern" else "Via its indicator"})
    return pd.DataFrame(rows)
