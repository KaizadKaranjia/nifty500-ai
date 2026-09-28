"""Indicator signals and candlestick patterns as dated events (same definitions as the v2 report card)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def up(x, lvl=0.0):
    x = pd.Series(x)
    return ((x > lvl) & (x.shift(1) <= lvl)).to_numpy()


def dn(x, lvl=0.0):
    x = pd.Series(x)
    return ((x < lvl) & (x.shift(1) >= lvl)).to_numpy()


def flip(x, to):
    x = pd.Series(x)
    return ((x == to) & (x.shift(1) != to) & x.shift(1).notna()).to_numpy()


def signals(F: pd.DataFrame) -> list[tuple[str, str, int, np.ndarray]]:
    g = lambda c: F[c].to_numpy(float)  # noqa: E731
    c = F["_close"].to_numpy(float)
    sma50, sma200, sma150, ema20 = g("_sma50"), g("_sma200"), g("_sma150"), g("_ema20")
    uptrend = (c > sma200) & (sma50 > sma200)
    dntrend = (c < sma200) & (sma50 < sma200)
    ret10 = g("ret_5") if "ret_10" not in F else g("ret_10")
    S = []
    add = lambda name, fam, d, m: S.append((name, fam, d, np.nan_to_num(m).astype(bool)))  # noqa: E731
    # moving averages
    add("Golden cross (50 > 200 DMA)", "Moving averages", 1, up(g("sma50_gt_sma200"), 0.5))
    add("Death cross (50 < 200 DMA)", "Moving averages", -1, dn(g("sma50_gt_sma200"), 0.5))
    add("Close crosses above 200 DMA", "Moving averages", 1, up(g("dist_sma200")))
    add("Close crosses below 200 DMA", "Moving averages", -1, dn(g("dist_sma200")))
    add("Close crosses above 50 DMA", "Moving averages", 1, up(g("dist_sma50")))
    add("Close crosses below 50 DMA", "Moving averages", -1, dn(g("dist_sma50")))
    add("EMA 9/21 bullish crossover", "Moving averages", 1, up(g("ema9_gt_ema21"), 0.5))
    add("EMA 9/21 bearish crossover", "Moving averages", -1, dn(g("ema9_gt_ema21"), 0.5))
    add("Moving averages stacked bullish (first day)", "Moving averages", 1, up(g("ma_stack_bull"), 0.5))
    add("Moving averages stacked bearish (first day)", "Moving averages", -1, up(g("ma_stack_bear"), 0.5))
    # MACD / momentum oscillators
    add("MACD crosses above signal", "MACD", 1, up(g("ta_MACD_macdhist")))
    add("MACD crosses below signal", "MACD", -1, dn(g("ta_MACD_macdhist")))
    add("MACD crosses above zero", "MACD", 1, up(g("ta_MACD_macd")))
    add("MACD crosses below zero", "MACD", -1, dn(g("ta_MACD_macd")))
    add("RSI(14) rises back above 30", "RSI", 1, up(g("ta_RSI"), 30))
    add("RSI(14) falls back below 70", "RSI", -1, dn(g("ta_RSI"), 70))
    add("RSI(14) crosses above 50", "RSI", 1, up(g("ta_RSI"), 50))
    add("RSI(14) crosses below 50", "RSI", -1, dn(g("ta_RSI"), 50))
    add("RSI(2) below 5 in an uptrend (Connors)", "RSI", 1, (g("rsi_2") < 5) & uptrend & ~(pd.Series(g("rsi_2") < 5).shift(1).fillna(False).to_numpy(bool)))
    add("RSI(2) above 95 in a downtrend", "RSI", -1, (g("rsi_2") > 95) & dntrend & ~(pd.Series(g("rsi_2") > 95).shift(1).fillna(False).to_numpy(bool)))
    add("Connors RSI below 10", "RSI", 1, dn(g("connors_rsi"), 10))
    add("Stochastic bullish cross below 20", "Stochastic", 1, up(g("stoch_k_14") - g("stoch_d_14")) & (g("stoch_k_14") < 25))
    add("Stochastic bearish cross above 80", "Stochastic", -1, dn(g("stoch_k_14") - g("stoch_d_14")) & (g("stoch_k_14") > 75))
    add("Williams %R exits oversold (-80)", "Williams %R", 1, up(g("ta_WILLR"), -80))
    add("Williams %R exits overbought (-20)", "Williams %R", -1, dn(g("ta_WILLR"), -20))
    add("CCI crosses above +100", "CCI", 1, up(g("ta_CCI"), 100))
    add("CCI crosses below -100", "CCI", -1, dn(g("ta_CCI"), -100))
    add("MFI exits oversold (20)", "Money flow", 1, up(g("ta_MFI"), 20))
    add("MFI exits overbought (80)", "Money flow", -1, dn(g("ta_MFI"), 80))
    add("Chaikin money flow turns positive", "Money flow", 1, up(g("ta_CMF")))
    add("Chaikin money flow turns negative", "Money flow", -1, dn(g("ta_CMF")))
    add("TRIX crosses above zero", "Momentum", 1, up(g("ta_TRIX")))
    add("TRIX crosses below zero", "Momentum", -1, dn(g("ta_TRIX")))
    add("KST bullish crossover", "Momentum", 1, up(g("kst_minus_sig")))
    add("KST bearish crossover", "Momentum", -1, dn(g("kst_minus_sig")))
    cop = g("ta_COPPOCK")
    add("Coppock turns up below zero", "Momentum", 1, (cop < 0) & up(pd.Series(cop).diff().to_numpy()))
    add("Awesome oscillator crosses above zero", "Momentum", 1, up(g("ta_AO")))
    add("Awesome oscillator crosses below zero", "Momentum", -1, dn(g("ta_AO")))
    add("TSI crosses above zero", "Momentum", 1, up(g("ta_TSI")))
    add("TSI crosses below zero", "Momentum", -1, dn(g("ta_TSI")))
    add("Schaff trend cycle rises above 25", "Momentum", 1, up(g("schaff_tc"), 25))
    add("Schaff trend cycle falls below 75", "Momentum", -1, dn(g("schaff_tc"), 75))
    add("Fisher transform bullish cross", "Momentum", 1, up(g("fisher_10")))
    add("Fisher transform bearish cross", "Momentum", -1, dn(g("fisher_10")))
    add("Elder impulse turns green", "Momentum", 1, flip(g("elder_impulse"), 2))
    add("Elder impulse turns red", "Momentum", -1, flip(g("elder_impulse"), -2))
    # trend-following flips
    add("Supertrend (10,3) turns up", "Trend", 1, flip(g("ta_SUPERTREND_trend"), 1))
    add("Supertrend (10,3) turns down", "Trend", -1, flip(g("ta_SUPERTREND_trend"), -1))
    add("Parabolic SAR flips up", "Trend", 1, flip(g("ta_SAREXT_dir"), 1))
    add("Parabolic SAR flips down", "Trend", -1, flip(g("ta_SAREXT_dir"), -1))
    add("ADX > 25 with +DI crossing above -DI", "Trend", 1, up(g("di_diff_14")) & (g("ta_ADX") > 25))
    add("ADX > 25 with -DI crossing above +DI", "Trend", -1, dn(g("di_diff_14")) & (g("ta_ADX") > 25))
    add("Aroon up crosses above Aroon down", "Trend", 1, up(g("ta_AROONOSC")))
    add("Aroon down crosses above Aroon up", "Trend", -1, dn(g("ta_AROONOSC")))
    add("Vortex bullish cross", "Trend", 1, up(g("ta_VORTEX_plusvi") - g("ta_VORTEX_minusvi")))
    add("Vortex bearish cross", "Trend", -1, dn(g("ta_VORTEX_plusvi") - g("ta_VORTEX_minusvi")))
    add("Gann HiLo activator turns up", "Trend", 1, flip(g("gann_hilo_dir"), 1))
    add("Gann HiLo activator turns down", "Trend", -1, flip(g("gann_hilo_dir"), -1))
    add("Volatility stop flips long", "Trend", 1, flip(g("volstop_dir"), 1))
    add("Volatility stop flips short", "Trend", -1, flip(g("volstop_dir"), -1))
    add("Renko reversal up", "Price-only charts", 1, flip(g("renko_dir"), 1))
    add("Renko reversal down", "Price-only charts", -1, flip(g("renko_dir"), -1))
    add("Three-line break reversal up", "Price-only charts", 1, flip(g("tlb_dir"), 1))
    add("Three-line break reversal down", "Price-only charts", -1, flip(g("tlb_dir"), -1))
    add("Kagi turns yang (thick)", "Price-only charts", 1, flip(g("kagi_state"), 1))
    add("Kagi turns yin (thin)", "Price-only charts", -1, flip(g("kagi_state"), -1))
    add("Heikin-Ashi turns bullish", "Price-only charts", 1, flip(g("ha_bull"), 1))
    add("Heikin-Ashi turns bearish", "Price-only charts", -1, flip(g("ha_bull"), 0))
    add("Alligator wakes up bullish", "Trend", 1, flip(g("alligator_bull"), 1))
    add("Alligator wakes up bearish", "Trend", -1, flip(g("alligator_bull"), -1))
    # Ichimoku
    tk = g("ichi_tk_diff_atr")
    above, below = g("ichi_above_cloud_atr") > 0, g("ichi_below_cloud_atr") > 0
    add("Ichimoku TK cross above the cloud", "Ichimoku", 1, up(tk) & above)
    add("Ichimoku TK cross below the cloud", "Ichimoku", -1, dn(tk) & below)
    add("Ichimoku Kumo breakout up", "Ichimoku", 1, up(g("ichi_above_cloud_atr")))
    add("Ichimoku Kumo breakdown", "Ichimoku", -1, up(g("ichi_below_cloud_atr")))
    # volatility / channels
    sq = g("ttm_squeeze_on")
    rel = pd.Series(sq).shift(1).eq(1).to_numpy() & (sq == 0)
    add("TTM squeeze fires up", "Volatility", 1, rel & (g("ta_MACD_macdhist") > 0))
    add("TTM squeeze fires down", "Volatility", -1, rel & (g("ta_MACD_macdhist") < 0))
    add("Close above upper Bollinger band", "Volatility", 1, up(g("bb_pctb_20_2"), 1.0))
    add("Close below lower Bollinger band", "Volatility", -1, dn(g("bb_pctb_20_2"), 0.0))
    add("Lower Bollinger band tag in an uptrend", "Volatility", 1, dn(g("bb_pctb_20_2"), 0.0) & uptrend)
    add("Keltner channel breakout up", "Volatility", 1, up(g("kc_pos"), 1.0))
    add("Keltner channel breakdown", "Volatility", -1, dn(g("kc_pos"), 0.0))
    add("Turtle 20-day breakout", "Channels", 1, up(g("donch_break_20"), 0.5))
    add("Turtle 20-day breakdown", "Channels", -1, dn(g("donch_break_20"), -0.5))
    add("Turtle 55-day breakout", "Channels", 1, up(g("donch_break_55"), 0.5))
    add("Turtle 55-day breakdown", "Channels", -1, dn(g("donch_break_55"), -0.5))
    # trend templates, stages, relative strength
    minerv = ((c > sma150) & (c > sma200) & (sma150 > sma200) & (g("slope_sma200") > 0) & (sma50 > sma150)
              & (c > sma50) & (g("dist_52w_low") >= 30) & (g("dist_52w_high") >= -25))
    add("Minervini trend template passes (first day)", "Trend templates", 1, up(minerv.astype(float), 0.5))
    stage2 = up(g("dist_sma150")) & (g("slope_sma150") > 0) & (g("vol_ratio_50") >= 1.5)
    add("Weinstein stage-2 breakout (30-wk MA, volume)", "Trend templates", 1, stage2)
    stage4 = dn(g("dist_sma150")) & (g("slope_sma150") < 0)
    add("Weinstein stage-4 breakdown", "Trend templates", -1, stage4)
    add("Mansfield RS turns positive", "Relative strength", 1, up(g("mansfield_rs")))
    add("Mansfield RS turns negative", "Relative strength", -1, dn(g("mansfield_rs")))
    add("RS line at a 52-week high (first day)", "Relative strength", 1, up(g("rs_line_new_high_252"), 0.5))
    add("RRG: enters Leading quadrant", "Relative strength", 1, flip(g("rrg_quadrant"), 2))
    add("RRG: enters Lagging quadrant", "Relative strength", -1, flip(g("rrg_quadrant"), -2))
    add("RRG: enters Improving quadrant", "Relative strength", 1, flip(g("rrg_quadrant"), -1))
    add("RRG: enters Weakening quadrant", "Relative strength", -1, flip(g("rrg_quadrant"), 1))
    # mean reversion / swing setups
    c7lo = pd.Series(c).rolling(7).min().to_numpy()
    add("Double 7s: 7-day closing low above 200 DMA", "Swing setups", 1, (c <= c7lo) & (c > sma200) & ~(pd.Series((c <= c7lo)).shift(1).fillna(False).to_numpy(bool)))
    add("IBS below 0.2 in an uptrend", "Swing setups", 1, (g("cdlx_ibs") < 0.2) & uptrend)
    add("Three lower closes in an uptrend", "Swing setups", 1, (g("updown_streak") == -3) & uptrend)
    di_up5 = pd.Series(g("di_diff_14")).rolling(5).min().to_numpy() > 0      # +DI above -DI for a week
    add("Holy Grail: ADX > 30 and pullback to 20 EMA", "Swing setups", 1,
        (g("ta_ADX") > 30) & di_up5 & (g("dist_sma50") > 0) & dn(g("dist_ema20"), 0.5) & (g("dist_ema20") >= -1.0)
        & (g("ret_1") > -2.0 * g("atr_pct_14")))
    # volume / delivery
    add("Delivery spike on an up day (1.5x qty)", "Volume & delivery", 1, up(g("deliv_up_day_spike"), 0.5))
    add("Volume > 2x average on an up day", "Volume & delivery", 1, (g("vol_ratio_50") >= 2) & (g("ret_1") > 0))
    add("Volume > 2x average on a down day", "Volume & delivery", -1, (g("vol_ratio_50") >= 2) & (g("ret_1") < 0))
    add("Earnings-like gap up (1.5 ATR, 3x volume)", "Volume & delivery", 1, (g("evgap_days_since") == 0) & (g("evgap_dir") > 0))
    add("Earnings-like gap down (1.5 ATR, 3x volume)", "Volume & delivery", -1, (g("evgap_days_since") == 0) & (g("evgap_dir") < 0))
    # candlesticks: every TA-Lib pattern, raw and in the "right" trend context
    down_ctx = (c < ema20) & (ret10 < 0)
    up_ctx = (c > ema20) & (ret10 > 0)
    for col in [x for x in F.columns if x.startswith("cdl_")]:
        v = g(col)
        nm = col[4:]
        add(f"Candle: {nm} (bullish)", "Candlesticks", 1, v > 0)
        add(f"Candle: {nm} (bearish)", "Candlesticks", -1, v < 0)
        add(f"Candle: {nm} (bullish, after a decline)", "Candlesticks in context", 1, (v > 0) & down_ctx)
        add(f"Candle: {nm} (bearish, after a rise)", "Candlesticks in context", -1, (v < 0) & up_ctx)
    for col, d in (("cdlx_tweezer_bottom", 1), ("cdlx_tweezer_top", -1), ("cdlx_pin_bull", 1), ("cdlx_pin_bear", -1),
                   ("cdlx_outside_bull", 1), ("cdlx_outside_bear", -1)):
        v = g(col) > 0
        add(f"Candle: {col[5:]}", "Candlesticks", d, v)
        add(f"Candle: {col[5:]} (in context)", "Candlesticks in context", d, v & (down_ctx if d > 0 else up_ctx))
    return S
