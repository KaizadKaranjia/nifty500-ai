"""Shared constants for the Nifty 500 AI daily job."""
from __future__ import annotations

# timeframe -> (label, stop x ATR, target x ATR, no-chase x ATR, max sessions)
TF_RULES = {
    "ST": ("Short term (up to 2 weeks)", 1.5, 3.0, 1.0, 10),
    "MT": ("Mid term (up to 3 months)", 3.0, 6.0, 1.5, 60),
    "LT": ("Long term (up to 1 year)", 5.0, 10.0, 2.0, 250),
}
TF_SHORT = {"ST": "2w", "MT": "3m", "LT": "1y"}
HOR = {"ST": "2 weeks", "MT": "3 months", "LT": "1 year"}
HOR_ADJ = {"ST": "2-week", "MT": "3-month", "LT": "1-year"}
COST = 0.003                                  # round-trip cost assumed per paper trade (0.3%)

BUY_TYPES = ("Strong Buy", "Buy", "Accumulate (buy on dips)")
SHORT_TYPES = ("Short Sell", "Strong Short Sell")
AVOID_TYPES = ("Avoid (likely underperformer)", "Reduce / Avoid")
INDEX_NAMES = {                               # bench key -> official NSE index name
    "NIFTY50": "Nifty 50",
    "NIFTY500": "Nifty 500",
    "MIDCAP150": "Nifty Midcap 150",
    "SMALLCAP250": "Nifty Smallcap 250",
    "NEXT50": "Nifty Next 50",
}


def call_kind(call_type: str) -> str | None:
    """BUY / SHORT / AVOID for calls that are tracked in the success rate, None for holds."""
    if call_type in BUY_TYPES:
        return "BUY"
    if call_type in SHORT_TYPES:
        return "SHORT"
    if call_type in AVOID_TYPES:
        return "AVOID"
    return None
