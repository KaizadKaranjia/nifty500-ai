"""Annotated charts for the top picks: candles, moving averages, detected patterns and trade levels."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BULL, BEAR, BLUE, GREY = "#15803d", "#b91c1c", "#1d4ed8", "#6b7280"


def pick_chart(sym: str, company: str, df: pd.DataFrame, events: pd.DataFrame, levels: dict, title_extra: str,
               out_png, bars: int = 160):
    d = df.iloc[-bars:]
    x0 = len(df) - len(d)
    x = np.arange(len(d))
    o, h, l, c, v = (d[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
    fig = plt.figure(figsize=(10.5, 5.8), dpi=110)
    ax = fig.add_axes([0.06, 0.30, 0.80, 0.56])
    axv = fig.add_axes([0.06, 0.10, 0.80, 0.17], sharex=ax)
    up = c >= o
    ax.vlines(x, l, h, color=np.where(up, BULL, BEAR), linewidth=0.7)
    ax.bar(x, np.maximum(np.abs(c - o), 1e-9), bottom=np.minimum(o, c), width=0.65,
           color=np.where(up, "#86efac", "#fca5a5"), edgecolor=np.where(up, BULL, BEAR), linewidth=0.5)
    full_c = df["close"].astype(float)
    for n, col, lab in ((20, "#f59e0b", "EMA 20"), (50, "#2563eb", "SMA 50"), (200, "#7c3aed", "SMA 200")):
        ma = (full_c.ewm(span=n, adjust=False).mean() if n == 20 else full_c.rolling(n).mean()).iloc[-bars:]
        ax.plot(x, ma.to_numpy(), color=col, lw=1.1, label=lab)
    # recent pattern events (last 20 sessions): classic chart patterns first, noise-type signals skipped
    rec = events[events["t"] >= len(df) - 20] if len(events) else events
    if len(rec):
        rec = rec[~rec["family"].isin(["Range contraction"])].copy()
        low_pri = {"Divergence", "Reversal bars", "Volume patterns"}
        rec["pri"] = rec["family"].isin(low_pri).astype(int)
        rec = rec.sort_values(["pri", "t"], ascending=[True, False]).drop_duplicates("pattern")
    lo = np.nanmin(l) if levels.get("stop") is None else min(np.nanmin(l), levels["stop"])
    hi = np.nanmax(h) if levels.get("target") is None else max(np.nanmax(h), levels["target"])
    y0, y1 = lo * 0.97, hi * 1.03
    for k, (_, e) in enumerate(rec.head(3).iterrows()):
        col = BULL if e["dir"] > 0 else BEAR
        pv = e.get("pivots")
        if isinstance(pv, list) and pv and pv[0][0] >= x0:
            px, py = zip(*pv)
            ax.plot(np.array(px) - x0, py, "-o", color=BLUE, lw=1.3, ms=3.5, alpha=0.9)
        tt = int(e["t"]) - x0
        ytxt = y1 - (0.07 + 0.065 * k) * (y1 - y0)
        ax.annotate(("▲ " if e["dir"] > 0 else "▼ ") + str(e["pattern"])[:40] + f" ({e['date']:%d-%b})",
                    xy=(tt, e["close"]), xytext=(len(d) * 0.62, ytxt), fontsize=7.4, color=col, ha="right",
                    va="center", arrowprops=dict(arrowstyle="-", color=col, lw=0.6, alpha=0.8),
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    # trade levels
    for key, col, lab in (("entry", GREY, "Entry"), ("target", BULL, "Target"), ("stop", BEAR, "Stop")):
        val = levels.get(key)
        if val is not None and np.isfinite(val):
            ax.axhline(val, color=col, lw=1.0, ls="--", alpha=0.9)
            ax.text(len(d) + 0.5, val, f" {lab} {val:,.2f}", color=col, fontsize=8, va="center")
    ax.set_xlim(-1, len(d) + 1)
    ax.set_ylim(y0, y1)
    ax.legend(loc="upper left", fontsize=7, frameon=False, ncol=3)
    ax.set_title(f"{sym} - {company}\n{title_extra}", fontsize=9.5, loc="left")
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=7, labelbottom=False)
    axv.bar(x, v / 1e5, color=np.where(up, "#86efac", "#fca5a5"), width=0.65)
    axv.set_ylabel("Vol (lakh)", fontsize=7)
    axv.tick_params(labelsize=7)
    ticks = np.linspace(0, len(d) - 1, 6).astype(int)
    axv.set_xticks(ticks)
    axv.set_xticklabels([d.index[i].strftime("%d-%b-%y") for i in ticks])
    axv.grid(alpha=0.25)
    fig.savefig(out_png)
    plt.close(fig)
