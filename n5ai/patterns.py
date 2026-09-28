
"""v2 chart-pattern engine - causal detection of classical, harmonic and bar-based chart patterns.

Swing structure comes from a causal zigzag at three scales (minor 1.5, intermediate 3, major 6
x ATR(50)). A pivot is only known once price has reversed by the threshold, and a pattern
"fires" on the bar whose CLOSE breaks the pattern's trigger line (neckline, rim, trendline...),
exactly as a chartist would act at the close. Nothing uses future bars.

Two alignments are scanned so breakouts are not delayed by the zigzag confirmation lag:
  A  pattern = last m confirmed pivots, the breakout leg is in progress
  B  pattern = last m-1 confirmed pivots + the running extreme of the current leg
Every template is written for the bullish orientation and mirrored for the bearish one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import talib

SCALES = {"minor": 1.5, "intermediate": 3.0, "major": 6.0}
MINW = {"minor": 8, "intermediate": 15, "major": 30}
MAXW = {"minor": 150, "intermediate": 300, "major": 600}
MINW_COMPLEX = {"minor": 15, "intermediate": 25, "major": 40}     # H&S, triple tops/bottoms


@dataclass
class Pv:
    idx: int
    px: float
    typ: int          # +1 swing high, -1 swing low (in the oriented space)
    conf: int = -1


@dataclass
class Cand:
    name: str
    family: str
    a: float          # trigger line level = a + b * t (oriented space)
    b: float
    dirn: int         # +1 close must cross above the line (oriented space), -1 below
    height: float     # measured-move height (oriented space, positive)
    stop: float       # invalidation level (oriented space)
    start: int
    end: int
    pivots: list = field(default_factory=list)
    variant: str = ""
    target_abs: float | None = None   # fixed target (oriented) when not level +/- height
    expire: int | None = None


# ------------------------------------------------------------------------------------------------
# zigzag
# ------------------------------------------------------------------------------------------------
def zigzag(h: np.ndarray, l: np.ndarray, thr: np.ndarray):
    n = len(h)
    piv: list[Pv] = []
    ext_idx = np.full(n, -1, dtype=np.int64)
    ext_px = np.full(n, np.nan)
    leg = np.zeros(n, dtype=np.int8)
    nconf = np.zeros(n, dtype=np.int64)
    state = 0
    hi, lo, hi_i, lo_i = h[0], l[0], 0, 0
    for t in range(n):
        th = thr[t]
        if state == 0:
            if h[t] > hi:
                hi, hi_i = h[t], t
            if l[t] < lo:
                lo, lo_i = l[t], t
            if np.isfinite(th) and hi - lo >= th:
                if hi_i > lo_i:
                    piv.append(Pv(lo_i, lo, -1, t))
                    state = 1
                else:
                    piv.append(Pv(hi_i, hi, 1, t))
                    state = -1
        elif state == 1:
            if h[t] > hi:
                hi, hi_i = h[t], t
            elif np.isfinite(th) and hi - l[t] >= th:
                piv.append(Pv(hi_i, hi, 1, t))
                state, lo, lo_i = -1, l[t], t
        else:
            if l[t] < lo:
                lo, lo_i = l[t], t
            elif np.isfinite(th) and h[t] - lo >= th:
                piv.append(Pv(lo_i, lo, -1, t))
                state, hi, hi_i = 1, h[t], t
        leg[t] = state
        if state == 1:
            ext_idx[t], ext_px[t] = hi_i, hi
        elif state == -1:
            ext_idx[t], ext_px[t] = lo_i, lo
        nconf[t] = len(piv)
    return piv, ext_idx, ext_px, leg, nconf


# ------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------
def line_through(p: Pv, q: Pv):
    if q.idx == p.idx:
        return p.px, 0.0
    b = (q.px - p.px) / (q.idx - p.idx)
    return p.px - b * p.idx, b


def fit_line(pts: list[Pv]):
    if len(pts) == 2:
        a, b = line_through(pts[0], pts[1])
        return a, b, 0.0
    x = np.array([p.idx for p in pts], float)
    y = np.array([p.px for p in pts], float)
    b, a = np.polyfit(x, y, 1)
    res = np.abs(y - (a + b * x)).max()
    return a, b, res


def types_ok(P: list[Pv], seq: tuple) -> bool:
    return len(P) == len(seq) and all(p.typ == t for p, t in zip(P, seq))


def in_range(x, lo, hi, tol=0.03):
    return (lo - tol) <= x <= (hi + tol)


class Ctx:
    """Oriented price arrays for one side (s=+1 bullish view, s=-1 bearish view)."""

    def __init__(self, s, H, L, C, atr, atr14, scale, extra):
        self.s = s
        self.H, self.L, self.C = (H, L, C) if s > 0 else (-L, -H, -C)
        self.atr, self.atr14 = atr, atr14
        self.scale, self.k = scale, SCALES.get(scale, 1.0)
        self.minw, self.maxw = MINW.get(scale, 8), MAXW.get(scale, 300)
        self.x = extra


# ------------------------------------------------------------------------------------------------
# bullish-orientation templates (mirrored automatically). Each gets the pattern pivots P.
# ------------------------------------------------------------------------------------------------
def t_double_bottom(P, g):
    if not types_ok(P, (1, -1, 1, -1)):
        return None
    H0, L1, H1, L2 = P
    height = H1.px - min(L1.px, L2.px)
    sep = L2.idx - L1.idx
    if height <= 0 or not (g.minw <= sep <= g.maxw) or H0.px <= H1.px:
        return None
    if abs(L1.px - L2.px) > 0.2 * height:
        return None
    # Adam (sharp, narrow) vs Eve (rounded, wide) bottoms
    def width(p):
        w = max(2, sep // 2)
        seg = g.L[max(0, p.idx - w): p.idx + w + 1]
        return int((seg <= p.px + 0.25 * height).sum())
    lab = "".join("A" if width(p) <= max(3, 0.12 * sep) else "E" for p in (L1, L2))
    variant = {"AA": "Adam & Adam", "AE": "Adam & Eve", "EA": "Eve & Adam", "EE": "Eve & Eve"}[lab]
    return Cand("Double bottom|Double top", "Double bottom/top", H1.px, 0.0, 1, height, min(L1.px, L2.px),
                L1.idx, L2.idx, [H0, L1, H1, L2], variant)


def t_triple_bottom(P, g):
    if not types_ok(P, (1, -1, 1, -1, 1, -1)):
        return None
    H0, L1, H1, L2, H2, L3 = P
    lows = [L1.px, L2.px, L3.px]
    lvl = max(H1.px, H2.px)
    height = lvl - min(lows)
    sep = L3.idx - L1.idx
    if height <= 0 or not (MINW_COMPLEX.get(g.scale, 15) <= sep <= g.maxw * 1.5) or H0.px <= lvl:
        return None
    if max(lows) - min(lows) > 0.2 * height or max(H1.px, H2.px) - min(H1.px, H2.px) > 0.4 * height:
        return None
    return Cand("Triple bottom|Triple top", "Triple bottom/top", lvl, 0.0, 1, height, min(lows), L1.idx, L3.idx,
                [H0, L1, H1, L2, H2, L3])


def t_head_shoulders(P, g):
    if not types_ok(P, (1, -1, 1, -1, 1, -1)):
        return None
    H0, LS, N1, HD, N2, RS = P
    a, b = line_through(N1, N2)
    neck_hd = a + b * HD.idx
    height = neck_hd - HD.px
    if height <= 0:
        return None
    if not (HD.px < LS.px - 0.1 * height and HD.px < RS.px - 0.1 * height):
        return None
    if (a + b * LS.idx) - LS.px < 0.3 * height or (a + b * RS.idx) - RS.px < 0.3 * height:
        return None
    if abs(LS.px - RS.px) > 0.6 * height or abs(N2.px - N1.px) > 0.6 * height:
        return None
    t1, t2 = HD.idx - LS.idx, RS.idx - HD.idx
    if t2 <= 0 or not (0.33 <= t1 / t2 <= 3.0):
        return None
    w = RS.idx - LS.idx
    if not (MINW_COMPLEX.get(g.scale, 15) <= w <= g.maxw * 1.5) or H0.px <= max(N1.px, N2.px):
        return None
    return Cand("Inverse head & shoulders|Head & shoulders top", "Head & shoulders", a, b, 1, height, RS.px,
                LS.idx, RS.idx, [H0, LS, N1, HD, N2, RS])


def t_cup_handle(P, g):
    if not types_ok(P, (1, -1, 1, -1)):
        return None
    HL, LC, HR, LH = P
    rim = min(HL.px, HR.px)
    depth = rim - LC.px
    base = abs(HL.px)
    if depth <= 0 or not (0.12 <= depth / base <= 0.50):
        return None
    if abs(HR.px - HL.px) > 0.15 * depth:
        return None
    dur = HR.idx - HL.idx
    if not (25 <= dur <= 325):
        return None
    hd = LH.idx - HR.idx
    if not (3 <= hd <= max(5, 0.4 * dur)):
        return None
    if HR.px - LH.px > 0.4 * depth or LH.px < LC.px + 0.5 * depth:
        return None
    seg = g.C[HL.idx: HR.idx + 1]
    if (seg <= LC.px + depth / 3).mean() < 0.4:          # U-shaped (a V spends only 1/3 of its time there)
        return None
    return Cand("Cup & handle|Inverse cup & handle", "Cup & handle", HR.px, 0.0, 1, depth, LH.px, HL.idx, LH.idx,
                [HL, LC, HR, LH])


def t_cup_no_handle(P, g):
    """Rounded cup whose right side climbs back through the left rim (no handle)."""
    if not types_ok(P, (-1, 1, -1)):
        return None
    L0, HL, LC = P
    depth = HL.px - LC.px
    base = abs(HL.px)
    if depth <= 0 or not (0.12 <= depth / base <= 0.50) or L0.px >= HL.px - 0.3 * depth:
        return None
    left = LC.idx - HL.idx
    if not (12 <= left <= 200):
        return None
    if (g.C[HL.idx: LC.idx + 1] <= LC.px + depth / 3).mean() < 0.45:   # rounded left side, not a straight fall
        return None
    return Cand("Cup (no handle) breakout|Inverted cup breakdown", "Cup (no handle)", HL.px, 0.0, 1, depth,
                LC.px, HL.idx, LC.idx, [L0, HL, LC], expire=HL.idx + 4 * left)


def t_v_bottom(P, g):
    if not types_ok(P, (1, -1)):
        return None
    H0, L1 = P
    dec = H0.px - L1.px
    D = L1.idx - H0.idx
    if D <= 0 or dec < 2.0 * g.k * g.atr[L1.idx] or dec / D < 0.5 * g.atr[L1.idx]:
        return None
    lvl = L1.px + 0.618 * dec
    return Cand("V-bottom|V-top (spike)", "V reversal", lvl, 0.0, 1, dec, L1.px, H0.idx, L1.idx, [H0, L1],
                target_abs=H0.px, expire=L1.idx + max(D, 5))


def t_abc_wave3(P, g):
    if not types_ok(P, (-1, 1, -1)):
        return None
    A, B, C = P
    ab = B.px - A.px
    if ab <= 1.5 * g.k * g.atr[B.idx] or C.px <= A.px:
        return None
    r = (B.px - C.px) / ab
    if not (0.382 - 0.03 <= r <= 0.786 + 0.03):
        return None
    return Cand("Wave-3 / ABC breakout up|Wave-3 / ABC breakdown", "Wave-3 / ABC measured move", B.px, 0.0, 1, ab,
                C.px, A.idx, C.idx, [A, B, C], target_abs=C.px + ab)


def t_dow_reversal(P, g):
    if not types_ok(P, (1, -1, 1, -1)):
        return None
    H0, L1, H1, L2 = P
    if not (H1.px < H0.px and L2.px > L1.px):
        return None
    if abs(L2.px - L1.px) <= 0.2 * (H1.px - min(L1.px, L2.px)):     # that is a double bottom
        return None
    return Cand("Dow trend reversal up|Dow trend reversal down", "Dow theory reversal", H1.px, 0.0, 1,
                H1.px - L2.px, L2.px, H0.idx, L2.idx, [H0, L1, H1, L2])


def t_three_drives(P, g):
    if not types_ok(P, (1, -1, 1, -1, 1, -1)):
        return None
    H0, L1, H1, L2, H2, L3 = P
    if not (L1.px > L2.px > L3.px and H0.px > H1.px > H2.px):
        return None
    try:
        e2 = (H1.px - L2.px) / (H1.px - L1.px)
        e3 = (H2.px - L3.px) / (H2.px - L2.px)
        r1 = (H1.px - L1.px) / (H0.px - L1.px)
        r2 = (H2.px - L2.px) / (H1.px - L2.px)
    except ZeroDivisionError:
        return None
    if not (in_range(e2, 1.13, 1.9) and in_range(e3, 1.13, 1.9) and in_range(r1, 0.382, 0.786)
            and in_range(r2, 0.382, 0.786)):
        return None
    return Cand("Three drives bottom|Three drives top", "Three drives", H2.px, 0.0, 1, H0.px - L3.px, L3.px,
                H0.idx, L3.idx, [H0, L1, H1, L2, H2, L3], target_abs=L3.px + 0.618 * (H0.px - L3.px))


HARMONICS = {
    # name: (AB/XA range, BC/AB range, CD/BC range, AD/XA range)  (AD/XA = D retracement of XA)
    "Gartley": ((0.55, 0.68), (0.382, 0.886), (1.13, 1.618), (0.72, 0.85)),
    "Bat": ((0.35, 0.55), (0.382, 0.886), (1.618, 2.618), (0.83, 0.93)),
    "Butterfly": ((0.72, 0.85), (0.382, 0.886), (1.618, 2.618), (1.20, 1.618)),
    "Crab": ((0.382, 0.618), (0.382, 0.886), (2.24, 3.618), (1.55, 1.70)),
    "Deep crab": ((0.83, 0.93), (0.382, 0.886), (2.0, 3.618), (1.55, 1.70)),
}


def t_harmonics(P, g):
    if not types_ok(P, (-1, 1, -1, 1, -1)):
        return None
    X, A, B, C, D = P
    xa = A.px - X.px
    ab = A.px - B.px
    bc = C.px - B.px
    cd = C.px - D.px
    if xa <= 0 or ab <= 0 or bc <= 0 or cd <= 0:
        return None
    r_ab, r_bc, r_cd, r_ad = ab / xa, bc / ab, cd / bc, (A.px - D.px) / xa
    out = []
    lvl = D.px + 1.0 * g.atr14[D.idx]
    for nm, (rab, rbc, rcd, rad) in HARMONICS.items():
        if in_range(r_ab, *rab) and in_range(r_bc, *rbc) and in_range(r_cd, *rcd) and in_range(r_ad, *rad, tol=0.02):
            out.append(Cand(f"Bullish {nm}|Bearish {nm}", "Harmonic", lvl, 0.0, 1, A.px - D.px,
                            min(D.px, X.px) - 0.5 * g.atr14[D.idx], X.idx, D.idx, [X, A, B, C, D], variant=nm,
                            target_abs=D.px + 0.618 * (A.px - D.px), expire=D.idx + 10))
    # Cypher: C beyond A (1.13-1.414 XA), D = 0.786 of XC
    xc = C.px - X.px
    if xc > 0 and in_range(r_ab, 0.382, 0.618) and in_range(xc / xa, 1.13, 1.414) \
            and in_range((C.px - D.px) / xc, 0.72, 0.85):
        out.append(Cand("Bullish Cypher|Bearish Cypher", "Harmonic", lvl, 0.0, 1, C.px - D.px,
                        X.px, X.idx, D.idx, [X, A, B, C, D], variant="Cypher",
                        target_abs=D.px + 0.618 * (C.px - D.px), expire=D.idx + 10))
    return out or None


def t_abcd(P, g):
    if not types_ok(P, (1, -1, 1, -1)):
        return None
    A, B, C, D = P
    ab, bc, cd = A.px - B.px, C.px - B.px, C.px - D.px
    if ab <= 0 or bc <= 0 or cd <= 0 or D.px >= B.px:
        return None
    if not (in_range(bc / ab, 0.382, 0.886) and in_range(cd / ab, 0.9, 1.1, tol=0.0)
            and in_range(cd / bc, 1.13, 2.618)):
        return None
    lvl = D.px + 1.0 * g.atr14[D.idx]
    return Cand("Bullish AB=CD|Bearish AB=CD", "Harmonic", lvl, 0.0, 1, A.px - D.px,
                D.px - 0.5 * g.atr14[D.idx], A.idx, D.idx, [A, B, C, D], variant="AB=CD",
                target_abs=D.px + 0.618 * (A.px - D.px), expire=D.idx + 10)


def t_wolfe(P, g):
    if not types_ok(P, (-1, 1, -1, 1, -1)):
        return None
    p1, p2, p3, p4, p5 = P
    if not (p3.px < p1.px and p4.px < p2.px and p5.px < p3.px and p4.px > p1.px):
        return None
    a13, b13 = line_through(p1, p3)
    a24, b24 = line_through(p2, p4)
    if not (b24 < b13) or p5.px >= a13 + b13 * p5.idx:          # converging, 5 overshoots the 1-3 line
        return None
    a14, b14 = line_through(p1, p4)
    return Cand("Bullish Wolfe wave|Bearish Wolfe wave", "Wolfe wave", a13, b13, 1, abs(p4.px - p5.px),
                p5.px - 0.5 * g.atr14[p5.idx], p1.idx, p5.idx, [p1, p2, p3, p4, p5],
                target_abs=a14 + b14 * (p5.idx + (p5.idx - p1.idx) * 0.5), expire=p5.idx + 15)


def t_quasimodo(P, g):
    if not types_ok(P, (-1, 1, -1, 1, -1)):
        return None
    L1, H1, L2, H2, L3 = P
    rng = H2.px - L2.px
    if not (L2.px < L1.px and H2.px > H1.px and L3.px > L2.px) or rng <= 0:
        return None
    if abs(L3.px - L1.px) > 0.35 * rng:
        return None
    lvl = L3.px + 1.0 * g.atr14[L3.idx]
    return Cand("Bullish Quasimodo|Bearish Quasimodo", "Quasimodo", lvl, 0.0, 1, H2.px - L3.px, L2.px,
                L1.idx, L3.idx, [L1, H1, L2, H2, L3], target_abs=H2.px, expire=L3.idx + 15)


def t_elliott5(P, g):
    """Five-wave decline that obeys the Elliott rules -> bullish reversal once half of wave 5 is retraced."""
    if not types_ok(P, (1, -1, 1, -1, 1, -1)):
        return None
    p0, p1, p2, p3, p4, p5 = P
    if not (p2.px < p0.px and p3.px < p1.px and p4.px < p1.px and p5.px < p3.px):
        return None
    w1, w3, w5 = p0.px - p1.px, p2.px - p3.px, p4.px - p5.px
    if w3 < min(w1, w5):
        return None
    lvl = p5.px + 0.5 * w5
    return Cand("Elliott 5-wave decline complete|Elliott 5-wave advance complete", "Elliott wave", lvl, 0.0, 1,
                p0.px - p5.px, p5.px, p0.idx, p5.idx, [p0, p1, p2, p3, p4, p5],
                target_abs=p5.px + 0.382 * (p0.px - p5.px), expire=p5.idx + 30)


def t_trendline(P, g):
    """Break of the line through the last two descending swing highs."""
    if not types_ok(P, (1, -1, 1, -1)):
        return None
    Ha, La, Hb, Lb = P
    if Hb.px >= Ha.px or Hb.idx - Ha.idx < g.minw:
        return None
    a, b = line_through(Ha, Hb)
    seg = np.arange(Ha.idx + 1, Lb.idx + 1)
    if len(seg) and (g.C[seg] > a + b * seg + 0.25 * g.atr[Lb.idx]).any():
        return None
    return Cand("Downtrend-line breakout|Uptrend-line breakdown", "Trendline break", a, b, 1,
                (a + b * Lb.idx) - Lb.px, Lb.px, Ha.idx, Lb.idx, [Ha, La, Hb, Lb])


def t_vcp(P, g):
    """Minervini volatility contraction pattern (bullish only)."""
    if g.s < 0 or g.scale != "minor":
        return None
    for m in (6, 4):
        Q = P[-m:]
        if len(Q) < m or not types_ok(Q, (1, -1) * (m // 2)):
            continue
        highs, lows = Q[0::2], Q[1::2]
        d = [(hh.px - ll.px) / hh.px for hh, ll in zip(highs, lows)]
        if not all(d[i + 1] <= 0.8 * d[i] for i in range(len(d) - 1)):
            continue
        if d[-1] > 0.12 or d[0] > 0.35 or d[0] < 0.08:
            continue
        if max(x.px for x in highs[1:]) > highs[0].px * 1.03 or not all(lows[i + 1].px > lows[i].px for i in range(len(lows) - 1)):
            continue
        t = Q[-1].idx
        sma50, sma150, sma200 = g.x["sma50"][t], g.x["sma150"][t], g.x["sma200"][t]
        if not (g.C[t] > sma50 > sma150 > sma200):
            continue
        return Cand("VCP breakout|-", "VCP", highs[-1].px, 0.0, 1, highs[0].px - lows[0].px, lows[-1].px,
                    Q[0].idx, Q[-1].idx, list(Q), variant=f"{len(d)} contractions")
    return None


MIRRORED = [(t_double_bottom, 4), (t_triple_bottom, 6), (t_head_shoulders, 6), (t_cup_handle, 4),
            (t_cup_no_handle, 3), (t_v_bottom, 2), (t_abc_wave3, 3), (t_dow_reversal, 4), (t_three_drives, 6),
            (t_harmonics, 5), (t_abcd, 4), (t_wolfe, 5), (t_quasimodo, 5), (t_elliott5, 6), (t_trendline, 4),
            (t_vcp, 6)]


# ------------------------------------------------------------------------------------------------
# neutral trendline geometry: triangles, wedges, channels, rectangles, broadening, diamonds
# ------------------------------------------------------------------------------------------------
GEOM = {
    ("flat", "flat", "par"): "Rectangle",
    ("flat", "up", "conv"): "Ascending triangle",
    ("down", "flat", "conv"): "Descending triangle",
    ("down", "up", "conv"): "Symmetrical triangle",
    ("up", "up", "conv"): "Rising wedge",
    ("down", "down", "conv"): "Falling wedge",
    ("up", "up", "par"): "Channel up",
    ("down", "down", "par"): "Channel down",
    ("up", "down", "div"): "Broadening formation",
    ("up", "flat", "div"): "Right-angled broadening (ascending)",
    ("flat", "down", "div"): "Right-angled broadening (descending)",
    ("up", "up", "div"): "Ascending broadening wedge",
    ("down", "down", "div"): "Descending broadening wedge",
}


def geometry(P: list[Pv], g, C: np.ndarray):
    """Classify the last pivots into a two-trendline pattern; returns up/down candidates."""
    for m in (6, 5, 4):
        Q = P[-m:]
        if len(Q) < m:
            continue
        hs = [p for p in Q if p.typ == 1]
        ls = [p for p in Q if p.typ == -1]
        if len(hs) < 2 or len(ls) < 2:
            continue
        au, bu, ru = fit_line(hs)
        al, bl, rl = fit_line(ls)
        x0, x1 = Q[0].idx, Q[-1].idx
        W = x1 - x0
        if not (g.minw <= W <= g.maxw):
            continue
        # both trendlines must be touched near the start and near the end of the pattern
        if Q[1].idx - Q[0].idx > 0.45 * W or Q[-1].idx - Q[-2].idx > 0.45 * W:
            continue
        h0 = (au + bu * x0) - (al + bl * x0)
        h1 = (au + bu * x1) - (al + bl * x1)
        if h0 <= 0 or h1 <= 0:
            continue
        tol = max(0.5 * g.atr[x1], 0.12 * max(h0, h1))
        if ru > tol or rl > tol:
            continue
        seg = np.arange(x0, x1 + 1)
        up_l, lo_l = au + bu * seg, al + bl * seg
        viol = ((C[seg] > up_l + tol) | (C[seg] < lo_l - tol)).sum()
        if viol > 1:
            continue
        du, dl = bu * W / max(h0, h1), bl * W / max(h0, h1)
        cls = lambda d: "flat" if abs(d) < 0.2 else ("up" if d > 0 else "down")  # noqa: E731
        ratio = h1 / h0
        shape = "conv" if ratio < 0.8 else ("div" if ratio > 1.25 else "par")
        name = GEOM.get((cls(du), cls(dl), shape))
        if name is None:
            continue
        if m < 5 and name not in ("Rectangle", "Ascending triangle", "Descending triangle", "Symmetrical triangle"):
            continue                                   # wedges, channels, broadening need 5+ touches
        height = max(h0, h1)
        apex = None
        if shape == "conv" and (bl - bu) > 0:
            apex = int((au - al) / (bl - bu))
        exp = apex if apex is not None else x1 + W
        c_up = Cand(f"{name} - up breakout", name, au, bu, 1, height, 0.0, x0, x1, list(Q), expire=exp)
        c_dn = Cand(f"{name} - down breakout", name, al, bl, -1, height, 0.0, x0, x1, list(Q), expire=exp)
        c_up.variant = c_dn.variant = f"{m} pivots"
        c_up.x_other = (al, bl)
        c_dn.x_other = (au, bu)
        return [c_up, c_dn]
    return None


def diamond(P: list[Pv], g, C):
    Q = P[-6:]
    if len(Q) < 6:
        return None
    hs = [p for p in Q if p.typ == 1]
    ls = [p for p in Q if p.typ == -1]
    if len(hs) != 3 or len(ls) != 3:
        return None
    if not (hs[1].px > hs[0].px and hs[1].px > hs[2].px and ls[1].px < ls[0].px and ls[1].px < ls[2].px):
        return None
    W = Q[-1].idx - Q[0].idx
    if not (g.minw <= W <= g.maxw):
        return None
    span = hs[1].px - ls[1].px
    if abs(hs[0].px - hs[2].px) > 0.5 * span or abs(ls[0].px - ls[2].px) > 0.5 * span:
        return None
    if hs[0].px - ls[0].px < 0.25 * span or hs[2].px - ls[2].px < 0.25 * span:
        return None
    mid = (hs[1].idx + ls[1].idx) / 2
    if not (0.5 <= (mid - Q[0].idx) / max(Q[-1].idx - mid, 1) <= 2.0):
        return None
    au, bu = line_through(hs[1], hs[2])
    al, bl = line_through(ls[1], ls[2])
    height = hs[1].px - ls[1].px
    c_up = Cand("Diamond - up breakout", "Diamond", au, bu, 1, height, 0.0, Q[0].idx, Q[-1].idx, list(Q),
                expire=Q[-1].idx + W)
    c_dn = Cand("Diamond - down breakout", "Diamond", al, bl, -1, height, 0.0, Q[0].idx, Q[-1].idx, list(Q),
                expire=Q[-1].idx + W)
    c_up.x_other, c_dn.x_other = (al, bl), (au, bu)
    return [c_up, c_dn]


# ------------------------------------------------------------------------------------------------
# engine
# ------------------------------------------------------------------------------------------------
class PatternScanner:
    def __init__(self, df: pd.DataFrame, extra: dict):
        self.df = df
        self.o, self.h, self.l, self.c = (df[x].to_numpy(float) for x in ("open", "high", "low", "close"))
        self.v = np.nan_to_num(df["volume"].to_numpy(float))
        self.n = len(self.c)
        self.atr14 = np.asarray(talib.ATR(self.h, self.l, self.c, 14), float)
        self.atr50 = np.asarray(talib.ATR(self.h, self.l, self.c, 50), float)
        # early bars: fall back to ATR14 so short histories still get pivots
        self.atr50 = np.where(np.isfinite(self.atr50), self.atr50, self.atr14)
        self.extra = extra
        self.events: list[dict] = []
        self.keys: set = set()
        self.zz: dict = {}

    # -- emit --------------------------------------------------------------------------------------
    def emit(self, t, name, family, direction, scale, level, target, stop, start, end, height, variant="",
             pivots=None):
        key = (name, scale, start)
        if key in self.keys:
            return
        self.keys.add(key)
        self.events.append({"t": int(t), "pattern": name, "family": family, "dir": int(direction), "scale": scale,
                            "level": float(level), "target": float(target) if target is not None else np.nan,
                            "stop": float(stop) if stop is not None else np.nan, "start": int(start),
                            "end": int(end), "height": float(height), "variant": variant,
                            "pivots": [(int(p.idx), float(p.px)) for p in (pivots or [])]})

    # -- trigger search --------------------------------------------------------------------------
    @staticmethod
    def first_cross(C, a, b, dirn, t0, t1):
        if t1 < t0:
            return -1
        t = np.arange(t0, t1 + 1)
        lev = a + b * t
        lev_p = a + b * (t - 1)
        prev = C[np.maximum(t - 1, 0)]
        if dirn > 0:
            hit = (C[t] > lev) & ~(prev > lev_p)
        else:
            hit = (C[t] < lev) & ~(prev < lev_p)
        k = np.flatnonzero(hit)
        return int(t[k[0]]) if len(k) else -1

    def run_candidates(self, cands, g: Ctx, t0, t1, scale):
        for cd in cands:
            if cd is None:
                continue
            te = t1
            if cd.expire is not None:
                te = min(te, int(cd.expire))
            te = min(te, cd.end + max(15, cd.end - cd.start), self.n - 1)
            tt = self.first_cross(g.C, cd.a, cd.b, cd.dirn, max(t0, 1), te)
            if tt < 0:
                continue
            s = g.s
            lev = cd.a + cd.b * tt
            if hasattr(cd, "x_other"):
                # neutral geometry: stop at the opposite line, target = breakout +/- height
                ao, bo = cd.x_other
                stop_o = ao + bo * tt
                tgt_o = lev + cd.dirn * cd.height
            else:
                stop_o = cd.stop
                tgt_o = cd.target_abs if cd.target_abs is not None else lev + cd.dirn * cd.height
            names = cd.name.split("|")
            nm = names[0] if s > 0 or len(names) == 1 else names[1]
            if nm == "-":
                continue
            real = lambda x: s * x  # noqa: E731
            pv = [Pv(p.idx, real(p.px), s * p.typ) for p in cd.pivots]
            self.emit(tt, nm, cd.family, s * cd.dirn, scale, real(lev), real(tgt_o), real(stop_o), cd.start, cd.end,
                      abs(cd.height), cd.variant, pv)

    def scan_zigzag(self, scale: str):
        k = SCALES[scale]
        piv, ext_idx, ext_px, leg, nconf = zigzag(self.h, self.l, k * self.atr50)
        self.zz[scale] = (piv, ext_idx, ext_px, leg, nconf)
        if len(piv) < 3:
            return
        ctxs = {s: Ctx(s, self.h, self.l, self.c, self.atr50, self.atr14, scale, self.extra) for s in (1, -1)}
        for j in range(len(piv)):
            tc = piv[j].conf
            t_next = piv[j + 1].conf if j + 1 < len(piv) else self.n
            confirmed = piv[: j + 1]
            # ---------- alignment A: last m confirmed pivots ----------
            for s, g in ctxs.items():
                Po = [Pv(p.idx, s * p.px, s * p.typ) for p in confirmed[-7:]]
                cands = []
                for fn, m in MIRRORED:
                    if len(Po) >= m:
                        r = fn(Po[-m:] if fn is not t_vcp else Po, g)
                        if r is not None:
                            cands.extend(r if isinstance(r, list) else [r])
                if s > 0:
                    for fn in (geometry, diamond):
                        r = fn(Po, g, g.C)
                        if r:
                            cands.extend(r)
                self.run_candidates(cands, g, tc, t_next - 1, scale)
            # ---------- alignment B: last m-1 confirmed + running extreme ----------
            t_lo, t_hi = tc + 1, min(t_next, self.n - 1)
            if t_hi < t_lo:
                continue
            e_prev = ext_idx[t_lo - 1: t_hi]              # extreme as of t-1 for t in [t_lo, t_hi]
            bounds = np.flatnonzero(np.diff(e_prev)) + 1
            starts = np.r_[0, bounds]
            ends = np.r_[bounds - 1, len(e_prev) - 1]
            for a_, b_ in zip(starts, ends):
                ei = int(e_prev[a_])
                if ei < 0 or ei <= confirmed[-1].idx:
                    continue
                E = Pv(ei, float(ext_px[t_lo - 1 + a_]), -confirmed[-1].typ)
                for s, g in ctxs.items():
                    Po = [Pv(p.idx, s * p.px, s * p.typ) for p in confirmed[-6:]] + [Pv(E.idx, s * E.px, s * E.typ)]
                    cands = []
                    for fn, m in MIRRORED:
                        if len(Po) >= m:
                            r = fn(Po[-m:] if fn is not t_vcp else Po, g)
                            if r is not None:
                                cands.extend(r if isinstance(r, list) else [r])
                    if s > 0:
                        for fn in (geometry, diamond):
                            r = fn(Po, g, g.C)
                            if r:
                                # the breakout must reverse the leg that made the running extreme
                                cands.extend([c_ for c_ in r if c_.dirn == -E.typ])
                    self.run_candidates(cands, g, t_lo + a_, t_lo + b_, scale)

    # ------------------------------------------------------------------------------------------------
    # bar-based patterns (no zigzag)
    # ------------------------------------------------------------------------------------------------
    def bar_patterns(self):
        o, h, l, c, v, n = self.o, self.h, self.l, self.c, self.v, self.n
        atr = self.atr14
        idx = np.arange(n)
        vs = pd.Series(v)
        v50 = vs.rolling(50, min_periods=20).mean().to_numpy()
        sma50, sma200 = self.extra["sma50"], self.extra["sma200"]

        def add(mask, name, fam, d, level, target=None, stop=None, start=None, height=None, variant=""):
            for t in np.flatnonzero(mask):
                lv = level[t] if np.ndim(level) else level
                tg = None if target is None else (target[t] if np.ndim(target) else target)
                st = None if stop is None else (stop[t] if np.ndim(stop) else stop)
                s0 = t if start is None else int(start[t] if np.ndim(start) else start)
                ht = 0.0 if height is None else float(height[t] if np.ndim(height) else height)
                self.emit(t, name, fam, d, "bar", lv, tg, st, s0, t, ht, variant)

        prev_c = np.r_[np.nan, c[:-1]]
        prev_h, prev_l = np.r_[np.nan, h[:-1]], np.r_[np.nan, l[:-1]]

        # --- flags & pennants: strong pole then tight counter-trend consolidation, then breakout ---
        for d in (1, -1):
            C = c if d > 0 else -c
            Hh = h if d > 0 else -l
            Ll = l if d > 0 else -h
            for t in range(60, n):
                for fl in range(3, 16):                        # flag length
                    p_end = t - fl                             # last bar of the pole
                    for pl in (5, 10, 15):                     # pole length
                        p0 = p_end - pl
                        if p0 < 1:
                            continue
                        pole = Hh[p_end] - Ll[p0]
                        a_ = atr[p_end]
                        if not np.isfinite(a_) or pole < 5 * a_:
                            continue
                        if Hh[p_end] < Hh[p0: p_end + 1].max() - 1e-9:
                            continue
                        fh = Hh[p_end: t].max()
                        fl_lo = Ll[p_end: t].min()
                        if fh > Hh[p_end] + 0.25 * a_ or fh - fl_lo > 0.5 * pole:
                            continue
                        if not (C[t] > fh and C[t - 1] <= fh):
                            continue
                        seg_h, seg_l = Hh[p_end + 1: t], Ll[p_end + 1: t]
                        kind = "flag"
                        if len(seg_h) >= 3:
                            xh = np.arange(len(seg_h))
                            bh = np.polyfit(xh, seg_h, 1)[0]
                            bl_ = np.polyfit(xh, seg_l, 1)[0]
                            if bh < 0 < bl_:
                                kind = "pennant"
                        gain = pole / abs(Ll[p0]) if Ll[p0] != 0 else 0
                        nm = {("flag", 1): "Bull flag", ("flag", -1): "Bear flag",
                              ("pennant", 1): "Bull pennant", ("pennant", -1): "Bear pennant"}[(kind, d)]
                        if d > 0 and gain >= 0.9 and pl >= 5:
                            nm = "High tight flag"
                        self.emit(t, nm, "Flag / pennant", d, "bar", d * fh, d * (fh + pole), d * fl_lo, p0, t,
                                  pole, f"pole {pl}d, flag {fl}d")
                        break
                    else:
                        continue
                    break
        # --- high tight flag (O'Neil): +90% in <= 40 bars, <= 25% pullback over 10-25 bars, breakout ---
        for t in range(80, n):
            for fl in (10, 15, 20, 25):
                pk = t - fl
                base = c[max(0, pk - 40): pk + 1].min()
                if base <= 0 or c[pk] / base < 1.9:
                    continue
                top = h[pk - 5: t].max()
                if l[pk: t].min() < top * 0.75:
                    continue
                if c[t] > top and c[t - 1] <= top:
                    self.emit(t, "High tight flag", "Flag / pennant", 1, "bar", top, top * 1.2, l[pk: t].min(), pk - 40,
                              t, top - base, "O'Neil")
                    break
        # --- rounding bottom / top (saucer): quadratic fit of closes ---
        for w in (60, 120, 250):
            x = np.arange(w, dtype=float)
            X = np.vstack([x ** 2, x, np.ones(w)]).T
            pinv = np.linalg.pinv(X)
            for t in range(w, n, 1):
                y = c[t - w: t]
                coef = pinv @ y
                a2, b1, c0 = coef
                if a2 == 0:
                    continue
                xv = -b1 / (2 * a2)
                if not (0.3 * w <= xv <= 0.7 * w):
                    continue
                fit = X @ coef
                ss = ((y - fit) ** 2).sum()
                st = ((y - y.mean()) ** 2).sum()
                if st <= 0 or 1 - ss / st < 0.75:
                    continue
                rim_l = y[: max(3, w // 10)].max() if a2 > 0 else y[: max(3, w // 10)].min()
                vert = c0 + b1 * xv + a2 * xv * xv
                depth = abs(rim_l - vert) / abs(rim_l)
                if depth < 0.10:
                    continue
                if a2 > 0 and c[t] > rim_l and c[t - 1] <= rim_l:
                    self.emit(t, "Rounding bottom (saucer)", "Rounding", 1, "bar", rim_l, rim_l + (rim_l - vert),
                              vert, t - w, t, rim_l - vert, f"{w}d")
                elif a2 < 0 and c[t] < rim_l and c[t - 1] >= rim_l:
                    self.emit(t, "Rounding top", "Rounding", -1, "bar", rim_l, rim_l - (vert - rim_l), vert, t - w, t,
                              vert - rim_l, f"{w}d")
        # --- island reversals ---
        for t in range(2, n):
            for k in range(1, 16):
                s0 = t - k                      # first bar of the island
                if s0 < 1:
                    break
                isl_h, isl_l = h[s0: t].max(), l[s0: t].min()
                if isl_h < l[s0 - 1] and isl_h < l[t]:
                    self.emit(t, "Island bottom", "Island reversal", 1, "bar", l[t], None, isl_l, s0, t,
                              l[s0 - 1] - isl_l)
                    break
                if isl_l > h[s0 - 1] and isl_l > h[t]:
                    self.emit(t, "Island top", "Island reversal", -1, "bar", h[t], None, isl_h, s0, t,
                              isl_h - h[s0 - 1])
                    break
        # --- gaps: breakaway / runaway / exhaustion ---
        hi40, lo40 = pd.Series(h).rolling(40).max().shift(1).to_numpy(), pd.Series(l).rolling(40).min().shift(1).to_numpy()
        rng40 = (hi40 - lo40) / np.maximum(c, 1e-9)
        up_gap = l > prev_h
        dn_gap = h < prev_l
        big = np.abs(o - prev_c) >= 0.5 * atr
        volx = v / np.maximum(v50, 1)
        ret20 = c / np.r_[np.full(20, np.nan), c[:-20]] - 1
        add(up_gap & big & (c > hi40) & (rng40 < 0.25) & (volx >= 1.5), "Breakaway gap up", "Gaps", 1, prev_h,
            stop=prev_h - atr)
        add(dn_gap & big & (c < lo40) & (rng40 < 0.25) & (volx >= 1.5), "Breakaway gap down", "Gaps", -1, prev_l,
            stop=prev_l + atr)
        add(up_gap & big & (ret20 > 0.15) & ~(c > hi40), "Runaway gap up", "Gaps", 1, prev_h, stop=prev_h - atr)
        add(dn_gap & big & (ret20 < -0.15) & ~(c < lo40), "Runaway gap down", "Gaps", -1, prev_l, stop=prev_l + atr)
        # exhaustion: an up-gap after a >25% run that is filled within 5 bars -> bearish at the fill
        for t in np.flatnonzero(up_gap & big & (ret20 > 0.25)):
            for j in range(t + 1, min(n, t + 6)):
                if c[j] < prev_h[t]:
                    self.emit(j, "Exhaustion gap (up, filled)", "Gaps", -1, "bar", prev_h[t], None, h[t: j + 1].max(),
                              t, j, h[t: j + 1].max() - prev_h[t])
                    break
        for t in np.flatnonzero(dn_gap & big & (ret20 < -0.25)):
            for j in range(t + 1, min(n, t + 6)):
                if c[j] > prev_l[t]:
                    self.emit(j, "Exhaustion gap (down, filled)", "Gaps", 1, "bar", prev_l[t], None, l[t: j + 1].min(),
                              t, j, prev_l[t] - l[t: j + 1].min())
                    break
        # --- event decline & dead-cat bounce ---
        drop2 = c / np.r_[np.full(2, np.nan), c[:-2]] - 1
        for t in np.flatnonzero((drop2 <= -0.15) & (np.r_[np.nan, drop2[:-1]] > -0.15)):
            drop = c[t - 2] - c[t]
            self.emit(t, "Event decline (>=15% in 2 days)", "Event decline", 1, "bar", c[t], None, None, t - 2, t, drop)
            lo, bhi, bounced = l[t], -np.inf, False
            for j in range(t + 1, min(n, t + 41)):
                if not bounced:
                    if l[j] < lo:
                        lo, bhi = l[j], -np.inf
                    bhi = max(bhi, h[j])
                    bounced = bhi - lo >= 0.25 * drop
                elif c[j] < lo:
                    self.emit(j, "Dead-cat bounce breakdown", "Event decline", -1, "bar", lo, None, bhi, t - 2, j,
                              bhi - lo)
                    break
        # --- key reversal days ---
        ll20 = pd.Series(l).rolling(20).min().shift(1).to_numpy()
        hh20 = pd.Series(h).rolling(20).max().shift(1).to_numpy()
        add((l < ll20) & (c > prev_c) & (h > prev_h), "Key reversal day (bullish)", "Reversal bars", 1, c, stop=l)
        add((h > hh20) & (c < prev_c) & (l < prev_l), "Key reversal day (bearish)", "Reversal bars", -1, c, stop=h)
        # --- 2B / turtle soup: new 20-day low that closes back above the prior 20-day low ---
        prior_low_age = 19 - pd.Series(l).rolling(20).apply(np.argmin, raw=True).shift(1).to_numpy()
        prior_high_age = 19 - pd.Series(h).rolling(20).apply(np.argmax, raw=True).shift(1).to_numpy()
        for d in (1, -1):
            if d > 0:
                cond = np.zeros(n, bool)
                for lagk in (0, 1, 2):
                    li = np.r_[np.full(lagk, np.nan), l[: n - lagk]]
                    ref = np.r_[np.full(lagk, np.nan), ll20[: n - lagk]]
                    age = np.r_[np.full(lagk, np.nan), prior_low_age[: n - lagk]]
                    cond |= (li < ref) & (c > ref) & (age >= 3)
                add(cond & ~np.r_[False, cond[:-1]], "2B bottom / Turtle soup", "False breakout", 1, ll20,
                    stop=pd.Series(l).rolling(3).min().to_numpy())
            else:
                cond = np.zeros(n, bool)
                for lagk in (0, 1, 2):
                    hi_ = np.r_[np.full(lagk, np.nan), h[: n - lagk]]
                    ref = np.r_[np.full(lagk, np.nan), hh20[: n - lagk]]
                    age = np.r_[np.full(lagk, np.nan), prior_high_age[: n - lagk]]
                    cond |= (hi_ > ref) & (c < ref) & (age >= 3)
                add(cond & ~np.r_[False, cond[:-1]], "2B top / Turtle soup", "False breakout", -1, hh20,
                    stop=pd.Series(h).rolling(3).max().to_numpy())
        # --- Wyckoff spring / upthrust: false break of a 40-day trading range ---
        rng_ok = rng40 < 0.20
        add(rng_ok & (l < lo40) & (l > lo40 - 1.5 * atr) & (c > lo40), "Wyckoff spring", "Wyckoff", 1, lo40,
            target=hi40, stop=l)
        add(rng_ok & (h > hi40) & (h < hi40 + 1.5 * atr) & (c < hi40), "Wyckoff upthrust", "Wyckoff", -1, hi40,
            target=lo40, stop=h)
        # --- NR7 / inside-day breakouts (Crabel) ---
        rng = h - l
        nr7 = rng <= pd.Series(rng).rolling(7).min().to_numpy()
        inside = (h <= prev_h) & (l >= prev_l)
        for base_mask, nm in ((nr7, "NR7"), (inside, "Inside day")):
            for d in (1, -1):
                hit = np.zeros(n, bool)
                lvl = np.full(n, np.nan)
                for t in np.flatnonzero(base_mask):
                    for j in range(t + 1, min(n, t + 4)):
                        if d > 0 and c[j] > h[t]:
                            hit[j], lvl[j] = True, h[t]
                            break
                        if d < 0 and c[j] < l[t]:
                            hit[j], lvl[j] = True, l[t]
                            break
                add(hit, f"{nm} breakout {'up' if d > 0 else 'down'}", "Range contraction", d, lvl)
        # --- Darvas box ---
        self._darvas()
        # --- 52-week / multi-year breakouts ---
        hi252 = pd.Series(h).rolling(252, min_periods=200).max().shift(1).to_numpy()
        lo252 = pd.Series(l).rolling(252, min_periods=200).min().shift(1).to_numpy()
        ath = np.r_[np.nan, np.fmax.accumulate(h)[:-1]]
        fresh_h = ~(pd.Series(c > hi252).rolling(20).max().shift(1).fillna(0).astype(bool).to_numpy())
        fresh_l = ~(pd.Series(c < lo252).rolling(20).max().shift(1).fillna(0).astype(bool).to_numpy())
        add((c > hi252) & fresh_h, "52-week high breakout", "New highs / lows", 1, hi252)
        add((c < lo252) & fresh_l, "52-week low breakdown", "New highs / lows", -1, lo252)
        fresh_a = ~(pd.Series(c > ath).rolling(20).max().shift(1).fillna(0).astype(bool).to_numpy())
        add((c > ath) & (idx >= 500) & fresh_a, "Multi-year high breakout (since 2016)", "New highs / lows", 1, ath)
        # --- volume patterns: pocket pivot, climaxes ---
        down_vol = np.where(c < prev_c, v, 0.0)
        max_dv10 = pd.Series(down_vol).rolling(10).max().shift(1).to_numpy()
        pp = (c > prev_c) & (v > max_dv10) & (c > sma50) & (np.abs(c / sma50 - 1) < 0.10) & (sma50 > sma200)
        add(pp, "Pocket pivot", "Volume patterns", 1, c)
        wide = rng > 2.0 * atr
        ret10 = c / np.r_[np.full(10, np.nan), c[:-10]] - 1
        sell_clx = (volx >= 3) & wide & (l <= ll20) & (c > l + 0.5 * rng) & (ret10 < -0.10)
        buy_clx = (volx >= 3) & wide & (h >= hh20) & (c < h - 0.5 * rng) & (ret10 > 0.10)
        add(sell_clx, "Selling climax (reversal)", "Volume patterns", 1, c, stop=l)
        add(buy_clx, "Buying climax (reversal)", "Volume patterns", -1, c, stop=h)
        # --- flat base breakout (IBD): >= 25 bars, range <= 15%, prior uptrend >= 20% ---
        for w in (25, 40, 60):
            bh = pd.Series(h).rolling(w).max().shift(1).to_numpy()
            bl = pd.Series(l).rolling(w).min().shift(1).to_numpy()
            ret60 = c / np.r_[np.full(60, np.nan), c[:-60]] - 1
            prior = np.r_[np.full(w, np.nan), ret60[:-w]]
            add((bh / bl - 1 <= 0.15) & (c > bh) & (prev_c <= bh) & (prior >= 0.20), f"Flat base breakout ({w}d)",
                "IBD bases", 1, bh, stop=bl)
        # --- weekly pipe & horn bottoms / tops ---
        self._weekly_pipes()

    def _darvas(self):
        h, l, c, n = self.h, self.l, self.c, self.n
        hi252 = pd.Series(h).rolling(252, min_periods=120).max().to_numpy()
        state, top, bot, t_top = 0, np.nan, np.nan, -1
        for t in range(3, n):
            if state == 0:
                if h[t] >= hi252[t] and np.isfinite(hi252[t]):
                    state, top, t_top = 1, h[t], t
            elif state == 1:                        # establishing the box top (3 bars not exceeded)
                if h[t] > top:
                    top, t_top = h[t], t
                elif t - t_top >= 3:
                    state, bot = 2, l[t_top + 1: t + 1].min()
            elif state == 2:                        # establishing the bottom (3 bars not broken)
                if h[t] > top:
                    state, top, t_top = 1, h[t], t
                    continue
                lo_now = l[t_top + 1: t + 1].min()
                if lo_now < bot:
                    bot = lo_now
                if t - int(np.argmin(l[t_top + 1: t + 1])) - (t_top + 1) >= 3:
                    state = 3
            elif state == 3:                        # box complete: wait for the break
                if c[t] > top:
                    self.emit(t, "Darvas box breakout", "Darvas box", 1, "bar", top, top + (top - bot), bot, t_top, t,
                              top - bot)
                    state, top, t_top = 1, h[t], t
                elif c[t] < bot:
                    self.emit(t, "Darvas box breakdown", "Darvas box", -1, "bar", bot, bot - (top - bot), top, t_top,
                              t, top - bot)
                    state = 0

    def _weekly_pipes(self):
        df = self.df
        wk = df.index.to_period("W-FRI")
        g = pd.DataFrame({"h": self.h, "l": self.l, "c": self.c, "t": np.arange(self.n)}, index=df.index).groupby(wk)
        W = g.agg(h=("h", "max"), l=("l", "min"), c=("c", "last"), t=("t", "last"))
        if len(W) < 20:
            return
        H, L, C, T = W.h.to_numpy(), W.l.to_numpy(), W.c.to_numpy(), W.t.to_numpy()
        rng = H - L
        avg = pd.Series(rng).rolling(10).mean().shift(2).to_numpy()
        for i in range(12, len(W)):
            # pipe bottom: two adjacent weeks with long downward spikes well below the surrounding lows
            a, b = i - 1, i
            around_l = min(L[i - 6: i - 1].min(), L[i - 6: i - 1].min())
            if (rng[a] > 1.5 * avg[i] and rng[b] > 1.5 * avg[i] and abs(L[a] - L[b]) <= 0.25 * rng[a]
                    and max(L[a], L[b]) < around_l - 0.3 * avg[i] and C[b] > (H[b] + L[b]) / 2):
                self.emit(int(T[b]), "Pipe bottom (weekly)", "Pipes & horns", 1, "bar", C[b], None, min(L[a], L[b]),
                          int(T[a - 1]) if a > 0 else 0, int(T[b]), rng[a])
            around_h = H[i - 6: i - 1].max()
            if (rng[a] > 1.5 * avg[i] and rng[b] > 1.5 * avg[i] and abs(H[a] - H[b]) <= 0.25 * rng[a]
                    and min(H[a], H[b]) > around_h + 0.3 * avg[i] and C[b] < (H[b] + L[b]) / 2):
                self.emit(int(T[b]), "Pipe top (weekly)", "Pipes & horns", -1, "bar", C[b], None, max(H[a], H[b]),
                          int(T[a - 1]) if a > 0 else 0, int(T[b]), rng[a])
            # horn: two spikes separated by one smaller week
            if i >= 13:
                a, m_, b = i - 2, i - 1, i
                if (L[a] < L[m_] and L[b] < L[m_] and abs(L[a] - L[b]) <= 0.25 * rng[a]
                        and min(rng[a], rng[b]) > 1.3 * avg[i] and C[b] > (H[b] + L[b]) / 2
                        and max(L[a], L[b]) < L[i - 7: i - 2].min()):
                    self.emit(int(T[b]), "Horn bottom (weekly)", "Pipes & horns", 1, "bar", C[b], None,
                              min(L[a], L[b]), int(T[a]), int(T[b]), rng[a])
                if (H[a] > H[m_] and H[b] > H[m_] and abs(H[a] - H[b]) <= 0.25 * rng[a]
                        and min(rng[a], rng[b]) > 1.3 * avg[i] and C[b] < (H[b] + L[b]) / 2
                        and min(H[a], H[b]) > H[i - 7: i - 2].max()):
                    self.emit(int(T[b]), "Horn top (weekly)", "Pipes & horns", -1, "bar", C[b], None,
                              max(H[a], H[b]), int(T[a]), int(T[b]), rng[a])

    # ------------------------------------------------------------------------------------------------
    # multi-scale & indicator-assisted patterns
    # ------------------------------------------------------------------------------------------------
    def fib_bounce(self):
        """Pullback of 38.2-61.8% of the last major swing that forms a confirmed minor pivot inside the zone."""
        if "major" not in self.zz or "minor" not in self.zz:
            return
        piv_M, _, _, _, nconf_M = self.zz["major"]
        piv_m = self.zz["minor"][0]
        for p in piv_m:
            t = p.conf
            j = nconf_M[t] - 1
            if j < 1:
                continue
            last, prev = piv_M[j], piv_M[j - 1]
            if p.idx <= last.idx:
                continue
            swing = last.px - prev.px
            if last.typ == 1 and p.typ == -1 and swing > 0:              # up-swing, minor low in the zone
                r = (last.px - p.px) / swing
                if 0.382 - 0.02 <= r <= 0.618 + 0.02:
                    self.emit(t, "Fibonacci 38-62% pullback bounce (uptrend)", "Fibonacci", 1, "minor", self.c[t],
                              last.px, prev.px + 0.214 * swing, prev.idx, p.idx, swing, f"{r:.0%} retrace",
                              [prev, last, p])
            if last.typ == -1 and p.typ == 1 and swing < 0:              # down-swing, minor high in the zone
                r = (p.px - last.px) / (-swing)
                if 0.382 - 0.02 <= r <= 0.618 + 0.02:
                    self.emit(t, "Fibonacci 38-62% rally failure (downtrend)", "Fibonacci", -1, "minor", self.c[t],
                              last.px, prev.px + 0.214 * swing, prev.idx, p.idx, -swing, f"{r:.0%} retrace",
                              [prev, last, p])

    def divergences(self, osc: dict):
        """Regular and hidden divergences between price swings and RSI / MACD histogram / OBV / MFI."""
        for scale in ("minor", "intermediate"):
            if scale not in self.zz:
                continue
            piv = self.zz[scale][0]
            lows = [p for p in piv if p.typ == -1]
            highs = [p for p in piv if p.typ == 1]
            for name, x in osc.items():
                for seq, typ in ((lows, -1), (highs, 1)):
                    for a, b in zip(seq[:-1], seq[1:]):
                        if b.idx - a.idx > MAXW[scale] or b.idx - a.idx < 5:
                            continue
                        xa, xb = x[a.idx], x[b.idx]
                        if not (np.isfinite(xa) and np.isfinite(xb)):
                            continue
                        t = b.conf
                        if typ == -1 and b.px < a.px and xb > xa:
                            self.emit(t, f"Bullish divergence ({name})", "Divergence", 1, scale, self.c[t], None, b.px,
                                      a.idx, b.idx, 0.0, "regular", [a, b])
                        elif typ == -1 and b.px > a.px and xb < xa:
                            self.emit(t, f"Hidden bullish divergence ({name})", "Divergence", 1, scale, self.c[t], None,
                                      b.px, a.idx, b.idx, 0.0, "hidden", [a, b])
                        elif typ == 1 and b.px > a.px and xb < xa:
                            self.emit(t, f"Bearish divergence ({name})", "Divergence", -1, scale, self.c[t], None, b.px,
                                      a.idx, b.idx, 0.0, "regular", [a, b])
                        elif typ == 1 and b.px < a.px and xb > xa:
                            self.emit(t, f"Hidden bearish divergence ({name})", "Divergence", -1, scale, self.c[t],
                                      None, b.px, a.idx, b.idx, 0.0, "hidden", [a, b])

    def sr_breakouts(self):
        """Horizontal resistance / support tested 3+ times by minor & intermediate swing points."""
        for scale in ("minor", "intermediate"):
            if scale not in self.zz:
                continue
            piv = self.zz[scale][0]
            for d in (1, -1):
                pts = [p for p in piv if p.typ == d]
                for j in range(2, len(pts)):
                    last = pts[j]
                    tol = 0.5 * self.atr50[last.idx]
                    cl = [p for p in pts[max(0, j - 8): j + 1] if abs(p.px - last.px) <= tol and last.idx - p.idx <= 250]
                    if len(cl) < 3:
                        continue
                    lvl = max(p.px for p in cl) if d > 0 else min(p.px for p in cl)
                    t0 = last.conf
                    t1 = min(self.n - 1, last.idx + 120)
                    nxt = [p for p in pts[j + 1: j + 2]]
                    if nxt:
                        t1 = min(t1, nxt[0].conf)
                    tt = self.first_cross(self.c * d, lvl * d, 0.0, 1, max(t0, 1), t1)
                    if tt > 0:
                        nm = "Horizontal resistance breakout (3+ touches)" if d > 0 else \
                             "Horizontal support breakdown (3+ touches)"
                        self.emit(tt, nm, "Support / resistance", d, scale, lvl, None, None, cl[0].idx, last.idx,
                                  0.0, f"{len(cl)} touches", cl)

    def point_figure(self, box_pct=2.0, rev=3):
        """Point & figure (log boxes, high/low method, 3-box reversal) with the classic signals."""
        h, l, n = self.h, self.l, self.n
        lb = np.log(1 + box_pct / 100)
        px = lambda b: float(np.exp(b * lb))  # noqa: E731
        cols = []                       # dicts: dir, top, bot, t0, fired
        for t in range(n):
            hb, lbx = int(np.floor(np.log(h[t]) / lb)), int(np.ceil(np.log(l[t]) / lb))
            if not cols:
                cols.append({"dir": 1, "top": hb, "bot": lbx, "t0": t, "fired": False})
                continue
            cur = cols[-1]
            if cur["dir"] == 1:
                if hb > cur["top"]:
                    cur["top"] = hb
                elif cur["top"] - lbx >= rev:
                    cols.append({"dir": -1, "top": cur["top"] - 1, "bot": lbx, "t0": t, "fired": False})
            else:
                if lbx < cur["bot"]:
                    cur["bot"] = lbx
                elif hb - cur["bot"] >= rev:
                    cols.append({"dir": 1, "top": hb, "bot": cur["bot"] + 1, "t0": t, "fired": False})
            cur = cols[-1]
            if len(cols) < 5 or cur["fired"]:
                continue
            same = cols[-3::-2][:2]      # the two previous columns of the same direction
            opp = cols[-2::-2][:2]       # the two previous opposite columns
            tag = f"{box_pct:g}% x {rev}"
            if cur["dir"] == 1 and cur["top"] > same[0]["top"]:
                cur["fired"] = True
                if same[0]["top"] == same[1]["top"]:
                    nm = "P&F triple-top buy"
                elif same[0]["top"] > same[1]["top"] and opp[0]["bot"] > opp[1]["bot"]:
                    nm = "P&F bullish ascending breakout"
                else:
                    nm = "P&F double-top buy"
                self.emit(t, nm, "Point & figure", 1, "bar", px(same[0]["top"]), None, px(cur["bot"]), same[1]["t0"],
                          t, 0.0, tag)
            elif cur["dir"] == -1 and cur["bot"] < same[0]["bot"]:
                cur["fired"] = True
                if same[0]["bot"] == same[1]["bot"]:
                    nm = "P&F triple-bottom sell"
                elif same[0]["bot"] < same[1]["bot"] and opp[0]["top"] < opp[1]["top"]:
                    nm = "P&F bearish descending breakdown"
                else:
                    nm = "P&F double-bottom sell"
                self.emit(t, nm, "Point & figure", -1, "bar", px(same[0]["bot"]), None, px(cur["top"]), same[1]["t0"],
                          t, 0.0, tag)

    def run(self, osc: dict) -> pd.DataFrame:
        for scale in SCALES:
            self.scan_zigzag(scale)
        self.bar_patterns()
        self.fib_bounce()
        self.divergences(osc)
        self.sr_breakouts()
        self.point_figure()
        E = pd.DataFrame(self.events)
        if E.empty:
            return E
        E["date"] = self.df.index[E["t"].to_numpy()]
        E["close"] = self.c[E["t"].to_numpy()]
        E["atr"] = self.atr14[E["t"].to_numpy()]
        return E.sort_values(["t", "pattern"]).reset_index(drop=True)


def detect_patterns(df: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    extra = {"sma50": feats["_sma50"].to_numpy(float), "sma150": feats["_sma150"].to_numpy(float),
             "sma200": feats["_sma200"].to_numpy(float)}
    osc = {"RSI": feats["ta_RSI"].to_numpy(float), "MACD hist": feats["ta_MACD_macdhist"].to_numpy(float),
           "OBV": pd.Series(talib.OBV(df["close"].to_numpy(float), np.nan_to_num(df["volume"].to_numpy(float))))
           .to_numpy(float), "MFI": feats["ta_MFI"].to_numpy(float)}
    return PatternScanner(df, extra).run(osc)
