"""
textstats.py — shared, dependency-light text statistics.

Pure functions used by the Phase 1 (describe) and Phase 2 (features_word)
scripts. Everything operates on the whitespace-tokenised `text_norm` column
(lowercased, accent-stripped, punctuation/elision removed, ς→σ folded).
"""

from __future__ import annotations

import math
from collections import Counter

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Tokenisation & n-grams
# ─────────────────────────────────────────────────────────────────────────────

def tokens(text: str) -> list[str]:
    """Whitespace tokens. `text_norm` is already cleaned, so split is enough."""
    return text.split()


def book_tokens(df: pd.DataFrame, col: str = "text_norm") -> list[str]:
    """All tokens of a book (or any line subset), in order."""
    out: list[str] = []
    for line in df[col]:
        out.extend(line.split())
    return out


def ngrams(seq: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]


# ─────────────────────────────────────────────────────────────────────────────
# Lexical richness (length-robust where noted)
# ─────────────────────────────────────────────────────────────────────────────

def ttr(toks: list[str]) -> float:
    """Raw type-token ratio. Length-DEPENDENT — reported only for reference."""
    return len(set(toks)) / len(toks) if toks else 0.0


def mattr(toks: list[str], window: int = 100) -> float:
    """
    Moving-Average Type-Token Ratio (Covington & McFall).
    Length-robust: average TTR over all sliding windows of fixed size.
    Falls back to plain TTR when the text is shorter than one window.
    """
    n = len(toks)
    if n == 0:
        return 0.0
    if n < window:
        return ttr(toks)
    ratios = []
    for i in range(n - window + 1):
        w = toks[i:i + window]
        ratios.append(len(set(w)) / window)
    return sum(ratios) / len(ratios)


def yule_k(toks: list[str]) -> float:
    """
    Yule's Characteristic K (×10^4). Length-robust measure of repetition:
    higher K → vocabulary reused more (less rich). K = 10^4 (Σ f² − N) / N².
    """
    n = len(toks)
    if n == 0:
        return 0.0
    freqs = Counter(toks)
    m2 = sum(f * f for f in freqs.values())
    return 1e4 * (m2 - n) / (n * n)


def hapax_stats(toks: list[str]) -> dict:
    """Hapax (once) and dis (twice) legomena, as rates per token and per type."""
    n = len(toks)
    freqs = Counter(toks)
    v = len(freqs)
    v1 = sum(1 for f in freqs.values() if f == 1)
    v2 = sum(1 for f in freqs.values() if f == 2)
    return {
        "types": v,
        "hapax_per_token": v1 / n if n else 0.0,
        "hapax_frac_types": v1 / v if v else 0.0,
        "dis_per_token": v2 / n if n else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Keyness (log-likelihood G²) — target subset vs. the rest of the corpus
# ─────────────────────────────────────────────────────────────────────────────

def log_likelihood_keyness(
    target_counts: Counter,
    rest_counts: Counter,
    min_count: int = 3,
    top_n: int = 40,
) -> pd.DataFrame:
    """
    Dunning's G² keyness for words OVER-represented in `target` relative to
    `rest`. Returns the top_n over-used words with G², both relative
    frequencies, and the log-ratio effect size.
    """
    c = sum(target_counts.values())   # target size
    d = sum(rest_counts.values())     # rest size
    rows = []
    for w, a in target_counts.items():
        if a < min_count:
            continue
        b = rest_counts.get(w, 0)
        e1 = c * (a + b) / (c + d)
        e2 = d * (a + b) / (c + d)
        ll = 0.0
        if a > 0:
            ll += a * math.log(a / e1)
        if b > 0:
            ll += b * math.log(b / e2)
        ll *= 2.0
        f_target = a / c
        f_rest = b / d if d else 0.0
        overused = f_target > f_rest
        if not overused:
            continue
        # +1 smoothing for a stable log-ratio effect size
        log_ratio = math.log2(((a + 0.5) / c) / ((b + 0.5) / d))
        rows.append({
            "word": w,
            "g2": ll,
            "count_target": a,
            "count_rest": b,
            "freq_target_pm": f_target * 1000,
            "freq_rest_pm": f_rest * 1000,
            "log_ratio": log_ratio,
        })
    out = pd.DataFrame(rows).sort_values("g2", ascending=False).head(top_n)
    return out.reset_index(drop=True)
