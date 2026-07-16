"""
within_book.py — Phase 2b: stylometry INSIDE a single book.

Where the cross-book phase asks "are the 48 books one voice or many?", this asks
"does the voice shift WITHIN a book, and where?" — the local seam detector for
interpolations and register changes.

For one book it produces:
  * a rolling stylistic-distance curve (each window vs the book's own centroid),
  * a window×window self-distance heatmap (block structure = a seam),
  * change points detected with `ruptures`, mapped back to line numbers.
Across all books it produces an internal-heterogeneity ranking.

Within a book, content and register
(narration vs. speech) churn constantly, so the rolling signal partly tracks
those rather than authorship. Features are therefore content-light (character
n-grams by default, or function words); genuine authorial seams need the later
narrator/speech split to disentangle.

Run one book:   python src/within_book.py --poem iliad   --book 2
Run all books:  python src/within_book.py --all
Options: --window 60 --step 10 --ngram 3 --topk 200 --pen 12 --channel char|func
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import ruptures as rpt

from textstats import ngrams
from features_word import PARTICLES, book_label, POEM_ORDER
from viz import savefig, POEM_COLORS

# Traditionally athetised / much-discussed spans, for optional shading on plots.
# (Approximate line ranges; illustrative overlays, not claims.)
SUSPECT_SPANS = {
    ("iliad", 2): [(484, 785, "Catalogue of Ships")],
    ("iliad", 10): [(1, 579, "Doloneia (whole book suspect)")],
    ("iliad", 18): [(478, 608, "Shield of Achilles")],
    ("odyssey", 11): [(225, 332, "Catalogue of heroines"),
                      (333, 384, "Intermezzo")],
    ("odyssey", 24): [(1, 204, "Second Nekyia")],
}


# ─────────────────────────────────────────────────────────────────────────────
# Windowing & features
# ─────────────────────────────────────────────────────────────────────────────

def get_book(df: pd.DataFrame, poem: str, book_no: int) -> pd.DataFrame:
    sub = df[(df.poem == poem) & (df.book_no == book_no)].sort_values("line_no")
    if sub.empty:
        raise ValueError(f"No lines for {poem} book {book_no}")
    return sub.reset_index(drop=True)


def line_windows(sub: pd.DataFrame, window: int, step: int):
    """Yield (center_line, start_line, end_line, window_df) sliding over lines."""
    n = len(sub)
    if n <= window:
        yield (int(sub.line_no.iloc[n // 2]),
               int(sub.line_no.iloc[0]), int(sub.line_no.iloc[-1]), sub)
        return
    for a in range(0, n - window + 1, step):
        w = sub.iloc[a:a + window]
        yield (int(w.line_no.iloc[len(w) // 2]),
               int(w.line_no.iloc[0]), int(w.line_no.iloc[-1]), w)


def char_ngram_space(sub: pd.DataFrame, n: int, top_k: int) -> list[str]:
    """Top-K character n-grams over the whole book (shared feature space)."""
    text = " ".join(sub.text_norm)
    grams = [text[i:i + n] for i in range(len(text) - n + 1)]
    return [g for g, _ in Counter(grams).most_common(top_k)]


def window_matrix(windows, channel: str, ngram: int, ng_space: list[str]):
    """
    Build a (W × F) relative-frequency matrix over windows.
      channel='char' → character n-gram profile (robust for short windows)
      channel='func' → function-word / particle profile (interpretable)
    Returns matrix, centers (line numbers), spans (start,end).
    """
    rows, centers, spans = [], [], []
    for center, s, e, w in windows:
        if channel == "char":
            text = " ".join(w.text_norm)
            grams = Counter(text[i:i + ngram] for i in range(len(text) - ngram + 1))
            total = sum(grams.values()) or 1
            rows.append([grams.get(g, 0) / total for g in ng_space])
        else:  # func
            toks = [t for line in w.text_norm for t in line.split()]
            toks = " ".join(w.text_norm).split()
            total = len(toks) or 1
            counts = Counter(toks)
            rows.append([counts.get(p, 0) / total for p in PARTICLES])
        centers.append(center)
        spans.append((s, e))
    return np.asarray(rows), centers, spans


# ─────────────────────────────────────────────────────────────────────────────
# Distances & change points
# ─────────────────────────────────────────────────────────────────────────────

def _cos_dist(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return 1.0 - float(a @ b) / (na * nb)


def dist_to_centroid(M: np.ndarray) -> np.ndarray:
    c = M.mean(axis=0)
    return np.array([_cos_dist(row, c) for row in M])


def dist_adjacent(M: np.ndarray) -> np.ndarray:
    """Cosine distance between consecutive windows (a 'novelty' signal)."""
    d = np.zeros(len(M))
    for i in range(1, len(M)):
        d[i] = _cos_dist(M[i], M[i - 1])
    return d


def self_distance(M: np.ndarray) -> np.ndarray:
    W = len(M)
    D = np.zeros((W, W))
    for i in range(W):
        for j in range(i + 1, W):
            D[i, j] = D[j, i] = _cos_dist(M[i], M[j])
    return D


def change_points(M: np.ndarray, pen: float) -> list[int]:
    """PELT change-point detection (rbf cost) on z-scored window features."""
    if len(M) < 4:
        return []
    Z = M.copy()
    std = Z.std(axis=0)
    std[std == 0] = 1.0
    Z = (Z - Z.mean(axis=0)) / std
    algo = rpt.Pelt(model="rbf", min_size=2).fit(Z)
    bkps = algo.predict(pen=pen)
    return [b for b in bkps if b < len(M)]  # drop the trailing end index


# ─────────────────────────────────────────────────────────────────────────────
# Plot for a single book
# ─────────────────────────────────────────────────────────────────────────────

def plot_book(poem, book_no, centers, spans, curve, adj, cps_idx,
              D, channel, figures_dir, voice="all"):
    label = book_label(poem, book_no)
    color = POEM_COLORS[poem]
    cp_lines = [centers[i] for i in cps_idx if i < len(centers)]
    vtag = "" if voice == "all" else f" [{voice} only]"

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [1, 1.4]})

    # (a) rolling distance-to-centroid + novelty + change points
    ax1.plot(centers, curve, color=color, lw=1.8, label="distance to book centroid")
    ax1.plot(centers, adj, color="grey", lw=1.0, alpha=0.7,
             label="adjacent-window novelty")
    for sp in SUSPECT_SPANS.get((poem, book_no), []):
        ax1.axvspan(sp[0], sp[1], color="gold", alpha=0.20)
        ax1.text((sp[0] + sp[1]) / 2, ax1.get_ylim()[1] * 0.95, sp[2],
                 ha="center", va="top", fontsize=8, color="#8a6d00")
    for ln in cp_lines:
        ax1.axvline(ln, color="crimson", ls="--", lw=1.1)
    ax1.set_xlabel("line number")
    ax1.set_ylabel("cosine distance")
    ax1.set_title(f"{label} — rolling stylistic distance within the book "
                  f"({channel} channel).  Dashed red = detected change points; "
                  f"gold = traditionally discussed spans.", fontsize=10)
    ax1.legend(fontsize=8, loc="upper right")

    # (b) window×window self-distance heatmap
    im = ax2.imshow(D, cmap="magma", aspect="auto",
                    extent=[centers[0], centers[-1], centers[-1], centers[0]])
    for ln in cp_lines:
        ax2.axvline(ln, color="cyan", ls="--", lw=0.9)
        ax2.axhline(ln, color="cyan", ls="--", lw=0.9)
    ax2.set_xlabel("line number")
    ax2.set_ylabel("line number")
    ax2.set_title("window × window self-distance (block structure = a seam)",
                  fontsize=10)
    fig.colorbar(im, ax=ax2, shrink=0.7, label="cosine distance")

    fig.suptitle(f"Within-book stylometry: {poem.title()} {book_no}{vtag}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    suffix = "" if voice == "all" else f"_{voice}"
    return savefig(fig, figures_dir, f"within_{label}{suffix}")


# ─────────────────────────────────────────────────────────────────────────────
# Drivers
# ─────────────────────────────────────────────────────────────────────────────

def analyse_book(df, poem, book_no, cfg_win, figures_dir):
    window, step, ngram, topk, pen, channel, voice = cfg_win
    sub = get_book(df, poem, book_no)
    ng_space = char_ngram_space(sub, ngram, topk) if channel == "char" else []
    windows = list(line_windows(sub, window, step))
    M, centers, spans = window_matrix(windows, channel, ngram, ng_space)
    curve = dist_to_centroid(M)
    adj = dist_adjacent(M)
    cps = change_points(M, pen)
    D = self_distance(M)
    fig_path = plot_book(poem, book_no, centers, spans, curve, adj, cps, D,
                         channel, figures_dir, voice=voice)
    cp_lines = [centers[i] for i in cps if i < len(centers)]
    summary = {
        "book": book_label(poem, book_no), "poem": poem, "book_no": book_no,
        "voice": voice, "n_lines": len(sub), "n_windows": len(M),
        "n_changepoints": len(cp_lines),
        "changepoint_lines": ";".join(map(str, cp_lines)),
        "max_centroid_dist": float(curve.max()),
        "mean_adjacent_dist": float(adj[1:].mean()) if len(adj) > 1 else 0.0,
    }
    return summary, fig_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--poem", choices=POEM_ORDER)
    ap.add_argument("--book", type=int)
    ap.add_argument("--all", action="store_true", help="run every book, rank heterogeneity")
    ap.add_argument("--window", type=int, default=60, help="window size in lines")
    ap.add_argument("--step", type=int, default=10, help="step in lines")
    ap.add_argument("--ngram", type=int, default=3, help="character n-gram order")
    ap.add_argument("--topk", type=int, default=200, help="char n-gram vocab size")
    ap.add_argument("--pen", type=float, default=5.0, help="PELT penalty (higher = fewer change points; try 2-3 for finer sensitivity)")
    ap.add_argument("--channel", choices=["char", "func"], default="char")
    ap.add_argument("--voice", choices=["all", "N", "S", "narration", "speech"],
                    default="all",
                    help="restrict to narration (N) or speech (S) via lines_voice.parquet")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(args.config).resolve().parent
    figures = root / cfg["paths"]["figures_dir"]
    tables = root / cfg["paths"]["tables_dir"]
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    voice = {"narration": "N", "speech": "S"}.get(args.voice, args.voice)
    if voice == "all":
        df = pd.read_parquet(root / cfg["paths"]["processed_dir"] / "lines.parquet")
    else:
        vpath = root / cfg["paths"]["processed_dir"] / "lines_voice.parquet"
        if not vpath.exists():
            ap.error("lines_voice.parquet not found — run src/speech_split.py first")
        df = pd.read_parquet(vpath)
        df = df[df.voice == voice].reset_index(drop=True)
    vlabel = {"N": "narration", "S": "speech", "all": "all"}[voice]
    cfg_win = (args.window, args.step, args.ngram, args.topk, args.pen,
               args.channel, vlabel)

    if args.all:
        rows = []
        for poem in POEM_ORDER:
            for bno in sorted(df[df.poem == poem].book_no.unique()):
                s, _ = analyse_book(df, poem, int(bno), cfg_win, figures)
                rows.append(s)
        summ = pd.DataFrame(rows)
        summ["heterogeneity_z"] = (
            (summ.max_centroid_dist - summ.max_centroid_dist.mean())
            / summ.max_centroid_dist.std(ddof=0))
        summ = summ.sort_values("max_centroid_dist", ascending=False)
        summ.to_csv(tables / "within_book_heterogeneity.csv", index=False)

        # ranking bar chart
        fig, ax = plt.subplots(figsize=(13, 5))
        colors = [POEM_COLORS[p] for p in summ.poem]
        ax.bar(range(len(summ)), summ.max_centroid_dist, color=colors)
        ax.set_xticks(range(len(summ)))
        ax.set_xticklabels(summ.book, rotation=90, fontsize=7)
        ax.set_ylabel("max distance of any window to book centroid")
        ax.set_title("Internal heterogeneity by book "
                     "(higher = contains a more stylistically divergent stretch)")
        ax.legend(handles=[
            plt.Rectangle((0, 0), 1, 1, color=POEM_COLORS["iliad"], label="Iliad"),
            plt.Rectangle((0, 0), 1, 1, color=POEM_COLORS["odyssey"], label="Odyssey"),
        ], fontsize=9)
        savefig(fig, figures, "within_book_heterogeneity")

        print("=== WITHIN-BOOK HETEROGENEITY (top 8 with most divergent internal stretch) ===\n")
        print(summ[["book", "n_lines", "n_windows", "n_changepoints",
                    "max_centroid_dist", "mean_adjacent_dist"]].head(8).to_string(index=False))
        print(f"\nWrote per-book figures + {tables/'within_book_heterogeneity.csv'}")
    else:
        if not (args.poem and args.book):
            ap.error("give --poem and --book, or use --all")
        s, fig_path = analyse_book(df, args.poem, args.book, cfg_win, figures)
        pd.DataFrame([s]).to_csv(tables / "within_book_changepoints.csv", index=False)
        print(f"=== {s['book']} — within-book analysis ({args.channel} channel) ===\n")
        print(f"lines: {s['n_lines']}   windows: {s['n_windows']}")
        print(f"change points at lines: {s['changepoint_lines'] or '(none)'}")
        print(f"max distance to centroid: {s['max_centroid_dist']:.3f}")
        print(f"mean adjacent-window distance: {s['mean_adjacent_dist']:.3f}")
        print(f"\nFigure → {fig_path}")


if __name__ == "__main__":
    main()
