"""
seam_scan.py — a systematic map of *where* the internal variation lives.

Runs the within-book seam detector over every book of both poems, twice: on the
full text, and on **narration only** (speech removed). A book that stays
heterogeneous with speech stripped out cannot owe its divergence merely to the
narrator/character alternation — it is a stronger candidate for a compositional
seam. This is the register-controlled companion to `within_book.py --all`, and it
skips the 48 per-book figures for speed.

Run:  python src/seam_scan.py --config config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

from within_book import (get_book, line_windows, char_ngram_space, window_matrix,
                         dist_to_centroid, change_points)
from viz import savefig, POEM_COLORS
from features_word import book_label


def scan_book(df, poem, b, window, step, ngram, topk, pen):
    sub = get_book(df, poem, b)
    if len(sub) < window + step:
        return None
    ng = char_ngram_space(sub, ngram, topk)
    windows = list(line_windows(sub, window, step))
    M, centers, spans = window_matrix(windows, "char", ngram, ng)
    if len(M) < 3:
        return None
    curve = dist_to_centroid(M)
    cps = [centers[i] for i in change_points(M, pen) if i < len(centers)]
    return {"n_lines": len(sub), "n_windows": len(M),
            "max_dist": float(curve.max()), "n_cp": len(cps),
            "cp_lines": ";".join(map(str, cps))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--ngram", type=int, default=3)
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--pen", type=float, default=5.0)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(args.config).resolve().parent
    figures = root / cfg["paths"]["figures_dir"]
    tables = root / cfg["paths"]["tables_dir"]

    df_all = pd.read_parquet(root / cfg["paths"]["processed_dir"] / "lines.parquet")
    vpath = root / cfg["paths"]["processed_dir"] / "lines_voice.parquet"
    if not vpath.exists():
        raise SystemExit("lines_voice.parquet not found — run src/speech_split.py first")
    df_narr = pd.read_parquet(vpath)
    df_narr = df_narr[df_narr.voice == "N"].reset_index(drop=True)

    rows = []
    for poem in ["iliad", "odyssey"]:
        for b in sorted(df_all[df_all.poem == poem].book_no.unique()):
            a = scan_book(df_all, poem, int(b), args.window, args.step, args.ngram, args.topk, args.pen)
            n = scan_book(df_narr, poem, int(b), args.window, args.step, args.ngram, args.topk, args.pen)
            rows.append({
                "poem": poem, "book": book_label(poem, int(b)), "book_no": int(b),
                "max_dist_all": a["max_dist"] if a else np.nan,
                "max_dist_narr": n["max_dist"] if n else np.nan,
                "n_cp_narr": n["n_cp"] if n else np.nan,
                "cp_lines_narr": n["cp_lines"] if n else "",
            })
    summ = pd.DataFrame(rows)
    # flag books still heterogeneous in narration (above the narration mean + 1 SD)
    thr = summ.max_dist_narr.mean() + summ.max_dist_narr.std()
    summ["narration_outlier"] = summ.max_dist_narr >= thr
    summ.to_csv(tables / "seam_scan.csv", index=False)

    # overview: all-text vs narration max divergence, per book
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax, poem in zip(axes, ["iliad", "odyssey"]):
        sp = summ[summ.poem == poem]
        x = np.arange(len(sp))
        ax.bar(x - 0.2, sp.max_dist_all, width=0.4, color="#bbbbbb", label="all text")
        ax.bar(x + 0.2, sp.max_dist_narr, width=0.4, color=POEM_COLORS[poem], label="narration only")
        ax.axhline(thr, color="crimson", ls="--", lw=1, label="narration mean + 1 SD")
        ax.set_xticks(x); ax.set_xticklabels(sp.book_no, fontsize=8)
        ax.set_xlabel(f"{poem.capitalize()} book"); ax.set_title(poem.capitalize(), fontsize=11)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("max window distance to book centroid")
    fig.suptitle("Where does the internal variation live? All-text vs narration-only, by book",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, figures, "seam_scan")

    print("=== BOOKS STILL HETEROGENEOUS IN NARRATION (speech removed) ===\n")
    out = summ[summ.narration_outlier].sort_values("max_dist_narr", ascending=False)
    print(out[["book", "n_cp_narr", "max_dist_narr", "cp_lines_narr"]].to_string(index=False))
    print(f"\nFull table: {tables/'seam_scan.csv'} ; figure: {figures/'seam_scan.png'}")


if __name__ == "__main__":
    main()
