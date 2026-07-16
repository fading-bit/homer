"""
metre_localize.py — a content-free map of metrical anomaly.

The calibrated test says the poems are metrically over-varied *globally*. This
asks *where*: for each book, how far does its dactyl/spondee foot-pattern profile
sit from its own poem's centroid? Because it uses only metre, the ranking cannot
be driven by subject matter. Books that stand out are candidates for a different
metrical hand (or a metrically distinctive genre-moment: catalogue, ekphrasis).

Run:  python src/metre_localize.py --config config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

from metre import scan_lines, pattern_index
from viz import savefig, POEM_COLORS


def book_profile(lines):
    pats, _ = scan_lines(lines)
    v = np.zeros(16)
    for p in pats:
        if p is not None:
            v[pattern_index(p)] += 1
    return v / v.sum() if v.sum() else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(args.config).resolve().parent
    figures = root / cfg["paths"]["figures_dir"]
    tables = root / cfg["paths"]["tables_dir"]
    df = pd.read_parquet(root / cfg["paths"]["processed_dir"] / "lines.parquet")

    mats = {}
    for poem in ["iliad", "odyssey"]:
        sub = df[df.poem == poem]
        books = sorted(sub.book_no.unique())
        M = np.vstack([book_profile(list(sub[sub.book_no == b]
                       .sort_values("line_no").text_norm)) for b in books])
        mats[poem] = (books, M)

    # standardise foot-pattern features across all 48 books of both poems
    allmat = np.vstack([mats["iliad"][1], mats["odyssey"][1]])
    mu, sd = allmat.mean(0), allmat.std(0) + 1e-9

    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, poem in zip(axes, ["iliad", "odyssey"]):
        books, M = mats[poem]
        Z = (M - mu) / sd
        cen = Z.mean(0)
        dist = np.linalg.norm(Z - cen, axis=1)
        thr = dist.mean() + dist.std()
        colors = [POEM_COLORS[poem] if d < thr else "crimson" for d in dist]
        ax.bar(range(len(books)), dist, color=colors)
        ax.axhline(thr, color="grey", ls="--", lw=1, label="mean + 1 SD")
        ax.set_xticks(range(len(books)))
        ax.set_xticklabels(books, fontsize=8)
        ax.set_xlabel(f"{poem.capitalize()} book"); ax.set_ylabel("metrical distance from poem centroid")
        ax.set_title(f"{poem.capitalize()}: metrical anomaly by book", fontsize=11)
        ax.legend(fontsize=8)
        for b, d in zip(books, dist):
            rows.append({"poem": poem, "book": b, "metrical_distance": round(float(d), 3),
                         "outlier": bool(d >= thr)})
    fig.suptitle("Content-free metrical outlier map: which books' dactyl/spondee "
                 "profile is unusual for their poem", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    savefig(fig, figures, "metre_localize")

    out = pd.DataFrame(rows)
    out.to_csv(tables / "metre_localize.csv", index=False)
    print("=== METRICAL ANOMALY BY BOOK (content-free) ===")
    for poem in ["iliad", "odyssey"]:
        top = out[out.poem == poem].sort_values("metrical_distance", ascending=False).head(4)
        print(f"\n{poem.capitalize()} — most metrically distinctive books:")
        print(top.to_string(index=False))
    print(f"\nWrote {tables/'metre_localize.csv'} and {figures/'metre_localize.png'}")


if __name__ == "__main__":
    main()
