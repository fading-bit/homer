"""
perplexity.py — a language-model reproduction of the seam detector.

Pavlopoulos & Konstantinidou (2022) find outlier passages in Homer with a
character-level language model: train on the poem, measure perplexity over a
rolling window, and flag windows above the upper 95% confidence bound. This is an
independent reproduction of that idea, to cross-check our char-trigram *cosine*
seam detector (within_book.py) with a completely different statistic.

Two outputs:
  1. leave-one-book-out perplexity per book (which book is most 'surprising' to a
     model trained on the rest of its poem) — a content-bearing anomaly ranking;
  2. a rolling-window perplexity trace inside a chosen book, with a 95% threshold,
     to see whether the same passages spike (e.g. the Shield in Iliad 18).

Char-level modelling stays *within* a poem (same edition), so it is not disturbed
by the cross-poem edition artifact discussed in the report.

Run:  python src/perplexity.py --config config.yaml [--book iliad:18]
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

from viz import savefig, POEM_COLORS


class CharLM:
    """Linearly-interpolated character n-gram model (orders 0..N-1 + uniform floor)."""

    def __init__(self, N=4):
        self.N = N
        self.counts = [defaultdict(Counter) for _ in range(N)]
        self.vocab = set()
        self.lmbda = [0.02, 0.18, 0.30, 0.30, 0.20][:N + 1]  # uniform, then orders 0..N-1
        s = sum(self.lmbda)
        self.lmbda = [x / s for x in self.lmbda]

    def train(self, text):
        self.vocab |= set(text)
        t = "^" * (self.N - 1) + text
        for i in range(self.N - 1, len(t)):
            ch = t[i]
            for k in range(self.N):
                self.counts[k][t[i - k:i]][ch] += 1

    def _logp(self, text):
        V = len(self.vocab) or 1
        t = "^" * (self.N - 1) + text
        lp, n = 0.0, 0
        for i in range(self.N - 1, len(t)):
            ch = t[i]
            p = self.lmbda[0] / V
            for k in range(self.N):
                c = self.counts[k].get(t[i - k:i])
                if c:
                    tot = sum(c.values())
                    if tot:
                        p += self.lmbda[k + 1] * c.get(ch, 0) / tot
            lp += math.log(p); n += 1
        return lp, n

    def perplexity(self, text):
        lp, n = self._logp(text)
        return math.exp(-lp / n) if n else float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--book", default="iliad:18", help="poem:book for the rolling trace")
    ap.add_argument("--window", type=int, default=600)
    ap.add_argument("--step", type=int, default=100)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(args.config).resolve().parent
    figures = root / cfg["paths"]["figures_dir"]
    tables = root / cfg["paths"]["tables_dir"]
    df = pd.read_parquet(root / cfg["paths"]["processed_dir"] / "lines.parquet")

    def book_text(poem, b):
        return " ".join(df[(df.poem == poem) & (df.book_no == b)]
                        .sort_values("line_no").text_norm)

    # 1. leave-one-book-out perplexity per book
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, poem in zip(axes, ["iliad", "odyssey"]):
        books = sorted(df[df.poem == poem].book_no.unique())
        texts = {b: book_text(poem, b) for b in books}
        ppl = {}
        for b in books:
            lm = CharLM(N=args.order)
            for ob in books:
                if ob != b:
                    lm.train(texts[ob])
            ppl[b] = lm.perplexity(texts[b])
        vals = np.array([ppl[b] for b in books])
        thr = vals.mean() + vals.std()
        colors = [POEM_COLORS[poem] if v < thr else "crimson" for v in vals]
        ax.bar(range(len(books)), vals, color=colors)
        ax.axhline(thr, color="grey", ls="--", lw=1, label="mean + 1 SD")
        ax.set_xticks(range(len(books))); ax.set_xticklabels(books, fontsize=8)
        ax.set_xlabel(f"{poem.capitalize()} book")
        ax.set_ylabel("leave-one-out perplexity")
        ax.set_title(f"{poem.capitalize()}: how surprising is each book to a model "
                     f"trained on the rest?", fontsize=10)
        ax.legend(fontsize=8)
        for b in books:
            rows.append({"poem": poem, "book": b, "loo_perplexity": round(ppl[b], 2),
                         "outlier": bool(ppl[b] >= thr)})
    fig.suptitle("Language-model anomaly by book (character 4-gram, leave-one-book-out)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, figures, "perplexity_books")
    pd.DataFrame(rows).to_csv(tables / "perplexity_books.csv", index=False)

    # 2. rolling-window perplexity inside one book
    poem, b = args.book.split(":"); b = int(b)
    books = sorted(df[df.poem == poem].book_no.unique())
    lm = CharLM(N=args.order)
    train_windows_ppl = []
    for ob in books:
        t = book_text(poem, ob)
        if ob != b:
            lm.train(t)
    # threshold from windows of the OTHER books (what's normal for the poem)
    for ob in books:
        if ob == b:
            continue
        t = book_text(poem, ob)
        for i in range(0, max(1, len(t) - args.window), args.step):
            train_windows_ppl.append(lm.perplexity(t[i:i + args.window]))
    thr = float(np.percentile(train_windows_ppl, 95))

    target = book_text(poem, b)
    xs, ys = [], []
    for i in range(0, max(1, len(target) - args.window), args.step):
        xs.append(i); ys.append(lm.perplexity(target[i:i + args.window]))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(xs, ys, color=POEM_COLORS[poem], lw=1.5)
    ax.axhline(thr, color="crimson", ls="--", lw=1, label="95th-percentile of the rest of the poem")
    ax.fill_between(xs, thr, ys, where=[y > thr for y in ys], color="crimson", alpha=0.2)
    ax.set_xlabel(f"character offset within {poem.capitalize()} {b}")
    ax.set_ylabel("rolling-window perplexity")
    ax.set_title(f"Rolling LM perplexity in {poem.capitalize()} {b} "
                 f"(window {args.window} chars) — spikes = passages unusual for the poem",
                 fontsize=11)
    ax.legend(fontsize=9)
    savefig(fig, figures, f"perplexity_rolling_{poem}{b}")

    print("=== LANGUAGE-MODEL ANOMALY BY BOOK (leave-one-out perplexity) ===")
    out = pd.DataFrame(rows)
    for poem in ["iliad", "odyssey"]:
        top = out[out.poem == poem].sort_values("loo_perplexity", ascending=False).head(4)
        print(f"\n{poem.capitalize()} — most surprising books:")
        print(top.to_string(index=False))
    print(f"\nWrote perplexity_books.(csv/png) and perplexity_rolling_{args.book.replace(':','')}.png")


if __name__ == "__main__":
    main()
