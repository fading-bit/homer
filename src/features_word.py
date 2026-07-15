"""
features_word.py — Phase 2: word-level features.

Produces the word-level "voice" signal, per book:
  1. Particle / function-word frequency profile  (the primary habit signal)
  2. Lexical richness  (Yule's K, MATTR, hapax rate)
  3. Formularity  (repeated-line rate, repeated n-gram density)
  4. A most-frequent-word (MFW) count matrix for the later distance phase

Run:  python src/features_word.py --config config.yaml
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

from textstats import book_tokens, ngrams, mattr, yule_k, hapax_stats, ttr
from viz import savefig, POEM_COLORS

# Homeric particles / connectives / modal & negation words, in NORMALISED form
# (lowercase, accent-stripped, ς→σ). A few are ambiguous after normalisation
# (η, ως, ου, που collapse with articles/relatives/interrogatives); they are
# included but flagged in the saved table's `ambiguous` column.
PARTICLES = [
    "δε", "μεν", "και", "τε", "γαρ", "αλλα", "δη", "αρα", "ρα", "αυταρ",
    "αταρ", "ουν", "τοι", "γε", "περ", "κε", "κεν", "αν", "νυ", "ουδε",
    "μηδε", "ουτε", "μητε", "ητοι", "ως", "ου", "ουκ", "ουχ", "μη", "η",
]
AMBIGUOUS = {"η", "ως", "ου", "που"}

POEM_ORDER = ["iliad", "odyssey"]
SUSPECT = {("iliad", 10), ("odyssey", 23), ("odyssey", 24)}  # oft-debated books


def book_label(poem: str, book_no: int) -> str:
    return f"{'IL' if poem == 'iliad' else 'OD'}{book_no:02d}"


def iter_books(df: pd.DataFrame):
    """Yield (label, poem, book_no, sub_df) in fixed Iliad-then-Odyssey order."""
    for poem in POEM_ORDER:
        sub = df[df.poem == poem]
        for book_no in sorted(sub.book_no.unique()):
            g = sub[sub.book_no == book_no]
            yield book_label(poem, book_no), poem, int(book_no), g


# ─────────────────────────────────────────────────────────────────────────────
# 1. Particle profile
# ─────────────────────────────────────────────────────────────────────────────

def particle_matrix(df: pd.DataFrame):
    labels, poems, rows = [], [], []
    for label, poem, _bno, g in iter_books(df):
        toks = book_tokens(g)
        n = len(toks)
        counts = Counter(toks)
        rows.append([1000 * counts.get(p, 0) / n for p in PARTICLES])  # per-mille
        labels.append(label)
        poems.append(poem)
    freq = pd.DataFrame(rows, index=labels, columns=PARTICLES)
    freq.insert(0, "poem", poems)
    # z-score each particle across the 48 books
    z = freq[PARTICLES].apply(lambda c: (c - c.mean()) / (c.std(ddof=0) or 1.0))
    return freq, z


def plot_particle_heatmap(z: pd.DataFrame, poems: list[str], figures_dir):
    fig, ax = plt.subplots(figsize=(12, 13))
    im = ax.imshow(z.values, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    ax.set_xticks(range(len(z.columns)))
    ax.set_xticklabels(z.columns, rotation=90, fontsize=9)
    ax.set_yticks(range(len(z.index)))
    ax.set_yticklabels(z.index, fontsize=7)
    # Line between Iliad (first 24) and Odyssey.
    split = poems.index("odyssey") - 0.5
    ax.axhline(split, color="black", lw=1.5)
    ax.text(len(z.columns) - 0.5, split / 2, "ILIAD", rotation=90,
            va="center", ha="left", fontsize=10, color=POEM_COLORS["iliad"])
    ax.text(len(z.columns) - 0.5, (split + len(z.index)) / 2, "ODYSSEY",
            rotation=90, va="center", ha="left", fontsize=10,
            color=POEM_COLORS["odyssey"])
    fig.colorbar(im, ax=ax, shrink=0.5, label="z-score (per-book, per-particle)")
    ax.set_title("Particle / function-word profile by book (per-mille, z-scored)")
    return savefig(fig, figures_dir, "particle_heatmap")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Lexical richness
# ─────────────────────────────────────────────────────────────────────────────

def richness_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, poem, bno, g in iter_books(df):
        toks = book_tokens(g)
        h = hapax_stats(toks)
        rows.append({
            "book": label, "poem": poem, "book_no": bno,
            "n_tokens": len(toks), "n_types": h["types"],
            "ttr_raw": ttr(toks), "mattr_w100": mattr(toks, 100),
            "yule_k": yule_k(toks),
            "hapax_per_token": h["hapax_per_token"],
            "hapax_frac_types": h["hapax_frac_types"],
        })
    return pd.DataFrame(rows)


def plot_richness(rt: pd.DataFrame, figures_dir):
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    x = range(len(rt))
    colors = [POEM_COLORS[p] for p in rt.poem]
    for ax, col, title in [
        (axes[0], "yule_k", "Yule's K  (higher = more repetition / less rich)"),
        (axes[1], "mattr_w100", "MATTR window=100  (higher = more varied)"),
    ]:
        ax.bar(x, rt[col], color=colors)
        ax.set_title(title, fontsize=11)
        for i, (_, r) in enumerate(rt.iterrows()):
            if (r.poem, r.book_no) in SUSPECT:
                ax.annotate("*", (i, r[col]), ha="center", va="bottom",
                            fontsize=14, color="crimson")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(rt.book, rotation=90, fontsize=7)
    axes[0].legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color=POEM_COLORS["iliad"], label="Iliad"),
        plt.Rectangle((0, 0), 1, 1, color=POEM_COLORS["odyssey"], label="Odyssey"),
    ], loc="upper right", fontsize=9)
    fig.suptitle("Lexical richness by book  (* = often-debated books)", fontsize=13)
    return savefig(fig, figures_dir, "lexical_richness")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Formularity
# ─────────────────────────────────────────────────────────────────────────────

def formularity_table(df: pd.DataFrame, ns=(3, 4, 5)) -> pd.DataFrame:
    # Corpus-wide repetition indices.
    line_counts = Counter(df.text_norm)
    ngram_counts = {n: Counter() for n in ns}
    for _label, _poem, _bno, g in iter_books(df):
        toks = book_tokens(g)
        for n in ns:
            ngram_counts[n].update(ngrams(toks, n))

    rows = []
    for label, poem, bno, g in iter_books(df):
        lines = list(g.text_norm)
        # A line is "repeated" if its normalised form occurs >=2x in the corpus.
        rep_lines = sum(1 for ln in lines if line_counts[ln] >= 2)
        row = {
            "book": label, "poem": poem, "book_no": bno,
            "n_lines": len(lines),
            "repeated_line_rate": rep_lines / len(lines) if lines else 0.0,
        }
        toks = book_tokens(g)
        for n in ns:
            grams = ngrams(toks, n)
            if grams:
                repeated = sum(1 for gr in grams if ngram_counts[n][gr] >= 2)
                row[f"rep_{n}gram_rate"] = repeated / len(grams)
            else:
                row[f"rep_{n}gram_rate"] = 0.0
        rows.append(row)
    ft = pd.DataFrame(rows)
    # A single composite "formularity index": mean of the repeated-line and
    # repeated-4gram rates (both z-scored so they contribute equally).
    z_line = (ft.repeated_line_rate - ft.repeated_line_rate.mean()) / ft.repeated_line_rate.std(ddof=0)
    z_4g = (ft.rep_4gram_rate - ft.rep_4gram_rate.mean()) / ft.rep_4gram_rate.std(ddof=0)
    ft["formularity_index_z"] = (z_line + z_4g) / 2
    return ft


def plot_formularity(ft: pd.DataFrame, figures_dir):
    fig, ax = plt.subplots(figsize=(13, 5))
    x = range(len(ft))
    colors = [POEM_COLORS[p] for p in ft.poem]
    ax.bar(x, ft.repeated_line_rate, color=colors)
    for i, (_, r) in enumerate(ft.iterrows()):
        if (r.poem, r.book_no) in SUSPECT:
            ax.annotate("*", (i, r.repeated_line_rate), ha="center",
                        va="bottom", fontsize=14, color="crimson")
    ax.set_xticks(list(x))
    ax.set_xticklabels(ft.book, rotation=90, fontsize=7)
    ax.set_ylabel("share of lines that recur ≥2× in corpus")
    ax.set_title("Formularity: repeated whole-line rate by book  "
                 "(* = often-debated books)")
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color=POEM_COLORS["iliad"], label="Iliad"),
        plt.Rectangle((0, 0), 1, 1, color=POEM_COLORS["odyssey"], label="Odyssey"),
    ], loc="upper right", fontsize=9)
    return savefig(fig, figures_dir, "formularity")


# ─────────────────────────────────────────────────────────────────────────────
# 4. MFW count matrix (for the later distance phase)
# ─────────────────────────────────────────────────────────────────────────────

def mfw_matrix(df: pd.DataFrame, top_n: int = 200):
    corpus = Counter()
    for _l, _p, _b, g in iter_books(df):
        corpus.update(book_tokens(g))
    vocab = [w for w, _ in corpus.most_common(top_n)]
    labels, rows = [], []
    for label, _p, _b, g in iter_books(df):
        toks = book_tokens(g)
        n = len(toks)
        counts = Counter(toks)
        rows.append([1000 * counts.get(w, 0) / n for w in vocab])
        labels.append(label)
    return pd.DataFrame(rows, index=labels, columns=vocab)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(args.config).resolve().parent
    tables = root / cfg["paths"]["tables_dir"]
    figures = root / cfg["paths"]["figures_dir"]
    matrices = root / cfg["paths"]["matrices_dir"]
    for d in (tables, figures, matrices):
        d.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(root / cfg["paths"]["processed_dir"] / "lines.parquet")

    # 1. Particles
    freq, z = particle_matrix(df)
    freq.to_csv(tables / "particles_freq_permille.csv")
    z.to_csv(tables / "particles_zscore.csv")
    np.savez(matrices / "particles_z.npz",
             matrix=z.values, labels=np.array(z.index), features=np.array(z.columns))
    plot_particle_heatmap(z, list(freq.poem), figures)

    # 2. Richness
    rt = richness_table(df)
    rt.to_csv(tables / "lexical_richness.csv", index=False)
    plot_richness(rt, figures)

    # 3. Formularity
    ft = formularity_table(df)
    ft.to_csv(tables / "formularity.csv", index=False)
    plot_formularity(ft, figures)

    # 4. MFW matrix
    mfw = mfw_matrix(df, top_n=200)
    mfw.to_csv(tables / "mfw200_permille.csv")
    np.savez(matrices / "mfw200.npz",
             matrix=mfw.values, labels=np.array(mfw.index), features=np.array(mfw.columns))

    # Console findings.
    print("=== PHASE 2 SUMMARY ===\n")
    print("Iliad vs Odyssey mean particle rate (per-mille), top divergences:")
    diff = (freq[freq.poem == "iliad"][PARTICLES].mean()
            - freq[freq.poem == "odyssey"][PARTICLES].mean()).sort_values()
    for p in list(diff.index[:4]) + list(diff.index[-4:]):
        il = freq[freq.poem == "iliad"][p].mean()
        od = freq[freq.poem == "odyssey"][p].mean()
        print(f"  {p:6s}  Iliad {il:6.2f}  Odyssey {od:6.2f}  (Δ {il-od:+.2f})")
    print("\nMost formulaic books (repeated-line rate):")
    print(ft.sort_values("repeated_line_rate", ascending=False)
          [["book", "repeated_line_rate", "rep_4gram_rate"]].head(6).to_string(index=False))
    print("\nLeast formulaic books:")
    print(ft.sort_values("repeated_line_rate")
          [["book", "repeated_line_rate", "rep_4gram_rate"]].head(6).to_string(index=False))
    print("\nRichness extremes (Yule's K — high = repetitive):")
    print(rt.sort_values("yule_k", ascending=False)[["book", "yule_k", "mattr_w100"]].head(4).to_string(index=False))
    print("...")
    print(rt.sort_values("yule_k")[["book", "yule_k", "mattr_w100"]].head(4).to_string(index=False))
    print(f"\nWrote tables → {tables}")
    print(f"Wrote figures → {figures}")
    print(f"Wrote matrices → {matrices}")


if __name__ == "__main__":
    main()
