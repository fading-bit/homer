"""
describe.py — Phase 1: descriptive orientation.

  1. Per-book summary table (lines, tokens, types, line length)
  2. Keyness table: the words most OVER-represented in each book vs the rest
     of the corpus, by Dunning's log-likelihood G²
  3. Keyness-weighted word-cloud grids (one panel per book), i.e. the
     distinctiveness cloud — words sized by how characteristic they are of the
     book, not by raw frequency (which would just show proper names / plot).

Run:  python src/describe.py --config config.yaml
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from textstats import book_tokens, log_likelihood_keyness
from viz import savefig, GREEK_FONT_PATH, POEM_COLORS
from features_word import iter_books, POEM_ORDER


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, poem, bno, g in iter_books(df):
        toks = book_tokens(g)
        rows.append({
            "book": label, "poem": poem, "book_no": bno,
            "n_lines": len(g), "n_tokens": len(toks),
            "n_types": len(set(toks)),
            "tokens_per_line": len(toks) / len(g) if len(g) else 0.0,
            "chars_per_line": g.text_raw.str.len().mean(),
        })
    return pd.DataFrame(rows)


def keyness_all_books(df: pd.DataFrame, top_n: int = 40):
    """Per-book counters + keyness of each book vs the rest of the corpus."""
    per_book = {}
    for label, _poem, _bno, g in iter_books(df):
        per_book[label] = Counter(book_tokens(g))
    corpus = Counter()
    for c in per_book.values():
        corpus.update(c)

    frames, key_by_book = [], {}
    for label, counts in per_book.items():
        rest = corpus.copy()
        rest.subtract(counts)  # rest-of-corpus counts
        rest += Counter()      # drop zero/negative entries
        kdf = log_likelihood_keyness(counts, rest, min_count=3, top_n=top_n)
        kdf.insert(0, "book", label)
        frames.append(kdf)
        key_by_book[label] = dict(zip(kdf.word, kdf.g2))
    return pd.concat(frames, ignore_index=True), key_by_book


def cloud_grid(key_by_book: dict, poem: str, figures_dir):
    """A 4×6 grid of keyness clouds for one poem's 24 books."""
    books = [b for b in key_by_book if b.startswith("IL" if poem == "iliad" else "OD")]
    books = sorted(books)
    fig, axes = plt.subplots(6, 4, figsize=(16, 20))
    for ax, b in zip(axes.flat, books):
        freqs = key_by_book[b]
        if freqs:
            wc = WordCloud(
                font_path=GREEK_FONT_PATH, width=400, height=300,
                background_color="white", prefer_horizontal=0.95,
                max_words=35, colormap="copper" if poem == "iliad" else "ocean",
            ).generate_from_frequencies(freqs)
            ax.imshow(wc, interpolation="bilinear")
        ax.set_title(b, fontsize=11, color=POEM_COLORS[poem])
        ax.axis("off")
    for ax in axes.flat[len(books):]:
        ax.axis("off")
    fig.suptitle(f"Keyness clouds — {poem.title()} "
                 f"(words sized by log-likelihood vs rest of corpus)",
                 fontsize=15, y=0.995)
    return savefig(fig, figures_dir, f"keyness_clouds_{poem}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(args.config).resolve().parent
    tables = root / cfg["paths"]["tables_dir"]
    figures = root / cfg["paths"]["figures_dir"]
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(root / cfg["paths"]["processed_dir"] / "lines.parquet")

    st = summary_table(df)
    st.to_csv(tables / "book_summary.csv", index=False)

    key_df, key_by_book = keyness_all_books(df, top_n=40)
    key_df.to_csv(tables / "keyness_by_book.csv", index=False)

    for poem in POEM_ORDER:
        cloud_grid(key_by_book, poem, figures)

    print("=== PHASE 1 SUMMARY ===\n")
    print(f"{len(df):,} lines across {df.book_no.nunique()} books × 2 poems")
    print(f"\nCorpus size: {st.n_tokens.sum():,} tokens, "
          f"{st.n_lines.sum():,} lines")
    print("\nBook-length range (tokens):",
          f"{st.n_tokens.min():,} ({st.loc[st.n_tokens.idxmin(),'book']}) – "
          f"{st.n_tokens.max():,} ({st.loc[st.n_tokens.idxmax(),'book']})")
    print("\nSample keyness — top distinctive words:")
    for b in ["IL01", "IL22", "OD09", "OD11"]:
        top = key_df[key_df.book == b].head(6).word.tolist()
        print(f"  {b}: {' · '.join(top)}")
    print(f"\nWrote tables → {tables}")
    print(f"Wrote figures → {figures}")


if __name__ == "__main__":
    main()
