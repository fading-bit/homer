"""
distances.py — Phase 3: inter-book distances and clustering (the payoff).

Builds three INDEPENDENT book×book distance matrices over the 48 books and asks
whether they cluster as one voice or several:

  1. Burrows's Delta  — z-scored most-frequent-word profiles, Manhattan distance
  2. Character n-gram cosine — top-K character-trigram profiles
  3. Compression distance (NCD) — "how much does knowing book i help compress j"

For each it produces a clustered heatmap and a dendrogram; it then projects the
books to 2-D (MDS) coloured by poem, measures how much the three matrices agree
(Mantel-style correlation), and reports the 2-cluster split purity and any book
that lands with the "wrong" poem.

CAVEAT (see the report §6): the two poems come from different editions, so the
Iliad-vs-Odyssey separation may be partly editorial. WITHIN-poem structure —
which books cluster together inside each poem — is edition-robust and the more
interpretable signal here.

Run:  python src/distances.py --config config.yaml
"""

from __future__ import annotations

import argparse
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS

from textstats import ngrams
from features_word import iter_books, book_label, POEM_ORDER
from viz import savefig, POEM_COLORS


# ── Feature builders ────────────────────────────────────────────────────────

def book_token_lists(df):
    labels, poems, toklists, texts = [], [], [], []
    for label, poem, _b, g in iter_books(df):
        toks = [t for line in g.text_norm for t in [line]]  # placeholder
        toks = " ".join(g.text_norm).split()
        labels.append(label); poems.append(poem)
        toklists.append(toks); texts.append(" ".join(g.text_norm))
    return labels, poems, toklists, texts


def delta_matrix(toklists, top_n=150):
    corpus = Counter()
    for t in toklists:
        corpus.update(t)
    vocab = [w for w, _ in corpus.most_common(top_n)]
    # relative frequencies → z-score per feature across books
    F = np.array([[t.count(w) / len(t) for w in vocab] for t in toklists])
    mu, sd = F.mean(0), F.std(0)
    sd[sd == 0] = 1.0
    Z = (F - mu) / sd
    W = len(toklists)
    D = np.zeros((W, W))
    for i in range(W):
        for j in range(i + 1, W):
            D[i, j] = D[j, i] = np.abs(Z[i] - Z[j]).mean()  # Burrows's Delta
    return D, Z


def charcos_matrix(texts, n=3, top_k=300):
    corpus = Counter()
    grams_per = []
    for txt in texts:
        g = Counter(txt[i:i + n] for i in range(len(txt) - n + 1))
        grams_per.append(g); corpus.update(g)
    vocab = [x for x, _ in corpus.most_common(top_k)]
    V = np.array([[g.get(x, 0) for x in vocab] for g in grams_per], float)
    V /= V.sum(1, keepdims=True)
    # cosine distance
    norm = np.linalg.norm(V, axis=1, keepdims=True); norm[norm == 0] = 1
    U = V / norm
    S = U @ U.T
    return 1.0 - S


def ncd_matrix(texts):
    def C(b): return len(zlib.compress(b, 9))
    enc = [t.encode("utf-8") for t in texts]
    cx = [C(e) for e in enc]
    W = len(texts)
    D = np.zeros((W, W))
    for i in range(W):
        for j in range(i + 1, W):
            cxy = C(enc[i] + enc[j])
            ncd = (cxy - min(cx[i], cx[j])) / max(cx[i], cx[j])
            D[i, j] = D[j, i] = ncd
    return D


# ── Clustering helpers ──────────────────────────────────────────────────────

def leaf_colors(labels):
    return ["#b5651d" if l.startswith("IL") else "#1d6fb5" for l in labels]


def clustered_heatmap(D, labels, title, name, figures_dir):
    Z = linkage(squareform(D, checks=False), method="average")
    colors = leaf_colors(labels)
    g = sns.clustermap(
        pd.DataFrame(D, index=labels, columns=labels),
        row_linkage=Z, col_linkage=Z, cmap="mako_r",
        row_colors=colors, col_colors=colors,
        xticklabels=True, yticklabels=True, figsize=(12, 12),
        cbar_pos=(0.02, 0.83, 0.03, 0.12),
    )
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=6)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=6)
    g.figure.suptitle(title, fontsize=13, y=1.0)
    path = Path(figures_dir) / f"{name}.png"
    g.figure.savefig(path, bbox_inches="tight", facecolor="white", dpi=120)
    plt.close(g.figure)
    return Z, path


def two_cluster_report(D, labels, poems, Z):
    cl = fcluster(Z, 2, criterion="maxclust")
    tab = pd.crosstab(pd.Series(cl, name="cluster"), pd.Series(poems, name="poem"))
    # majority poem per cluster
    maj = {c: tab.loc[c].idxmax() for c in tab.index}
    crossovers = [labels[i] for i, (c, p) in enumerate(zip(cl, poems)) if maj[c] != p]
    purity = sum(maj[c] == p for c, p in zip(cl, poems)) / len(poems)
    return purity, crossovers, tab


def mantel_like(matrices: dict):
    keys = list(matrices)
    iu = np.triu_indices(next(iter(matrices.values())).shape[0], k=1)
    vecs = {k: m[iu] for k, m in matrices.items()}
    out = {}
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            r = np.corrcoef(vecs[keys[a]], vecs[keys[b]])[0, 1]
            out[f"{keys[a]}~{keys[b]}"] = r
    return out


def mds_scatter(D, labels, poems, figures_dir, name="mds_delta"):
    coords = MDS(n_components=2, dissimilarity="precomputed", random_state=0,
                 n_init=4, init="random", normalized_stress="auto").fit_transform(D)
    fig, ax = plt.subplots(figsize=(10, 8))
    for poem in POEM_ORDER:
        idx = [i for i, p in enumerate(poems) if p == poem]
        ax.scatter(coords[idx, 0], coords[idx, 1], s=90, alpha=0.8,
                   color=POEM_COLORS[poem], label=poem.title(),
                   edgecolor="white", linewidth=0.8)
    for i, l in enumerate(labels):
        ax.annotate(l[2:], (coords[i, 0], coords[i, 1]), fontsize=6,
                    ha="center", va="center", color="white", weight="bold")
    ax.set_title("Books in stylistic space (MDS on Burrows's Delta)\n"
                 "colour = poem; within-poem spread is edition-robust, "
                 "the poem split may be partly editorial", fontsize=11)
    ax.legend(fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    return savefig(fig, figures_dir, name)


# ────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--top_n", type=int, default=150, help="MFW count for Delta")
    ap.add_argument("--ngram", type=int, default=3)
    ap.add_argument("--top_k", type=int, default=300, help="char n-gram vocab")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(args.config).resolve().parent
    figures = root / cfg["paths"]["figures_dir"]
    tables = root / cfg["paths"]["tables_dir"]
    matrices = root / cfg["paths"]["matrices_dir"]
    for d in (figures, tables, matrices):
        d.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(root / cfg["paths"]["processed_dir"] / "lines.parquet")
    labels, poems, toklists, texts = book_token_lists(df)

    D_delta, _ = delta_matrix(toklists, top_n=args.top_n)
    D_char = charcos_matrix(texts, n=args.ngram, top_k=args.top_k)
    D_ncd = ncd_matrix(texts)

    for name, D in [("dist_delta", D_delta), ("dist_charcos", D_char), ("dist_ncd", D_ncd)]:
        np.savez(matrices / f"{name}.npz", matrix=D, labels=np.array(labels))

    Z_delta, _ = clustered_heatmap(D_delta, labels,
        "Burrows's Delta (most-frequent words)", "clustermap_delta", figures)
    Z_char, _ = clustered_heatmap(D_char, labels,
        "Character-trigram cosine distance", "clustermap_charcos", figures)
    Z_ncd, _ = clustered_heatmap(D_ncd, labels,
        "Compression distance (NCD)", "clustermap_ncd", figures)
    mds_scatter(D_delta, labels, poems, figures)

    # agreement + 2-cluster purity
    corr = mantel_like({"delta": D_delta, "char": D_char, "ncd": D_ncd})
    rows = []
    for nm, D, Z in [("delta", D_delta, Z_delta), ("char", D_char, Z_char), ("ncd", D_ncd, Z_ncd)]:
        purity, crossovers, _ = two_cluster_report(D, labels, poems, Z)
        rows.append({"matrix": nm, "two_cluster_purity": round(purity, 3),
                     "crossover_books": ";".join(crossovers) or "(none)"})
    summ = pd.DataFrame(rows)
    summ.to_csv(tables / "distance_summary.csv", index=False)
    pd.DataFrame([corr]).to_csv(tables / "distance_matrix_correlations.csv", index=False)

    print("=== PHASE 3: DISTANCES & CLUSTERING ===\n")
    print("Cross-matrix agreement (correlation of the three distance matrices):")
    for k, v in corr.items():
        print(f"  {k:14s} r = {v:.3f}")
    print("\nDoes each matrix separate the two poems? (2-cluster split)")
    print(summ.to_string(index=False))
    print(f"\nWrote matrices → {matrices}")
    print(f"Wrote figures  → {figures}  (clustermap_*.png, mds_delta.png)")
    print(f"Wrote tables   → {tables}")


if __name__ == "__main__":
    main()
