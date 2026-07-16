"""
morphosyntax.py — a third, content-light channel: inflectional word-endings.

The ideal second content-free channel after metre is dependency syntax. Neural
Greek parsers (stanza, spaCy/odyCy) need models hosted on HuggingFace, which this
environment cannot reach, and the gold Perseus treebank does not cover the
Hellenistic calibrators — so a fully parsed, calibrated syntax test is not
runnable here. This is the self-contained stand-in.

Ancient Greek is heavily inflected: a word's grammatical role (case, number,
tense, mood, person) is carried by its ending. The distribution of word-endings
over a passage is therefore a MORPHOSYNTACTIC fingerprint — it reflects the mix of
finite verbs, cases, participles and infinitives (grammatical habit) far more than
subject matter. It is not full dependency syntax, but it is a distinct, largely
content-light grammatical channel, and it needs no external model.

Each chunk is described by the relative frequency of the commonest word-final
2- and 3-character suffixes; the rest of the calibrated test (internal dispersion
vs. single-author epics) is identical to metre.py.

Run:  python src/morphosyntax.py --config config.yaml
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from calibrate import tei_lines, balanced_zscore, delta_matrix, COLORS, SINGLE_AUTHOR
from viz import savefig


def word_suffixes(word):
    out = []
    if len(word) >= 2:
        out.append(word[-2:])
    if len(word) >= 3:
        out.append(word[-3:])
    return out


def suffix_features(works, size, top_n=150):
    """Per `size`-line chunk: relative frequency of the top word-ending suffixes."""
    chunks, owner = [], []
    for name, lines in works.items():
        for i in range(0, len(lines) - size + 1, size):
            toks = " ".join(lines[i:i + size]).split()
            sufs = []
            for w in toks:
                sufs += word_suffixes(w)
            chunks.append(sufs); owner.append(name)
    pool = Counter()
    for c in chunks:
        pool.update(c)
    vocab = [s for s, _ in pool.most_common(top_n)]
    F = np.array([[c.count(s) / len(c) for s in vocab] for c in chunks])
    return F, np.array(owner)


def load_works(cfg, root, norm_cfg):
    df = pd.read_parquet(root / cfg["paths"]["processed_dir"] / "lines.parquet")
    W = {}
    W["Iliad"] = list(df[df.poem == "iliad"].sort_values(["book_no", "line_no"]).text_norm)
    W["Odyssey"] = list(df[df.poem == "odyssey"].sort_values(["book_no", "line_no"]).text_norm)
    c = root / "data" / "raw" / "calib"
    W["Apollonius"] = tei_lines(c / "Apollonius_Argonautica.xml", norm_cfg)
    W["Quintus"] = tei_lines(c / "Quintus_Posthomerica.xml", norm_cfg)
    W["Hesiod"] = (tei_lines(c / "Hesiod_Theogony.xml", norm_cfg)
                   + tei_lines(c / "Hesiod_WorksDays.xml", norm_cfg))
    return W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--size", type=int, default=400)
    ap.add_argument("--top_n", type=int, default=150)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(args.config).resolve().parent
    figures = root / cfg["paths"]["figures_dir"]
    tables = root / cfg["paths"]["tables_dir"]
    norm_cfg = cfg["normalize"]["normalized"]

    works = load_works(cfg, root, norm_cfg)
    order = list(works.keys())
    F, owner = suffix_features(works, args.size, args.top_n)
    Z = balanced_zscore(F, owner, order)
    D = delta_matrix(Z)

    rows, within = [], {}
    for name in order:
        idx = np.where(owner == name)[0]
        sub = D[np.ix_(idx, idx)]
        w = sub[np.triu_indices(len(idx), 1)]
        within[name] = w
        sil = np.nan
        if len(idx) >= 4:
            lab = AgglomerativeClustering(n_clusters=2, metric="precomputed",
                                          linkage="average").fit_predict(sub)
            if len(set(lab)) == 2:
                sil = float(silhouette_score(sub, lab, metric="precomputed"))
        rows.append({"work": name, "single_author": name in SINGLE_AUTHOR,
                     "n_chunks": len(idx), "internal_dispersion": float(w.mean()),
                     "best2_silhouette": sil})
    summ = pd.DataFrame(rows)
    summ.to_csv(tables / "morphosyntax_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    op = ["Apollonius", "Quintus", "Hesiod", "Iliad", "Odyssey"]
    lbl = {"Hesiod": "Hesiod\n(2 poems)"}
    bp = ax.boxplot([within[w] for w in op], patch_artist=True, widths=0.6, showfliers=False)
    for patch, w in zip(bp["boxes"], op):
        patch.set_facecolor(COLORS[w]); patch.set_alpha(0.65)
    for m in bp["medians"]:
        m.set_color("black")
    ax.set_xticklabels([lbl.get(w, w) for w in op])
    unified = summ[summ.work.isin(["Apollonius", "Quintus"])].internal_dispersion.max()
    ax.axhline(unified, color="crimson", ls="--", lw=1,
               label="max unified single-epic (Apollonius, Quintus)")
    ax.set_ylabel("pairwise Delta on word-ending (morphosyntactic) profiles")
    ax.set_title("Internal MORPHOSYNTACTIC dispersion: the two poems vs. single-author "
                 "epics\n(inflectional word-endings; higher = more internally variable)",
                 fontsize=11)
    ax.legend(fontsize=9)
    savefig(fig, figures, "morphosyntax_dispersion")

    print("=== MORPHOSYNTAX (word-endings): calibrated plurality test ===\n")
    print(summ.round(4).to_string(index=False))
    unified_hi = summ[summ.work.isin(["Apollonius", "Quintus"])].internal_dispersion.max()
    for poem in ["Iliad", "Odyssey"]:
        d = summ.set_index("work").loc[poem, "internal_dispersion"]
        print(f"{poem}: morphosyntactic dispersion {d:.4f} "
              f"({'ABOVE' if d > unified_hi else 'within'} the unified single-epic level)")
    print(f"\nWrote {tables/'morphosyntax_summary.csv'} and "
          f"{figures/'morphosyntax_dispersion.png'}")


if __name__ == "__main__":
    main()
