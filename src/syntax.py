"""
syntax.py — the real content-free channel: dependency SYNTAX.

This is the principled successor to the morphosyntax word-ending proxy. It parses
the Greek with a neural dependency parser and builds features from grammatical
STRUCTURE, following Gorman & Gorman (2016) and R. Gorman (2020): the **sWord**,
the path of dependency-relation labels from a token up to the root of its sentence
tree. Syntax is largely independent of subject matter, so if the Iliad/Odyssey are
still more internally variable than single-author epics here, the "varied content"
explanation is finished — and, in particular, this is the channel that could firm
up the Odyssey, which only marginally separated on word-endings.

It could not be run in the environment where the rest of the pipeline was built,
because the neural Greek models are hosted on HuggingFace, which was blocked there.
Run it where that host is reachable.

────────────────────────────────────────────────────────────────────────────
SETUP (once; needs internet to fetch the model):
    pip install stanza
    python -c "import stanza; stanza.download('grc', package='proiel')"
      # 'proiel' parses running text well; 'perseus' is the other grc treebank.

RUN:
    python src/syntax.py --config config.yaml                 # sWord features (default)
    python src/syntax.py --config config.yaml --feature deprel
    python src/syntax.py --config config.yaml --feature pos
Parses are cached under data/processed/parse_cache/, so the slow step happens once.

WHAT TO SEND BACK (to fold into the report/article):
    • the printed summary block (dispersion per work, bootstrap 95% CIs, P-values)
    • outputs/tables/syntax_summary.csv
    • outputs/figures/syntax_dispersion.png
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from lxml import etree

from calibrate import balanced_zscore, delta_matrix, COLORS, SINGLE_AUTHOR, TEI_NS

ORDER = ["Iliad", "Odyssey", "Apollonius", "Quintus", "Hesiod"]
CALIBRATORS = ["Apollonius", "Quintus"]


# ── corpus (ACCENTED text — the parser needs diacritics and punctuation) ──
def tei_lines_accented(path):
    tree = etree.parse(str(path))
    out = []
    for l in tree.xpath("//tei:l", namespaces=TEI_NS):
        t = "".join(l.itertext()).strip()
        if t:
            out.append(t)
    return out


def load_accented(cfg, root):
    df = pd.read_parquet(root / cfg["paths"]["processed_dir"] / "lines.parquet")
    W = {}
    W["Iliad"] = list(df[df.poem == "iliad"].sort_values(["book_no", "line_no"]).text_dip)
    W["Odyssey"] = list(df[df.poem == "odyssey"].sort_values(["book_no", "line_no"]).text_dip)
    c = root / "data" / "raw" / "calib"
    W["Apollonius"] = tei_lines_accented(c / "Apollonius_Argonautica.xml")
    W["Quintus"] = tei_lines_accented(c / "Quintus_Posthomerica.xml")
    W["Hesiod"] = (tei_lines_accented(c / "Hesiod_Theogony.xml")
                   + tei_lines_accented(c / "Hesiod_WorksDays.xml"))
    return W


# ── the sWord: path of deprel labels from a token up to the root ──
def sword(word, byid, maxdepth=4):
    labels, cur, d = [], word, 0
    while cur is not None and cur.head not in (0, None) and d < maxdepth:
        labels.append((cur.deprel or "dep").split(":")[0])
        cur = byid.get(cur.head)
        d += 1
    labels.append("root")
    return "<".join(labels)


def parse_work(name, lines, nlp, cache_dir, batch=200):
    cache = cache_dir / f"{name}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    toks = []
    for i in range(0, len(lines), batch):
        doc = nlp("\n".join(lines[i:i + batch]))
        for sent in doc.sentences:
            byid = {w.id: w for w in sent.words}
            for w in sent.words:
                toks.append({
                    "upos": w.upos or "X",
                    "deprel": (w.deprel or "dep").split(":")[0],
                    "sword": sword(w, byid),
                })
        print(f"  {name}: parsed ~{min(i + batch, len(lines))}/{len(lines)} lines", end="\r")
    print()
    cache.write_bytes(pickle.dumps(toks))
    return toks


# ── same calibrated dispersion + bootstrap as the other channels ──
def boot_dispersion(D, idx, B, rng):
    out = np.empty(B)
    for b in range(B):
        s = rng.choice(idx, size=len(idx), replace=True)
        sub = D[np.ix_(s, s)]
        iu = np.triu_indices(len(s), 1)
        pv = sub[iu]
        pv = pv[s[iu[0]] != s[iu[1]]]
        out[b] = pv.mean() if len(pv) else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--feature", choices=["sword", "deprel", "pos"], default="sword")
    ap.add_argument("--chunk_tokens", type=int, default=2600,
                    help="tokens per chunk (~400 verse lines); keeps chunk counts "
                         "comparable to the word/metre channels")
    ap.add_argument("--top_n", type=int, default=150)
    ap.add_argument("--package", default="proiel", help="stanza grc treebank: proiel|perseus")
    ap.add_argument("--B", type=int, default=2000)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(args.config).resolve().parent
    figures = root / cfg["paths"]["figures_dir"]
    tables = root / cfg["paths"]["tables_dir"]
    cache_dir = root / cfg["paths"]["processed_dir"] / "parse_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    import stanza
    nlp = stanza.Pipeline(lang="grc", package=args.package,
                          processors="tokenize,pos,lemma,depparse", verbose=False)

    works = load_accented(cfg, root)
    key = {"sword": "sword", "deprel": "deprel", "pos": "upos"}[args.feature]

    # parse + chunk by tokens
    all_chunks, owner = [], []
    for name in ORDER:
        toks = parse_work(name, works[name], nlp, cache_dir)
        for i in range(0, len(toks) - args.chunk_tokens + 1, args.chunk_tokens):
            all_chunks.append([t[key] for t in toks[i:i + args.chunk_tokens]])
            owner.append(name)
    owner = np.array(owner)

    pool = Counter()
    for ch in all_chunks:
        pool.update(ch)
    vocab = [v for v, _ in pool.most_common(args.top_n)]
    F = np.array([[Counter(ch)[v] / len(ch) for v in vocab] for ch in all_chunks])

    Z = balanced_zscore(F, owner, ORDER)
    D = delta_matrix(Z)

    rows, within, boot = [], {}, {}
    for k, name in enumerate(ORDER):
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
        dist = boot_dispersion(D, idx, args.B, np.random.default_rng(100 + k))
        boot[name] = dist
        rows.append({"work": name, "single_author": name in SINGLE_AUTHOR,
                     "n_chunks": len(idx), "internal_dispersion": float(w.mean()),
                     "ci_lo": float(np.nanpercentile(dist, 2.5)),
                     "ci_hi": float(np.nanpercentile(dist, 97.5)),
                     "best2_silhouette": sil})
    summ = pd.DataFrame(rows)
    summ.to_csv(tables / f"syntax_summary_{args.feature}.csv", index=False)

    # figure
    fig, ax = plt.subplots(figsize=(10, 6))
    op = ["Apollonius", "Quintus", "Hesiod", "Iliad", "Odyssey"]
    lbl = {"Hesiod": "Hesiod\n(2 poems)"}
    bp = ax.boxplot([within[w] for w in op], patch_artist=True, widths=0.6, showfliers=False)
    for patch, w in zip(bp["boxes"], op):
        patch.set_facecolor(COLORS[w]); patch.set_alpha(0.65)
    for m in bp["medians"]:
        m.set_color("black")
    ax.set_xticklabels([lbl.get(w, w) for w in op])
    unified = summ[summ.work.isin(CALIBRATORS)].internal_dispersion.max()
    ax.axhline(unified, color="crimson", ls="--", lw=1,
               label="max unified single-epic (Apollonius, Quintus)")
    ax.set_ylabel(f"pairwise Delta on syntactic ({args.feature}) profiles")
    ax.set_title(f"Internal SYNTACTIC dispersion ({args.feature}): the two poems vs. "
                 f"single-author epics\n(content-free channel; higher = more variable)",
                 fontsize=11)
    ax.legend(fontsize=9)
    fig.savefig(figures / f"syntax_dispersion_{args.feature}.png", dpi=150, bbox_inches="tight")

    # report
    print(f"\n=== SYNTAX ({args.feature}): calibrated plurality test ===\n")
    print(summ.round(4).to_string(index=False))
    cal = np.maximum(boot["Apollonius"], boot["Quintus"])
    unified_hi = summ[summ.work.isin(CALIBRATORS)].internal_dispersion.max()
    print(f"\nUnified single-epic dispersion (Apollonius, Quintus): "
          f"{summ.set_index('work').loc['Apollonius','internal_dispersion']:.3f}, "
          f"{summ.set_index('work').loc['Quintus','internal_dispersion']:.3f}")
    for poem in ["Iliad", "Odyssey"]:
        d = summ.set_index("work").loc[poem, "internal_dispersion"]
        p = float(np.mean(boot[poem] > cal))
        print(f"{poem}: dispersion {d:.3f} "
              f"({'ABOVE' if d > unified_hi else 'within'} single-epic level); "
              f"P(> both calibrators) = {p:.3f}")
    print(f"\nWrote {tables/f'syntax_summary_{args.feature}.csv'} and "
          f"{figures/f'syntax_dispersion_{args.feature}.png'}")
    print("\nSend those two files + this printout back to fold into the paper.")


if __name__ == "__main__":
    main()
