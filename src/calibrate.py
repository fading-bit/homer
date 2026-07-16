"""
calibrate.py — Is there more than one Homer WITHIN the Iliad / within the Odyssey?

A stylometric "seam" inside a poem is uninterpretable on its own: a single poet
writing a battle, a simile and a shield-description will vary too, hence calibration
needed and to ask whether each poem's internal stylistic variation is larger than 
what known single-author hexameter epics show.

Method:
  * Split every poem into equal 400-line chunks.
  * Describe each chunk by a 150 most-frequent-word profile; standardise features
    with EQUAL WEIGHT PER WORK (so the majority work — Homer — cannot look
    artificially cohesive), then Burrows's Delta between chunks.
  * For each work measure INTERNAL DISPERSION (mean pairwise Delta among its own
    chunks) and its best 2-cluster silhouette (how cleanly its chunks split into
    two stylistic groups).
  * Compare the Iliad and Odyssey to the single-author calibrators
    (Apollonius' Argonautica, Quintus' Posthomerica, Hesiod).

Comparing WITHIN-work spread sidesteps the edition problem: each work's spread is
measured inside its own edition.

Run:  python src/calibrate.py --config config.yaml
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from lxml import etree
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.manifold import MDS

from corpus import normalize_line
from speech_split import tag_lines
from viz import savefig

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}

# single-author calibrators vs. the two poems under test
SINGLE_AUTHOR = {"Apollonius", "Quintus", "Hesiod"}
COLORS = {
    "Iliad": "#b5651d", "Odyssey": "#1d6fb5",
    "Apollonius": "#4a7c59", "Quintus": "#7a5c9e", "Hesiod": "#999999",
}


def tei_lines(path, norm_cfg):
    tree = etree.parse(str(path))
    out = []
    for l in tree.xpath("//tei:l", namespaces=TEI_NS):
        text = "".join(l.itertext()).strip()
        if not text:
            continue
        n = normalize_line(text, norm_cfg)
        if n:
            out.append(n)
    return out


def chunk_tokens(lines, size):
    """Token-lists from consecutive `size`-line blocks (drop trailing partial)."""
    out = []
    for i in range(0, len(lines) - size + 1, size):
        out.append(" ".join(lines[i:i + size]).split())
    return out


def balanced_zscore(F, owner, works):
    """Standardise each feature with equal weight per work (mean of per-work
    means / mean of per-work stds), so a majority work cannot dominate the scale."""
    means = np.vstack([F[owner == w].mean(0) for w in works])
    stds = np.vstack([F[owner == w].std(0) for w in works])
    mu = means.mean(0)
    sd = stds.mean(0)
    sd[sd == 0] = 1.0
    return (F - mu) / sd


def delta_matrix(Z):
    n = len(Z)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = np.abs(Z[i] - Z[j]).mean()
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--size", type=int, default=400, help="chunk size in lines")
    ap.add_argument("--top_n", type=int, default=150, help="MFW features")
    ap.add_argument("--voice", choices=["all", "narration", "speech"], default="all",
                    help="restrict every work to one register before chunking")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(args.config).resolve().parent
    figures = root / cfg["paths"]["figures_dir"]
    tables = root / cfg["paths"]["tables_dir"]
    norm_cfg = cfg["normalize"]["normalized"]

    # Homer (already normalised) + single-author calibrators
    df = pd.read_parquet(root / cfg["paths"]["processed_dir"] / "lines.parquet")
    works = {}
    works["Iliad"] = list(df[df.poem == "iliad"].sort_values(["book_no", "line_no"]).text_norm)
    works["Odyssey"] = list(df[df.poem == "odyssey"].sort_values(["book_no", "line_no"]).text_norm)
    calib = root / "data" / "raw" / "calib"
    works["Apollonius"] = tei_lines(calib / "Apollonius_Argonautica.xml", norm_cfg)
    works["Quintus"] = tei_lines(calib / "Quintus_Posthomerica.xml", norm_cfg)
    works["Hesiod"] = (tei_lines(calib / "Hesiod_Theogony.xml", norm_cfg)
                       + tei_lines(calib / "Hesiod_WorksDays.xml", norm_cfg))

    # optional register filter (same speech tagger applied to every work)
    vsuffix = ""
    if args.voice != "all":
        vc = "N" if args.voice == "narration" else "S"
        for name in list(works):
            tags = tag_lines(works[name])
            works[name] = [ln for ln, t in zip(works[name], tags) if t == vc]
        vsuffix = f"_{args.voice}"

    # chunk everything
    chunks, owner = [], []
    for name, lines in works.items():
        for c in chunk_tokens(lines, args.size):
            chunks.append(c)
            owner.append(name)
    owner = np.array(owner)
    order = list(works.keys())

    # MFW feature space (pooled), balanced standardisation, Delta
    pool = Counter()
    for c in chunks:
        pool.update(c)
    vocab = [w for w, _ in pool.most_common(args.top_n)]
    F = np.array([[c.count(w) / len(c) for w in vocab] for c in chunks])
    Z = balanced_zscore(F, owner, order)
    D = delta_matrix(Z)

    # per-work internal dispersion + best 2-cluster silhouette
    rows, within_by_work = [], {}
    for name in order:
        idx = np.where(owner == name)[0]
        sub = D[np.ix_(idx, idx)]
        iu = np.triu_indices(len(idx), 1)
        within = sub[iu]
        within_by_work[name] = within
        sil = np.nan
        if len(idx) >= 4:
            lab = AgglomerativeClustering(
                n_clusters=2, metric="precomputed", linkage="average").fit_predict(sub)
            if len(set(lab)) == 2:
                sil = float(silhouette_score(sub, lab, metric="precomputed"))
        rows.append({
            "work": name, "single_author": name in SINGLE_AUTHOR,
            "n_chunks": len(idx), "internal_dispersion": float(within.mean()),
            "dispersion_sd": float(within.std()), "best2_silhouette": sil,
        })
    summ = pd.DataFrame(rows)
    summ.to_csv(tables / f"calibration_summary{vsuffix}.csv", index=False)

    # ── figure 1: dispersion distributions per work ──
    fig, ax = plt.subplots(figsize=(10, 6))
    order_plot = ["Apollonius", "Quintus", "Hesiod", "Iliad", "Odyssey"]
    data = [within_by_work[w] for w in order_plot]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
    for patch, w in zip(bp["boxes"], order_plot):
        patch.set_facecolor(COLORS[w]); patch.set_alpha(0.65)
    for med in bp["medians"]:
        med.set_color("black")
    ax.set_xticklabels(order_plot)
    sa_hi = summ[summ.single_author].internal_dispersion.max()
    ax.axhline(sa_hi, color="crimson", ls="--", lw=1,
               label="max single-author internal dispersion")
    ax.set_ylabel("pairwise Burrows's Delta between a work's own chunks")
    ax.set_title("Internal stylistic dispersion: the two poems vs. known "
                 "single-author hexameter epics\n(higher = more internally variable)",
                 fontsize=11)
    ax.legend(fontsize=9)
    savefig(fig, figures, f"calibration_dispersion{vsuffix}")

    # ── figure 2: MDS of all chunks coloured by work ──
    coords = MDS(n_components=2, dissimilarity="precomputed", random_state=0,
                 n_init=4, init="random", normalized_stress="auto").fit_transform(D)
    fig, ax = plt.subplots(figsize=(10, 8))
    for name in order:
        idx = owner == name
        ax.scatter(coords[idx, 0], coords[idx, 1], s=70, alpha=0.8,
                   color=COLORS[name],
                   label=f"{name}{' (single author)' if name in SINGLE_AUTHOR else ''}",
                   edgecolor="white", linewidth=0.7)
    ax.set_title("Every 400-line chunk in stylistic space (MDS on Burrows's Delta)\n"
                 "do the Iliad / Odyssey chunks stay as tight as a single author's?",
                 fontsize=11)
    ax.legend(fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    savefig(fig, figures, f"calibration_mds{vsuffix}")

    # ── the calibrated answer ──
    sa = summ[summ.single_author]
    lo, hi = sa.internal_dispersion.min(), sa.internal_dispersion.max()
    sil_hi = sa.best2_silhouette.max()
    print(f"=== IS THERE MORE THAN ONE HOMER? — CALIBRATED TEST [{args.voice}] ===\n")
    print(f"chunk size {args.size} lines, {args.top_n} MFW features, "
          f"{len(chunks)} chunks total\n")
    print(summ.round(4).to_string(index=False))
    print(f"\nSingle-author internal-dispersion range: {lo:.4f} – {hi:.4f}")
    print(f"Single-author best-2-cluster silhouette (max): {sil_hi:.3f}\n")
    for poem in ["Iliad", "Odyssey"]:
        r = summ.set_index("work").loc[poem]
        d, s = r.internal_dispersion, r.best2_silhouette
        d_ok = d <= hi * 1.05
        s_ok = (s <= sil_hi + 0.02) or np.isnan(s)
        print(f"{poem}: dispersion {d:.4f} "
              f"({'within' if d_ok else 'ABOVE'} single-author range), "
              f"best-2 silhouette {s:.3f} "
              f"({'not stronger' if s_ok else 'STRONGER'} than single-author).")
        if d_ok and s_ok:
            print(f"   → internal variation is normal for one author; "
                  f"no stylometric evidence of more than one hand in the {poem}.")
        else:
            print(f"   → internal variation exceeds the single-author baseline; "
                  f"suggestive of plurality in the {poem}.")
    # ── length-matched control: is Homer still more dispersed at the SAME span? ──
    def matched_dispersion(work, k):
        idx = np.where(owner == work)[0]           # contiguous, in poem order
        if len(idx) < k:
            return np.nan
        vals = []
        for s in range(0, len(idx) - k + 1):
            w = idx[s:s + k]
            sub = D[np.ix_(w, w)]
            vals.append(sub[np.triu_indices(k, 1)].mean())
        return float(np.mean(vals))

    print("\nLength-matched control (dispersion within contiguous windows the size")
    print("of the calibrators — rules out a pure length/drift effect):")
    for k, ref in [(14, "Apollonius=%.3f" % summ.set_index('work').loc['Apollonius','internal_dispersion']),
                   (22, "Quintus=%.3f" % summ.set_index('work').loc['Quintus','internal_dispersion'])]:
        il = matched_dispersion("Iliad", k)
        od = matched_dispersion("Odyssey", k)
        print(f"  window={k} chunks:  Iliad {il:.3f}   Odyssey {od:.3f}   (vs {ref})")

    print(f"\nWrote {tables/'calibration_summary.csv'}, "
          f"{figures/'calibration_dispersion.png'}, {figures/'calibration_mds.png'}")


if __name__ == "__main__":
    main()
