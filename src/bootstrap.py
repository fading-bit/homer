"""
bootstrap.py — is the excess variation real, or sampling noise?

The calibrated test (calibrate.py, metre.py) reports point estimates of each work's
internal dispersion. This adds a chunk bootstrap: resample each work's chunks with
replacement (2000×), recompute its mean pairwise Delta, and read off a 95% CI. If
the Iliad's / Odyssey's interval sits above the single-author epics' intervals, the
gap is not an artefact of how few chunks we happen to have. Done on both the
word-frequency channel and the content-free metre channel.

Run:  python src/bootstrap.py --config config.yaml
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

from calibrate import tei_lines, balanced_zscore, delta_matrix, COLORS
from metre import scan_lines, chunk_profiles
from morphosyntax import suffix_features
from viz import savefig

ORDER = ["Iliad", "Odyssey", "Apollonius", "Quintus", "Hesiod"]
CALIBRATORS = ["Apollonius", "Quintus"]     # unified single-author epics


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


def word_features(works, size, top_n):
    chunks, owner = [], []
    for name, lines in works.items():
        for i in range(0, len(lines) - size + 1, size):
            chunks.append(" ".join(lines[i:i + size]).split()); owner.append(name)
    pool = Counter()
    for ch in chunks:
        pool.update(ch)
    vocab = [w for w, _ in pool.most_common(top_n)]
    F = np.array([[ch.count(w) / len(ch) for w in vocab] for ch in chunks])
    return F, np.array(owner)


def metre_features(works, size):
    profs, owner = [], []
    for name, lines in works.items():
        pats, _ = scan_lines(lines)
        for v in chunk_profiles(pats, size):
            profs.append(v); owner.append(name)
    return np.vstack(profs), np.array(owner)


def boot_dispersion(D, idx, B, rng):
    out = np.empty(B)
    for b in range(B):
        s = rng.choice(idx, size=len(idx), replace=True)
        sub = D[np.ix_(s, s)]
        iu = np.triu_indices(len(s), 1)
        pv = sub[iu]
        pv = pv[s[iu[0]] != s[iu[1]]]          # drop identical-chunk pairs
        out[b] = pv.mean() if len(pv) else np.nan
    return out


def channel(F, owner, B=2000):
    Z = balanced_zscore(F, owner, ORDER)
    D = delta_matrix(Z)
    res = {}
    for k, w in enumerate(ORDER):
        idx = np.where(owner == w)[0]
        dist = boot_dispersion(D, idx, B, np.random.default_rng(100 + k))
        res[w] = {"mean": float(np.nanmean(dist)),
                  "lo": float(np.nanpercentile(dist, 2.5)),
                  "hi": float(np.nanpercentile(dist, 97.5)),
                  "dist": dist}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--size", type=int, default=400)
    ap.add_argument("--top_n", type=int, default=150)
    ap.add_argument("--B", type=int, default=2000)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(args.config).resolve().parent
    figures = root / cfg["paths"]["figures_dir"]
    tables = root / cfg["paths"]["tables_dir"]
    norm_cfg = cfg["normalize"]["normalized"]

    works = load_works(cfg, root, norm_cfg)
    res_word = channel(*word_features(works, args.size, args.top_n), B=args.B)
    res_metre = channel(*metre_features(works, args.size), B=args.B)
    res_morph = channel(*suffix_features(works, args.size, args.top_n), B=args.B)

    rows = []
    print("=== BOOTSTRAP: is the excess internal variation real? ===\n")
    for label, res in [("word-frequency", res_word), ("metre", res_metre),
                       ("morphosyntax", res_morph)]:
        print(f"[{label}] internal dispersion, 95% CI (2000× chunk bootstrap):")
        for w in ORDER:
            r = res[w]
            print(f"  {w:11s} {r['mean']:.3f}  [{r['lo']:.3f}, {r['hi']:.3f}]")
            rows.append({"channel": label, "work": w, "mean": round(r["mean"], 4),
                         "ci_lo": round(r["lo"], 4), "ci_hi": round(r["hi"], 4)})
        cal = np.maximum(res["Apollonius"]["dist"], res["Quintus"]["dist"])
        pI = float(np.mean(res["Iliad"]["dist"] > cal))
        pO = float(np.mean(res["Odyssey"]["dist"] > cal))
        print(f"  P(Iliad > both calibrators)   = {pI:.3f}")
        print(f"  P(Odyssey > both calibrators) = {pO:.3f}\n")
        rows.append({"channel": label, "work": "P(Iliad>calib)", "mean": round(pI, 3),
                     "ci_lo": "", "ci_hi": ""})
        rows.append({"channel": label, "work": "P(Odyssey>calib)", "mean": round(pO, 3),
                     "ci_lo": "", "ci_hi": ""})
    pd.DataFrame(rows).to_csv(tables / "bootstrap_summary.csv", index=False)

    # forest plot: three panels (word, metre, morphosyntax)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ypos = {"Apollonius": 4, "Quintus": 3, "Hesiod": 2, "Odyssey": 1, "Iliad": 0}
    panels = [("word-frequency", res_word),
              ("metre (content-free)", res_metre),
              ("morphosyntax (word-endings)", res_morph)]
    for ax, (label, res) in zip(axes, panels):
        for w, y in ypos.items():
            r = res[w]
            ax.errorbar(r["mean"], y,
                        xerr=[[r["mean"] - r["lo"]], [r["hi"] - r["mean"]]],
                        fmt="o", color=COLORS[w], capsize=4, markersize=8, lw=2)
        cal_hi = max(res["Apollonius"]["hi"], res["Quintus"]["hi"])
        ax.axvline(cal_hi, color="crimson", ls="--", lw=1)
        ax.set_yticks(list(ypos.values()))
        ax.set_yticklabels([w if w != "Hesiod" else "Hesiod (2 poems)" for w in ypos])
        ax.set_xlabel("internal dispersion (mean pairwise Delta)")
        ax.set_title(label, fontsize=11)
        ax.set_ylim(-0.6, 4.6)
    axes[0].text(0.0, -0.6, "red line = upper 95% CI of the single-author epics",
                 transform=axes[0].get_xaxis_transform(), fontsize=8, color="crimson")
    fig.suptitle("Are the Iliad and Odyssey significantly more internally variable "
                 "than single-author epics?\n95% bootstrap intervals across three "
                 "channels", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    savefig(fig, figures, "bootstrap_forest")
    print(f"Wrote {tables/'bootstrap_summary.csv'} and {figures/'bootstrap_forest.png'}")


if __name__ == "__main__":
    main()
