"""
metre.py — hexameter scansion and a content-free calibrated test.

Word-frequency dispersion (calibrate.py) leaves one loophole: the Iliad may range
over more varied *content* than the calibrators. Metre is (largely) independent of
subject matter, so it is the channel that can adjudicate. If the Iliad and Odyssey
are STILL more internally variable than single-author epics on metre, the "varied
content" explanation weakens and the plurality reading is corroborated.

Scansion: each line is syllabified and each syllable given a quantity —
  * long by nature: η, ω, and diphthongs (αι ει οι υι αυ ευ ου ηυ ωυ)
  * short by nature: ε, ο
  * ambiguous (dichrona): α, ι, υ
  * long by position: a vowel followed by ≥2 consonants (double ζ ξ ψ = two),
    counted across word boundaries; a stop+liquid cluster is treated as optional;
  * correption: a long vowel/diphthong in hiatus (immediately before a vowel) may
    shorten.
The dactylic hexameter itself then resolves the ambiguous syllables: we search for
a parse of the line into five feet of dactyl (— ∪ ∪) or spondee (— —) plus a final
disyllable. The recovered pattern of feet 1–4 (16 possibilities) is the per-line
metrical fingerprint. Dichrona are resolved dactyl-first, uniformly for every work,
so any bias cancels in the cross-work comparison.

Run:  python src/metre.py --config config.yaml [--voice narration]
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
from sklearn.manifold import MDS

from corpus import normalize_line
from speech_split import tag_lines
from viz import savefig
from calibrate import tei_lines, balanced_zscore, delta_matrix, COLORS, SINGLE_AUTHOR

VOWELS = set("αεηιουω")
LONG_NAT = set("ηω")
SHORT_NAT = set("εο")
DIPH = {"αι", "ει", "οι", "υι", "αυ", "ευ", "ου", "ηυ", "ωυ"}
DOUBLE = set("ζξψ")
STOPS = set("βγδπτκφθχ")
LIQ = set("λρ")
CONS = set("βγδζθκλμνξπρστφχψ")


def segments(text):
    """Split a line into (nucleus, following-consonant-string) pairs."""
    s = [c for c in text.replace(" ", "") if c in VOWELS or c in CONS]
    out, i, n = [], 0, len(s)
    while i < n:
        if s[i] in VOWELS:
            if i + 1 < n and (s[i] + s[i + 1]) in DIPH:
                nuc = s[i] + s[i + 1]; i += 2
            else:
                nuc = s[i]; i += 1
            j = i
            while j < n and s[j] in CONS:
                j += 1
            out.append((nuc, "".join(s[i:j])))
            i = j
        else:
            i += 1  # leading consonants before first vowel (attach to nothing)
    return out


def quantities(text):
    seg = segments(text)
    Q = []
    for k, (nuc, gap) in enumerate(seg):
        if len(nuc) == 2 or nuc in LONG_NAT:
            base = "L"
        elif nuc in SHORT_NAT:
            base = "S"
        else:
            base = "A"  # α ι υ
        c = sum(2 if ch in DOUBLE else 1 for ch in gap)
        muta = len(gap) == 2 and gap[0] in STOPS and gap[1] in LIQ
        nxt = seg[k + 1][0] if k + 1 < len(seg) else None
        if k == len(seg) - 1:
            q = "L"                       # final syllable: anceps → treat long
        elif c >= 2 and not muta:
            q = "L"                       # long by position
        elif c >= 2 and muta:
            q = base if base == "L" else "A"
        elif c == 0 and base == "L" and nxt is not None:
            q = "A"                       # correption possible in hiatus
        else:
            q = base
        Q.append(q)
    return Q


def scan(Q):
    """Return the feet-1–5 D/S pattern (list) or None if the line won't scan."""
    N = len(Q)
    if not (12 <= N <= 17):
        return None
    okL = lambda q: q in ("L", "A")
    okS = lambda q: q in ("S", "A")

    def rec(p, foot):
        if foot == 6:
            return [] if (p == N - 2 and okL(Q[p])) else None
        # dactyl first (Homeric default), then spondee
        if p + 3 <= N and okL(Q[p]) and okS(Q[p + 1]) and okS(Q[p + 2]):
            r = rec(p + 3, foot + 1)
            if r is not None:
                return ["D"] + r
        if p + 2 <= N and okL(Q[p]) and okL(Q[p + 1]):
            r = rec(p + 2, foot + 1)
            if r is not None:
                return ["S"] + r
        return None

    return rec(0, 1)


def pattern_index(feet):
    """16 patterns of feet 1–4 → index 0..15 (bit set = spondee)."""
    return sum((1 << i) for i, f in enumerate(feet[:4]) if f == "S")


def scan_lines(lines):
    """Return (patterns list with None for failures, scansion rate)."""
    pats = [scan(quantities(ln)) for ln in lines]
    ok = sum(p is not None for p in pats)
    return pats, (ok / len(pats) if pats else 0.0)


def chunk_profiles(pats, size):
    """16-dim metrical-pattern distribution per `size`-line chunk (scanned lines)."""
    profs = []
    for i in range(0, len(pats) - size + 1, size):
        block = [p for p in pats[i:i + size] if p is not None]
        v = np.zeros(16)
        for p in block:
            v[pattern_index(p)] += 1
        if v.sum() > 0:
            v /= v.sum()
        profs.append(v)
    return profs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--size", type=int, default=400)
    ap.add_argument("--voice", choices=["all", "narration", "speech"], default="all")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(args.config).resolve().parent
    figures = root / cfg["paths"]["figures_dir"]
    tables = root / cfg["paths"]["tables_dir"]
    norm_cfg = cfg["normalize"]["normalized"]

    df = pd.read_parquet(root / cfg["paths"]["processed_dir"] / "lines.parquet")
    works = {}
    works["Iliad"] = list(df[df.poem == "iliad"].sort_values(["book_no", "line_no"]).text_norm)
    works["Odyssey"] = list(df[df.poem == "odyssey"].sort_values(["book_no", "line_no"]).text_norm)
    calib = root / "data" / "raw" / "calib"
    works["Apollonius"] = tei_lines(calib / "Apollonius_Argonautica.xml", norm_cfg)
    works["Quintus"] = tei_lines(calib / "Quintus_Posthomerica.xml", norm_cfg)
    works["Hesiod"] = (tei_lines(calib / "Hesiod_Theogony.xml", norm_cfg)
                       + tei_lines(calib / "Hesiod_WorksDays.xml", norm_cfg))

    vsuffix = ""
    if args.voice != "all":
        vc = "N" if args.voice == "narration" else "S"
        for name in list(works):
            tags = tag_lines(works[name])
            works[name] = [ln for ln, t in zip(works[name], tags) if t == vc]
        vsuffix = f"_{args.voice}"

    # scan + chunk
    order = list(works.keys())
    profs, owner, rates, foot_rates = [], [], {}, {}
    for name in order:
        pats, rate = scan_lines(works[name])
        rates[name] = rate
        # per-foot dactyl rate (feet 1-4) for description
        scanned = [p for p in pats if p is not None]
        fr = [np.mean([p[f] == "D" for p in scanned]) for f in range(4)] if scanned else [np.nan]*4
        foot_rates[name] = fr
        for v in chunk_profiles(pats, args.size):
            profs.append(v); owner.append(name)
    owner = np.array(owner)
    F = np.vstack(profs)
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
                     "scansion_rate": round(rates[name], 3), "n_chunks": len(idx),
                     "dactyl_rate_f1_4": ";".join(f"{x:.2f}" for x in foot_rates[name]),
                     "internal_dispersion": float(w.mean()),
                     "best2_silhouette": sil})
    summ = pd.DataFrame(rows)
    summ.to_csv(tables / f"metre_summary{vsuffix}.csv", index=False)

    # figure: metrical dispersion per work
    fig, ax = plt.subplots(figsize=(10, 6))
    op = ["Apollonius", "Quintus", "Hesiod", "Iliad", "Odyssey"]
    disp_lbl = {"Hesiod": "Hesiod\n(2 poems)"}
    bp = ax.boxplot([within[w] for w in op], patch_artist=True, widths=0.6, showfliers=False)
    for patch, w in zip(bp["boxes"], op):
        patch.set_facecolor(COLORS[w]); patch.set_alpha(0.65)
    for m in bp["medians"]:
        m.set_color("black")
    ax.set_xticklabels([disp_lbl.get(w, w) for w in op])
    # baseline = the unified single-author epics (Hesiod is two separate poems)
    unified = summ[summ.work.isin(["Apollonius", "Quintus"])].internal_dispersion.max()
    ax.axhline(unified, color="crimson", ls="--", lw=1,
               label="max unified single-epic (Apollonius, Quintus)")
    ax.set_ylabel("pairwise Delta on metrical (foot-pattern) profiles")
    ax.set_title(f"Internal METRICAL dispersion: the two poems vs. single-author epics"
                 f"{'  ['+args.voice+']' if args.voice!='all' else ''}\n"
                 f"(content-free channel; higher = more internally variable)", fontsize=11)
    ax.legend(fontsize=9)
    savefig(fig, figures, f"metre_dispersion{vsuffix}")

    print(f"=== METRE: content-free plurality test [{args.voice}] ===\n")
    print(summ.to_string(index=False))
    sa = summ[summ.single_author]
    lo, hi = sa.internal_dispersion.min(), sa.internal_dispersion.max()
    unified_hi = summ[summ.work.isin(["Apollonius", "Quintus"])].internal_dispersion.max()
    print(f"\nScansion rates: " + ", ".join(f"{r.work} {r.scansion_rate:.0%}"
                                            for _, r in summ.iterrows()))
    print(f"Unified single-epic (Apollonius, Quintus) metrical dispersion: "
          f"{summ.set_index('work').loc['Apollonius','internal_dispersion']:.3f}, "
          f"{summ.set_index('work').loc['Quintus','internal_dispersion']:.3f}  "
          f"(Hesiod = two poems combined, {summ.set_index('work').loc['Hesiod','internal_dispersion']:.3f})")
    for poem in ["Iliad", "Odyssey"]:
        d = summ.set_index("work").loc[poem, "internal_dispersion"]
        print(f"{poem}: metrical dispersion {d:.4f} "
              f"({'ABOVE' if d > unified_hi else 'within'} the unified single-epic level)")
    print(f"\nWrote {tables/f'metre_summary{vsuffix}.csv'} and "
          f"{figures/f'metre_dispersion{vsuffix}.png'}")


if __name__ == "__main__":
    main()
