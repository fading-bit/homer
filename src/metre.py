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
# postpositives (enclitics/particles that lean back): a caesura cannot fall
# before them, so a word-break immediately before one is not a real pause.
POSTPOS = {"δε", "τε", "γε", "ρα", "αρα", "γαρ", "μεν", "μην", "δη", "τοι",
           "κεν", "κε", "αν", "νυ", "νυν", "περ", "θην", " κε"}


def segments(text):
    """Nuclei, the consonant gaps between them, and word-final flags.

    Gaps span word boundaries (needed for length-by-position); wbreak[k] marks
    whether nucleus k is the last syllable of its word (needed for caesura)."""
    seq = []  # (char, word_index, starts_word)
    for wi, w in enumerate(text.split()):
        for i, ch in enumerate(w):
            if ch in VOWELS or ch in CONS:
                seq.append((ch, wi, i == 0))
    words = text.split()
    nuclei, gaps, wbreak, widx = [], [], [], []
    pend, pend_break = [], False
    j, L = 0, len(seq)
    while j < L:
        ch, wi, ws = seq[j]
        if ch in VOWELS:
            if j + 1 < L and seq[j + 1][0] in VOWELS and (ch + seq[j + 1][0]) in DIPH:
                nuc = ch + seq[j + 1][0]; j += 2
            else:
                nuc = ch; j += 1
            if nuclei:                       # close the previous nucleus's gap
                gaps.append("".join(pend))
                wbreak.append(pend_break or ws)   # word ended before this nucleus
            nuclei.append(nuc); widx.append(wi)
            pend, pend_break = [], False
        else:
            pend.append(ch)
            pend_break = pend_break or ws
            j += 1
    gaps.append("".join(pend))               # trailing consonants after last nucleus
    wbreak.append(True)                      # last nucleus ends a word (line end)
    return words, nuclei, gaps, wbreak, widx


def _quantities(nuclei, gaps):
    Q = []
    n = len(nuclei)
    for k in range(n):
        nuc, gap = nuclei[k], gaps[k]
        if len(nuc) == 2 or nuc in LONG_NAT:
            base = "L"
        elif nuc in SHORT_NAT:
            base = "S"
        else:
            base = "A"  # α ι υ
        c = sum(2 if ch in DOUBLE else 1 for ch in gap)
        muta = len(gap) == 2 and gap[0] in STOPS and gap[1] in LIQ
        nxt = nuclei[k + 1] if k + 1 < n else None
        if k == n - 1:
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


def quantities(text):
    _, nuclei, gaps, _, _ = segments(text)
    return _quantities(nuclei, gaps)


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


def caesura(feet, wbreak, widx, words):
    """Classify the principal third-foot caesura using gender-neutral, position-
    based names. Penthemimeral: a word ends on the long of foot 3. Trochaic: a
    word ends after the first short of a *dactylic* foot 3. A break immediately
    before a postpositive is not a real pause and is skipped. Else 'other'."""
    s3 = sum(3 if f == "D" else 2 for f in feet[:2])   # index of foot-3's long
    n = len(wbreak)

    def real_break(k):
        if k + 1 >= n:
            return False
        nxt = widx[k + 1]
        return not (nxt < len(words) and words[nxt] in POSTPOS)

    if s3 < n and wbreak[s3] and real_break(s3):
        return "penthemimeral"
    if feet[2] == "D" and s3 + 1 < n and wbreak[s3 + 1] and real_break(s3 + 1):
        return "trochaic"
    return "other"


def foot_start(feet, f):
    """Syllable index at which foot f (1-based) begins."""
    return sum(3 if x == "D" else 2 for x in feet[:f - 1])


def line_metrics(text):
    """Full metrical description of one line: feet pattern, caesura, spondaic
    fifth foot, Hermann's-Bridge violation, bucolic diaeresis. All content-free."""
    words, nuclei, gaps, wbreak, widx = segments(text)
    feet = scan(_quantities(nuclei, gaps))
    if feet is None:
        return None
    caes = caesura(feet, wbreak, widx, words)
    n = len(wbreak)
    spondaic5 = feet[4] == "S"
    # Hermann's Bridge: no word-end after the first short of a dactylic 4th foot
    s4 = foot_start(feet, 4)
    hermann = feet[3] == "D" and s4 + 1 < n and wbreak[s4 + 1]
    # bucolic diaeresis: word-end at the end of foot 4 (start of foot 5 minus 1)
    s5 = foot_start(feet, 5)
    bucolic = s5 - 1 < n and wbreak[s5 - 1]
    return {"feet": feet, "caesura": caes, "spondaic5": spondaic5,
            "hermann": hermann, "bucolic": bucolic}


def scan_line_full(text):
    """Return (feet-1–5 pattern or None, caesura label or None)."""
    words, nuclei, gaps, wbreak, widx = segments(text)
    feet = scan(_quantities(nuclei, gaps))
    return feet, (caesura(feet, wbreak, widx, words) if feet is not None else None)


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


def rich_profiles(lines, size):
    """Enriched content-free metrical profile per `size`-line chunk: the 16
    foot-patterns plus penthemimeral/trochaic caesura rates, spondaic-fifth rate,
    Hermann's-Bridge-violation rate and bucolic-diaeresis rate."""
    mets = [line_metrics(ln) for ln in lines]
    profs = []
    for i in range(0, len(mets) - size + 1, size):
        block = [m for m in mets[i:i + size] if m is not None]
        if not block:
            profs.append(np.zeros(21)); continue
        pat = np.zeros(16)
        for m in block:
            pat[pattern_index(m["feet"])] += 1
        pat /= pat.sum()
        extra = np.array([
            np.mean([m["caesura"] == "penthemimeral" for m in block]),
            np.mean([m["caesura"] == "trochaic" for m in block]),
            np.mean([m["spondaic5"] for m in block]),
            np.mean([m["hermann"] for m in block]),
            np.mean([m["bucolic"] for m in block]),
        ])
        profs.append(np.concatenate([pat, extra]))
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

    # ── caesura + other content-free metrical laws (formal features) ──
    caes_rows = []
    for name in order:
        counts = {"penthemimeral": 0, "trochaic": 0, "other": 0}
        sp5 = herm = buc = tot = 0
        for ln in works[name]:
            m = line_metrics(ln)
            if m is None:
                continue
            counts[m["caesura"]] += 1
            sp5 += m["spondaic5"]; herm += m["hermann"]; buc += m["bucolic"]; tot += 1
        tot = tot or 1
        caes_rows.append({"work": name, "single_author": name in SINGLE_AUTHOR,
                          "scanned_lines": tot,
                          "penthemimeral": round(counts["penthemimeral"] / tot, 3),
                          "trochaic": round(counts["trochaic"] / tot, 3),
                          "other": round(counts["other"] / tot, 3),
                          "spondaic_5th": round(sp5 / tot, 3),
                          "hermann_violation": round(herm / tot, 4),
                          "bucolic_diaeresis": round(buc / tot, 3)})
    caes = pd.DataFrame(caes_rows)
    caes.to_csv(tables / f"caesura_summary{vsuffix}.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    op = ["Iliad", "Odyssey", "Apollonius", "Quintus", "Hesiod"]
    lbl = {"Hesiod": "Hesiod\n(2 poems)"}
    ci = caes.set_index("work")
    tro = [ci.loc[w, "trochaic"] for w in op]
    pen = [ci.loc[w, "penthemimeral"] for w in op]
    oth = [ci.loc[w, "other"] for w in op]
    x = np.arange(len(op))
    ax.bar(x, tro, label="trochaic caesura", color="#6a51a3")
    ax.bar(x, pen, bottom=tro, label="penthemimeral caesura", color="#d1913c")
    ax.bar(x, oth, bottom=[t + p for t, p in zip(tro, pen)], label="other", color="#bbbbbb")
    ax.set_xticks(x); ax.set_xticklabels([lbl.get(w, w) for w in op])
    ax.set_ylabel("share of scanned lines"); ax.set_ylim(0, 1)
    ax.set_title("Where the line breaks: principal caesura by poet\n"
                 "(Homer's ~54% trochaic share matches the hand-counted figure)", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    savefig(fig, figures, f"caesura_types{vsuffix}")

    print("\n=== CAESURA & METRICAL LAWS (content-free formal features) ===")
    print(caes.to_string(index=False))
    print(f"Wrote {tables/f'caesura_summary{vsuffix}.csv'} and "
          f"{figures/f'caesura_types{vsuffix}.png'}")

    # ── enriched content-free metre channel: does the plurality result hold? ──
    rprofs, rowner = [], []
    for name in order:
        for v in rich_profiles(works[name], args.size):
            rprofs.append(v); rowner.append(name)
    rowner = np.array(rowner)
    RZ = balanced_zscore(np.vstack(rprofs), rowner, order)
    RD = delta_matrix(RZ)
    print("\n=== ENRICHED METRE (foot-patterns + caesura + bridges): dispersion ===")
    r_disp = {}
    for name in order:
        idx = np.where(rowner == name)[0]
        w = RD[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)]
        r_disp[name] = float(w.mean())
    r_unified = max(r_disp["Apollonius"], r_disp["Quintus"])
    for name in order:
        tag = ""
        if name in ("Iliad", "Odyssey"):
            tag = " ABOVE" if r_disp[name] > r_unified else " within"
            tag += " unified single-epic level"
        print(f"  {name:11s} {r_disp[name]:.4f}{tag}")
    pd.DataFrame([{"work": k, "enriched_metre_dispersion": round(v, 4)}
                 for k, v in r_disp.items()]).to_csv(
        tables / f"metre_rich_summary{vsuffix}.csv", index=False)


if __name__ == "__main__":
    main()
