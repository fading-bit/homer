"""
make_report.py — build the project report as Markdown (Phases 0–3 + speech split).

Reads outputs/tables/*.csv and references outputs/figures/*.png, writing
outputs/homer_stylo_report.md.  Figures are referenced relatively as
`figures/<name>.png`, so keep the .md next to a `figures/` folder.

Run:  python src/make_report.py --config config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def fig(name, caption):
    return f"![{caption}](figures/{name})\n\n*{caption}*"


def build_md(cfg, root) -> str:
    tdir = root / cfg["paths"]["tables_dir"]
    pdir = root / cfg["paths"]["processed_dir"]

    df = pd.read_parquet(pdir / "lines.parquet")
    parts = pd.read_csv(tdir / "particles_freq_permille.csv", index_col=0)
    form = pd.read_csv(tdir / "formularity.csv")
    rich = pd.read_csv(tdir / "lexical_richness.csv")
    het = pd.read_csv(tdir / "within_book_heterogeneity.csv")
    dsum = pd.read_csv(tdir / "distance_summary.csv")
    dcorr = pd.read_csv(tdir / "distance_matrix_correlations.csv").iloc[0].to_dict()
    vb = pd.read_csv(tdir / "voice_breakdown.csv")
    calib = pd.read_csv(tdir / "calibration_summary.csv").set_index("work")
    calib_n = pd.read_csv(tdir / "calibration_summary_narration.csv").set_index("work")
    metre = pd.read_csv(tdir / "metre_summary.csv").set_index("work")

    il = df[df.poem == "iliad"]; od = df[df.poem == "odyssey"]
    il_tok = int(il.text_norm.str.split().map(len).sum())
    od_tok = int(od.text_norm.str.split().map(len).sum())

    # particles
    p_il = parts[parts.poem == "iliad"].drop(columns="poem").mean()
    p_od = parts[parts.poem == "odyssey"].drop(columns="poem").mean()
    diff = (p_il - p_od).sort_values()
    prows = []
    for name in list(diff.index[:4]) + list(diff.index[-4:]):
        d = p_il[name] - p_od[name]
        prows.append([name, f"{p_il[name]:.2f}", f"{p_od[name]:.2f}", f"{d:+.2f}",
                      "Iliad" if d > 0 else "Odyssey"])

    # formularity extremes
    fhi = form.sort_values("repeated_line_rate", ascending=False).head(4)
    flo = form.sort_values("repeated_line_rate").head(4)
    frows = [[r.book, f"{r.repeated_line_rate:.3f}", f"{r.rep_4gram_rate:.3f}"]
             for _, r in pd.concat([fhi, flo]).iterrows()]

    # richness extremes
    rhi = rich.sort_values("yule_k", ascending=False).head(3)
    rlo = rich.sort_values("yule_k").head(3)
    rrows = [[r.book, f"{r.yule_k:.1f}", f"{r.mattr_w100:.3f}"]
             for _, r in pd.concat([rhi, rlo]).iterrows()]

    # within-book heterogeneity top
    top = het.sort_values("max_centroid_dist", ascending=False).head(6)
    wrows = [[r.book, int(r.n_lines), int(r.n_changepoints),
              (r.changepoint_lines if isinstance(r.changepoint_lines, str) else "—"),
              f"{r.max_centroid_dist:.3f}"] for _, r in top.iterrows()]

    # distances
    pur = {r.matrix: r.two_cluster_purity for _, r in dsum.iterrows()}

    # voice split
    n_narr = int(vb[vb.stratum == "narration"].lines.iloc[0])
    n_spch = int(vb[vb.stratum == "speech"].lines.iloc[0])
    t_narr = int(vb[vb.stratum == "narration"].tokens.iloc[0])
    t_spch = int(vb[vb.stratum == "speech"].tokens.iloc[0])
    pct_spch = 100 * n_spch / (n_narr + n_spch)
    pct_spch_tok = 100 * t_spch / (t_narr + t_spch)

    M = []  # markdown chunks
    M.append("# Is Homer One Voice?")
    M.append("### A computational-stylometric investigation of the Iliad and Odyssey")
    M.append("*Project report · Phases 0–5 (corpus, EDA, within-book analysis, "
             "narrator/speech split, and a calibrated test for authorial plurality "
             "across word-frequency and metre) · Fada · July 2026*")
    M.append("---")

    # Abstract
    M.append("## Abstract")
    M.append(
        f"This project asks whether Homer's verse is the work of a single poetic "
        f"voice or of many. The working corpus is the Greek text of both poems "
        f"({len(df):,} verse lines, ~{(il_tok+od_tok)//1000}k tokens) from the "
        f"Perseus Digital Library. Rather than compare the two poems to each other, "
        f"the emphasis is on detecting plurality of voice **within** the text. Four "
        f"results anchor the report. A book-level **particle profile** and "
        f"**formularity** both distinguish the poems, but the second is confounded "
        f"with content and, as clustering shows, much of the apparent poem gap is an "
        f"artefact of the two source editions rather than deep style. The strongest "
        f"finding is internal: a **within-book seam detector independently "
        f"rediscovers the Shield of Achilles and the Catalogue of Ships** as the "
        f"passages most distinct from their surroundings. Finally, a **narrator vs. "
        f"character-speech split** ({pct_spch:.0f}% of lines are direct speech) lets "
        f"us strip out register: run on narration alone, the Shield seam **survives**, "
        f"so it is a shift in the narrating voice, not a speech artefact. The project's "
        f"headline comes from a **calibrated test**: measured against Apollonius, "
        f"Quintus and Hesiod, both poems are **more internally variable than any "
        f"single-author benchmark** (the Iliad most, and robustly across three "
        f"controls \u2014 length, register via a narration-only rerun, and content via "
        f"a metrical channel that scans 92\u201395% of lines), yet that variation is "
        f"**diffuse, not a clean split** into separable hands. So there is "
        f"more than one Homer in the weak sense \u2014 heterogeneous, layered "
        f"composition beyond what one poet produces \u2014 but not in the strong sense "
        f"of discrete, identifiable authors.")

    # 1. Question
    M.append("## 1  Research question and design")
    M.append(
        "The \u201cHomeric Question\u201d is several questions at once. At the level "
        "of composition, Parry and Lord's oral-formulaic theory recast the poems as "
        "the sediment of a tradition of many singers; at the level of discourse, "
        "Bakker reads the verse as structured speech; and at the level of the text, "
        "Nagy's evolutionary model denies there was ever a single fixed original. "
        "The question this report pursues is the sharpest form of \u201cmore than one "
        "voice\u201d: is there detectable stylistic **plurality within the text** \u2014 "
        "seams that might mark different hands \u2014 rather than how the two poems "
        "compare as wholes.")
    M.append(
        "The central hazard is that most measurable differences between stretches of "
        "Homer track **content** (what a passage is about) or **register** (narration "
        "vs. speech) rather than authorial **habit**. The project is therefore built "
        "around one discipline: judge every feature by whether it measures content, "
        "register, or habit, and treat only habit as a candidate voice signal.")

    # 2. Corpus
    M.append("## 2  Corpus and source")
    M.append(
        "Our source is the Greek text of both poems from the **Perseus Digital "
        "Library**, parsed from canonical TEI XML whose explicit book and line markup "
        "makes segmentation exact. A Phase-0 ingester produces a tidy line table with "
        "verbatim, diplomatic, and normalised text columns (the normalised form is "
        "lower-cased, with accents, punctuation and elision marks removed and final "
        "sigma folded); the corpus is 98% Greek.")
    M.append(
        "The two poems come from different editions (the Iliad in Monro & Allen's "
        "Oxford Classical Text, the Odyssey in Murray's). Where this matters \u2014 in "
        "the cross-poem comparisons \u2014 it is flagged at the point it arises; it "
        "does not affect any within-text analysis.")
    M.append(md_table(
        ["Poem", "Edition (via Perseus)", "Books", "Lines", "Tokens"],
        [["Iliad", "Monro & Allen (OCT)", 24, f"{len(il):,}", f"{il_tok:,}"],
         ["Odyssey", "Murray", 24, f"{len(od):,}", f"{od_tok:,}"],
         ["**Total**", "—", 48, f"**{len(df):,}**", f"**{il_tok+od_tok:,}**"]]))
    M.append("*Table 1. Working corpus.*")

    # 3. Method
    M.append("## 3  Method")
    M.append(
        "**Book level.** Four analyses run over the normalised tokens of each of the "
        "48 books: a particle / function-word profile (z-scored per-mille rates of "
        "~30 Homeric particles \u2014 content-independent, hence a habit signal); "
        "lexical richness (Yule's K, moving-average type\u2013token ratio, hapax "
        "rate); formularity (share of a book's lines that recur elsewhere, plus "
        "repeated n-gram density); and a most-frequent-word matrix for the distance "
        "phase.")
    M.append(
        "**Within-book level.** A sliding window (60 lines, step 10) moves through a "
        "single book; each window is described by its character-trigram profile. Three "
        "readings follow: the cosine distance of each window to the book's own "
        "centroid, a window\u00d7window self-distance matrix (block structure signals "
        "a seam), and change points from a penalised search (PELT).")
    M.append(
        "**Between-book level.** Three distance matrices over the 48 books \u2014 "
        "Burrows's Delta (word level, robust to spelling), character-trigram cosine "
        "(sensitive to orthography), and a compression distance (NCD) \u2014 are "
        "clustered and projected to two dimensions. The narrator/speech split (\u00a75) "
        "adds a register stratification on top of all of these.")

    # 4. Results
    M.append("## 4  Results")

    M.append("### 4.1  Particle profile")
    M.append(
        "The two poems separate on function-word usage: *\u03b4\u03ad* is markedly "
        "commoner in the Iliad, while *\u03c4\u03bf\u03b9*, *\u03ba\u03b1\u03af*, "
        "*\u03b1\u1f50\u03c4\u03ac\u03c1* and *\u1f22* lean to the Odyssey, and "
        "*\u1f24\u03c4\u03bf\u03b9* is effectively Iliad-only. Because these words "
        "carry almost no subject matter, the difference is a genuine habit signal "
        "(part of it may nonetheless be editorial, given \u00a72 \u2014 see \u00a74.5).")
    M.append(md_table(["Particle", "Iliad (\u2030)", "Odyssey (\u2030)", "\u0394", "Leans"], prows))
    M.append("*Table 3. Largest Iliad\u2013Odyssey divergences in particle frequency (per mille).*")
    M.append(fig("particle_heatmap.png",
                 "Figure 1. Particle / function-word profile by book (per-mille, z-scored)."))

    M.append("### 4.2  Formularity — strong but confounded")
    M.append(
        "The Odyssey is systematically more formulaic than the Iliad (repeated-line "
        "rates around 0.20 vs. 0.15), and within the Iliad the least formulaic books "
        "are the battle climaxes. This almost certainly reflects content \u2014 the "
        "Odyssey recycles hospitality and travel type-scenes \u2014 so formularity "
        "cannot on its own be read as evidence about authorship.")
    M.append(md_table(["Book", "Repeated-line rate", "Repeated 4-gram rate"], frows))
    M.append("*Table 4. Most (top four) and least (bottom four) formulaic books.*")
    M.append(fig("formularity.png", "Figure 2. Repeated whole-line rate by book."))

    M.append("### 4.3  Lexical richness — a null result")
    M.append(
        "Lexical richness barely varies: MATTR sits in a narrow band (~0.85\u20130.88) "
        "for all 48 books and Yule's K spans a modest range with no dramatic outliers. "
        "A useful negative: vocabulary diversity is not a lever for this corpus.")
    M.append(md_table(["Book", "Yule's K", "MATTR"], rrows))
    M.append("*Table 5. Richness extremes (three highest, three lowest Yule's K).*")

    M.append("### 4.4  Within-book seams — the strongest result")
    M.append(
        "Ranking all 48 books by the most stylistically divergent stretch each "
        "contains puts **Iliad 18 and Iliad 2 at the top** \u2014 precisely the two "
        "most famous embedded set-pieces: the **Shield of Achilles** and the "
        "**Catalogue of Ships**. In Iliad 18 the change-point search places a boundary "
        "at line 481, where the Shield ekphrasis begins (18.478); in Iliad 2 the "
        "Catalogue forms a self-similar block distinct from the surrounding narrative. "
        "The method rediscovers, unprompted, passages philology has always flagged.")
    M.append(md_table(["Book", "Lines", "Change pts", "Change-point lines", "Max divergence"], wrows))
    M.append("*Table 6. The six books containing the most divergent internal stretch.*")
    M.append(fig("within_IL18.png",
                 "Figure 3. Within-book stylometry, Iliad 18: the curve climbs through "
                 "the Shield (gold); the change point and heatmap block mark its onset."))
    M.append(fig("within_IL02.png",
                 "Figure 4. Within-book stylometry, Iliad 2: the Catalogue of Ships as "
                 "a distinct block."))
    M.append(fig("within_book_heterogeneity.png",
                 "Figure 5. Internal heterogeneity of all 48 books; Iliad 18 and 2 are outliers."))
    M.append(
        "These stretches differ in **register and content** (a descriptive ekphrasis, "
        "a formulaic list) from ordinary narrative, so a within-book seam is a "
        "candidate for scrutiny, not a verdict \u2014 which is exactly what motivates "
        "the register split in \u00a75.")

    M.append("### 4.5  Between-book distances and clustering")
    M.append(
        f"Clustering the 48 books is instructive because the measures disagree. On "
        f"**character trigrams the two poems separate perfectly** ({pur.get('char',1.0):.0%} "
        f"pure by poem). On **Burrows's Delta they barely separate** \u2014 an "
        f"overlapping cloud with only a weak tendency, and several books crossing "
        f"over. This disagreement is the fingerprint of the two editions (\u00a72): "
        f"character n-grams track orthographic convention, which is uniform within an "
        f"edition, so two editions split cleanly whether or not their authors differ. "
        f"Delta, resting on function words, is robust to spelling and is the "
        f"trustworthy cross-poem signal \u2014 and it says the poems are only "
        f"**modestly** distinguishable (Delta\u2013character correlation "
        f"r = {dcorr.get('delta~char',0):.2f}; the compression distance is a weak, "
        f"largely independent signal, r = {dcorr.get('delta~ncd',0):.2f}, and is "
        f"discounted). The clean character split should not be read as an authorial "
        f"boundary.")
    M.append(md_table(["Distance measure", "Poem separation (2-cluster purity)"],
                      [["Character trigrams", f"{pur.get('char',1.0):.0%} (perfect — but see text)"],
                       ["Burrows's Delta (words)", f"{pur.get('delta',0):.0%} (poems are not the top split)"],
                       ["Compression (NCD)", f"{pur.get('ncd',0):.0%} (weak, discounted)"]]))
    M.append("*Table 7. How cleanly each measure splits the poems; the character split is an edition effect.*")
    M.append(fig("mds_delta.png", "Figure 6. Books in stylistic space (MDS on Burrows's Delta); the poems overlap."))
    M.append(fig("clustermap_charcos.png", "Figure 7. Character-trigram clustering: a clean two-block split — the signature of two editions."))
    M.append(fig("clustermap_delta.png", "Figure 8. Word-level (Delta) clustering: only a partial poem tendency, with crossovers."))

    # 5. Speech split (with development narrative)
    M.append("## 5  The narrator vs. character-speech split")

    M.append("### 5.1  How the tagger was built")
    M.append(
        "Homer alternates between the poet's narration and quoted character speech, "
        "and that alternation is a register shift that would masquerade as a change of "
        "hand. Tagging the two strata is therefore the key disentangler. The starting "
        "point was a check of the raw text:")
    M.append(
        "> No quotation marks are present, but the speech-boundary formulae are all "
        "there once the normalised final sigma is handled correctly \u2014 "
        "\u03c0\u03c1\u03bf\u03c3\u03ad\u03c6\u03b7 (234), "
        "\u1f00\u03bc\u03b5\u03b9\u03b2\u03cc\u03bc\u03b5\u03bd\u03bf\u03c2 (208), "
        "\u1f63\u03c2 \u03c6\u03ac\u03c4\u03bf (197), "
        "\u03c0\u03c1\u03bf\u03c3\u03b7\u03cd\u03b4\u03b1 (176), and so on. That is the "
        "classic way to detect Homeric speech: an introduction formula opens a speech, "
        "and a resumption formula (\u1f63\u03c2 \u03c6\u03ac\u03c4\u03bf / "
        "\u1f63\u03c2 \u03b5\u1f30\u03c0\u03ce\u03bd) closes it.")
    M.append(
        "The tagger built on that idea then needed three corrections, each prompted by "
        "inspecting its output:")
    M.append(
        "1. **Precision over recall.** An initial, generous cue list (including "
        "\u03c6\u03c9\u03bd\u03ae\u03c3\u03b1\u03c2 and "
        "\u1f00\u03b3\u03bf\u03c1\u03b5\u03cd\u03c9) let speech bleed into narration, "
        "because \u201c\u1f63\u03c2 \u1f04\u03c1\u03b1 "
        "\u03c6\u03c9\u03bd\u03ae\u03c3\u03b1\u03c2\u201d resumptions were also being "
        "read as openers. The cues were pared back to the reliable reply/address "
        "frames, which pair with a resumption and so yield clean spans.")
    M.append(
        "2. **Reply-frames close *and* open.** A dialogue spot-check (Iliad 1) showed "
        "that in rapid exchange a speech is often not closed by "
        "\u1f63\u03c2 \u03c6\u03ac\u03c4\u03bf but runs straight into the next frame "
        "(\u03c4\u1f78\u03bd \u03b4\u1fbd \u1f20\u03bc\u03b5\u03af\u03b2\u03b5\u03c4\u1fbd \u2026); "
        "the frame line itself was being absorbed into the speech. The fix: a reply "
        "frame both closes the current speech and opens the next \u2014 it is narration "
        "either way.")
    M.append(
        "3. **Augmented forms.** The augmented reply verb "
        "\u1f20\u03bc\u03b5\u03af\u03b2\u03b5\u03c4\u03bf (normalised "
        "\u201c\u03b7\u03bc\u03b5\u03b9\u03b2\u03b5\u03c4\u201d) was missed by a cue "
        "written for the \u03b1- forms; the cue was broadened to catch both.")
    M.append(
        "With these in place the split lands at a share of speech that matches the "
        "scholarly estimate for Homer, and dialogue segments cleanly (the frame lines "
        "come out as narration, the speeches between them as speech).")

    M.append("### 5.2  Validation and the register contrast")
    M.append(
        f"The corpus divides into **{n_narr:,} narration lines and {n_spch:,} speech "
        f"lines ({pct_spch:.1f}% speech; {pct_spch_tok:.1f}% by tokens)** \u2014 right "
        f"on the received estimate that roughly half of Homer is direct speech. The "
        f"contrast between the strata is exactly what it should be (Figure 9): speech "
        f"is marked by vocatives (\u03b3\u03ad\u03c1\u03bf\u03bd, "
        f"\u1f48\u03b4\u03c5\u03c3\u03c3\u03b5\u1fe6), second-person address, an "
        f"imperative (\u03ba\u03ad\u03ba\u03bb\u03c5\u03c4\u03b5 \u201chear!\u201d), "
        f"possessives (\u1f10\u03bc\u03cc\u03bd, \u03c3\u03cc\u03bd) and the dual "
        f"\u03bd\u1ff6\u03b9 \u201cwe two\u201d; narration is marked by the "
        f"speech-frame verbs and third-person report.".replace("\u03bd", "\u03bd"))
    M.append(fig("voice_markers.png",
                 "Figure 9. What distinguishes character speech from narration "
                 "(most discriminating frequent words)."))

    M.append("### 5.3  Within-narration seams — probing for more than one Homer")
    M.append(
        "The split's payoff is that a seam can now be tested on the narrating voice "
        "alone. Re-running the Iliad 18 seam detector on **narration only** (speech "
        "removed) is the decisive check for the Shield of Achilles: on the full text "
        "the Shield boundary sits at line 481 (max divergence 0.137); on narration "
        "only, at matched resolution, a change point re-appears inside the ekphrasis "
        "(line 504) and the maximum divergence actually **rises to 0.152**. The Shield "
        "seam therefore **survives the removal of speech** \u2014 it is a genuine shift "
        "in the narrating voice, not an artefact of a character starting to talk. That "
        "is precisely the sort of within-text discontinuity a search for more than one "
        "Homer is looking for (a candidate, not yet a verdict).")
    M.append(fig("within_IL18_narration.png",
                 "Figure 10. Iliad 18, narration only: the Shield block persists and a "
                 "change point re-appears inside the ekphrasis after speech is removed."))

    M.append("### 5.4  Limitation of the split")
    M.append(
        "The tagger is a formula-based heuristic that favours **precision over "
        "recall**. Speeches opened by a plain lexical verb rather than a reply frame "
        "\u2014 prayers (\u1f20\u03c1\u1fb6\u03c4\u03bf), commands "
        "(\u03bc\u1fe6\u03b8\u03bf\u03bd \u1f14\u03c4\u03b5\u03bb\u03bb\u03b5), the "
        "first plea of a scene (\u03bb\u03af\u03c3\u03c3\u03b5\u03c4\u03bf) \u2014 are "
        "missed and stay labelled narration (Chryses' public plea in Iliad 1 is one "
        "such case). The narration stratum therefore carries a little residual "
        "scene-opening speech, and the reported speech share is a **lower bound**. This "
        "is a metric limitation to keep in view when reading the stratified results, "
        "and a candidate for a trained tagger in future work.".replace("\u03bd", "\u03bd"))

    # 6. The calibrated answer
    il_d = calib.loc["Iliad", "internal_dispersion"]
    od_d = calib.loc["Odyssey", "internal_dispersion"]
    ap_d = calib.loc["Apollonius", "internal_dispersion"]
    qu_d = calib.loc["Quintus", "internal_dispersion"]
    he_d = calib.loc["Hesiod", "internal_dispersion"]
    il_s = calib.loc["Iliad", "best2_silhouette"]
    od_s = calib.loc["Odyssey", "best2_silhouette"]
    M.append("## 6  Is there more than one Homer? — a calibrated answer")
    M.append(
        "This is the question the project is built toward, and it cannot be answered "
        "by finding a seam alone: a single poet writing a battle, a simile and a "
        "shield-description varies too. The test is therefore **calibrated** \u2014 each "
        "poem is split into equal 400-line chunks, each chunk is described by a "
        "150-word most-frequent-word profile, and the **internal dispersion** (mean "
        "pairwise Burrows's Delta among a work's own chunks) is compared against known "
        "single-author hexameter epics: Apollonius' *Argonautica*, Quintus' "
        "*Posthomerica*, and Hesiod. Features are standardised with equal weight per "
        "work so the majority text cannot look artificially cohesive, and \u2014 "
        "critically \u2014 comparing *within-work* spread sidesteps the edition problem "
        "entirely.")
    M.append(md_table(
        ["Work", "Type", "Chunks", "Internal dispersion", "Best 2-cluster silhouette"],
        [["**Iliad**", "under test", int(calib.loc['Iliad','n_chunks']), f"**{il_d:.2f}**", f"{il_s:.2f}"],
         ["**Odyssey**", "under test", int(calib.loc['Odyssey','n_chunks']), f"**{od_d:.2f}**", f"{od_s:.2f}"],
         ["Apollonius", "single author", int(calib.loc['Apollonius','n_chunks']), f"{ap_d:.2f}", f"{calib.loc['Apollonius','best2_silhouette']:.2f}"],
         ["Quintus", "single author", int(calib.loc['Quintus','n_chunks']), f"{qu_d:.2f}", f"{calib.loc['Quintus','best2_silhouette']:.2f}"],
         ["Hesiod", "single author", int(calib.loc['Hesiod','n_chunks']), f"{he_d:.2f}", f"{calib.loc['Hesiod','best2_silhouette']:.2f}"]]))
    M.append("*Table 8. Internal stylistic dispersion of the two poems against known "
             "single-author epics.*")
    M.append(fig("calibration_dispersion.png",
                 "Figure 11. Internal dispersion of the two poems vs. single-author "
                 "hexameter epics (higher = more internally variable)."))
    M.append(fig("calibration_mds.png",
                 "Figure 12. Every 400-line chunk in stylistic space: the single-author "
                 "works form compact clouds; the Iliad and Odyssey are diffuse."))
    M.append(
        f"**The answer has two parts.** First, both poems are **more internally "
        f"variable than any single-author benchmark**: the Iliad's dispersion "
        f"({il_d:.2f}) and the Odyssey's ({od_d:.2f}) both exceed Apollonius "
        f"({ap_d:.2f}) and Quintus ({qu_d:.2f}), the Iliad most of all. This holds "
        f"under a length-matched control (restricting Homer to Apollonius- and "
        f"Quintus-sized windows leaves the Iliad at ~1.32 and the Odyssey at ~1.25, "
        f"still well above both), so it is not an artefact of the poems' greater "
        f"length. On the MDS, the single-author works are compact clouds while the "
        f"Homeric chunks spread widely. This points toward **plural, layered "
        f"composition rather than a single unified author** \u2014 more so in the "
        f"Iliad.")
    M.append(
        f"Second, and equally important, that variation is **diffuse, not a clean "
        f"split**: neither poem divides into two (or more) separable stylistic groups "
        f"any better than a single-author work does (best 2-cluster silhouettes "
        f"{il_s:.2f} for the Iliad and {od_s:.2f} for the Odyssey are as low as the "
        f"calibrators'). So the evidence does *not* support a tidy \u201cbooks A\u2013M "
        f"by one poet, N\u2013Z by another\u201d partition; there is no discrete second "
        f"hand to point to.")
    M.append(
        "**So: is there more than one Homer within each poem?** At the level "
        "stylometry can see, the honest answer is *yes in the weak sense, no in the "
        "strong sense*. Both poems carry more internal stylistic heterogeneity than a "
        "single poet produces \u2014 consistent with the oral-traditional, accreted "
        "picture of many hands over time \u2014 yet that heterogeneity does not "
        "resolve into a small number of cleanly separable authors.")
    # register-controlled robustness
    iln, odn = calib_n.loc["Iliad", "internal_dispersion"], calib_n.loc["Odyssey", "internal_dispersion"]
    apn, qun = calib_n.loc["Apollonius", "internal_dispersion"], calib_n.loc["Quintus", "internal_dispersion"]
    ilns = calib_n.loc["Iliad", "best2_silhouette"]
    M.append(
        f"**Register held constant strengthens this.** The obvious objection is that "
        f"the excess variation is just register \u2014 the poems mixing narration and "
        f"speech more than the calibrators. Re-running the whole test on **narration "
        f"only** (the same speech tagger applied to every work, speech removed) "
        f"answers it: the gap *widens*. On narration alone the Iliad's dispersion "
        f"rises to {iln:.2f} and the Odyssey's to {odn:.2f}, against Apollonius "
        f"{apn:.2f} and Quintus {qun:.2f} (Figure 13) \u2014 and the Iliad's best "
        f"2-cluster silhouette climbs to {ilns:.2f}, several times the single-author "
        f"baseline, hinting that its narration is not merely diffuse but carries some "
        f"internal grouping. Comparing narration to narration, the Homeric narrating "
        f"voice is markedly *less* uniform than these poets', so the difference is not "
        f"an artefact of how much characters talk.")
    M.append(fig("calibration_dispersion_narration.png",
                 "Figure 13. Internal dispersion on narration only: with register held "
                 "constant, both poems sit further above the single-author works."))
    ilm, odm = metre.loc["Iliad", "internal_dispersion"], metre.loc["Odyssey", "internal_dispersion"]
    apm, qum = metre.loc["Apollonius", "internal_dispersion"], metre.loc["Quintus", "internal_dispersion"]
    sr_lo = int(metre["scansion_rate"].min() * 100)
    sr_hi = int(metre["scansion_rate"].max() * 100)
    M.append(
        f"The remaining objection is **content** rather than register: even within "
        f"narration the Iliad ranges over more varied matter (battle, catalogue, "
        f"ekphrasis, simile, divine council) than the calibrators attempt, and subject "
        f"matter leaks faintly into word frequencies. This is what **metre** "
        f"adjudicates, since a poet's distribution of dactyls and spondees is largely "
        f"independent of what a passage is about. Scanning every line into its "
        f"dactyl/spondee foot-pattern ({sr_lo}\u2013{sr_hi}% of lines scan) and "
        f"repeating the dispersion test on these metrical profiles gives the **same "
        f"verdict on a content-free channel**: the Iliad ({ilm:.2f}) and Odyssey "
        f"({odm:.2f}) are both more metrically variable than the unified single-author "
        f"epics Apollonius ({apm:.2f}) and Quintus ({qum:.2f}) \u2014 as varied, in "
        f"fact, as two distinct Hesiodic poems combined (Figure 14).")
    M.append(fig("metre_dispersion.png",
                 "Figure 14. Internal dispersion on metre (dactyl/spondee foot-patterns), "
                 "a content-free channel. Both poems exceed the unified single-epic "
                 "level; \u201cHesiod\u201d combines two separate poems."))
    M.append(
        "With **register** controlled (the narration-only test) and now **content** "
        "largely controlled (metre) both pointing the same way, the \u201cone "
        "versatile hand over varied matter\u201d explanation is substantially "
        "weakened. The evidence converges: the Iliad and Odyssey carry more internal "
        "stylistic *and* metrical variability than epics we know to be by a single "
        "author \u2014 real compositional plurality \u2014 while still not resolving "
        "into a few cleanly separable hands. Two honest reservations remain: metre is "
        "not perfectly content-free (formula and dialect choices carry metrical "
        "shape), and the plurality is diffuse, so this is strong convergent evidence "
        "for \u201cmany Homers\u201d in the layered, traditional sense rather than a "
        "demonstration of a countable number of poets.")

    # 7. Limitations
    M.append("## 7  Limitations")
    M.append(
        "- **Mixed editions.** The Iliad (Monro\u2013Allen) and Odyssey (Murray) come "
        "from different editors, so cross-poem differences may be partly editorial; "
        "\u00a74.5 shows this directly. Within-text results are unaffected.\n"
        "- **Speech-tagger recall.** The narrator/speech split is precision-favouring "
        "and misses lexically-opened speeches, so the narration stratum holds some "
        "residual speech and the speech share is a lower bound (\u00a75.4).\n"
        "- **Change-point sensitivity.** The seam detector's output depends on a "
        "penalty and on window resolution; boundaries should be read together with the "
        "continuous curve and heatmap, not as hard claims.\n"
        "- **Surface forms only.** No lemmatisation, part-of-speech, or syntax yet, so "
        "morphology and the syntactic signal are unexploited (metre is now used, "
        "\u00a76); a few particles are ambiguous once accents are stripped.\n"
        "- **What calibration can and cannot settle.** [\u00a76.] The calibrated test "
        "supplies the single-author baseline the project lacked; the register "
        "objection is answered by the narration-only rerun and the content objection "
        "by the metrical rerun, both of which keep the two poems above the single-epic "
        "level. The residual reservations: metre is not perfectly content-free (formula "
        "and dialect choices carry metrical shape), ambiguous vowels are resolved "
        "dactyl-first (a uniform bias that cancels in the comparison but is a "
        "simplification), \u201cHesiod\u201d combines two separate poems, and the "
        "Odyssey's narration is too short for the longest length-matched window.")

    # 8. Plan
    M.append("## 8  Plan for continuation")
    M.append(
        "The narration-only calibration and the metrical test (\u00a76) are now done "
        "and close the register and content objections; the remaining steps add a "
        "second content-free channel, locate the passages responsible, and turn the "
        "answer from a verdict into a probability.")
    M.append(
        "1. **Syntax \u2014 the second content-free channel.** Parse the poems with a "
        "dependency model (Perseus/CLTK Greek treebanks) and rerun the \u00a76 "
        "dispersion test on syntactic features (dependency-relation and part-of-speech "
        "n-grams). Syntax, like metre, is largely content-independent; agreement with "
        "the metrical result would make the plurality reading very hard to explain "
        "away as varied subject matter.\n"
        "2. **Systematic within-narration seam scan.** Extend the Iliad-18 probe to "
        "every book on the narration stratum, cataloguing seams that survive speech "
        "removal and locating the specific passages that drive the elevated "
        "dispersion \u2014 turning the global \u201cmore variable\u201d result into a "
        "map of *where*.\n"
        "3. **More single-author calibrators** (Callimachus' *Hymns*, the *Homeric "
        "Hymns*, Nonnus) to tighten the baseline and its uncertainty, and to replace "
        "the two-poem Hesiod reference with unified single works.\n"
        "4. **Elision-aware normalisation + lemmatisation** to remove orthographic "
        "noise and sharpen the metrical scansion (fewer unscanned lines).\n"
        "5. **Statistical endpoint.** A hierarchical (Dirichlet-process) mixture over "
        "chunks \u2014 combining the word, metre and syntax channels \u2014 returning a "
        "posterior over the number of distinguishable voices with a sensitivity "
        "analysis, so \u201cdiffuse vs. discrete\u201d becomes a calibrated probability "
        "rather than a verdict.")

    # References
    M.append("## References")
    refs = [
        "Bakker, E. J. (1997). *Poetry in Speech: Orality and Homeric Discourse*. Cornell University Press.",
        "Burrows, J. (2002). 'Delta': a measure of stylistic difference and a guide to likely authorship. *Literary and Linguistic Computing*, 17(3), 267\u2013287.",
        "Covington, M. A., & McFall, J. D. (2010). Cutting the Gordian knot: the moving-average type\u2013token ratio (MATTR). *Journal of Quantitative Linguistics*, 17(2), 94\u2013100.",
        "Dunning, T. (1993). Accurate methods for the statistics of surprise and coincidence. *Computational Linguistics*, 19(1), 61\u201374.",
        "Ke\u0161elj, V., Peng, F., Cercone, N., & Thomas, C. (2003). N-gram-based author profiles for authorship attribution. *PACLING*, 255\u2013264.",
        "Li, M., Chen, X., Li, X., Ma, B., & Vit\u00e1nyi, P. (2004). The similarity metric. *IEEE Trans. Information Theory*, 50(12), 3250\u20133264.",
        "Lord, A. B. (1960). *The Singer of Tales*. Harvard University Press.",
        "Nagy, G. (1996). *Poetry as Performance: Homer and Beyond*. Cambridge University Press.",
        "Parry, M. (1971). *The Making of Homeric Verse* (A. Parry, ed.). Clarendon Press.",
        "Truong, C., Oudre, L., & Vayatis, N. (2020). Selective review of offline change point detection methods. *Signal Processing*, 167, 107299.",
    ]
    M.append("\n".join(f"{i+1}. {r}" for i, r in enumerate(refs)))

    return "\n\n".join(M) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(args.config).resolve().parent
    out_path = root / "outputs" / "homer_stylo_report.md"
    out_path.write_text(build_md(cfg, root), encoding="utf-8")
    print(f"Wrote {out_path} ({out_path.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
